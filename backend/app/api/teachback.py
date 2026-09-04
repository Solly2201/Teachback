"""The guided TeachBack session flow.

start   -> create a session and a concept-by-concept conversation plan; return
           the first short question plus the concept timeline
respond -> NLP-analyse the (short) answer, let the rule-based conversation
           engine pick encouraging feedback and the next adaptive question
finish  -> combine NLP features + self-reports into an observation vector,
           run HMM inference over the student's whole history, and return the
           estimated learning state, session summary and explained
           recommendation
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..hmm.model import hmm_available, infer_sequence
from ..models import Observation, Quiz, Response, Student, TeachSession, Topic
from ..nlp.analyzer import (analyze_response, is_term_list,
                            merge_session_analyses, targeted_concept_check)
from ..llm.settings import generative_probes_enabled
from ..nlp.conversation import (MAX_QUESTIONS, _update_relationships, build_plan,
                                first_question, play_turn, timeline_out)
from ..probe.generate import maybe_generate_probe
from ..recommend.rules import recommend
from ..states import FEATURE_NAMES, STATE_NAMES
from .helpers import (DEMONSTRATED, NEEDS_CLARIFICATION, NOT_DISCUSSED,
                      observation_out, relationship_summary, student_state_label,
                      topic_def)

router = APIRouter(prefix="/api/sessions", tags=["teachback"])

# The teacher has finished evaluating this topic and its raw responses have
# been deleted; nothing new may be recorded against it (see api/topics.py).
EVALUATION_CLOSED = "The evaluation for this lecture is closed. No new TeachBack can be recorded."


class StartIn(BaseModel):
    student_id: int
    topic_id: int


class RespondIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class FinishIn(BaseModel):
    attention: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=10)
    difficulty: float = Field(ge=0, le=10)
    # the student's own lecture takeaway (optional, never penalised)
    summary: str = Field(default="", max_length=5000)
    # fast lecture feedback for the teacher
    pace: str = Field(default="", max_length=20)
    feedback_choices: list[str] = Field(default_factory=list)
    feedback_text: str = Field(default="", max_length=2000)


def _prior_posterior(db: Session, student_id: int) -> list[float] | None:
    """The student's HMM state posterior from their history so far, for the
    experimental probe controller's difficulty choice. Read-only: mid-session
    generation never writes an observation or changes any HMM behavior, and
    a student with no history (or no trained model) simply yields None."""
    try:
        if not hmm_available():
            return None
        history = (
            db.query(Observation)
            .filter(Observation.student_id == student_id)
            .order_by(Observation.created_at, Observation.id)
            .all()
        )
        if not history:
            return None
        return infer_sequence([o.features for o in history])["current_posterior"]
    except Exception:
        return None


def _progress_out(plan: dict) -> dict:
    total = len(plan["concepts"]) or 1
    done_or_seen = sum(1 for c in plan["concepts"] if c["status"] != "pending")
    concept_no = min(plan["current"] + 1, total) if not plan["done"] else total
    return {
        "timeline": timeline_out(plan),
        "concept_no": concept_no,
        "total_concepts": total,
        "concepts_visited": done_or_seen,
        "question_no": plan["question_no"],
        "max_questions": MAX_QUESTIONS,
    }


@router.post("/start")
def start_session(data: StartIn, db: Session = Depends(get_db)):
    student = db.get(Student, data.student_id)
    topic = db.get(Topic, data.topic_id)
    if not student or not topic:
        raise HTTPException(404, "Student or topic not found")
    if topic.archived_at is not None:
        # the lecture behind this topic was removed; existing sessions stay
        # readable, but no new one may start on retired material
        raise HTTPException(400, "This lecture is no longer available for TeachBack.")
    if topic.evaluation_closed_at is not None:
        raise HTTPException(400, EVALUATION_CLOSED)

    tdef = topic_def(topic)
    plan = build_plan(tdef)
    question = first_question(plan, tdef)

    session = TeachSession(student_id=student.id, topic_id=topic.id, plan=plan)
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "session_id": session.id,
        "student": {"id": student.id, "name": student.name},
        "topic": tdef,
        "intro": (
            f"Let's see what you took away from {topic.name}. You don't need textbook definitions — "
            "explain things in your own words, and short answers are completely fine. "
            "I'll ask simple questions one at a time."
        ),
        "prompt": question["text"],
        "question": question,
        **_progress_out(plan),
    }


@router.post("/{session_id}/respond")
def respond(session_id: int, data: RespondIn, db: Session = Depends(get_db)):
    session = db.get(TeachSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session.completed:
        raise HTTPException(400, "Session already completed")
    if session.topic is not None and session.topic.evaluation_closed_at is not None:
        raise HTTPException(400, EVALUATION_CLOSED)
    plan = session.plan or build_plan(topic_def(session.topic))
    if plan.get("done"):
        raise HTTPException(400, "The conversation is over; finish the session")

    tdef = topic_def(session.topic)
    analysis = analyze_response(data.text, tdef)

    # short answers rely on the question for context — evaluate the answer
    # against the concept currently being asked about with that context
    if plan.get("concepts") and plan.get("asked_kind") != "extension":
        cur = plan["concepts"][plan["current"]]
        cdef = next(
            (c for c in tdef["concepts"]
             if (cur.get("id") is not None and c.get("id") == cur["id"]) or c["name"] == cur["name"]),
            None,
        )
        if cdef:
            # sibling_names lets the check tell "explaining this concept"
            # apart from "listing the lecture's other headings": a bare list
            # of the topic's own labels is naming, not explaining.
            analysis["target_check"] = targeted_concept_check(
                data.text, cdef, topic_name=tdef.get("name", ""),
                misconceptions=tdef.get("misconceptions"),
                sibling_names=[c["name"] for c in tdef["concepts"]])

    prev = session.responses[-1].analysis.get("turn", {}).get("followup") if session.responses else None
    prompt = (prev or {}).get("text") or (first_question(plan, tdef)["text"])

    plan, turn = play_turn(plan, analysis, tdef)

    # EXPERIMENTAL (off by default): swap the deterministic follow-up's
    # wording for an LLM-generated, teacher-grounded probe. Only the TEXT of
    # the question changes — the plan, the evidence pipeline and the HMM are
    # untouched, and any failure keeps the v1 question. The generated text
    # is a question, not an answer, so exactly like v1 questions it is never
    # analyzed as student evidence (only data.text ever reaches the analyzer).
    if turn.get("followup") and generative_probes_enabled():
        generated = maybe_generate_probe(
            plan=plan, topic_def=tdef, followup=turn["followup"],
            student_answer=data.text,
            posterior=_prior_posterior(db, session.student_id))
        if generated:
            turn["followup"] = {**turn["followup"], "text": generated["question"],
                                "generated": generated["meta"]}

    session.plan = plan
    session.exchange_count += 1

    stored = dict(analysis)
    stored["turn"] = {k: turn.get(k) for k in
                      ("feedback", "followup", "misconception", "resolved_misconception", "incidental")}
    resp = Response(
        session_id=session.id,
        exchange_no=session.exchange_count,
        prompt=prompt,
        text=data.text,
        analysis=stored,
    )
    db.add(resp)
    db.commit()

    return {
        "session_id": session.id,
        "feedback": turn.get("feedback"),
        "followup": turn.get("followup"),
        "misconception": turn.get("misconception"),
        "resolved_misconception": turn.get("resolved_misconception"),
        "incidental": turn.get("incidental", []),
        "closing": turn.get("closing"),
        "awaiting_self_report": turn["done"],
        "analysis": {
            "concepts": analysis["concepts"],
            "misconceptions": analysis["misconceptions"],
            "detected_misconceptions": analysis["detected_misconceptions"],
            "features": analysis["features"],
            "word_count": analysis["word_count"],
        },
        **_progress_out(plan),
    }


def _apply_summary_to_plan(plan: dict, analysis: dict) -> tuple[list[str], list[str]]:
    """Fold the student's own summary into the conversation evidence.

    Upgrade-only: the summary can add or strengthen evidence for a concept or
    relationship, but it never downgrades what the conversation established —
    a short summary is never a penalty. Returns (upgraded, mentioned) names.
    """
    rank = {"pending": 0, "unclear": 0, "partial": 1, "covered": 2}
    upgraded, mentioned = [], []
    before = {id(entry): entry.get("status") for entry in plan.get("concepts", [])}
    for entry in plan.get("concepts", []):
        res = next(
            (c for c in analysis.get("concepts", [])
             if (entry.get("id") is not None and c.get("id") == entry["id"]) or c["name"] == entry["name"]),
            None,
        )
        if res is None or res["status"] == "missing":
            continue
        mentioned.append(entry["name"])
        new_rank = {"covered": 2, "partial": 1}[res["status"]]
        if new_rank > rank.get(entry["status"], 0):
            if res["status"] == "covered":
                upgraded.append(entry["name"])
            entry["status"] = res["status"]
            # provenance: this evidence came from the takeaway summary, not
            # from the conversation. Summary evidence still counts, but the
            # student and the teacher can see which is which.
            entry["evidence_source"] = ("summary" if before[id(entry)] in (None, "pending")
                                        else "teachback+summary")
    # relationship evidence accumulates the same way it does per turn
    _update_relationships(plan, analysis)
    return upgraded, mentioned


@router.post("/{session_id}/finish")
def finish(session_id: int, data: FinishIn, db: Session = Depends(get_db)):
    session = db.get(TeachSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.responses:
        raise HTTPException(400, "No responses submitted yet")
    if session.completed:
        raise HTTPException(400, "Session already completed")
    if session.topic is not None and session.topic.evaluation_closed_at is not None:
        raise HTTPException(400, EVALUATION_CLOSED)
    if not hmm_available():
        raise HTTPException(503, "HMM model not trained yet - run scripts/build_all.py")

    analyses = [r.analysis for r in session.responses]
    tdef = topic_def(session.topic)
    plan = session.plan or {}
    # deep-copy so SQLAlchemy notices the JSON column change on reassignment
    plan = {**plan, "concepts": [dict(c) for c in plan.get("concepts", [])],
            "relationships": [dict(r) for r in plan.get("relationships", [])]}

    # --- the student's own lecture takeaway: another evidence source ---
    summary_text = (data.summary or "").strip()
    summary_insights: dict = {}
    summary_analysis = None
    # A takeaway that is a bare list of the lecture's own terms
    # ("backpropagation gradient weight loss optimization") repeats vocabulary
    # without saying anything, and must not manufacture evidence. It is still
    # stored and shown back to the student — it just adds nothing.
    if summary_text and not is_term_list(summary_text):
        summary_analysis = analyze_response(summary_text, tdef)
        upgraded, mentioned = _apply_summary_to_plan(plan, summary_analysis)
        summary_insights = {
            "concepts_mentioned": mentioned,
            "new_concepts_demonstrated": upgraded,
            "relationships_demonstrated": [
                f"{r['source']} → {r['target']}" for r in summary_analysis.get("relationships", [])
                if r["status"] == "demonstrated"
            ],
            "misconceptions": summary_analysis.get("detected_misconceptions", []),
        }
    session.plan = plan
    session.summary_text = summary_text
    session.summary_insights = summary_insights
    session.pace = data.pace
    session.feedback_choices = data.feedback_choices
    session.feedback_text = data.feedback_text

    nlp_feats = merge_session_analyses(analyses)
    if summary_analysis is not None:
        # the summary may only ADD evidence: coverage takes the cumulative
        # best including the summary; a short summary never lowers features
        with_summary = merge_session_analyses(analyses + [summary_analysis])
        nlp_feats["concept_coverage"] = max(nlp_feats["concept_coverage"],
                                            with_summary["concept_coverage"])
        nlp_feats["misconception_score"] = max(
            nlp_feats["misconception_score"],
            summary_analysis["features"]["misconception_score"])

    detected = sorted({m for a in analyses for m in a.get("detected_misconceptions", [])}
                      | set(summary_insights.get("misconceptions", [])))
    resolved = sorted(set(plan.get("resolved", [])))
    # unresolved misconceptions count fully against the student; a misconception
    # corrected during the session is attenuated rather than kept at its peak —
    # the correction itself is evidence of understanding
    unresolved = [m for m in detected if m not in resolved]
    miscon_score = nlp_feats["misconception_score"]
    if detected:
        miscon_score = round(miscon_score * (0.3 + 0.7 * len(unresolved) / len(detected)), 3)

    features = [
        nlp_feats["concept_coverage"],
        nlp_feats["semantic_correctness"],
        miscon_score,
        nlp_feats["explanation_depth"],
        nlp_feats["response_effort"],
        round(data.attention / 10.0, 3),
        round(data.confidence / 10.0, 3),
        round(data.difficulty / 10.0, 3),
    ]

    # the state before this session, for the "learning journey" display
    prev_obs = (
        db.query(Observation)
        .filter(Observation.student_id == session.student_id)
        .order_by(Observation.created_at.desc(), Observation.id.desc())
        .first()
    )
    previous_state_label = prev_obs.state_label if prev_obs else None

    obs = Observation(
        student_id=session.student_id,
        topic_id=session.topic_id,
        session_id=session.id,
        features=features,
        misconception_names=unresolved,
        source="live",
    )
    db.add(obs)
    session.completed = True
    db.commit()

    # HMM inference over the student's entire observation history (Viterbi).
    history = (
        db.query(Observation)
        .filter(Observation.student_id == session.student_id)
        .order_by(Observation.created_at, Observation.id)
        .all()
    )
    inference = infer_sequence([o.features for o in history])
    # re-label the whole history: Viterbi smooths past states given new evidence
    for o, s_idx in zip(history, inference["states"]):
        o.state_index = int(s_idx)
        o.state_label = STATE_NAMES[int(s_idx)]
    db.commit()

    # fact-level evidence: which reviewed lecture facts did the student
    # express anywhere in the session (conversation or takeaway summary)?
    facts_by_concept: dict[str, list[str]] = {}
    for a in analyses + ([summary_analysis] if summary_analysis is not None else []):
        for c in a.get("concepts", []):
            bucket = facts_by_concept.setdefault(c["name"], [])
            for f in c.get("facts_matched") or []:
                if f not in bucket:
                    bucket.append(f)

    # concept summary from the conversation plan (falls back to NLP statuses)
    plan_concepts = plan.get("concepts") or []
    if plan_concepts:
        status_map = {"covered": "covered", "partial": "partial", "unclear": "unclear", "pending": "missing"}
        concept_summary = [
            {"name": c["name"], "status": status_map.get(c["status"], "missing"),
             # where the evidence came from: the conversation, the takeaway
             # summary, or both — summary evidence counts but stays labelled
             "evidence_source": c.get("evidence_source") or "teachback",
             "facts_matched": facts_by_concept.get(c["name"], [])}
            for c in plan_concepts
        ]
    else:
        best: dict = {}
        for a in analyses:
            for c in a["concepts"]:
                rank = {"covered": 2, "partial": 1, "missing": 0}[c["status"]]
                if rank >= best.get(c["name"], ("missing", -1))[1]:
                    best[c["name"]] = (c["status"], rank)
        concept_summary = [{"name": k, "status": v[0]} for k, v in best.items()]

    demonstrated = [c["name"] for c in concept_summary if c["status"] == "covered"]
    needs_clarification = [c["name"] for c in concept_summary
                           if c["status"] in ("partial", "unclear", "missing")]
    # RECOMMENDATION SAFETY: only a concept the student actually attempted and
    # left incomplete is evidence of a gap. A concept that simply never came
    # up ("missing") is an absence of evidence — turning it into a specific
    # remediation task would invent a learning problem out of silence.
    evidenced_gaps = [c["name"] for c in concept_summary if c["status"] in ("partial", "unclear")]
    not_discussed_concepts = [c["name"] for c in concept_summary if c["status"] == "missing"]

    # Relationship evidence accumulated by the conversation plan (see
    # helpers.REL_STATUS_MAP): demonstrated / not_discussed / needs_clarification.
    # Only needs_clarification is a learning gap — "not discussed" is an absence
    # of evidence and is never treated, worded or counted as a mistake.
    rel_summary = relationship_summary(plan)
    rels_demonstrated = [r for r in rel_summary if r["status"] == DEMONSTRATED]
    rels_unclear = [r for r in rel_summary if r["status"] == NEEDS_CLARIFICATION]
    rels_not_discussed = [r for r in rel_summary if r["status"] == NOT_DISCUSSED]

    # confidence/difficulty are separate observations, never a state: they
    # only steer WHICH activity style is recommended (see recommend/rules.py)
    understanding_evidence = len(demonstrated) / len(concept_summary) if concept_summary else 0.0

    # The learning state is a reading of the student's WHOLE history (Viterbi
    # over every session), so one strong session after a weak run does not
    # flip it — that is the point of using an HMM. Shown next to "you
    # demonstrated 6 of 6 concepts" it reads as a contradiction, so say
    # plainly which question each number answers.
    state_index = inference["current_state"]
    state_note = ""
    if concept_summary:
        if understanding_evidence >= 0.7 and state_index <= 1:
            state_note = (
                f"This session went well — you demonstrated {len(demonstrated)} of "
                f"{len(concept_summary)} concepts. The learning condition above reads your "
                "recent sessions together rather than this one alone, so it moves gradually; "
                "another session like this one will shift it.")
        elif understanding_evidence <= 0.3 and state_index >= 3:
            state_note = (
                "This session showed less than usual. The learning condition above reflects "
                "your recent sessions together, so one quieter session does not undo them.")
    rec = recommend(
        inference["current_state"], tdef["activities"], unresolved,
        evidence={
            "demonstrated": demonstrated,
            # only real, evidenced gaps steer a concept-specific remediation
            "unclear": evidenced_gaps
            + [f"the connection {r['source']} → {r['target']}" for r in rels_unclear],
            "not_discussed": not_discussed_concepts
            + [f"the connection {r['source']} → {r['target']}" for r in rels_not_discussed],
        },
        signals={
            "understanding": round(understanding_evidence, 3),
            "confidence": round(data.confidence / 10.0, 3),
            "difficulty": round(data.difficulty / 10.0, 3),
        },
        topic_def=tdef,
    )

    # conceptual evidence bullets stored with the observation, shown on the
    # progress page to explain WHY the learning state is what it is
    evidence_notes = [f"{len(demonstrated)}/{len(concept_summary)} concepts demonstrated"] if concept_summary else []
    if rel_summary:
        note = f"{len(rels_demonstrated)}/{len(rel_summary)} key relationships demonstrated"
        if rels_not_discussed:
            # said plainly so it never reads as a failure on the progress page
            note += f" ({len(rels_not_discussed)} not discussed — no evidence either way)"
        evidence_notes.append(note)
    evidence_notes += [f"Mentioned from the lecture: {f}"
                       for facts in facts_by_concept.values() for f in facts][:2]
    evidence_notes += [f"Needs clarification: {name}" for name in needs_clarification[:2]]
    evidence_notes += [f"Connection needing clarification: {r['source']} → {r['target']}" for r in rels_unclear[:2]]
    evidence_notes += [f"Misconception resolved: {m}" for m in resolved]
    evidence_notes += [f"Misconception still open: {m}" for m in unresolved]
    evidence_notes += [f"Own summary demonstrated: {n}"
                       for n in summary_insights.get("new_concepts_demonstrated", [])[:2]]
    if features[4] >= 0.6:
        evidence_notes.append("High effort")
    obs.evidence_notes = evidence_notes
    db.commit()

    miscon_details = [
        {"name": m["name"], "clarification": m.get("clarification", ""),
         "resolved": m["name"] in resolved}
        for m in tdef.get("misconceptions", []) if m["name"] in detected
    ]

    # the optional Quick knowledge check for this topic (secondary evidence;
    # offered after the conversation + takeaway, never required)
    topic_quiz = db.query(Quiz).filter(Quiz.topic_id == session.topic_id).first()
    quiz_info = ({"quiz_id": topic_quiz.id, "n_questions": len(topic_quiz.questions)}
                 if topic_quiz and topic_quiz.questions else None)

    return {
        "session_id": session.id,
        "quiz": quiz_info,
        "observation": observation_out(obs),
        "session_features": dict(zip(FEATURE_NAMES, features)),
        "concept_summary": concept_summary,
        "relationship_summary": rel_summary,
        "detected_misconceptions": unresolved,
        "resolved_misconceptions": resolved,
        "misconception_details": miscon_details,
        "previous_state_label": previous_state_label,
        "previous_student_state_label": (
            student_state_label(prev_obs.state_index) if prev_obs else None),
        "state": {
            "index": inference["current_state"],
            "label": inference["current_label"],
            "student_label": student_state_label(inference["current_state"]),
            # The HMM posterior is the model's confidence in WHICH learning
            # condition best explains the observed session sequence. It is not
            # "the probability the student understands", and is labelled that
            # way everywhere it is shown.
            "posterior": dict(zip(STATE_NAMES, inference["current_posterior"])),
            "posterior_meaning": ("Model confidence in the current learning condition, "
                                  "given this student's session history — not a probability "
                                  "of understanding."),
            # reconciles "you demonstrated everything" with a state that reads
            # the whole trajectory; empty when the two already agree
            "note": state_note,
        },
        "timeline": [observation_out(o) for o in history[-10:]],
        "recommendation": rec,
        "summary_insights": summary_insights,
    }
