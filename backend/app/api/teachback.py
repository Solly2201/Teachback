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
from ..models import Observation, Response, Student, TeachSession, Topic
from ..nlp.analyzer import analyze_response, merge_session_analyses, targeted_concept_check
from ..nlp.conversation import MAX_QUESTIONS, build_plan, first_question, play_turn, timeline_out
from ..recommend.rules import recommend
from ..states import FEATURE_NAMES, STATE_NAMES
from .helpers import observation_out, topic_def

router = APIRouter(prefix="/api/sessions", tags=["teachback"])


class StartIn(BaseModel):
    student_id: int
    topic_id: int


class RespondIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class FinishIn(BaseModel):
    attention: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=10)
    difficulty: float = Field(ge=0, le=10)


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
            f"Let's talk through {topic.name} together — short answers are perfectly fine. "
            "I'll ask simple questions one at a time and we'll figure out what you understand."
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
            analysis["target_check"] = targeted_concept_check(data.text, cdef)

    prev = session.responses[-1].analysis.get("turn", {}).get("followup") if session.responses else None
    prompt = (prev or {}).get("text") or (first_question(plan, tdef)["text"])

    plan, turn = play_turn(plan, analysis, tdef)
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


@router.post("/{session_id}/finish")
def finish(session_id: int, data: FinishIn, db: Session = Depends(get_db)):
    session = db.get(TeachSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.responses:
        raise HTTPException(400, "No responses submitted yet")
    if session.completed:
        raise HTTPException(400, "Session already completed")
    if not hmm_available():
        raise HTTPException(503, "HMM model not trained yet - run scripts/build_all.py")

    analyses = [r.analysis for r in session.responses]
    nlp_feats = merge_session_analyses(analyses)
    plan = session.plan or {}

    detected = sorted({m for a in analyses for m in a.get("detected_misconceptions", [])})
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

    tdef = topic_def(session.topic)

    # concept summary from the conversation plan (falls back to NLP statuses)
    plan_concepts = plan.get("concepts") or []
    if plan_concepts:
        status_map = {"covered": "covered", "partial": "partial", "unclear": "unclear", "pending": "missing"}
        concept_summary = [
            {"name": c["name"], "status": status_map.get(c["status"], "missing")} for c in plan_concepts
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
    needs_clarification = [c["name"] for c in concept_summary if c["status"] in ("partial", "unclear", "missing")]

    # relationship evidence accumulated by the conversation plan
    rel_status_map = {"demonstrated": "demonstrated", "contradicted": "needs_clarification",
                      "unclear": "needs_clarification", "pending": "not_shown"}
    relationship_summary = [
        {"source": r["source"], "label": r.get("label", "relates to"), "target": r["target"],
         "status": rel_status_map.get(r["status"], "not_shown")}
        for r in plan.get("relationships", [])
    ]
    rels_demonstrated = [r for r in relationship_summary if r["status"] == "demonstrated"]
    rels_unclear = [r for r in relationship_summary if r["status"] == "needs_clarification"]

    rec = recommend(
        inference["current_state"], tdef["activities"], unresolved,
        evidence={
            "demonstrated": demonstrated,
            "unclear": needs_clarification
            + [f"the connection {r['source']} → {r['target']}" for r in rels_unclear],
        },
    )

    # conceptual evidence bullets stored with the observation, shown on the
    # progress page to explain WHY the learning state is what it is
    evidence_notes = [f"{len(demonstrated)}/{len(concept_summary)} concepts demonstrated"] if concept_summary else []
    if relationship_summary:
        evidence_notes.append(
            f"{len(rels_demonstrated)}/{len(relationship_summary)} key relationships demonstrated")
    evidence_notes += [f"Needs clarification: {name}" for name in needs_clarification[:2]]
    evidence_notes += [f"Connection needing clarification: {r['source']} → {r['target']}" for r in rels_unclear[:2]]
    evidence_notes += [f"Misconception resolved: {m}" for m in resolved]
    evidence_notes += [f"Misconception still open: {m}" for m in unresolved]
    if features[4] >= 0.6:
        evidence_notes.append("High effort")
    obs.evidence_notes = evidence_notes
    db.commit()

    miscon_details = [
        {"name": m["name"], "clarification": m.get("clarification", ""),
         "resolved": m["name"] in resolved}
        for m in tdef.get("misconceptions", []) if m["name"] in detected
    ]

    return {
        "session_id": session.id,
        "observation": observation_out(obs),
        "session_features": dict(zip(FEATURE_NAMES, features)),
        "concept_summary": concept_summary,
        "relationship_summary": relationship_summary,
        "detected_misconceptions": unresolved,
        "resolved_misconceptions": resolved,
        "misconception_details": miscon_details,
        "previous_state_label": previous_state_label,
        "state": {
            "index": inference["current_state"],
            "label": inference["current_label"],
            "posterior": dict(zip(STATE_NAMES, inference["current_posterior"])),
        },
        "timeline": [observation_out(o) for o in history[-10:]],
        "recommendation": rec,
    }
