"""The optional "Quick knowledge check" (10 teacher-reviewed MCQs).

Secondary evidence only: TeachBack (explaining in your own words) remains the
primary understanding signal. The quiz asks "what can the student correctly
recognise/apply?" — its results never overwrite TeachBack evidence, never
touch the HMM observation vector, and are always reported per concept, not as
a single mastery number.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Observation, Quiz, QuizAnswer, QuizAttempt, QuizQuestion,
                      Student, TeachSession, Topic)
from ..nlp.quiz_gen import generate_quiz_questions, validate_question
from ..recommend.rules import recommend
from .helpers import (DEMONSTRATED, NEEDS_CLARIFICATION, NOT_DISCUSSED,
                      REL_STATUS_LABEL, relationship_status, topic_def)

router = APIRouter(prefix="/api", tags=["quiz"])

QUIZ_INTRO = ("This is a short check of the material covered in the lecture. It is not a "
              "replacement for your explanation — it just gives us another way to "
              "understand what you picked up.")


def question_out(q: QuizQuestion, with_answer: bool = False) -> dict:
    out = {"id": q.id, "concept_name": q.concept_name, "kind": q.kind,
           "question": q.question, "options": q.options or []}
    if with_answer:
        out["correct_index"] = q.correct_index
        out["explanation"] = q.explanation
    return out


def build_quiz_for_topic(db: Session, topic: Topic, questions: list[dict] | None = None) -> Quiz | None:
    """Create/replace the topic's quiz from reviewed (or generated) questions.

    Invalid questions are skipped — a malformed edit can never publish an
    ambiguous MCQ. Returns None when nothing valid remains.
    """
    if questions is None:
        questions = generate_quiz_questions(topic_def(topic))
    valid = [q for q in questions if validate_question(q)]

    existing = db.query(Quiz).filter(Quiz.topic_id == topic.id).first()
    if not valid:
        if existing:
            db.delete(existing)
        return None
    quiz = existing or Quiz(topic_id=topic.id)
    if existing is None:
        db.add(quiz)
    quiz.questions = [
        QuizQuestion(concept_name=q.get("concept_name", ""), kind=q.get("kind", "basic"),
                     question=q["question"], options=list(q["options"]),
                     correct_index=int(q["correct_index"]),
                     explanation=q.get("explanation", ""), position=i)
        for i, q in enumerate(valid)
    ]
    return quiz


@router.get("/topics/{topic_id}/quiz")
def get_topic_quiz(topic_id: int, db: Session = Depends(get_db)):
    """Student view of the quiz — correct answers are NOT included."""
    topic = db.get(Topic, topic_id)
    if not topic:
        raise HTTPException(404, "Topic not found")
    quiz = db.query(Quiz).filter(Quiz.topic_id == topic_id).first()
    if not quiz or not quiz.questions:
        return {"available": False}
    return {
        "available": True,
        "quiz_id": quiz.id,
        "title": quiz.title,
        "intro": QUIZ_INTRO,
        "questions": [question_out(q) for q in quiz.questions],
    }


@router.post("/topics/{topic_id}/quiz/generate")
def generate_topic_quiz(topic_id: int, db: Session = Depends(get_db)):
    """(Re)generate the quiz from the topic's current reviewed structure."""
    topic = db.get(Topic, topic_id)
    if not topic:
        raise HTTPException(404, "Topic not found")
    quiz = build_quiz_for_topic(db, topic)
    db.commit()
    if quiz is None:
        return {"available": False, "questions": []}
    db.refresh(quiz)
    return {"available": True, "quiz_id": quiz.id,
            "questions": [question_out(q, with_answer=True) for q in quiz.questions]}


class AnswerIn(BaseModel):
    question_id: int
    selected_index: int = Field(ge=0, le=3)


class SubmitIn(BaseModel):
    student_id: int
    session_id: int | None = None
    answers: list[AnswerIn]


STATUS_RANK = {"covered": 2, "partial": 1, "unclear": 0, "pending": 0}


def _combined_concept_view(plan: dict | None, per_concept: dict) -> list[dict]:
    """Merge TeachBack concept evidence with MCQ results, preserving both.

    MCQ results never erase TeachBack evidence: a concept the student
    explained well but missed an MCQ on gets a gentle "worth a quick review",
    and a concept they recognised but couldn't explain is NOT called confused.
    """
    teachback = {c["name"]: c["status"] for c in (plan or {}).get("concepts", [])}
    names = list(dict.fromkeys(list(teachback) + list(per_concept)))
    out = []
    for name in names:
        tb = teachback.get(name)
        mcq = per_concept.get(name)
        entry = {"name": name, "teachback_status": tb,
                 "mcq_correct": mcq["correct"] if mcq else None,
                 "mcq_total": mcq["total"] if mcq else None}
        tb_good = tb in ("covered", "partial")
        mcq_perfect = mcq is not None and mcq["correct"] == mcq["total"]
        mcq_good = mcq is not None and mcq["correct"] * 2 >= mcq["total"]
        if mcq is None:
            entry["verdict"] = "revisit" if tb in ("unclear", "pending") else "solid"
            entry["message"] = ""
        elif tb_good and mcq_perfect:
            entry["verdict"] = "solid"
            entry["message"] = "Your explanation and the knowledge check both support this — strong evidence."
        elif tb_good:
            # any MCQ miss on a concept the student explained well: the
            # TeachBack evidence stays, the gap is named gently
            entry["verdict"] = "quick_review"
            entry["message"] = ("Your explanation showed understanding, but the knowledge check "
                               "suggests one part may need a quick review.")
        elif mcq_good:
            entry["verdict"] = "practice_explaining"
            entry["message"] = ("You answered the knowledge check correctly — your explanation may "
                               "benefit from a little more practice putting the idea into your own words.")
        else:
            entry["verdict"] = "revisit"
            entry["message"] = "Both signals suggest this concept is worth revisiting with the material."
        out.append(entry)
    return out


# --- relationships: two independent evidence channels -----------------------
# TeachBack asks the student to SAY the connection; the knowledge check asks
# them to RECOGNISE it. Neither is rewritten as the other: a correct MCQ never
# becomes "you explained this connection", and a wrong MCQ never deletes an
# explanation the student actually gave.
def _relationship_evidence(topic_def: dict, plan: dict | None,
                           per_question: list[dict]) -> list[dict]:
    """Per-relationship view combining TeachBack evidence with MCQ evidence.

    A relationship MCQ is tied back to its relationship structurally: the
    question was generated from that pair, so it carries the source as its
    concept name and the target as its correct option.
    """
    plan_status = {(r["source"], r["target"]): relationship_status(r.get("status"))
                   for r in (plan or {}).get("relationships", [])}
    out = []
    for r in topic_def.get("relationships", []):
        key = (r["source"], r["target"])
        tb = plan_status.get(key, NOT_DISCUSSED)
        matched = [q for q in per_question
                   if q.get("kind") == "relationship"
                   and (q.get("concept_name") or "").strip() == r["source"].strip()
                   and (q["options"][q["correct_index"]] or "").strip() == r["target"].strip()]
        entry = {
            "source": r["source"], "label": r.get("label", "relates to"), "target": r["target"],
            "teachback_status": tb,
            "teachback_label": REL_STATUS_LABEL[tb],
            "mcq_total": len(matched) or None,
            "mcq_correct": sum(1 for q in matched if q["correct"]) if matched else None,
            "message": "",
        }
        if matched:
            all_right = all(q["correct"] for q in matched)
            if tb == DEMONSTRATED and all_right:
                entry["message"] = ("You explained this connection and the knowledge check "
                                    "supported it.")
            elif tb == DEMONSTRATED:
                entry["message"] = ("You explained this connection during TeachBack; the knowledge "
                                    "check missed it, so it is worth one quick look — your "
                                    "explanation still stands.")
            elif all_right:
                entry["message"] = ("Your knowledge check supported this connection, although you "
                                    "didn't explicitly discuss it during TeachBack.")
            elif tb == NEEDS_CLARIFICATION:
                entry["message"] = "Worth revisiting: both signals point at this connection."
            else:
                entry["message"] = ("This connection didn't come up in your explanation, and the "
                                    "knowledge check didn't confirm it either.")
        out.append(entry)
    return out


@router.post("/quiz/{quiz_id}/submit")
def submit_quiz(quiz_id: int, data: SubmitIn, db: Session = Depends(get_db)):
    quiz = db.get(Quiz, quiz_id)
    if not quiz:
        raise HTTPException(404, "Quiz not found")
    student = db.get(Student, data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    questions = {q.id: q for q in quiz.questions}

    attempt = QuizAttempt(quiz_id=quiz.id, student_id=student.id,
                          session_id=data.session_id, n_questions=len(quiz.questions))
    db.add(attempt)
    per_question = []
    per_concept: dict[str, dict] = {}
    n_correct = 0
    seen_qids = set()
    for a in data.answers:
        q = questions.get(a.question_id)
        if q is None or q.id in seen_qids:
            continue
        seen_qids.add(q.id)
        correct = a.selected_index == q.correct_index
        n_correct += int(correct)
        attempt.answers.append(QuizAnswer(question_id=q.id,
                                          selected_index=a.selected_index, correct=correct))
        per_question.append({**question_out(q, with_answer=True),
                             "selected_index": a.selected_index, "correct": correct})
        bucket = per_concept.setdefault(q.concept_name or "General", {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(correct)
    attempt.n_correct = n_correct
    db.commit()
    db.refresh(attempt)

    # combined evidence with the TeachBack session (both signals preserved)
    session = db.get(TeachSession, data.session_id) if data.session_id else None
    combined = _combined_concept_view(session.plan if session else None, per_concept)
    relationship_evidence = (
        _relationship_evidence(topic_def(session.topic), session.plan, per_question)
        if session is not None else []
    )

    # the knowledge-check result becomes an evidence NOTE on the session's
    # observation — never a feature: the 8-dim observation vector and the HMM
    # state are untouched
    updated_recommendation = None
    if session is not None and session.completed:
        obs = db.query(Observation).filter(Observation.session_id == session.id).first()
        if obs is not None:
            notes = list(obs.evidence_notes or [])
            note = f"Knowledge check: {n_correct}/{attempt.n_questions} correct"
            if note not in notes:
                obs.evidence_notes = notes + [note]
                db.commit()
        # refresh the recommendation with the combined evidence: MCQ gaps are
        # added as focus areas, MCQ strengths soften the framing — the
        # HMM-estimated state itself is reused, never recomputed from the score
        if obs is not None and obs.state_index is not None:
            tdef = topic_def(session.topic)
            demonstrated = [c["name"] for c in combined
                            if c["teachback_status"] == "covered"]
            unclear = [c["name"] for c in combined if c["verdict"] == "revisit"]
            unclear += [c["name"] for c in combined if c["verdict"] == "quick_review"]
            # only connections the student actually got wrong or left incomplete
            # count as gaps — a connection that simply never came up does not
            unclear += [f"the connection {r['source']} → {r['target']}"
                        for r in relationship_evidence
                        if r["teachback_status"] == NEEDS_CLARIFICATION]
            feats = obs.features or []
            signals = None
            if len(feats) >= 8:
                signals = {"understanding": feats[0], "confidence": feats[6], "difficulty": feats[7]}
            updated_recommendation = recommend(
                obs.state_index, tdef["activities"], obs.misconception_names or [],
                evidence={"demonstrated": demonstrated, "unclear": unclear},
                signals=signals, topic_def=tdef)
            if any(c["verdict"] == "practice_explaining" for c in combined):
                updated_recommendation["notes"].append(
                    "You recognised the right answers in the knowledge check — the next step is "
                    "practising saying those ideas in your own words.")

    solid = [c["name"] for c in combined if c["verdict"] == "solid"]
    revisit = [c["name"] for c in combined if c["verdict"] in ("revisit", "quick_review")]
    return {
        "attempt_id": attempt.id,
        "n_correct": n_correct,
        "n_questions": attempt.n_questions,
        "headline": f"{n_correct}/{attempt.n_questions} questions correct",
        "message": ("Your explanations and knowledge check together give us a better picture "
                    "of what you understood."),
        "per_question": per_question,
        "per_concept": [{"name": k, **v} for k, v in per_concept.items()],
        "combined": combined,
        "relationship_evidence": relationship_evidence,
        "solid_concepts": solid,
        "revisit_concepts": revisit,
        "updated_recommendation": updated_recommendation,
    }
