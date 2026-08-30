"""Teacher dashboard aggregates and meta/evaluation endpoints.

Every faculty-facing aggregate is scoped by subject: when the teacher/subject
switcher selects a subject, only observations, sessions, topics, misconceptions
and feedback belonging to that subject's topics are aggregated. Scoping happens
in these queries, not in the frontend, so one subject's data can never leak
into another subject's dashboard.
"""
import json
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..config import EVAL_RESULTS_PATH, HMM_MAPPING_PATH
from ..database import get_db
from ..models import (Observation, Quiz, QuizAnswer, QuizAttempt, Student,
                      TeachSession, Topic)
from ..states import (FEATURE_NAMES, STATE_KEYS, STATE_NAMES, STATE_PROFILES,
                      STATE_STUDENT_DESCRIPTIONS, STATE_STUDENT_NAMES)
from .helpers import observation_out

router = APIRouter(prefix="/api", tags=["teacher", "meta"])


@router.get("/teacher/overview")
def teacher_overview(subject_id: int | None = None, db: Session = Depends(get_db)):
    """Class overview. With subject_id, every aggregate is scoped to that
    subject's topics — the cross-subject isolation the switcher promises."""
    # archived topics (their lecture was deleted) are excluded from every
    # aggregate, so a removed lecture cannot reappear in the dashboard — the
    # underlying student records are untouched and still visible per student
    topic_q = db.query(Topic).filter(Topic.archived_at.is_(None)).order_by(Topic.id)
    if subject_id is not None:
        topic_q = topic_q.filter(Topic.subject_id == subject_id)
    topics = topic_q.all()
    topic_ids = {t.id for t in topics}

    archived_ids = {t.id for t in db.query(Topic).filter(Topic.archived_at.isnot(None)).all()}

    obs_q = db.query(Observation).order_by(Observation.created_at, Observation.id)
    if subject_id is not None:
        obs_q = obs_q.filter(Observation.topic_id.in_(topic_ids))
    elif archived_ids:
        obs_q = obs_q.filter(Observation.topic_id.notin_(archived_ids))
    observations = obs_q.all()

    by_student: dict[int, list[Observation]] = defaultdict(list)
    for o in observations:
        by_student[o.student_id].append(o)

    # students who have interacted with this subject (all students when unscoped)
    if subject_id is None:
        students = db.query(Student).all()
    else:
        students = db.query(Student).filter(Student.id.in_(by_student.keys())).all()
    id_to_name = {s.id: s.name for s in students}

    # current state distribution (latest in-subject observation per student)
    dist = Counter()
    for obs_list in by_student.values():
        if obs_list and obs_list[-1].state_index is not None:
            dist[obs_list[-1].state_index] += 1
    total = sum(dist.values()) or 1
    distribution = [
        {"state": STATE_NAMES[i], "key": STATE_KEYS[i], "count": dist.get(i, 0),
         "percent": round(100 * dist.get(i, 0) / total, 1)}
        for i in range(5)
    ]

    # common misconceptions across this subject's observations
    miscon_counts = Counter()
    for o in observations:
        for name in o.misconception_names or []:
            miscon_counts[name] += 1
    common_misconceptions = [
        {"name": name, "count": count} for name, count in miscon_counts.most_common(8)
    ]

    # students whose state is deteriorating (mean of last 2 vs previous 2 states)
    declining = []
    for sid, obs_list in by_student.items():
        states = [o.state_index for o in obs_list if o.state_index is not None]
        if len(states) >= 4:
            recent = sum(states[-2:]) / 2
            before = sum(states[-4:-2]) / 2
            if recent - before <= -1:
                declining.append(
                    {"id": sid, "name": id_to_name.get(sid, "?"),
                     "from_state": STATE_NAMES[round(before)], "to_state": STATE_NAMES[states[-1]],
                     "drop": round(before - recent, 1)}
                )
    declining.sort(key=lambda d: -d["drop"])

    # topic-level statistics
    topic_stats = []
    for t in topics:
        t_obs = [o for o in observations if o.topic_id == t.id]
        if t_obs:
            avg_cov = sum(o.features[0] for o in t_obs) / len(t_obs)
            avg_mis = sum(o.features[2] for o in t_obs) / len(t_obs)
            avg_state = sum(o.state_index or 0 for o in t_obs) / len(t_obs)
        else:
            avg_cov = avg_mis = avg_state = 0
        topic_stats.append(
            {"id": t.id, "name": t.name,
             "subject_name": t.subject.name if t.subject else None,
             "sessions": len(t_obs),
             "avg_concept_coverage": round(avg_cov, 3),
             "avg_misconception_score": round(avg_mis, 3),
             "avg_state": round(avg_state, 2)}
        )

    # lecture feedback aggregates (confidence/difficulty from live
    # observation features; pace + request chips from completed sessions)
    session_q = db.query(TeachSession).filter(TeachSession.completed)
    if subject_id is not None:
        session_q = session_q.filter(TeachSession.topic_id.in_(topic_ids))
    elif archived_ids:
        session_q = session_q.filter(TeachSession.topic_id.notin_(archived_ids))
    completed_sessions = session_q.all()
    sessions_by_topic = defaultdict(list)
    for ts in completed_sessions:
        sessions_by_topic[ts.topic_id].append(ts)
    topic_feedback = []
    for t in topics:
        t_live = [o for o in observations
                  if o.topic_id == t.id and o.source == "live" and len(o.features or []) >= 8]
        t_sessions = sessions_by_topic.get(t.id, [])
        paces = Counter(ts.pace for ts in t_sessions if ts.pace)
        choices = Counter(ch for ts in t_sessions for ch in (ts.feedback_choices or []))
        comments = [ts.feedback_text for ts in t_sessions if (ts.feedback_text or "").strip()][-3:]
        if not (t_live or paces or choices or comments):
            continue
        topic_feedback.append({
            "id": t.id, "name": t.name,
            "subject_name": t.subject.name if t.subject else None,
            "responses": len(t_sessions),
            "avg_confidence": round(10 * sum(o.features[6] for o in t_live) / len(t_live), 1) if t_live else None,
            "avg_difficulty": round(10 * sum(o.features[7] for o in t_live) / len(t_live), 1) if t_live else None,
            "pace": [{"label": k, "count": v} for k, v in paces.most_common()],
            "common_requests": [{"label": k, "count": v} for k, v in choices.most_common(5)],
            "recent_comments": comments,
        })

    # knowledge-check performance per topic, kept SEPARATE from TeachBack
    # evidence: for each concept both the MCQ correctness rate and the share
    # of completed TeachBack sessions that demonstrated the concept are shown
    # — deliberately two numbers, never one "mastery" score
    knowledge_checks = []
    for t in topics:
        t_quiz = db.query(Quiz).filter(Quiz.topic_id == t.id).first()
        if not t_quiz:
            continue
        attempts = db.query(QuizAttempt).filter(QuizAttempt.quiz_id == t_quiz.id).all()
        if not attempts:
            continue
        answers = (db.query(QuizAnswer)
                   .filter(QuizAnswer.attempt_id.in_([a.id for a in attempts])).all())
        q_concept = {q.id: (q.concept_name or "General") for q in t_quiz.questions}
        mcq: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # concept -> [correct, total]
        for ans in answers:
            bucket = mcq[q_concept.get(ans.question_id, "General")]
            bucket[1] += 1
            bucket[0] += int(ans.correct)
        # TeachBack demonstration rate per concept from completed sessions
        tb: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for ts in sessions_by_topic.get(t.id, []):
            for c in (ts.plan or {}).get("concepts", []):
                bucket = tb[c["name"]]
                bucket[1] += 1
                bucket[0] += int(c.get("status") in ("covered", "partial"))
        total_correct = sum(v[0] for v in mcq.values())
        total_answers = sum(v[1] for v in mcq.values()) or 1
        knowledge_checks.append({
            "id": t.id, "name": t.name,
            "attempts": len(attempts),
            "avg_percent": round(100 * total_correct / total_answers),
            "concepts": [
                {"name": name,
                 "mcq_percent": round(100 * c / max(n, 1)), "mcq_n": n,
                 "teachback_percent": (round(100 * tb[name][0] / tb[name][1])
                                       if tb.get(name, [0, 0])[1] else None),
                 "teachback_n": tb.get(name, [0, 0])[1]}
                for name, (c, n) in sorted(mcq.items())
            ],
        })

    recent = [observation_out(o) for o in observations[-10:]][::-1]
    for r, o in zip(recent, observations[-10:][::-1]):
        r["student_name"] = id_to_name.get(o.student_id, "?")

    return {
        "student_count": len(students),
        "live_session_count": len(completed_sessions),
        "distribution": distribution,
        "common_misconceptions": common_misconceptions,
        "declining_students": declining[:8],
        "topic_stats": topic_stats,
        "topic_feedback": topic_feedback,
        "knowledge_checks": knowledge_checks,
        "recent_interactions": recent,
    }


@router.get("/meta/states")
def meta_states():
    mapping = None
    if HMM_MAPPING_PATH.exists():
        with open(HMM_MAPPING_PATH, encoding="utf-8") as f:
            mapping = json.load(f)
    from ..hmm.model import hmm_available, validate_model

    return {
        "state_names": STATE_NAMES,
        "state_keys": STATE_KEYS,
        # student-facing wording: the same five states described by the
        # evidence observed, not by an inferred attitude
        "state_student_names": STATE_STUDENT_NAMES,
        "state_student_descriptions": STATE_STUDENT_DESCRIPTIONS,
        "feature_names": FEATURE_NAMES,
        "state_profiles": {STATE_NAMES[k]: v for k, v in STATE_PROFILES.items()},
        "hmm_state_mapping": mapping,
        "hmm_validation": validate_model() if hmm_available() else {"ok": False,
                                                                    "problems": ["model not trained"]},
    }


@router.get("/meta/evaluation")
def meta_evaluation():
    if not EVAL_RESULTS_PATH.exists():
        return {"available": False}
    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        return {"available": True, **json.load(f)}
