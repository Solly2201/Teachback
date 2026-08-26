from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ActivityCompletion, Observation, Student, TeachSession
from .activities import completion_out
from ..recommend.rules import recommend
from ..states import FEATURE_NAMES, STATE_NAMES
from .helpers import latest_state, observation_out, student_out, topic_def

router = APIRouter(prefix="/api/students", tags=["students"])


def _ordered_observations(db: Session, student_id: int) -> list[Observation]:
    return (
        db.query(Observation)
        .filter(Observation.student_id == student_id)
        .order_by(Observation.created_at, Observation.id)
        .all()
    )


@router.get("")
def list_students(db: Session = Depends(get_db)):
    students = db.query(Student).order_by(Student.id).all()
    out = []
    for s in students:
        out.append(student_out(s, latest_state(s.observations)))
    return out


@router.get("/{student_id}")
def get_student(student_id: int, db: Session = Depends(get_db)):
    s = db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    obs = _ordered_observations(db, student_id)
    current = obs[-1].state_index if obs else None

    rec = None
    if obs and current is not None:
        last_topic = obs[-1].topic
        tdef = topic_def(last_topic) if last_topic else None
        feats = obs[-1].features or []
        signals = None
        if len(feats) >= 8:
            signals = {"understanding": feats[0], "confidence": feats[6], "difficulty": feats[7]}
        rec = recommend(current, tdef["activities"] if tdef else [],
                        obs[-1].misconception_names or [], signals=signals, topic_def=tdef)
        if last_topic:
            rec["topic_name"] = last_topic.name
            rec["topic_id"] = last_topic.id

    recent_topics = []
    seen = set()
    for o in reversed(obs):
        if o.topic_id and o.topic_id not in seen:
            seen.add(o.topic_id)
            recent_topics.append({"id": o.topic_id, "name": o.topic.name, "last_state": o.state_label})
        if len(recent_topics) >= 4:
            break

    return {
        **student_out(s, current),
        "recent_topics": recent_topics,
        "timeline": [observation_out(o) for o in obs[-12:]],
        "recommendation": rec,
        "session_count": len(obs),
    }


@router.get("/{student_id}/progress")
def student_progress(student_id: int, db: Session = Depends(get_db)):
    s = db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    obs = _ordered_observations(db, student_id)
    completions = (
        db.query(ActivityCompletion)
        .filter(ActivityCompletion.student_id == student_id)
        .order_by(ActivityCompletion.created_at.desc(), ActivityCompletion.id.desc())
        .limit(10)
        .all()
    )
    recent_sessions = (
        db.query(TeachSession)
        .filter(TeachSession.student_id == student_id, TeachSession.completed)
        .order_by(TeachSession.id.desc())
        .limit(6)
        .all()
    )
    summaries = [
        {"topic_name": ts.topic.name if ts.topic else None,
         "text": ts.summary_text,
         "created_at": ts.started_at.isoformat() if ts.started_at else None}
        for ts in recent_sessions if (ts.summary_text or "").strip()
    ]
    return {
        "student": student_out(s, obs[-1].state_index if obs else None),
        "feature_names": FEATURE_NAMES,
        "state_names": STATE_NAMES,
        "timeline": [observation_out(o) for o in obs],
        "completions": [completion_out(c) for c in completions],
        "summaries": summaries,
    }
