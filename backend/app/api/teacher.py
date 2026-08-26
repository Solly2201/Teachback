"""Teacher dashboard aggregates and meta/evaluation endpoints."""
import json
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..config import EVAL_RESULTS_PATH, HMM_MAPPING_PATH
from ..database import get_db
from ..models import Observation, Student, TeachSession, Topic
from ..states import FEATURE_NAMES, STATE_KEYS, STATE_NAMES, STATE_PROFILES
from .helpers import observation_out

router = APIRouter(prefix="/api", tags=["teacher", "meta"])


@router.get("/teacher/overview")
def teacher_overview(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    observations = (
        db.query(Observation).order_by(Observation.created_at, Observation.id).all()
    )

    by_student: dict[int, list[Observation]] = defaultdict(list)
    for o in observations:
        by_student[o.student_id].append(o)

    # current state distribution (latest observation per student)
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

    # common misconceptions across all observations
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
                student = next(s for s in students if s.id == sid)
                declining.append(
                    {"id": sid, "name": student.name,
                     "from_state": STATE_NAMES[round(before)], "to_state": STATE_NAMES[states[-1]],
                     "drop": round(before - recent, 1)}
                )
    declining.sort(key=lambda d: -d["drop"])

    # topic-level statistics
    topics = db.query(Topic).all()
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
            {"id": t.id, "name": t.name, "sessions": len(t_obs),
             "avg_concept_coverage": round(avg_cov, 3),
             "avg_misconception_score": round(avg_mis, 3),
             "avg_state": round(avg_state, 2)}
        )

    recent = [observation_out(o) for o in observations[-10:]][::-1]
    id_to_name = {s.id: s.name for s in students}
    for r, o in zip(recent, observations[-10:][::-1]):
        r["student_name"] = id_to_name.get(o.student_id, "?")

    live_sessions = db.query(TeachSession).filter(TeachSession.completed).count()

    return {
        "student_count": len(students),
        "live_session_count": live_sessions,
        "distribution": distribution,
        "common_misconceptions": common_misconceptions,
        "declining_students": declining[:8],
        "topic_stats": topic_stats,
        "recent_interactions": recent,
    }


@router.get("/meta/states")
def meta_states():
    mapping = None
    if HMM_MAPPING_PATH.exists():
        with open(HMM_MAPPING_PATH, encoding="utf-8") as f:
            mapping = json.load(f)
    return {
        "state_names": STATE_NAMES,
        "state_keys": STATE_KEYS,
        "feature_names": FEATURE_NAMES,
        "state_profiles": {STATE_NAMES[k]: v for k, v in STATE_PROFILES.items()},
        "hmm_state_mapping": mapping,
    }


@router.get("/meta/evaluation")
def meta_evaluation():
    if not EVAL_RESULTS_PATH.exists():
        return {"available": False}
    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        return {"available": True, **json.load(f)}
