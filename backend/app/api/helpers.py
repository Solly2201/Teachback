"""Shared serialisation helpers for the API layer."""
from ..models import Observation, Student, Topic
from ..states import STATE_NAMES


def topic_def(topic: Topic) -> dict:
    """Structured topic definition consumed by the NLP analyzer and dialogue engine."""
    return {
        "id": topic.id,
        "name": topic.name,
        "description": topic.description,
        "reference_explanation": topic.reference_explanation,
        "opening_prompt": topic.opening_prompt,
        "extension_question": topic.extension_question,
        "concepts": [
            {"id": c.id, "name": c.name, "description": c.description,
             "main_question": c.main_question, "easier_question": c.easier_question,
             "probe_question": c.probe_question, "application_question": c.application_question}
            for c in topic.concepts
        ],
        "misconceptions": [
            {"id": m.id, "name": m.name, "description": m.description,
             "clarification": m.clarification, "probe_question": m.probe_question}
            for m in topic.misconceptions
        ],
        "activities": [
            {"id": a.id, "title": a.title, "description": a.description,
             "kind": a.kind, "target_state": a.target_state}
            for a in topic.activities
        ],
    }


def observation_evidence(o: Observation) -> list[str]:
    """Short human-readable bullets explaining what this session showed.

    Derived deterministically from the observation features so it works for
    both live TeachBack sessions and seeded histories.
    """
    f = o.features or []
    bullets: list[str] = []
    if len(f) >= 8:
        if f[0] >= 0.7:
            bullets.append("Most key concepts demonstrated")
        elif f[0] >= 0.4:
            bullets.append("Some concepts demonstrated, others unclear")
        else:
            bullets.append("Few concepts demonstrated")
        if f[1] >= 0.65:
            bullets.append("Accurate explanations")
        elif f[1] < 0.4:
            bullets.append("Explanations were off the mark")
        if f[4] >= 0.6:
            bullets.append("High effort")
        elif f[4] < 0.15:
            bullets.append("Very low engagement")
    for name in o.misconception_names or []:
        bullets.append(f"Misconception detected: {name}")
    return bullets


def observation_out(o: Observation) -> dict:
    return {
        "id": o.id,
        "topic_id": o.topic_id,
        "topic_name": o.topic.name if o.topic else None,
        "features": o.features,
        "state_index": o.state_index,
        "state_label": o.state_label,
        "misconceptions": o.misconception_names or [],
        "evidence": observation_evidence(o),
        "source": o.source,
        "created_at": o.created_at.isoformat() if o.created_at else None,
    }


def student_out(s: Student, current_state: int | None = None) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "program": s.program,
        "is_demo": s.is_demo,
        "current_state": current_state,
        "current_state_label": STATE_NAMES[current_state] if current_state is not None else None,
    }


def latest_state(observations: list[Observation]) -> int | None:
    if not observations:
        return None
    latest = max(observations, key=lambda o: (o.created_at, o.id))
    return latest.state_index
