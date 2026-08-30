"""Shared serialisation helpers for the API layer."""
from ..models import Observation, Student, Topic
from ..states import STATE_NAMES, STATE_STUDENT_NAMES


def student_state_label(state_index: int | None) -> str | None:
    """Student-facing wording for a learning state.

    Faculty views keep the formal state name; students see a description of
    the EVIDENCE ("Very Little Evidence Yet") rather than an inferred
    attitude ("Not Trying"), because the system observes answers and
    self-reports — never intent.
    """
    return STATE_STUDENT_NAMES[state_index] if state_index is not None else None

# --- how a concept relationship is reported to students and teachers -------
# Three states, and only ONE of them is a learning gap. "Not discussed" means
# the student neither showed the connection nor said anything against it: an
# absence of evidence. It must never be phrased, counted or acted on as a
# misunderstanding, so it is a distinct value everywhere it is exposed.
DEMONSTRATED = "demonstrated"
NOT_DISCUSSED = "not_discussed"
NEEDS_CLARIFICATION = "needs_clarification"

# conversation-plan status -> reported status
REL_STATUS_MAP = {
    "demonstrated": DEMONSTRATED,
    "contradicted": NEEDS_CLARIFICATION,
    "unclear": NEEDS_CLARIFICATION,
    "pending": NOT_DISCUSSED,
}

REL_STATUS_LABEL = {
    DEMONSTRATED: "Demonstrated",
    NOT_DISCUSSED: "Not discussed",
    NEEDS_CLARIFICATION: "Needs clarification",
}


def relationship_status(plan_status: str | None) -> str:
    return REL_STATUS_MAP.get(plan_status, NOT_DISCUSSED)


def relationship_summary(plan: dict | None) -> list[dict]:
    """Reported relationship evidence from a TeachBack conversation plan."""
    return [
        {"source": r["source"], "label": r.get("label", "relates to"), "target": r["target"],
         "status": relationship_status(r.get("status")),
         "status_label": REL_STATUS_LABEL[relationship_status(r.get("status"))]}
        for r in (plan or {}).get("relationships", [])
    ]


def topic_def(topic: Topic) -> dict:
    """Structured topic definition consumed by the NLP analyzer and dialogue engine."""
    return {
        "id": topic.id,
        "name": topic.name,
        "subject_id": topic.subject_id,
        "subject_name": topic.subject.name if topic.subject else None,
        "description": topic.description,
        "reference_explanation": topic.reference_explanation,
        "opening_prompt": topic.opening_prompt,
        "extension_question": topic.extension_question,
        "concepts": [
            {"id": c.id, "name": c.name, "description": c.description,
             "main_question": c.main_question, "easier_question": c.easier_question,
             "probe_question": c.probe_question, "application_question": c.application_question,
             "facts": c.facts or [], "examples": c.examples or [], "source": c.source or {}}
            for c in topic.concepts
        ],
        "misconceptions": [
            {"id": m.id, "name": m.name, "description": m.description,
             "clarification": m.clarification, "probe_question": m.probe_question}
            for m in topic.misconceptions
        ],
        "relationships": [
            {"id": r.id, "source": r.source, "label": r.label, "target": r.target,
             "description": r.description, "contradiction": r.contradiction,
             "probe_question": r.probe_question}
            for r in topic.relationships
        ],
        "activities": [
            {"id": a.id, "title": a.title, "description": a.description,
             "kind": a.kind, "target_state": a.target_state,
             "content": a.content, "question": a.question}
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
        "student_state_label": student_state_label(o.state_index),
        "misconceptions": o.misconception_names or [],
        # live sessions store their own conceptual evidence bullets; seeded
        # observations fall back to feature-derived bullets
        "evidence": (o.evidence_notes or None) or observation_evidence(o),
        "source": o.source,
        "created_at": o.created_at.isoformat() if o.created_at else None,
    }


def student_out(s: Student, current_state: int | None = None) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "program": s.program,
        "roll_no": s.roll_no or "",
        "is_demo": s.is_demo,
        "current_state": current_state,
        "current_state_label": STATE_NAMES[current_state] if current_state is not None else None,
        "current_student_state_label": student_state_label(current_state),
    }


def latest_state(observations: list[Observation]) -> int | None:
    if not observations:
        return None
    latest = max(observations, key=lambda o: (o.created_at, o.id))
    return latest.state_index
