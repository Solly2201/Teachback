"""Transparent rule-based adaptive recommendation.

Given the HMM-estimated learning state (and the latest NLP analysis for
context), pick the next activity. Rules are deliberately simple: each state
maps to an activity style; if the topic has a stored activity targeting that
state it is used, otherwise a sensible generic fallback is returned. If a
misconception was just detected, a targeted clarification is attached.
"""
from ..states import STATE_KEYS

GENERIC_ACTIVITIES = {
    "not_trying": {
        "title": "Quick warm-up question",
        "description": "Answer one very short question to get moving again: in one sentence, what is this topic about?",
        "kind": "re_engagement",
    },
    "unclear": {
        "title": "Simple explanation with an analogy",
        "description": "Read a short, simplified explanation of the topic with an everyday analogy, then answer one basic conceptual question.",
        "kind": "concept_review",
    },
    "struggling": {
        "title": "Guided worked example",
        "description": "Work through a step-by-step example with hints at each step, focused on the parts you found hard.",
        "kind": "guided_practice",
    },
    "understanding": {
        "title": "Application problem",
        "description": "Solve a medium-difficulty problem that applies the concept to a new situation.",
        "kind": "application",
    },
    "confident": {
        "title": "Edge-case challenge",
        "description": "Tackle an advanced question about a tricky edge case, or explain the topic to a classmate and note what questions they ask.",
        "kind": "challenge",
    },
}


# Per-state explanation of why that style of activity is chosen.
STATE_WHY = {
    "not_trying": "Your recent sessions showed very low engagement, so a short warm-up is the easiest way back in.",
    "unclear": "The core ideas are not settled yet, so a simpler explanation will help more than practice right now.",
    "struggling": "You are putting in real effort but some ideas have gaps, so a guided exercise with hints fits best.",
    "understanding": "You demonstrated the core concepts, so the next step is applying them to a new situation.",
    "confident": "You consistently demonstrated understanding across recent sessions, so you are ready for a harder challenge.",
}


def recommend(state_index: int, topic_activities: list[dict] | None = None,
              detected_misconceptions: list[str] | None = None,
              evidence: dict | None = None) -> dict:
    """Pick the next activity for a learning state.

    evidence (optional) = {"demonstrated": [names], "unclear": [names]} — used
    only to make the returned "why" explanation concrete.
    """
    state_key = STATE_KEYS[state_index]

    activity = None
    for a in topic_activities or []:
        if a.get("target_state") == state_key:
            activity = {"title": a["title"], "description": a["description"], "kind": a.get("kind", "practice")}
            break
    if activity is None:
        activity = dict(GENERIC_ACTIVITIES[state_key])

    why_parts = []
    demonstrated = (evidence or {}).get("demonstrated") or []
    unclear = (evidence or {}).get("unclear") or []
    if demonstrated and unclear:
        why_parts.append(
            f"You showed understanding of {', '.join(demonstrated[:3])}, "
            f"but {', '.join(unclear[:2])} still needs clarification."
        )
    elif unclear:
        why_parts.append(f"{', '.join(unclear[:3])} still needs clarification.")
    elif demonstrated:
        why_parts.append(f"You showed understanding of {', '.join(demonstrated[:3])}.")
    why_parts.append(STATE_WHY[state_key])

    result = {"state_key": state_key, "activity": activity, "notes": [], "why": " ".join(why_parts)}
    if detected_misconceptions:
        result["notes"].append(
            "Before the activity, revisit this point: " + "; ".join(detected_misconceptions) + "."
        )
    return result
