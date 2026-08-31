"""Transparent rule-based adaptive recommendation.

Given the HMM-estimated learning state (and the latest NLP analysis for
context), pick the next activity. Rules are deliberately simple: each state
maps to an activity style; if the topic has a stored activity targeting that
state it is used, otherwise a deterministic template activity is built from
the topic's own concepts, and only then does a generic fallback apply. If a
misconception was just detected, a targeted clarification is attached.

Optional `signals` (understanding evidence, self-reported confidence and
perceived difficulty, all 0-1) adjust WHICH activity style is recommended —
an under-challenged student gets an optional extension, a strong-but-unsure
student gets a confidence-building application — but they never change the
HMM-estimated state itself. Confidence is an observation, not ground truth.

`evidence` separates three things on purpose:

    demonstrated   - the student showed it
    unclear        - the student engaged with it and it did not come together:
                     this is real evidence of a gap and MAY drive a specific
                     remediation activity
    not_discussed  - it never came up. Absence of evidence. It never produces
                     a concept-specific "you are struggling with X" activity
                     and is never worded as a mistake; at most it is offered
                     neutrally as something still to talk about.
"""
from ..states import STATE_KEYS

# Generic fallbacks used when a topic has no stored activity for the state.
# They carry their own content/question so the student can still open and
# complete them (id is None because there is no stored Activity row).
GENERIC_ACTIVITIES = {
    "not_trying": {
        "title": "Quick warm-up question",
        "description": "Answer one very short question to get moving again: in one sentence, what is this topic about?",
        "kind": "re_engagement",
        "content": "Sometimes the hardest part is just starting. No pressure here — one honest sentence is enough.",
        "question": "In one sentence: what is this topic about?",
    },
    "unclear": {
        "title": "Simple explanation with an analogy",
        "description": "Read a short, simplified explanation of the topic with an everyday analogy, then answer one basic conceptual question.",
        "kind": "concept_review",
        "content": ("Go back to your notes or textbook and re-read only the first short explanation of this "
                    "topic — just the part that states the main idea, nothing more."),
        "question": "In your own words, what is the main idea of this topic?",
    },
    "struggling": {
        "title": "Guided worked example",
        "description": "Work through a step-by-step example with hints at each step, focused on the parts you found hard.",
        "kind": "guided_practice",
        "content": ("Pick one worked example of this topic from your notes and follow it line by line, "
                    "pausing at each step to say why that step happens."),
        "question": "Which single step of the example was hardest to justify, and what made it click (or not)?",
    },
    "understanding": {
        "title": "Application problem",
        "description": "Solve a medium-difficulty problem that applies the concept to a new situation.",
        "kind": "application",
        "content": "You understand the core idea — the next step is stretching it to a situation you haven't seen.",
        "question": "Describe one new situation where this topic applies, and how you would use it there.",
    },
    "confident": {
        "title": "Edge-case challenge",
        "description": "Tackle an advanced question about a tricky edge case, or explain the topic to a classmate and note what questions they ask.",
        "kind": "challenge",
        "content": ("The best test of understanding is the edges: where does this idea stop working, "
                    "or need extra care?"),
        "question": "Describe one edge case or limitation of this topic and how you would handle it.",
    },
}


# Per-state explanation of why that style of activity is chosen.
STATE_WHY = {
    # "low engagement" is a claim about the student's effort, which the system
    # never observes — it only ever sees how much evidence a session produced.
    # states.STATE_STUDENT_NAMES was rewritten for exactly this reason; this
    # line is the same sentence said a second time and had been missed.
    "not_trying": "Your recent sessions haven't given us much to go on yet, so a short warm-up is the easiest way back in.",
    "unclear": "The core ideas are not settled yet, so a simpler explanation will help more than practice right now.",
    "struggling": "You are putting in real effort but some ideas have gaps, so a guided exercise with hints fits best.",
    "understanding": "You demonstrated the core concepts, so the next step is applying them to a new situation.",
    "confident": "You consistently demonstrated understanding across recent sessions, so you are ready for a harder challenge.",
}


def _template_activity(state_key: str, topic_def: dict | None, evidence: dict | None) -> dict | None:
    """Deterministic activity built from the topic's own concepts.

    Keeps lecture-created topics (which have no stored activities) fully
    actionable without the teacher writing custom activities per lecture.
    Purely template-based — no topic-specific code.
    """
    concepts = (topic_def or {}).get("concepts") or []
    if not concepts:
        return None
    topic_name = topic_def.get("name", "this topic")
    by_name = {c["name"]: c for c in concepts}
    # focus on a concept there is actual evidence of a gap in; never on one
    # that merely never came up (that would invent a learning problem)
    unclear = (evidence or {}).get("unclear") or []
    focus = next((by_name[n] for n in unclear if n in by_name), concepts[0])

    def reminder(c: dict) -> str:
        return c.get("description") or f"{c['name']} is one of the key ideas in {topic_name}."

    base = {"id": None, "generated": True}
    if state_key == "not_trying":
        return {**base, "kind": "re_engagement", "title": f"One-line warm-up: {topic_name}",
                "description": "Ease back in with one very short question.",
                "content": f"No pressure here — one honest sentence about {topic_name} is enough to get moving again.",
                "question": f"In one sentence: what is {topic_name} about?"}
    if state_key == "unclear":
        return {**base, "kind": "concept_review", "title": f"{focus['name']} in simple words",
                "description": f"A short plain-language review of {focus['name']}.",
                "content": f"Here is the idea again, in plain words: {reminder(focus)}",
                "question": f"Now explain {focus['name']} in one sentence, using your own words."}
    if state_key == "struggling":
        return {**base, "kind": "guided_practice", "title": f"An everyday example of {focus['name']}",
                "description": f"Connect {focus['name']} to something familiar.",
                "content": f"As a reminder: {reminder(focus)}",
                "question": f"Give one simple real-world example or analogy for {focus['name']}."}
    if state_key == "understanding":
        return {**base, "kind": "application", "title": f"Apply {focus['name']}",
                "description": f"Use {focus['name']} in a new situation.",
                "content": f"You have the idea — the next step is using it. Reminder: {reminder(focus)}",
                "question": f"Describe one situation involving {focus['name']}, and what role it plays there."}
    # confident: connect two ideas — prefer a teacher-authored relationship pair
    rels = (topic_def or {}).get("relationships") or []
    if rels:
        a_name, b_name = rels[0]["source"], rels[0]["target"]
    elif len(concepts) >= 2:
        a_name, b_name = concepts[0]["name"], concepts[1]["name"]
    else:
        a_name, b_name = focus["name"], topic_name
    return {**base, "kind": "challenge", "title": f"Connect the ideas: {a_name} & {b_name}",
            "description": "An optional extension — connect two ideas from this topic.",
            "content": f"The strongest test of understanding is connecting ideas rather than repeating them.",
            "question": f"How does {a_name} relate to {b_name}? Explain the connection in one or two sentences."}


def _resolve_activity(target_key: str, topic_activities: list[dict] | None,
                      topic_def: dict | None, evidence: dict | None) -> dict:
    """Stored teacher activity > topic-derived template > generic fallback."""
    for a in topic_activities or []:
        if a.get("target_state") == target_key:
            return {"id": a.get("id"), "title": a["title"], "description": a["description"],
                    "kind": a.get("kind", "practice"),
                    "content": a.get("content", ""), "question": a.get("question", "")}
    template = _template_activity(target_key, topic_def, evidence)
    if template is not None:
        return template
    return {"id": None, **GENERIC_ACTIVITIES[target_key]}


def recommend(state_index: int, topic_activities: list[dict] | None = None,
              detected_misconceptions: list[str] | None = None,
              evidence: dict | None = None, signals: dict | None = None,
              topic_def: dict | None = None) -> dict:
    """Pick the next activity for a learning state.

    evidence (optional) = {"demonstrated": [names], "unclear": [names]} — used
    to make the "why" explanation concrete and to focus template activities.
    signals (optional) = {"understanding", "confidence", "difficulty"} in 0-1 —
    adjust the recommended activity style only; the HMM state is untouched.
    """
    state_key = STATE_KEYS[state_index]
    target_key = state_key
    signal_why = None
    notes = []

    s = signals or {}
    und, conf, diff = s.get("understanding"), s.get("confidence"), s.get("difficulty")
    if und is not None and conf is not None:
        if (und >= 0.65 and conf >= 0.75 and diff is not None and diff <= 0.35
                and state_key in ("understanding", "confident")):
            # under-challenged: strong evidence, high confidence, low difficulty
            target_key = "confident"
            signal_why = ("Today's material appears comfortable for you — strong explanations, "
                          "high confidence and low perceived difficulty — so here is an optional extension.")
        elif und >= 0.65 and conf <= 0.4 and state_key in ("understanding", "confident"):
            # understands but doesn't feel it: build confidence with an easy application
            target_key = "understanding"
            signal_why = ("Your explanations were strong even though your confidence was low — one easy "
                          "application task should help your confidence catch up with your understanding.")
        elif und <= 0.4 and conf >= 0.75:
            notes.append("Your confidence is high, but the explanations showed limited evidence so far — "
                         "the activity below is a quick way to double-check the ideas.")

    activity = _resolve_activity(target_key, topic_activities, topic_def, evidence)

    why_parts = []
    demonstrated = (evidence or {}).get("demonstrated") or []
    unclear = (evidence or {}).get("unclear") or []
    not_discussed = (evidence or {}).get("not_discussed") or []
    if demonstrated and unclear:
        why_parts.append(
            f"You showed understanding of {', '.join(demonstrated[:3])}, "
            f"but {', '.join(unclear[:2])} still needs clarification."
        )
    elif unclear:
        why_parts.append(f"{', '.join(unclear[:3])} still needs clarification.")
    elif demonstrated:
        why_parts.append(f"You showed understanding of {', '.join(demonstrated[:3])}.")
    why_parts.append(signal_why or STATE_WHY[state_key])
    if not_discussed and not unclear:
        # stated as an absence of evidence, never as a gap in understanding
        notes.append(
            f"We didn't get to {', '.join(not_discussed[:3])} this time — that isn't a mistake, "
            "just something still to talk about.")

    result = {"state_key": state_key, "activity_state_key": target_key,
              "activity": activity, "notes": notes, "why": " ".join(why_parts)}
    if detected_misconceptions:
        result["notes"].append(
            "Before the activity, revisit this point: " + "; ".join(detected_misconceptions) + "."
        )
    return result
