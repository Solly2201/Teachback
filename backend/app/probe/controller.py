"""Deterministic, inspectable pedagogical controller for generated probes.

The v1 conversation plan (nlp/conversation.py) remains the flow authority:
it decides when to move on, when to probe a misconception and when the
session ends — that machinery is untouched and identical with the feature
off. What this controller adds, when the experimental feature is ON, is an
explicit pedagogical decision for the follow-up the plan chose:

  * action        — what kind of intervention this follow-up is
  * target        — the concept / relationship / misconception it addresses,
                    resolved to a stable teacher-material id
  * difficulty    — easy vs standard, informed by the student's HMM state
                    posterior and how this concept has gone so far
  * candidates    — a utility ranking over EVERY currently probe-worthy
                    target, recording which probe the evidence gaps say
                    would be most informative right now

The utility score is an interpretable evidence-gap heuristic, not a formal
expected-information-gain computation, and is documented as such. Intuition:
a probe is most useful where current evidence leaves us most undecided —
an engaged-but-incomplete concept or a detected misconception — and least
useful where evidence is already firm. The ranking is recorded with every
generated probe (plan_agrees_with_ranking) so the research question "does
uncertainty-aware targeting pick differently from the fixed rules?" is
measurable from the stored metadata alone.

The LLM sees only the finished decision. It never chooses the learner state,
the target or the difficulty, and it cannot override this controller
(schema.validate_against_decision rejects any attempt).
"""

# Evidence-gap utility per status. 0 = probing would tell us almost nothing
# new, 1 = probing is maximally informative. Values are fixed and documented
# rather than tuned: partial evidence (engaged, incomplete) is the most
# undecided condition; a detected-and-unresolved misconception is the single
# most diagnostic thing to probe; firm evidence is barely worth re-probing.
CONCEPT_UTILITY = {"partial": 0.9, "unclear": 0.8, "pending": 0.55, "covered": 0.1}
RELATIONSHIP_UTILITY = {"contradicted": 0.95, "unclear": 0.85, "pending": 0.5,
                        "demonstrated": 0.1}
MISCONCEPTION_OPEN_UTILITY = 1.0
MISCONCEPTION_RESOLVED_UTILITY = 0.15

CONCEPT_WHY = {
    "partial": "the student engaged with it but the evidence is incomplete",
    "unclear": "the student attempted it and it did not come together",
    "pending": "no evidence either way yet",
    "covered": "already demonstrated",
}

ACTION_FOR_KIND = {
    "main": "ASK_MAIN_QUESTION",
    "easier": "ASK_EASIER_PROBE",
    "probe": "ASK_PROBE",
    "deepen": "ASK_DEEPEN",
    "relationship": "PROBE_RELATIONSHIP",
    "misconception": "PROBE_MISCONCEPTION",
}

# Posterior mass on the three low-evidence states above which probes are
# phrased easy: when the model itself is unsure the student is in a good
# place, an easier question produces cleaner evidence than a stretch one.
EASY_POSTERIOR_MASS = 0.5


def rank_candidates(plan: dict, topic_def: dict) -> list[dict]:
    """Utility ranking over every target worth probing right now."""
    candidates = []
    for entry in plan.get("concepts", []):
        status = entry.get("status", "pending")
        candidates.append({
            "target_type": "concept",
            "target_id": entry.get("id"),
            "target_name": entry["name"],
            "utility": CONCEPT_UTILITY.get(status, 0.55),
            "why": CONCEPT_WHY.get(status, CONCEPT_WHY["pending"]),
        })
    for rel in plan.get("relationships", []):
        status = rel.get("status", "pending")
        candidates.append({
            "target_type": "relationship",
            "target_id": rel.get("id"),
            "target_name": f"{rel['source']} → {rel['target']}",
            "utility": RELATIONSHIP_UTILITY.get(status, 0.5),
            "why": ("the connection sounded mixed up" if status == "contradicted"
                    else "the connection has evidence gaps" if status == "unclear"
                    else "the connection has not come up" if status == "pending"
                    else "already demonstrated"),
        })
    resolved = set(plan.get("resolved", []))
    for m in topic_def.get("misconceptions", []):
        if m["name"] not in plan.get("detected", []):
            continue  # never probe a misconception nobody showed signs of
        open_ = m["name"] not in resolved
        candidates.append({
            "target_type": "misconception",
            "target_id": m.get("id"),
            "target_name": m["name"],
            "utility": MISCONCEPTION_OPEN_UTILITY if open_ else MISCONCEPTION_RESOLVED_UTILITY,
            "why": ("it appeared in an answer and is not resolved" if open_
                    else "it appeared earlier but was cleared up"),
        })
    candidates.sort(key=lambda c: (-c["utility"], str(c["target_name"])))
    return candidates


def _pick_difficulty(kind: str, entry: dict | None, posterior: list[float] | None) -> str:
    if kind in ("easier", "deepen"):
        return "easy"
    if entry is not None and entry.get("attempts", 0) > 0:
        return "easy"  # a first probe already happened; step down, not up
    if posterior and len(posterior) == 5 and sum(posterior[:3]) >= EASY_POSTERIOR_MASS:
        return "easy"
    return "standard"


def decide(plan: dict, topic_def: dict, followup: dict,
           posterior: list[float] | None = None) -> dict | None:
    """Turn the plan's chosen follow-up into an explicit pedagogical decision.

    Returns None when this follow-up is not one the generative path handles
    (teacher-authored extension questions and open prompts stay verbatim, and
    a target that cannot be resolved to a stored teacher-material id is left
    to v1 wording).
    """
    kind = followup.get("kind")
    action = ACTION_FOR_KIND.get(kind)
    if action is None:
        return None

    target_type, target_id, target_name, entry = None, None, None, None
    if kind in ("main", "easier", "probe", "deepen"):
        concepts = plan.get("concepts") or []
        current = plan.get("current", 0)
        if not (0 <= current < len(concepts)):
            return None
        entry = concepts[current]
        target_type, target_id, target_name = "concept", entry.get("id"), entry["name"]
    elif kind == "relationship":
        asked = plan.get("asked_rel")
        if not asked:
            return None
        rid, rsource, rtarget = asked
        target_type, target_id = "relationship", rid
        target_name = f"{rsource} → {rtarget}"
    elif kind == "misconception":
        name = plan.get("asked_miscon")
        mdef = next((m for m in topic_def.get("misconceptions", []) if m["name"] == name), None)
        if mdef is None:
            return None
        target_type, target_id, target_name = "misconception", mdef.get("id"), name

    if target_id is None:
        return None

    candidates = rank_candidates(plan, topic_def)
    top = candidates[0] if candidates else None
    agrees = bool(top and top["target_type"] == target_type and top["target_id"] == target_id)
    selected = next((c for c in candidates
                     if c["target_type"] == target_type and c["target_id"] == target_id), None)

    return {
        "action": action,
        "target_type": target_type,
        "target_id": target_id,
        "target_name": target_name,
        "difficulty": _pick_difficulty(kind, entry, posterior),
        # plain language, teacher-facing; built from target names, never from
        # the student's own words
        "reason": (f"Probing {target_name} because " + selected["why"] + "."
                   if selected else f"Probing {target_name}."),
        "candidates": candidates[:8],
        "plan_agrees_with_ranking": agrees,
    }
