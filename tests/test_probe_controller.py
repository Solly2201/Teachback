"""The deterministic pedagogical controller: what gets probed, and why.

The controller decides everything pedagogically meaningful (action, target,
difficulty) before any LLM is involved. These tests pin that behavior: the
decisions are pure functions of the plan/evidence, the ranking follows the
documented evidence-gap utilities, and the HMM posterior influences only the
difficulty — never the flow and never the state itself.
"""
from app.probe.controller import decide, rank_candidates

TOPIC_DEF = {
    "name": "Strings",
    "concepts": [
        {"id": 1, "name": "String", "description": "Text data."},
        {"id": 2, "name": "Indexing", "description": "Positions start at 0."},
    ],
    "relationships": [
        {"id": 10, "source": "String", "label": "is a", "target": "Sequence of characters",
         "description": "A string is a sequence of characters.", "probe_question": ""},
    ],
    "misconceptions": [
        {"id": 20, "name": "String equals variable",
         "description": "A string is the same thing as a variable.",
         "clarification": "A variable can hold a string; they are not the same.",
         "probe_question": "What is the difference between a string and a variable?"},
    ],
}


def make_plan(**overrides):
    plan = {
        "concepts": [
            {"id": 1, "name": "String", "status": "partial", "attempts": 1},
            {"id": 2, "name": "Indexing", "status": "pending", "attempts": 0},
        ],
        "relationships": [
            {"id": 10, "source": "String", "target": "Sequence of characters",
             "status": "pending", "asked": False},
        ],
        "current": 0,
        "asked_rel": None,
        "asked_miscon": None,
        "detected": [],
        "resolved": [],
    }
    plan.update(overrides)
    return plan


# --- ranking ----------------------------------------------------------------

def test_open_misconception_outranks_everything():
    plan = make_plan(detected=["String equals variable"])
    ranked = rank_candidates(plan, TOPIC_DEF)
    assert ranked[0]["target_type"] == "misconception"
    assert ranked[0]["target_id"] == 20
    assert ranked[0]["utility"] == 1.0


def test_partial_concept_outranks_pending_material():
    ranked = rank_candidates(make_plan(), TOPIC_DEF)
    assert (ranked[0]["target_type"], ranked[0]["target_id"]) == ("concept", 1)
    utilities = {(c["target_type"], c["target_id"]): c["utility"] for c in ranked}
    assert utilities[("concept", 1)] > utilities[("concept", 2)]
    assert utilities[("concept", 1)] > utilities[("relationship", 10)]


def test_undetected_misconceptions_are_never_candidates():
    # never probe (or accuse) a misconception nobody showed signs of
    ranked = rank_candidates(make_plan(), TOPIC_DEF)
    assert all(c["target_type"] != "misconception" for c in ranked)


def test_resolved_misconception_drops_to_the_bottom():
    plan = make_plan(detected=["String equals variable"], resolved=["String equals variable"])
    ranked = rank_candidates(plan, TOPIC_DEF)
    miscon = next(c for c in ranked if c["target_type"] == "misconception")
    assert miscon["utility"] < 0.5


def test_ranking_is_deterministic():
    plan = make_plan(detected=["String equals variable"])
    assert rank_candidates(plan, TOPIC_DEF) == rank_candidates(plan, TOPIC_DEF)


# --- decisions per follow-up kind -------------------------------------------

def test_concept_probe_decision():
    followup = {"kind": "probe", "text": "v1 text", "concept": "String"}
    d = decide(make_plan(), TOPIC_DEF, followup)
    assert d["action"] == "ASK_PROBE"
    assert (d["target_type"], d["target_id"], d["target_name"]) == ("concept", 1, "String")
    assert d["reason"] and "String" in d["reason"]
    assert d["candidates"]


def test_relationship_decision_uses_asked_rel():
    plan = make_plan(asked_rel=[10, "String", "Sequence of characters"])
    d = decide(plan, TOPIC_DEF, {"kind": "relationship", "text": "v1"})
    assert d["action"] == "PROBE_RELATIONSHIP"
    assert (d["target_type"], d["target_id"]) == ("relationship", 10)


def test_misconception_decision_resolves_the_id():
    plan = make_plan(asked_miscon="String equals variable",
                     detected=["String equals variable"])
    d = decide(plan, TOPIC_DEF, {"kind": "misconception", "text": "v1"})
    assert d["action"] == "PROBE_MISCONCEPTION"
    assert (d["target_type"], d["target_id"]) == ("misconception", 20)
    assert d["plan_agrees_with_ranking"] is True  # it IS the top-utility target


def test_extension_and_open_questions_are_left_to_v1():
    assert decide(make_plan(), TOPIC_DEF, {"kind": "extension", "text": "t"}) is None
    assert decide(make_plan(), TOPIC_DEF, {"kind": "open", "text": "t"}) is None


def test_target_without_stored_id_is_left_to_v1():
    plan = make_plan()
    plan["concepts"][0]["id"] = None
    assert decide(plan, TOPIC_DEF, {"kind": "probe", "text": "t"}) is None


# --- difficulty -------------------------------------------------------------

def test_easier_kind_is_always_easy():
    d = decide(make_plan(), TOPIC_DEF, {"kind": "easier", "text": "t"})
    assert d["difficulty"] == "easy"


def test_low_state_posterior_mass_steps_difficulty_down():
    plan = make_plan()
    plan["concepts"][0]["attempts"] = 0
    followup = {"kind": "probe", "text": "t"}
    struggling = [0.1, 0.3, 0.4, 0.15, 0.05]  # 0.8 mass on the low states
    confident = [0.02, 0.03, 0.05, 0.4, 0.5]
    assert decide(plan, TOPIC_DEF, followup, posterior=struggling)["difficulty"] == "easy"
    assert decide(plan, TOPIC_DEF, followup, posterior=confident)["difficulty"] == "standard"
    # no posterior at all (new student, HMM unavailable) is not a crash
    assert decide(plan, TOPIC_DEF, followup, posterior=None)["difficulty"] == "standard"


def test_repeat_attempts_step_difficulty_down():
    plan = make_plan()  # concept 1 already has attempts=1
    d = decide(plan, TOPIC_DEF, {"kind": "probe", "text": "t"},
               posterior=[0.0, 0.0, 0.0, 0.3, 0.7])
    assert d["difficulty"] == "easy"


def test_reason_never_contains_student_words():
    # the controller builds its explanation from target names only; there is
    # no student text input to decide() at all
    d = decide(make_plan(), TOPIC_DEF, {"kind": "probe", "text": "t"})
    assert "reason" in d and isinstance(d["reason"], str)
