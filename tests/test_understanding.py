"""Focused tests for conceptual-understanding evidence.

These verify that the system accepts varied phrasings of the same concept
(technical / simple / informal), detects wrong relationships despite high
semantic similarity, accumulates evidence across turns, and resolves
misconceptions after a correction.
"""
from app.nlp.analyzer import analyze_response, contradiction_cues, targeted_concept_check
from app.nlp.conversation import _verdict, build_plan, play_turn
from app.seed_content import TOPICS

BACKPROP = next(t for t in TOPICS if t["name"] == "Backpropagation")
GRADIENT = next(c for c in BACKPROP["concepts"] if c["name"] == "Gradient")
GRADIENT_ENTRY = {"id": None, "name": "Gradient", "status": "pending", "attempts": 0}


def _judge(text: str) -> str:
    analysis = analyze_response(text, BACKPROP)
    analysis["target_check"] = targeted_concept_check(text, GRADIENT)
    return _verdict(analysis, GRADIENT_ENTRY)


# A. Technical wording is accepted
def test_technical_wording_demonstrates_concept():
    assert _judge("The gradient is the derivative of the loss with respect to the weight.") == "correct"


# B. Simple wording is equally valid evidence
def test_simple_wording_demonstrates_same_concept():
    assert _judge("It tells us how much the error changes when we change a weight.") == "correct"


# C. Non-textbook terminology is not rejected
def test_different_terminology_not_rejected():
    assert _judge("It shows how sensitive the error is to a weight.") in ("correct", "partial")


# An analogy with no shared content words gets no automatic credit — the
# tutor responds with an easier question instead of marking it right or wrong
def test_analogy_without_overlap_gets_easier_question():
    text = "It's like checking which direction makes the error go up or down."
    analysis = analyze_response(text, BACKPROP)
    analysis["target_check"] = targeted_concept_check(text, GRADIENT)
    plan = build_plan(BACKPROP)
    plan["current"] = 1  # Gradient is the current concept
    plan["concepts"][0]["status"] = "covered"
    plan, turn = play_turn(plan, analysis, BACKPROP)
    assert turn["followup"]["kind"] == "easier"
    assert analysis["detected_misconceptions"] == []


# D. A known misconception is detected as a wrong relationship, not a keyword
def test_misconception_detected():
    analysis = analyze_response("The gradient directly changes the weights.", BACKPROP)
    assert "Gradients directly change the weights" in analysis["detected_misconceptions"]


# E. A correct multi-relationship explanation earns relationship evidence
def test_correct_relationships_demonstrated():
    analysis = analyze_response(
        "Backpropagation calculates the gradients and gradient descent uses them to update the weights.",
        BACKPROP,
    )
    status = {(r["source"], r["target"]): r["status"] for r in analysis["relationships"]}
    assert status[("Backpropagation", "Gradient")] == "demonstrated"
    assert status[("Gradient descent", "Gradient")] == "demonstrated"
    assert status[("Gradient descent", "Weight")] == "demonstrated"
    assert not any(r["status"] == "contradicted" for r in analysis["relationships"])


# F. Semantically similar but conceptually wrong is NOT treated as correct
def test_wrong_direction_relationship_contradicted():
    analysis = analyze_response("Gradient descent uses the gradient to increase the loss.", BACKPROP)
    status = {(r["source"], r["target"]): r["status"] for r in analysis["relationships"]}
    assert status[("Gradient descent", "Weight")] == "contradicted"
    assert status[("Weight update", "Loss")] == "contradicted"
    # the correct polarity is not flagged
    ok = analyze_response("We change the weights to reduce the loss.", BACKPROP)
    assert not any(r["status"] == "contradicted" for r in ok["relationships"])


def test_contradiction_cues_derived_from_teacher_text():
    cues = contradiction_cues(
        "The weights are updated so that the loss decreases over time.",
        "The weights are updated so that the loss increases over time.",
    )
    assert "increases" in cues and "loss" not in cues


# G. A short contextual answer still counts as evidence
def test_short_contextual_answer_counts():
    assert _judge("It tells us how the loss changes.") == "correct"


# H. "I don't know" gets no inflated credit
def test_i_dont_know_is_unclear():
    assert _judge("I don't know.") == "unclear"
    tc = targeted_concept_check("I don't know.", GRADIENT)
    assert tc["overlap"] == 0


# I. Misconception followed by a correction becomes resolved
def test_misconception_resolved_after_correction():
    plan = build_plan(BACKPROP)
    analysis = analyze_response("Backpropagation and gradient descent are the same thing.", BACKPROP)
    plan, turn = play_turn(plan, analysis, BACKPROP)
    assert turn["followup"]["kind"] == "misconception"
    correction = analyze_response(
        "Backpropagation only computes the gradients, while gradient descent is the separate "
        "optimizer that uses those gradients to update the weights.",
        BACKPROP,
    )
    plan, turn = play_turn(plan, correction, BACKPROP)
    assert turn["resolved_misconception"] == "Backpropagation is the same as gradient descent"
    assert "Backpropagation is the same as gradient descent" in plan["resolved"]


# J. Evidence accumulates across multiple short answers
def test_multi_turn_evidence_accumulates():
    plan = build_plan(BACKPROP)
    turns = [
        "The loss function measures how wrong the network's prediction is.",
        "The gradient tells us how the loss changes when we change a weight.",
    ]
    for text in turns:
        analysis = analyze_response(text, BACKPROP)
        cur = BACKPROP["concepts"][min(plan["current"], len(BACKPROP["concepts"]) - 1)]
        analysis["target_check"] = targeted_concept_check(text, cur)
        plan, _ = play_turn(plan, analysis, BACKPROP)
    statuses = {c["name"]: c["status"] for c in plan["concepts"]}
    assert statuses["Loss / error"] == "covered"
    assert statuses["Gradient"] == "covered"
    # relationship evidence gathered along the way is kept on the plan
    rel_status = {(r["source"], r["target"]): r["status"] for r in plan["relationships"]}
    assert rel_status[("Gradient", "Weight")] == "demonstrated"


# Relationship gaps get targeted follow-up questions once concepts are done
def test_relationship_gap_gets_probed():
    plan = build_plan(BACKPROP)
    # all concepts except the first are already demonstrated
    for c in plan["concepts"][1:]:
        c["status"] = "covered"
    text = "The loss function measures how wrong the network's prediction is."
    analysis = analyze_response(text, BACKPROP)
    analysis["target_check"] = targeted_concept_check(text, BACKPROP["concepts"][0])
    plan, turn = play_turn(plan, analysis, BACKPROP)
    # the last concept was answered correctly, so the next question targets a
    # relationship that has not been demonstrated yet
    assert turn["followup"]["kind"] == "relationship"
    assert "→" in turn["followup"]["concept"]
