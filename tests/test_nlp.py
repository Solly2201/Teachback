"""Tests for the NLP analyzer and the rule-based conversation engine."""
from app.nlp.analyzer import analyze_response, merge_session_analyses, split_sentences
from app.seed_content import TOPICS

BACKPROP = next(t for t in TOPICS if t["name"] == "Backpropagation")

GOOD_ANSWER = (
    "The network makes a prediction and a loss function measures how wrong it is. "
    "The error is propagated backwards through the layers using the chain rule to compute "
    "the gradient of the loss with respect to each weight. An optimizer like gradient descent "
    "then updates the weights step by step, and repeating this many times minimises the loss."
)

MISCONCEPTION_ANSWER = (
    "Backpropagation works by changing the input data so that the network produces the right "
    "answer. The inputs get adjusted whenever the output is wrong."
)


def test_sentence_splitting():
    assert len(split_sentences("First sentence. Second sentence! Third one?")) == 3
    assert split_sentences("") == []


def test_good_answer_covers_concepts():
    analysis = analyze_response(GOOD_ANSWER, BACKPROP)
    covered = [c for c in analysis["concepts"] if c["status"] in ("covered", "partial")]
    assert len(covered) >= 4
    assert analysis["features"]["concept_coverage"] >= 0.6
    assert analysis["features"]["semantic_correctness"] >= 0.6
    assert analysis["detected_misconceptions"] == []


def test_misconception_is_detected():
    analysis = analyze_response(MISCONCEPTION_ANSWER, BACKPROP)
    assert "Backpropagation changes the input" in analysis["detected_misconceptions"]
    assert analysis["features"]["misconception_score"] > 0.4


def test_low_effort_features():
    analysis = analyze_response("idk", BACKPROP)
    assert analysis["features"]["response_effort"] < 0.1
    assert analysis["features"]["concept_coverage"] <= 0.2


def test_targeted_check_accepts_short_contextual_answer():
    from app.nlp.analyzer import targeted_concept_check

    gradient = next(c for c in BACKPROP["concepts"] if c["name"] == "Gradient")
    good = targeted_concept_check("It shows how changing a weight affects the error.", gradient)
    assert good["contextual"] >= 0.66 and good["overlap"] >= 1
    idk = targeted_concept_check("I don't know.", gradient)
    assert idk["overlap"] == 0


def test_conversation_advances_on_short_answers():
    from app.nlp.conversation import build_plan, play_turn

    plan = build_plan(BACKPROP)
    analysis = analyze_response("The loss measures how wrong the network's prediction is.", BACKPROP)
    plan, turn = play_turn(plan, analysis, BACKPROP)
    assert plan["concepts"][0]["status"] == "covered"
    assert turn["followup"]["concept"] == "Gradient"
    assert turn["feedback"]


def test_conversation_gives_easier_question_when_unclear():
    from app.nlp.analyzer import targeted_concept_check
    from app.nlp.conversation import build_plan, play_turn

    plan = build_plan(BACKPROP)
    analysis = analyze_response("I don't really know anything about that.", BACKPROP)
    analysis["target_check"] = targeted_concept_check(
        "I don't really know anything about that.", BACKPROP["concepts"][0])
    plan, turn = play_turn(plan, analysis, BACKPROP)
    assert turn["followup"]["kind"] == "easier"
    assert plan["concepts"][0]["attempts"] == 1


def test_conversation_probes_misconception():
    from app.nlp.conversation import build_plan, play_turn

    plan = build_plan(BACKPROP)
    analysis = analyze_response(
        "Backpropagation and gradient descent are the same thing.", BACKPROP)
    plan, turn = play_turn(plan, analysis, BACKPROP)
    assert turn["followup"]["kind"] == "misconception"
    assert turn["misconception"]["name"] == "Backpropagation is the same as gradient descent"
    assert turn["misconception"]["clarification"]


def test_conversation_terminates():
    from app.nlp.analyzer import targeted_concept_check
    from app.nlp.conversation import MAX_QUESTIONS, build_plan, play_turn

    plan = build_plan(BACKPROP)
    text = "Hmm, something about networks maybe."
    for i in range(MAX_QUESTIONS + 2):
        analysis = analyze_response(text, BACKPROP)
        if plan["concepts"]:
            cur = BACKPROP["concepts"][min(plan["current"], len(BACKPROP["concepts"]) - 1)]
            analysis["target_check"] = targeted_concept_check(text, cur)
        plan, turn = play_turn(plan, analysis, BACKPROP)
        if turn["done"]:
            break
    assert plan["done"]


def test_merge_accumulates_coverage():
    a1 = analyze_response("A loss function measures how wrong the prediction is.", BACKPROP)
    a2 = analyze_response(
        "The gradient of the loss with respect to each weight tells us how to adjust the weights.",
        BACKPROP,
    )
    merged = merge_session_analyses([a1, a2])
    assert merged["concept_coverage"] >= max(
        a1["features"]["concept_coverage"], a2["features"]["concept_coverage"]
    )
