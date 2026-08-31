"""The things TeachBack must never do, whatever the similarity scores say.

These come out of the adversarial audit (scripts/adversarial_audit.py). They
are not accuracy tests — accuracy is measured elsewhere and will never be
perfect. Each of these pins a way the system could mislead a student:

    FALSE CREDIT      telling someone they know something they do not
    FALSE ACCUSATION  correcting someone for something they did not say
    INVENTED GAPS     turning silence into a learning problem

They are written as behaviour, not as thresholds, so a future change to the
scoring is free to move numbers around as long as none of these breaks.
"""
import pytest

from app.nlp.analyzer import analyze_response, targeted_concept_check
from app.nlp.conversation import _verdict
from app.recommend.rules import recommend
from app.seed_content import PYTHON_LECTURE, TOPICS

STRINGS = {
    "name": PYTHON_LECTURE["title"],
    "reference_explanation": " ".join(c["description"]
                                      for c in PYTHON_LECTURE["reviewed_concepts"]),
    "concepts": [dict(c, id=i + 1)
                 for i, c in enumerate(PYTHON_LECTURE["reviewed_concepts"])],
    "misconceptions": [dict(m, id=i + 1)
                       for i, m in enumerate(PYTHON_LECTURE["reviewed_misconceptions"])],
    "relationships": [dict(r, id=i + 1)
                      for i, r in enumerate(PYTHON_LECTURE["reviewed_relationships"])],
}
TOPIC_BY_NAME = {t["name"]: t for t in TOPICS}
ALL_TOPICS = [("strings", STRINGS)] + [(t["name"], t) for t in TOPICS]


def outcome(tdef, concept_name, text):
    concept = next(c for c in tdef["concepts"] if c["name"] == concept_name)
    analysis = analyze_response(text, tdef)
    analysis["target_check"] = targeted_concept_check(
        text, concept, topic_name=tdef.get("name", ""),
        misconceptions=tdef.get("misconceptions"),
        sibling_names=[c["name"] for c in tdef["concepts"]])
    entry = {"id": concept.get("id"), "name": concept_name,
             "status": "pending", "attempts": 0}
    return _verdict(analysis, entry), analysis.get("detected_misconceptions", [])


# ------------------------------------------------- 1. no credit for nothing

@pytest.mark.parametrize("text", [
    "the canteen was closed today",
    "can we get the slides by email",
    "table chair window bottle",
    "yeah so basically like you know the thing",
    "What did you understand about Indexing?",   # the question echoed back
    "",
    "...",
])
def test_an_answer_with_no_content_never_earns_credit(text):
    verdict, _ = outcome(STRINGS, "Indexing", text)
    assert verdict == "unclear", f"{text!r} -> {verdict}"


@pytest.mark.parametrize("text", [
    "I don't know", "not sure", "I have no idea",
    "I don't remember this one", "i forgot", "no clue at all",
])
def test_saying_you_do_not_know_never_earns_credit(text):
    verdict, _ = outcome(STRINGS, "Slicing", text)
    assert verdict == "unclear", f"{text!r} -> {verdict}"


@pytest.mark.parametrize("topic_name,tdef", ALL_TOPICS)
def test_an_empty_answer_earns_nothing_on_every_topic(topic_name, tdef):
    concept = tdef["concepts"][0]["name"]
    verdict, _ = outcome(tdef, concept, "")
    assert verdict == "unclear", f"{topic_name}/{concept} -> {verdict}"


@pytest.mark.parametrize("concept,text", [
    ("Indexing", "python uses indexing"),
    ("Slicing", "slicing is important"),
    ("Strings", "strings come up a lot in python"),
    ("Characters", "characters were mentioned in class"),
])
def test_naming_the_concept_is_not_explaining_it(concept, text):
    verdict, _ = outcome(STRINGS, concept, text)
    assert verdict == "unclear", f"{text!r} -> {verdict}"


# ------------------------------------------- 2. no accusation without cause

@pytest.mark.parametrize("concept,text", [
    ("Strings", "A string is a sequence of characters enclosed in quotation marks."),
    ("Strings", "it's basically text inside quotes"),
    ("Indexing", "you use the position number to pull out one particular letter"),
    ("Indexing", "the first position is zero"),
    ("Indexing", "indexing does not start at one, it starts at zero"),
    ("Indexing", "i thought the first letter was at one, but actually it is at zero"),
    ("Slicing", "you take a part of the text between a start and an end position"),
    ("Slicing", "slicing reads part of the string, it never modifies the original"),
    ("Characters", "it is built from the separate letters sitting in a set order"),
    ("split() and join()", "split breaks the text into a list and join sticks them back"),
])
def test_a_correct_answer_is_never_called_a_misconception(concept, text):
    _, detected = outcome(STRINGS, concept, text)
    assert detected == [], f"{text!r} accused of: {detected}"


@pytest.mark.parametrize("text", [
    "I don't know", "the canteen was closed today", "not sure", "",
])
def test_silence_is_never_called_a_misconception(text):
    _, detected = outcome(STRINGS, "Indexing", text)
    assert detected == []


def test_an_unrelated_answer_creates_no_relationship_evidence():
    for text in ("we also went over the assignment deadline", "I don't know", ""):
        analysis = analyze_response(text, STRINGS)
        statuses = {r["status"] for r in analysis["relationships"]}
        assert statuses <= {"not_shown"}, f"{text!r} -> {statuses}"


# --------------------------------------------- 3. silence is not a mistake

def test_a_concept_that_never_came_up_is_not_a_remediation_target():
    rec = recommend(1, [], evidence={"demonstrated": ["Indexing"], "unclear": [],
                                     "not_discussed": ["Slicing"]},
                    topic_def=STRINGS)
    blob = (rec["activity"]["title"] + rec["activity"]["question"] + rec["why"]).lower()
    assert "slicing" not in blob
    note = " ".join(rec["notes"]).lower()
    for accusation in ("wrong", "misunderstood", "failed", "mistake"):
        assert accusation not in note or "not a mistake" in note or "isn't a mistake" in note


def test_confidence_alone_never_becomes_understanding():
    """High self-reported confidence with no evidence must produce a gentle
    check, never a claim that the student understands."""
    rec = recommend(1, [], evidence={"demonstrated": [], "unclear": ["Indexing"],
                                     "not_discussed": []},
                    signals={"understanding": 0.1, "confidence": 0.95, "difficulty": 0.2},
                    topic_def=STRINGS)
    assert rec["state_key"] == "unclear"
    assert "double-check" in " ".join(rec["notes"]).lower()


# ---------------------------------- 4. the misconception comparison is fair

def test_misconception_references_are_compared_on_equal_footing():
    """Concept reference texts are prefixed "Name: ..."; the wrong claims must
    be prefixed the same way, or the concept side wins on the shared prefix
    alone and a restated misconception looks like understanding."""
    concept = next(c for c in STRINGS["concepts"] if c["name"] == "Indexing")
    check = targeted_concept_check(
        "the first letter is at index 1", concept, topic_name=STRINGS["name"],
        misconceptions=STRINGS["misconceptions"])
    assert check["shadowed"] is True, check


def test_restating_a_taught_misconception_does_not_earn_full_credit():
    for concept, text in [
        ("Indexing", "the first letter is at index 1 and the second is at index 2"),
        ("Indexing", "you start counting the positions from one"),
    ]:
        verdict, detected = outcome(STRINGS, concept, text)
        assert verdict != "correct", f"{text!r} -> {verdict} (detected {detected})"


# ------------------------------- 5. a score must be earned, not started with

def test_every_similarity_score_is_reported_against_its_own_floor():
    """Both similarity scores start somewhere above zero before the student
    has said anything, because the concept's name is repeated inside every
    reference text. For a self-describing concept that floor is already past
    the credit bar. The check must report how far the ANSWER moved each score,
    or an absolute threshold means a different thing for every concept."""
    for tdef in (STRINGS, *TOPICS):
        for concept in tdef["concepts"]:
            check = targeted_concept_check(
                "", concept, topic_name=tdef.get("name", ""),
                misconceptions=tdef.get("misconceptions"))
            assert "plain_lift" in check and "contextual_lift" in check, concept["name"]
            # an empty answer contributed nothing, by definition
            assert check["contextual_lift"] <= 0.0, (concept["name"], check)


@pytest.mark.parametrize("topic_name,tdef", ALL_TOPICS)
def test_saying_nothing_never_clears_the_bar_on_any_concept(topic_name, tdef):
    """The regression this guards: "Hidden states:" with no answer after it
    scored 0.82 against the Hidden states reference texts — above the credit
    threshold — so filler like "umm well you know how it is" was reported as
    a demonstrated concept."""
    for concept in tdef["concepts"]:
        for text in ("umm well you know how it is", "is this going to be in the test"):
            verdict, _ = outcome(tdef, concept["name"], text)
            assert verdict != "correct", f"{topic_name}/{concept['name']}: {text!r}"


def test_listing_the_lectures_other_headings_is_not_evidence():
    """The whole-response analysis has always discounted the lecture's other
    concept names; the per-question check was not being given them, so a bare
    list of the topic's own labels counted as an explanation of whichever
    concept happened to be under discussion."""
    hmm = TOPIC_BY_NAME["Hidden Markov Models"]
    verdict, _ = outcome(hmm, "Hidden states",
                         "hidden states observations transitions markov")
    assert verdict != "correct", verdict


# --------------------------------------------- 6. polarity is not similarity

def test_asserting_the_totality_a_concept_rules_out_is_not_full_credit():
    """"The next state depends on everything that happened before it" is the
    Markov property inverted. It shares almost every word with the teacher's
    own sentence, so the embedding scores it CLOSER to the concept (0.91) than
    to the taught misconception it restates (0.80)."""
    hmm = TOPIC_BY_NAME["Hidden Markov Models"]
    verdict, _ = outcome(hmm, "Markov property",
                         "the next state depends on everything that happened before it")
    assert verdict != "correct", verdict


def test_the_polarity_rule_does_not_fire_on_correct_answers():
    """"Each hidden state emits observable outputs" is correct and universal-
    sounding. Quantifying the states is not the same as claiming the totality
    the concept excludes, so the rule must stay off."""
    from app.nlp.analyzer import inverts_exclusivity
    hmm = TOPIC_BY_NAME["Hidden Markov Models"]
    markov = next(c for c in hmm["concepts"] if c["name"] == "Markov property")
    emissions = next(c for c in hmm["concepts"]
                     if c["name"].startswith("Observations"))
    assert not inverts_exclusivity(
        emissions["description"],
        "each hidden state emits visible outputs with some probability")
    for correct in ("only the current state matters for what happens next, "
                    "not the whole history",
                    "the next state does not depend on the whole history, "
                    "only on the current one",
                    "only now matters not the whole past"):
        assert not inverts_exclusivity(markov["description"], correct), correct


# ------------------------- 7. when it is unsure, it must ask something NEW

def test_a_probe_is_not_the_main_question_asked_again():
    """The whole mitigation for an uncertain answer is "ask a better
    follow-up". Three seeded probes were the main question reworded — one of
    them differed by a single word — so a student who could not answer the
    first was handed it back verbatim."""
    import difflib

    groups = [(t["name"], t["concepts"]) for t in TOPICS]
    groups.append((PYTHON_LECTURE["title"], PYTHON_LECTURE["reviewed_concepts"]))
    for topic_name, concepts in groups:
        for c in concepts:
            main = (c.get("main_question") or "").strip().lower()
            probe = (c.get("probe_question") or "").strip().lower()
            if not main or not probe:
                continue
            overlap = difflib.SequenceMatcher(None, main, probe).ratio()
            assert overlap < 0.85, (
                f"{topic_name} / {c['name']}: the probe restates the main question "
                f"({overlap:.2f} similar)\n  main : {main}\n  probe: {probe}")


def test_accepting_a_partial_answer_does_not_claim_the_whole_idea():
    """This wording is used at the moment a concept is recorded as PARTIAL —
    the same concept then appears under "worth another look". Saying "you have
    the main idea" there tells the student the opposite of what was recorded."""
    from app.nlp.conversation import ACK_ACCEPT_PARTIAL

    overclaims = ("main idea", "core of it", "you understand", "you've got it",
                  "that's right", "exactly", "complete")
    for line in ACK_ACCEPT_PARTIAL:
        lowered = line.lower()
        assert not any(p in lowered for p in overclaims), line
        # it still has to sound like a person, not a verdict
        assert not any(p in lowered for p in ("fail", "wrong", "incorrect")), line


# --------------- 8. student-facing text describes evidence, not the person

JUDGEMENTAL = ("low engagement", "not trying", "lazy", "careless", "you failed",
               "you are confused", "low ability", "poor effort", "didn't bother")


def test_no_student_facing_text_judges_the_students_effort():
    """The system observes how much evidence a session produced. It never
    observes effort or motivation, so it must not report on them. states.py
    was rewritten for this reason; the recommendation's per-state explanation
    and the progress-page bullets were saying it again in two other places."""
    from app.api.helpers import observation_evidence
    from app.recommend.rules import GENERIC_ACTIVITIES, STATE_WHY
    from app.states import STATE_STUDENT_DESCRIPTIONS, STATE_STUDENT_NAMES

    texts = list(STATE_WHY.values()) + list(STATE_STUDENT_NAMES) \
        + list(STATE_STUDENT_DESCRIPTIONS)
    for activity in GENERIC_ACTIVITIES.values():
        texts += [str(v) for v in activity.values()]

    class _Obs:  # the lowest-effort observation the bullets can describe
        features = [0.0] * 8
        misconception_names: list = []

    texts += observation_evidence(_Obs())

    for text in texts:
        lowered = (text or "").lower()
        hit = [w for w in JUDGEMENTAL if w in lowered]
        assert not hit, f"{hit} in student-facing text: {text!r}"
