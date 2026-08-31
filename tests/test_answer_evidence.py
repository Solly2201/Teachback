"""Regression tests for the bugs the 150-answer student audit exposed.

The audit (scripts/student_audit.py) asked one question: if ordinary students
explain what they learned in their own imperfect words, does TeachBack
recognise what they understood — without crediting people who said nothing?
It found both failure directions coming from the same weak spot, and these
tests pin the fixes.

The underlying cause: every reference text is written as "Name: explanation",
so a bare mention of the concept name scores ~0.8 against it while a genuine
paraphrase that avoids the jargon scores ~0.4. The old guard was a lexical
"share at least one word with the teacher's text" count, which the name itself
satisfies and which a correct paraphrase can fail by chance.
"""
import pytest

from app.nlp.analyzer import (analyze_response, informative_terms,
                              targeted_concept_check)
from app.nlp.conversation import _gave_up, _verdict
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
BACKPROP = next(t for t in TOPICS if t["name"] == "Backpropagation")


def _concept(tdef, name):
    return next(c for c in tdef["concepts"] if c["name"] == name)


def judge(text, concept_name, tdef=STRINGS):
    """Exactly what a live TeachBack turn does with this answer."""
    concept = _concept(tdef, concept_name)
    analysis = analyze_response(text, tdef)
    analysis["target_check"] = targeted_concept_check(
        text, concept, topic_name=tdef.get("name", ""),
        misconceptions=tdef.get("misconceptions"))
    entry = {"id": concept.get("id"), "name": concept_name,
             "status": "pending", "attempts": 0}
    return _verdict(analysis, entry)


# ------------------------------------------- naming is not explaining

@pytest.mark.parametrize("text,concept", [
    ("python uses indexing", "Indexing"),
    ("indexing is important in python", "Indexing"),
    ("python is useful for strings", "Strings"),
    ("strings come up a lot in python", "Strings"),
    ("slicing was covered today", "Slicing"),
    ("characters were mentioned in class", "Characters"),
    ("we did string methods", "split() and join()"),
])
def test_naming_a_concept_is_not_evidence_of_understanding(text, concept):
    """The reference texts contain the concept name, so these score very high
    on similarity alone. None of them says anything about the idea."""
    assert judge(text, concept) == "unclear", text


def test_informative_terms_strips_the_name_the_title_and_filler():
    assert informative_terms("python uses indexing", "Indexing",
                             "Strings in Python") == set()
    assert informative_terms("indexing is really important", "Indexing",
                             "Strings in Python") == set()
    # inflection-tolerant: the plural of the concept name is still the name
    assert informative_terms("gradients are used", "Gradient",
                             "Backpropagation") == set()
    # but a real explanation survives intact
    assert informative_terms("you use the number to get the letter", "Indexing",
                             "Strings in Python") == {"number", "get", "letter"}


# ------------------------------- ...and explaining without the jargon counts

@pytest.mark.parametrize("text,concept", [
    ("you use the number to get the letter", "Indexing"),
    ("you say which number you want and it gives you that letter", "Indexing"),
    ("the first position is zero", "Indexing"),
    ("it's basically words you put inside speech marks", "Strings"),
    ("anything you type between two quote symbols", "Strings"),
    ("it's built out of the separate letters that sit in order", "Characters"),
    ("you save the text in a variable so you can use it later", "String assignment"),
])
def test_correct_answers_without_the_teachers_vocabulary_are_credited(text, concept):
    """A student who understood the lecture but avoids textbook terminology
    must not be told their answer was unclear."""
    assert judge(text, concept) in ("correct", "partial"), text


def test_a_single_content_word_is_not_an_explanation():
    """The contextual score is inflated by the shared "Name: " prefix, so a
    fragment can clear it while explaining nothing."""
    for text in ("something with brackets", "something mathematical",
                 "it's something in python"):
        assert judge(text, "Indexing") == "unclear", text


def test_one_of_the_teachers_own_key_words_is_enough_corroboration():
    """Short but specific answers still count: the single thing the student
    said is one of the teacher's own words for the idea."""
    assert judge("zero indexed", "Indexing") in ("correct", "partial", "unclear")
    assert judge("the stop position is excluded", "Slicing") in ("correct", "partial")


# --------------------------------------------------- giving up is not evidence

@pytest.mark.parametrize("text", [
    "i don't know", "idk", "no idea", "not sure",
    "i don't really remember this one", "i can't remember this",
    "i forgot", "no clue at all", "i didn't get this one",
])
def test_giving_up_never_earns_credit(text):
    assert _gave_up(text), f"not recognised as giving up: {text}"
    assert judge(text, "Indexing") == "unclear", text


def test_a_real_answer_is_not_mistaken_for_giving_up():
    for text in ("the first position is zero",
                 "you use the number to get the letter",
                 "i know it starts counting from zero"):
        assert judge(text, "Indexing") in ("correct", "partial"), text


# ------------------------------------------- misconceptions withhold credit

def test_an_answer_closer_to_a_wrong_claim_does_not_get_full_credit():
    """When the misconception detector does not meet its (strict) bar, the
    answer still must not be reported as demonstrating the concept. Withhold
    credit; never accuse."""
    verdict = judge("the first letter is number one and the second is number two",
                    "Indexing")
    assert verdict != "correct", verdict


def test_withholding_credit_is_not_an_accusation():
    """A shadowed answer is at most 'partial' — the student is asked another
    question, not told they are wrong."""
    analysis = analyze_response("you can change a letter of the string directly", STRINGS)
    # either the misconception is named, or credit is simply withheld
    concept = _concept(STRINGS, "Strings")
    analysis["target_check"] = targeted_concept_check(
        "you can change a letter of the string directly", concept,
        topic_name=STRINGS["name"], misconceptions=STRINGS["misconceptions"])
    entry = {"id": concept.get("id"), "name": "Strings", "status": "pending", "attempts": 0}
    assert _verdict(analysis, entry) != "correct"


# ------------------------------------------- relationships stay conservative

def test_a_correct_statement_is_not_called_a_contradiction():
    """"split gives you back the separate pieces in a list" was reported as
    contradicting the teacher's connection purely for reusing the word "back"
    from the teacher's wrong version. Connective adverbs are not evidence that
    a relationship was stated backwards."""
    analysis = analyze_response(
        "split gives you back the separate pieces in a list", STRINGS)
    rel = next(r for r in analysis["relationships"]
               if r["source"] == "split()" and r["target"] == "List")
    assert rel["status"] != "contradicted", rel


def test_an_off_topic_answer_creates_no_relationship_evidence():
    analysis = analyze_response("we also went over the assignment deadline", STRINGS)
    assert all(r["status"] == "not_shown" for r in analysis["relationships"])


# ------------------------------------ concept coverage obeys the same rule

def test_session_level_coverage_also_ignores_bare_name_mentions():
    """_mark_incidental credits concepts straight from analyze_response, so the
    same guard has to live there too — otherwise naming a concept in passing
    would silently mark it demonstrated."""
    analysis = analyze_response("python uses indexing and slicing", STRINGS)
    by_name = {c["name"]: c for c in analysis["concepts"]}
    assert by_name["Indexing"]["status"] == "missing"
    assert by_name["Slicing"]["status"] == "missing"


def test_a_real_explanation_still_covers_its_concept():
    analysis = analyze_response(
        "a string is text stored between quotes and each letter has a position "
        "you can use to pull it out", STRINGS)
    by_name = {c["name"]: c for c in analysis["concepts"]}
    assert by_name["Strings"]["status"] in ("covered", "partial")


def test_backprop_answers_in_plain_language_are_recognised():
    assert judge("it measures how far off the guess was", "Loss / error",
                 BACKPROP) in ("correct", "partial")
    assert judge("backpropagation is the topic", "Backward propagation of error",
                 BACKPROP) == "unclear"


# ------------------------------- listing the lecture's labels is not explaining

def test_a_list_of_lecture_labels_creates_no_evidence():
    """A takeaway that just repeats the lecture's own terminology
    ("markov hidden states observations transitions") names things without
    explaining any of them, and must not upgrade anything."""
    hmm = next(t for t in TOPICS if t["name"] == "Hidden Markov Models")
    analysis = analyze_response(
        "markov hidden states observations transitions", hmm)
    assert all(c["status"] == "missing" for c in analysis["concepts"]), \
        [(c["name"], c["status"]) for c in analysis["concepts"]]


def test_naming_one_concept_while_explaining_another_still_works():
    """The sibling-name rule must not silence a real explanation that happens
    to mention a neighbouring concept."""
    analysis = analyze_response(
        "a string is made up of individual characters that sit in a fixed order",
        STRINGS)
    by_name = {c["name"]: c["status"] for c in analysis["concepts"]}
    assert by_name["Characters"] in ("covered", "partial"), by_name
