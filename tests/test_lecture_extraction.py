"""Regression tests for the structured lecture parser and concept extraction.

The Strings notes (the seeded sample lecture material) are the fixture: the
extraction must find the taught concepts and must NOT primarily produce noise
like "Letters", "Values", "Python" or "Operator".
"""
from app.nlp.lecture_parser import parse_connection_line, parse_lecture
from app.nlp.lecture_prep import prepare_lecture
from app.seed_content import PYTHON_LECTURE

STRINGS_NOTES = PYTHON_LECTURE["material"]

NOISE_NAMES = {"letters", "letter", "values", "value", "python", "operator", "operators",
               "thing", "things", "example", "examples"}


def _prep(**kwargs):
    return prepare_lecture(STRINGS_NOTES, title="Strings in Python", **kwargs)


def test_headings_detected_as_structure():
    doc = parse_lecture(STRINGS_NOTES)
    assert doc["has_structure"]
    assert doc["title"] == "Strings in Python"
    headings = [s["heading"] for s in doc["sections"]]
    for expected in ("Strings", "Indexing", "Slicing"):
        assert expected in headings, f"heading {expected} not detected"


def test_objectives_and_special_sections_parsed():
    doc = parse_lecture(STRINGS_NOTES)
    assert len(doc["objectives"]) == 3
    assert len(doc["connections"]) == 4
    assert len(doc["mistakes"]) == 2
    assert "Strings store text" in doc["summary"]


def test_code_examples_not_treated_as_prose():
    doc = parse_lecture(STRINGS_NOTES)
    indexing = next(s for s in doc["sections"] if s["heading"] == "Indexing")
    # the code lines are kept as examples, not sentences
    assert any("s[0]" in e for e in indexing["examples"])
    assert not any("s[0]" in s for s in indexing["sentences"])


def test_extracted_concepts_are_the_taught_ones_not_noise():
    prep = _prep()
    names = {c["name"] for c in prep["concepts"]}
    lower = {n.lower() for n in names}
    # taught concepts present
    for expected in ("Strings", "Indexing", "Slicing"):
        assert expected in names, f"{expected} missing from {names}"
    # noise absent
    assert not (lower & NOISE_NAMES), f"noise concepts suggested: {lower & NOISE_NAMES}"


def test_concepts_keep_source_evidence_and_examples():
    prep = _prep()
    indexing = next(c for c in prep["concepts"] if c["name"] == "Indexing")
    assert indexing["source_section"] == "Indexing"
    assert "Indexes start at 0 in Python." in indexing["facts"]
    assert any("s[0]" in e for e in indexing["examples"])
    assert indexing["description"].startswith("Indexing means using a position")


def test_relationships_come_from_connection_lines():
    prep = _prep()
    pairs = {(r["source"], r["target"]) for r in prep["relationships"]}
    assert ("Strings", "Characters") in pairs
    assert any(r["source"] == "split()" for r in prep["relationships"])
    # every relationship carries its source line/sentence
    assert all(r.get("description") for r in prep["relationships"])


def test_misconceptions_parsed_with_clarifications():
    prep = _prep()
    miscons = {m["name"]: m for m in prep["misconception_suggestions"]}
    idx = next((m for name, m in miscons.items() if "1" in name or "one" in name.lower()), None)
    assert idx is not None, f"index-1 mistake not suggested: {list(miscons)}"
    assert "0" in idx["clarification"]


def test_questions_are_conversational_and_grounded():
    prep = _prep()
    for c in prep["concepts"]:
        assert c["main_question"].startswith("What did you understand about")
        assert c["easier_question"]
    indexing = next(c for c in prep["concepts"] if c["name"] == "Indexing")
    # the probe quotes an actual lecture fact
    assert "Indexes start at 0" in indexing["probe_question"]


def test_teacher_objectives_influence_unstructured_extraction():
    plain = ("today we talked about many things in class. pruning removes branches "
             "that do not help the tree make better predictions. pruning keeps models smaller. "
             "we also mentioned the cafeteria menu changes and the sports day schedule.")
    with_obj = prepare_lecture(plain, title="Trees", objectives=["Explain pruning."])
    names = {c["name"].lower() for c in with_obj["concepts"]}
    assert any("pruning" in n for n in names)


def test_unsupported_concepts_not_invented():
    prep = _prep()
    names = {c["name"].lower() for c in prep["concepts"]}
    # nothing outside the lecture (no classes/inheritance/decorators)
    for foreign in ("class", "inheritance", "decorator", "loop", "function"):
        assert not any(foreign in n for n in names)


def test_suggested_activities_grounded_in_lecture():
    prep = _prep()
    assert prep["activities"], "no suggested activities"
    states = {a["target_state"] for a in prep["activities"]}
    assert {"not_trying", "unclear", "struggling", "understanding", "confident"} <= states
    text = " ".join(a["title"] + a["content"] + a["question"] for a in prep["activities"])
    assert "Strings" in text or "strings" in text


def test_connection_line_parser():
    parsed = parse_connection_line("Slicing → extracts → a Substring")
    assert parsed == {"source": "Slicing", "label": "extracts", "target": "a Substring",
                      "description": "Slicing → extracts → a Substring"}
    assert parse_connection_line("Just a normal sentence.") is None
