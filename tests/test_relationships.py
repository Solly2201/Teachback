"""Relationship evidence semantics.

A concept relationship has three distinct outcomes, and the whole point of
these tests is that the middle one is NOT a failure:

    Demonstrated         the student showed the connection
    Not discussed        no evidence either way — never a gap, never a mistake
    Needs clarification  the student expressed the connection incompletely,
                         questionably or incorrectly

The knowledge check is a SEPARATE evidence channel: a correct MCQ never turns
into "you explained this connection", and a wrong MCQ never deletes an
explanation the student actually gave.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.helpers import (DEMONSTRATED, NEEDS_CLARIFICATION, NOT_DISCUSSED,
                             relationship_summary)
from app.api.quiz import _relationship_evidence
from app.hmm.model import hmm_available
from app.main import app
from app.nlp.analyzer import analyze_response, negated_terms
from app.nlp.conversation import build_plan, play_turn
from app.recommend.rules import recommend
from app.seed_content import PYTHON_LECTURE
from app.states import UNDERSTANDING

client = TestClient(app)

STRINGS = {
    "name": PYTHON_LECTURE["title"],
    "reference_explanation": PYTHON_LECTURE["material"],
    "concepts": PYTHON_LECTURE["reviewed_concepts"],
    "misconceptions": PYTHON_LECTURE["reviewed_misconceptions"],
    "relationships": PYTHON_LECTURE["reviewed_relationships"],
}
STRINGS_CHARACTERS = ("Strings", "Characters")


def _rel(text: str, pair=STRINGS_CHARACTERS, topic=STRINGS) -> str:
    """Analyzer status for one relationship after a single response."""
    analysis = analyze_response(text, topic)
    return {(r["source"], r["target"]): r["status"] for r in analysis["relationships"]}[pair]


# --------------------------------------------------------------- A. demonstrated

def test_a_explicit_relationship_is_demonstrated():
    assert _rel("A string is a sequence of characters, so each character is part of "
                "the string.") == "demonstrated"


def test_a_everyday_wording_is_demonstrated_too():
    """Below the direct-match bar, but both ends of the link are evidenced.

    This phrasing sits at cosine ~0.67 against the teacher's sentence — under
    the direct threshold — and used to be reported as no evidence at all.
    """
    assert _rel("They're basically individual letters or symbols inside the "
                "string.") == "demonstrated"


# -------------------------------------------------------------- B. not discussed

def test_b_unrelated_answer_leaves_relationship_not_shown():
    assert _rel("A string is text stored inside quotes.") == "not_shown"
    assert _rel("Strings are basically text that we store inside quotes.") == "not_shown"


def test_b_not_discussed_is_reported_as_its_own_state():
    plan = {"relationships": [
        {"source": "Strings", "label": "contain", "target": "Characters", "status": "pending"},
    ]}
    summary = relationship_summary(plan)
    assert summary[0]["status"] == NOT_DISCUSSED
    assert summary[0]["status_label"] == "Not discussed"
    # the wording must not suggest a mistake
    assert "clarif" not in summary[0]["status_label"].lower()


# ------------------------------------------------------------ C. contradicted

def test_c_explicit_contradiction_needs_clarification():
    """No teacher-authored wrong version here — the negation carries it."""
    assert _rel("A string is just a collection of separate variables, not "
                "characters.") == "contradicted"


def test_c_teacher_authored_contradiction_still_detected():
    assert _rel("split() joins a list of pieces back into one string.",
                pair=("split()", "List")) == "contradicted"


def test_c_correct_answer_sharing_a_word_with_the_wrong_version_is_not_accused():
    """"...join puts the pieces back together" shares "back" with the teacher's
    wrong version of split(), but states the right one with "breaks"."""
    assert _rel("split breaks the text into pieces and join puts the pieces back "
                "together.", pair=("split()", "List")) == "demonstrated"


def test_c_negation_of_something_else_is_not_a_contradiction():
    assert _rel("A string is not a number, it is a sequence of characters.") != "contradicted"


def test_c_negation_scope_is_local():
    assert "characters" in negated_terms("it is not characters")
    assert "characters" not in negated_terms("it is not a number but a long run of characters")


def test_c_reported_as_needs_clarification():
    plan = {"relationships": [
        {"source": "Strings", "label": "contain", "target": "Characters", "status": "contradicted"},
    ]}
    assert relationship_summary(plan)[0]["status"] == NEEDS_CLARIFICATION


# ------------------------------------------------------- D. partially explained

def _probe_answer(text: str) -> str:
    """Answer a direct probe of Strings -> Characters; returns the plan status."""
    plan = build_plan(STRINGS)
    for c in plan["concepts"]:
        c["status"] = "covered"
    entry = next(r for r in plan["relationships"]
                 if (r["source"], r["target"]) == STRINGS_CHARACTERS)
    plan["asked_kind"] = "relationship"
    plan["asked_rel"] = [entry.get("id"), entry["source"], entry["target"]]
    plan, _ = play_turn(plan, analyze_response(text, STRINGS), STRINGS)
    return next(r for r in plan["relationships"]
                if (r["source"], r["target"]) == STRINGS_CHARACTERS)["status"]


def test_d_probe_answered_off_target_stays_not_discussed():
    """Asked directly, answered about something else: still no evidence either
    way — being asked is not the same as getting it wrong."""
    assert _probe_answer("Strings are basically text that we store inside quotes.") == "pending"


def test_d_probe_given_up_on_stays_not_discussed():
    assert _probe_answer("I don't know.") == "pending"


def test_d_probe_answered_incompletely_needs_clarification():
    """Engaged with the connection but stopped short of establishing it."""
    assert _probe_answer("A string is a kind of container.") == "unclear"
    assert _probe_answer("A string is something you can look inside of, one bit "
                         "at a time.") == "unclear"


def test_d_partial_evidence_outside_a_probe_is_never_a_gap():
    """An answer that drifts near a connection nobody asked about must not be
    turned into a misunderstanding of it."""
    plan = build_plan(STRINGS)
    analysis = analyze_response("You can take a part of the string using start and end positions.",
                                STRINGS)
    partial = [r for r in analysis["relationships"] if r["status"] == "partial"]
    assert partial, "expected this answer to sit in the partial band"
    plan, _ = play_turn(plan, analysis, STRINGS)
    for p in partial:
        entry = next(r for r in plan["relationships"]
                     if (r["source"], r["target"]) == (p["source"], p["target"]))
        assert entry["status"] == "pending"


# ------------------------------------------------- E/F. MCQ is separate evidence

def _rel_mcq(target_correct: bool, teachback_status: str) -> dict:
    plan = {"relationships": [
        {"source": "Strings", "label": "contain", "target": "Characters",
         "status": teachback_status},
    ]}
    topic = {"relationships": [{"source": "Strings", "label": "contain", "target": "Characters",
                                "description": "A string is made of characters."}]}
    per_question = [{
        "id": 1, "kind": "relationship", "concept_name": "Strings",
        "options": ["Characters", "List", "Quotes", "Numbers"], "correct_index": 0,
        "selected_index": 0 if target_correct else 1, "correct": target_correct,
    }]
    return _relationship_evidence(topic, plan, per_question)[0]


def test_e_correct_mcq_does_not_fabricate_teachback_evidence():
    e = _rel_mcq(target_correct=True, teachback_status="pending")
    assert e["teachback_status"] == NOT_DISCUSSED
    assert e["teachback_label"] == "Not discussed"
    assert (e["mcq_correct"], e["mcq_total"]) == (1, 1)
    assert "didn't explicitly discuss it during TeachBack" in e["message"]
    # the MCQ must not be described as an explanation the student gave
    assert "you explained" not in e["message"].lower()


def test_f_wrong_mcq_does_not_erase_teachback_evidence():
    e = _rel_mcq(target_correct=False, teachback_status="demonstrated")
    assert e["teachback_status"] == DEMONSTRATED
    assert (e["mcq_correct"], e["mcq_total"]) == (0, 1)
    assert "your explanation still stands" in e["message"].lower()


def test_both_signals_agreeing_is_reported_as_such():
    e = _rel_mcq(target_correct=True, teachback_status="demonstrated")
    assert e["teachback_status"] == DEMONSTRATED and e["mcq_correct"] == 1


def test_relationship_with_no_mcq_carries_only_teachback_evidence():
    topic = {"relationships": [{"source": "Slicing", "label": "extracts", "target": "Substring",
                                "description": "Slicing extracts part of a string."}]}
    plan = {"relationships": [{"source": "Slicing", "label": "extracts", "target": "Substring",
                               "status": "pending"}]}
    e = _relationship_evidence(topic, plan, [])[0]
    assert e["teachback_status"] == NOT_DISCUSSED
    assert e["mcq_total"] is None and e["message"] == ""


# ------------------------------------------------------- recommendation impact

def test_not_discussed_relationships_are_not_learning_gaps():
    """Strong concept evidence with no relationship misconception must keep its
    positive interpretation, whatever went undiscussed."""
    rec = recommend(
        UNDERSTANDING, [], [],
        evidence={"demonstrated": ["Strings", "Characters", "Indexing"], "unclear": []},
        signals={"understanding": 1.0, "confidence": 0.7, "difficulty": 0.4},
        topic_def=STRINGS,
    )
    assert "needs clarification" not in rec["why"]
    assert "You showed understanding of" in rec["why"]
    assert not rec["notes"]


def test_relationship_needing_clarification_is_a_learning_gap():
    rec = recommend(
        UNDERSTANDING, [], [],
        evidence={"demonstrated": ["Strings"],
                  "unclear": ["the connection Strings → Characters"]},
        signals={"understanding": 0.6, "confidence": 0.6, "difficulty": 0.5},
        topic_def=STRINGS,
    )
    assert "the connection Strings → Characters still needs clarification" in rec["why"]


# ------------------------------------------------------------------ end to end

CASUAL = {
    "Strings": "Strings are basically text that we store inside quotes.",
    "String assignment": "You can put the string into a variable using =.",
    "Characters": "They're basically individual letters or symbols inside the string.",
    "Indexing": "You use the position to get a character, and Python starts from zero.",
    "Slicing": "You can take a part of the string using start and end positions.",
    "split() and join()": "split breaks the text into pieces and join puts the pieces back together.",
}


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_session_never_reports_an_undiscussed_connection_as_a_problem():
    teachers = client.get("/api/teachers").json()
    py = next(s for t in teachers for s in t["subjects"] if s["name"] == "Python Programming")
    topics = client.get(f"/api/topics?subject_id={py['id']}").json()
    topic = next(t for t in topics if "Strings" in t["name"])
    student = next(s for s in client.get("/api/students").json()
                   if s["name"] == "Shreshtha Bindal")

    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    concept = start["question"]["concept"]
    for _ in range(14):
        answer = CASUAL.get(concept and concept.split(" → ")[0],
                            "I think they are connected because one gives you the other.")
        step = client.post(f"/api/sessions/{sid}/respond", json={"text": answer}).json()
        if step["awaiting_self_report"]:
            break
        concept = step["followup"]["concept"]

    out = client.post(f"/api/sessions/{sid}/finish", json={
        "summary": "Strings are text in quotes and you can index and slice them.",
        "attention": 8, "confidence": 7, "difficulty": 4, "pace": "just right",
        "feedback_choices": [], "feedback_text": ""}).json()

    rels = out["relationship_summary"]
    assert {r["status"] for r in rels} <= {DEMONSTRATED, NOT_DISCUSSED, NEEDS_CLARIFICATION}
    # this student never says anything wrong about a connection
    assert not [r for r in rels if r["status"] == NEEDS_CLARIFICATION]
    assert [r for r in rels if r["status"] == DEMONSTRATED]

    evidence = " ".join(out["observation"]["evidence"])
    assert "no evidence either way" in evidence
    assert "Connection needing clarification" not in evidence
    # a relationship that simply never came up is not a reason to remediate
    assert "the connection" not in out["recommendation"]["why"]
