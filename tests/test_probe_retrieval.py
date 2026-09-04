"""Targeted teacher-grounded retrieval: the right material, only that
material, and every item traceable to its teacher-authored source."""
import numpy as np

from app.probe import retrieval
from app.probe.retrieval import retrieve

TOPIC_DEF = {
    "name": "Strings",
    "concepts": [
        {"id": 1, "name": "String", "description": "A string is a sequence of characters.",
         "main_question": "What is a string?", "easier_question": "",
         "probe_question": "How could you access one character of a string?",
         "application_question": "",
         "facts": ["Strings are immutable.", "Indexes start at 0."],
         "examples": ["'hello'[1] gives 'e'."]},
        {"id": 2, "name": "Variable", "description": "A name bound to a value.",
         "main_question": "What is a variable?", "easier_question": "",
         "probe_question": "", "application_question": "",
         "facts": ["A variable can be reassigned."], "examples": []},
    ],
    "relationships": [
        {"id": 10, "source": "String", "label": "is a", "target": "Sequence of characters",
         "description": "A string is an ordered sequence of characters.",
         "contradiction": "A string is a single indivisible value.",
         "probe_question": "If a string has several characters, how do you get one?"},
    ],
    "misconceptions": [
        {"id": 20, "name": "String equals variable",
         "description": "A string is the same thing as a variable.",
         "clarification": "A variable can hold a string; they are not the same thing.",
         "probe_question": "Can a variable hold something that is not a string?"},
    ],
}

PLAN = {"concepts": [{"id": 1, "name": "String", "status": "partial"}], "current": 0}


def decision(ttype, tid, name="String"):
    return {"action": "ASK_PROBE", "target_type": ttype, "target_id": tid,
            "target_name": name, "difficulty": "easy"}


def test_concept_retrieval_is_targeted_with_provenance():
    items = retrieve(TOPIC_DEF, decision("concept", 1), PLAN, "a string holds text")
    ids = [i["id"] for i in items]
    assert "concept:1" in ids
    assert "concept_fact:1:0" in ids and "concept_fact:1:1" in ids
    assert "concept_example:1:0" in ids
    assert "concept_question:1:main" in ids and "concept_question:1:probe" in ids
    # targeted means targeted: nothing from the OTHER concept leaks in
    assert all(":2" not in i["id"].split(":")[0] + ":" + i["id"].split(":")[1]
               for i in items if i["id"].startswith("concept"))
    texts = " ".join(i["text"] for i in items)
    assert "Variable" not in texts and "reassigned" not in texts


def test_relationship_retrieval_includes_endpoints_and_wrong_claim():
    items = retrieve(TOPIC_DEF, decision("relationship", 10), PLAN, "")
    ids = {i["id"] for i in items}
    assert "relationship:10" in ids
    assert "relationship_contradiction:10" in ids
    assert "relationship_question:10" in ids
    assert "concept:1" in ids  # the String endpoint's teacher explanation
    wrong = next(i for i in items if i["id"] == "relationship_contradiction:10")
    assert wrong["kind"] == "known_wrong_claim"


def test_misconception_retrieval_carries_the_clarification():
    items = retrieve(TOPIC_DEF, decision("misconception", 20), PLAN, "")
    ids = {i["id"] for i in items}
    assert {"misconception:20", "misconception_clarification:20",
            "misconception_question:20"} <= ids
    assert "concept:1" in ids  # the concept under discussion anchors context


def test_unknown_target_returns_nothing():
    assert retrieve(TOPIC_DEF, decision("concept", 999), PLAN, "") == []
    assert retrieve(TOPIC_DEF, decision("relationship", 999), PLAN, "") == []


def test_concept_with_no_substantive_material_returns_nothing():
    bare = {"name": "T", "concepts": [{"id": 5, "name": "Bare", "description": "",
                                       "main_question": "What is Bare?", "facts": [],
                                       "examples": []}],
            "relationships": [], "misconceptions": []}
    # a lone teacher question with no explanation/facts cannot ground a probe
    assert retrieve(bare, decision("concept", 5, "Bare"), PLAN, "") == []


def test_many_facts_are_ranked_against_the_answer_and_capped(monkeypatch):
    facts = [f"Fact number {i} about strings." for i in range(10)]
    tdef = {"name": "T", "concepts": [{"id": 1, "name": "String", "description": "desc",
                                       "facts": facts, "examples": []}],
            "relationships": [], "misconceptions": []}

    def fake_embed(texts):
        # make fact 9 most similar to the answer, then 8, 7, ...
        vectors = np.zeros((len(texts), 3), dtype=np.float32)
        for i in range(len(texts) - 1):
            vectors[i, 0] = i
        vectors[-1, 0] = 100.0
        return vectors

    monkeypatch.setattr("app.nlp.embedder.embed", fake_embed)
    items = retrieve(tdef, decision("concept", 1), PLAN, "the answer")
    fact_ids = [i["id"] for i in items if i["id"].startswith("concept_fact")]
    assert len(fact_ids) == retrieval.MAX_FACT_ITEMS
    assert "concept_fact:1:9" in fact_ids and "concept_fact:1:0" not in fact_ids


def test_embedder_failure_degrades_instead_of_crashing(monkeypatch):
    facts = [f"Fact {i}." for i in range(10)]
    tdef = {"name": "T", "concepts": [{"id": 1, "name": "String", "description": "desc",
                                       "facts": facts, "examples": []}],
            "relationships": [], "misconceptions": []}

    def broken_embed(texts):
        raise RuntimeError("model not available")

    monkeypatch.setattr("app.nlp.embedder.embed", broken_embed)
    items = retrieve(tdef, decision("concept", 1), PLAN, "answer")
    fact_ids = [i["id"] for i in items if i["id"].startswith("concept_fact")]
    assert len(fact_ids) == retrieval.MAX_FACT_ITEMS  # first N kept instead
