"""Regression tests for what the deterministic extraction is allowed to claim.

Covers the failure modes the pipeline is supposed to be immune to:
* a heading is not automatically a concept
* the first sentence under a heading is not automatically the meaning
* a leftover fragment is not automatically an "important fact"
* two concepts in one sentence is not automatically a relationship
* nothing is invented that the material does not support

The material used here is deliberately varied — Markdown, plain prose,
heading-only notes, metadata-heavy text, duplicate headings — and spans
several subjects, so nothing can pass by being tuned to one lecture.
"""
import pytest

from app.nlp.lecture_parser import is_generic_heading
from app.nlp.lecture_prep import best_meaning, prepare_lecture, rank_facts

STRUCTURED_NOTES = """# Databases

## Learning Objectives
- Explain what an index does.
- Explain what normalisation is for.

## Indexing

Slide 7 of 40
An index is a data structure that lets the database find rows without scanning
the whole table.
An index speeds up reads and slows down writes.

Example:
CREATE INDEX idx_name ON users(name)

## Normalisation

Normalisation means organising tables so that each fact is stored in exactly
one place.
Normalisation reduces duplicated data.

## Agenda

## Thank You

Questions?
"""

PLAIN_NOTES = (
    "today we talked about a few things in class. a hash function maps a key to a "
    "bucket index so lookups can go straight to the right place. collisions happen "
    "when two keys land in the same bucket, and chaining stores them in a list. "
    "we also mentioned the canteen timings and the sports day schedule."
)

METADATA_NOTES = """# Course Pack

## Cloud Computing

5 (c) Copyright 2014 EMC Corporation.
All rights reserved.

## Elasticity

Elasticity means capacity can grow and shrink automatically as demand changes.
organizations
the coming decade
Elastic systems add servers when traffic rises and remove them when it falls.
"""


# ----------------------------------------------- headings are not concepts

def test_structural_headings_are_recognised_as_structure():
    for heading in ("Agenda", "Thank You", "Lesson: Cloud Computing Overview",
                    "Module 3:", "Chapter 4 — Overview", "Table of Contents",
                    "Learning Objectives", "References", "Q&A", "12",
                    "(c) Copyright 2014 EMC Corporation"):
        assert is_generic_heading(heading), f"{heading!r} should be structural"


def test_real_concept_headings_are_not_treated_as_structure():
    for heading in ("Indexing", "Normalisation", "Bubble Sort", "Backpropagation",
                    "On-demand Self-service", "Hash Function", "Transaction Isolation"):
        assert not is_generic_heading(heading), f"{heading!r} should be a concept candidate"


def test_heading_without_explanatory_support_is_not_a_concept():
    notes = "# Topic\n\n## Essential Characteristics\n\n## Service Models\n\n## Summary\n"
    prep = prepare_lecture(notes)
    assert prep["concepts"] == []
    assert prep["structure"]["notes"], "the teacher should be told nothing was supported"


def test_structured_notes_keep_taught_ideas_and_drop_structure():
    names = {c["name"] for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]}
    assert {"Indexing", "Normalisation"} <= names
    assert not ({"Agenda", "Thank You", "Learning Objectives"} & names)


# ------------------------------------------------------- meaning selection

def test_meaning_is_chosen_not_taken_from_the_first_line():
    """The first line under the heading is metadata; the definition is second."""
    indexing = next(c for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]
                    if c["name"] == "Indexing")
    assert "Slide 7" not in indexing["description"]
    assert indexing["description"].startswith("An index is a data structure")


def test_meaning_never_becomes_copyright_or_a_fragment():
    prep = prepare_lecture(METADATA_NOTES)
    for c in prep["concepts"]:
        assert "copyright" not in c["description"].lower()
        assert c["description"].strip().lower() not in ("organizations", "the coming decade")
    # the concept whose only "content" was a copyright notice is not suggested
    assert "Cloud Computing" not in {c["name"] for c in prep["concepts"]}


def test_best_meaning_prefers_a_definition_over_a_bare_mention():
    candidates = [
        "See the notes for more.",
        "Recursion is a technique where a function calls itself on a smaller input.",
        "Recursion again.",
    ]
    meaning, score = best_meaning("Recursion", candidates)
    assert meaning.startswith("Recursion is a technique")
    assert score > 0


def test_best_meaning_returns_nothing_when_there_is_nothing_to_choose():
    meaning, _ = best_meaning("Cloud Computing", ["5 (c) Copyright 2014 EMC Corporation.",
                                                  "All rights reserved."])
    assert meaning == ""


def test_meaning_is_usable_as_a_reference_explanation():
    for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]:
        if c["description"]:
            assert len(c["description"].split()) >= 5
            assert not c["description"].strip().endswith(":")


# ---------------------------------------------------------- fact selection

def test_fragments_and_metadata_never_become_facts():
    prep = prepare_lecture(METADATA_NOTES)
    elasticity = next(c for c in prep["concepts"] if c["name"] == "Elasticity")
    assert elasticity["facts"], "the real supporting statement was dropped"
    for fact in elasticity["facts"]:
        assert fact.lower() not in ("organizations", "the coming decade")
        assert len(fact.split()) >= 4


def test_rank_facts_scores_rather_than_taking_the_first_n():
    meaning = "An index is a data structure that speeds up lookups."
    candidates = [
        "rows",                                                  # fragment
        "(c) 2019 University Press.",                            # metadata
        "An index costs extra space and slows down writes.",      # good
        "An index is a data structure that speeds up lookups.",   # duplicates meaning
        "Indexes are stored as B-trees in most databases.",       # good
    ]
    facts = rank_facts("Indexing", meaning, candidates)
    assert "rows" not in facts
    assert not any("University Press" in f for f in facts)
    assert meaning not in facts
    assert len(facts) == 2


def test_facts_do_not_simply_restate_the_meaning():
    for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]:
        assert c["description"] not in c["facts"]


def test_code_examples_are_examples_not_facts():
    indexing = next(c for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]
                    if c["name"] == "Indexing")
    assert any("CREATE INDEX" in e for e in indexing["examples"])
    assert not any("CREATE INDEX" in f for f in indexing["facts"])


# ------------------------------------------------------------- confidence

def test_every_suggestion_carries_an_honest_confidence_label():
    for c in prepare_lecture(STRUCTURED_NOTES)["concepts"]:
        assert c["confidence"] in ("strong", "moderate", "weak")
        assert c["confidence_label"]
        assert c["confidence_reason"]


def test_concept_count_is_not_forced_to_a_fixed_number():
    """A short lecture yields few concepts; padding with noise is worse."""
    two = prepare_lecture(STRUCTURED_NOTES)["concepts"]
    assert 1 <= len(two) <= 4, [c["name"] for c in two]
    long_notes = "# Networks\n" + "".join(
        f"\n## Layer {i}\n\nLayer {i} is responsible for a distinct part of moving a packet "
        f"between two hosts.\nLayer {i} adds its own header to the data it receives.\n"
        for i in range(1, 12))
    many = prepare_lecture(long_notes)["concepts"]
    assert len(many) > len(two)
    assert len(many) <= 8  # still bounded by MAX_CONCEPTS


# ---------------------------------------------------------- relationships

def test_co_mention_alone_does_not_create_a_relationship():
    """A false relationship is worse than a missing one: it later shapes how a
    student's answer is judged."""
    notes = """# Systems

## Cache

A cache is a small fast store that keeps recently used data close to the processor.
Today we will look at cache and memory, and then take a short break.

## Memory

Memory holds the data and instructions a program is currently working with.
"""
    rels = prepare_lecture(notes)["relationships"]
    assert not any({r["source"], r["target"]} == {"Cache", "Memory"} for r in rels), rels


def test_relationship_is_created_when_a_relational_cue_connects_two_concepts():
    notes = """# Systems

## Cache

A cache is a small fast store that keeps recently used data close to the processor.
The cache stores copies of values taken from memory.

## Memory

Memory holds the data and instructions a program is currently working with.
"""
    rels = prepare_lecture(notes)["relationships"]
    pair = next((r for r in rels if r["source"] == "Cache" and r["target"] == "Memory"), None)
    assert pair is not None, rels
    assert pair["label"] != "relates to"
    assert pair["description"]


def test_explicit_connection_section_stays_authoritative():
    notes = """# Strings

## Strings

A string is text stored between quotes.

## Characters

A character is one symbol inside a string.

## Important Connections

- Strings → contain → Characters
"""
    rels = prepare_lecture(notes)["relationships"]
    explicit = next(r for r in rels if r["source"] == "Strings")
    assert explicit["target"] == "Characters"
    assert explicit["label"] == "contain"
    assert explicit["origin"] == "explicit"


def test_no_relationship_is_invented_from_metadata():
    for r in prepare_lecture(METADATA_NOTES)["relationships"]:
        assert "copyright" not in r["description"].lower()


# ------------------------------------------------------------- objectives

def test_objectives_rank_candidates_without_inventing_them():
    prep = prepare_lecture(STRUCTURED_NOTES, objectives=["Explain what an index does."])
    by_name = {c["name"]: c for c in prep["concepts"]}
    assert "Indexing" in by_name
    assert by_name["Indexing"]["objective_match"] > by_name["Normalisation"]["objective_match"]
    # an objective about something the lecture never covered creates nothing
    other = prepare_lecture(STRUCTURED_NOTES,
                            objectives=["Explain how a neural network is trained."])
    assert not any("neural" in c["name"].lower() for c in other["concepts"])


def test_notes_without_objectives_still_work():
    prep = prepare_lecture(STRUCTURED_NOTES)
    assert prep["concepts"]
    assert prep["objectives"], "objectives should be drafted when the teacher gives none"


# ------------------------------------------------------ shapes of material

def test_plain_prose_without_headings_still_extracts_taught_ideas():
    prep = prepare_lecture(PLAIN_NOTES, title="Hash Tables")
    names = " ".join(c["name"].lower() for c in prep["concepts"])
    assert "hash" in names or "collision" in names or "chaining" in names
    assert "canteen" not in names and "sports" not in names


def test_duplicate_headings_are_merged_into_one_concept():
    notes = """# Trees

## Pruning

Pruning removes branches that do not improve the predictions of the tree.

## Depth

Depth is the number of splits from the root to a leaf.

## Pruning

Pruning is applied after the tree has been fully grown.
"""
    concepts = prepare_lecture(notes)["concepts"]
    names = [c["name"] for c in concepts]
    assert names.count("Pruning") == 1
    pruning = next(c for c in concepts if c["name"] == "Pruning")
    assert pruning["facts"], "evidence from the second occurrence should be merged in"


@pytest.mark.parametrize("material", ["", "   ", "# Title\n", "Lecture 4"])
def test_empty_or_near_empty_material_is_handled_without_inventing_anything(material):
    prep = prepare_lecture(material)
    assert prep["concepts"] == []
    assert prep["relationships"] == []
    assert prep["misconception_suggestions"] == []


def test_nothing_outside_the_material_is_invented():
    prep = prepare_lecture(STRUCTURED_NOTES)
    material = STRUCTURED_NOTES.lower()
    for c in prep["concepts"]:
        for text in [c["description"]] + c["facts"] + c["examples"]:
            if text:
                # every claim is a verbatim span of the teacher's own material
                assert text.strip().lower()[:40] in material.replace("\n", " ")
