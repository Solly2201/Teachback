"""Regression tests for PDF ingestion: layout-aware extraction + cleanup.

These reproduce the failure that motivated the work. A real corporate slide
deck carries a running header, a running footer, a slide number and an
"(c) Copyright ... All rights reserved." line on every slide. Flattening it
with plain text extraction made that decoration indistinguishable from
teaching content, so a copyright notice could end up as a concept's meaning.

The assertions are about STRUCTURE, not about cloud computing: the same
properties are checked on a completely different deck (sorting algorithms) so
nothing can be satisfied by hardcoding one subject's vocabulary.
"""
import base64

import pytest
from fastapi.testclient import TestClient
from pdf_decks import COPYRIGHT, FOOTER, HEADER, cloud_deck, scanned_deck, sorting_deck
from pdf_fixture import build_pdf, line, spaced_line

from app.main import app
from app.nlp.lecture_parser import parse_lecture
from app.nlp.lecture_prep import ScannedPdfError, extract_material, prepare_lecture
from app.nlp.pdf_clean import clean_document, document_to_markdown, pdf_to_notes
from app.nlp.pdf_extract import extract_document

client = TestClient(app)

DECKS = [("cloud", cloud_deck), ("sorting", sorting_deck)]


def _notes(deck) -> str:
    """Cleaned lecture notes from a deck factory or from raw PDF bytes."""
    data = deck() if callable(deck) else deck
    return pdf_to_notes(data)[0]


def _prep(deck, **kwargs):
    return prepare_lecture(_notes(deck), **kwargs)


# --------------------------------------------------------------- extraction

def test_extraction_preserves_pages_blocks_and_geometry():
    doc = extract_document(cloud_deck())
    assert doc["page_count"] == 10
    assert doc["extractor"] in ("pdfplumber", "pypdf")
    assert not doc["scanned"]
    for page in doc["pages"]:
        for block in page["blocks"]:
            assert block["page_number"] == page["number"]
            assert isinstance(block["order"], int)
            assert block["text"].strip()
    # layout-aware extractors give geometry and font size; both are what the
    # cleanup uses to tell a running footer from a slide title
    if doc["extractor"] == "pdfplumber":
        first = doc["pages"][0]["blocks"][0]
        assert first["bbox"] is not None and len(first["bbox"]) == 4
        assert first["size"] and doc["body_size"]


def test_extraction_is_deterministic():
    a, ra = pdf_to_notes(cloud_deck())
    b, rb = pdf_to_notes(cloud_deck())
    assert a == b
    assert ra["removed_total"] == rb["removed_total"]


# ------------------------------------------------------------------ cleanup

@pytest.mark.parametrize("name,deck", DECKS)
def test_copyright_headers_footers_and_page_numbers_are_removed(name, deck):
    text, report = pdf_to_notes(deck())
    lowered = text.lower()
    assert "all rights reserved" not in lowered
    assert "copyright" not in lowered
    assert "©" not in text
    reasons = {r["reason"] for r in report["removed_by_reason"]}
    assert "copyright" in reasons
    assert {"running_header", "running_footer"} & reasons
    assert report["removed_total"] >= report["page_count"]


def test_specific_boilerplate_lines_do_not_survive():
    text, _ = pdf_to_notes(cloud_deck())
    assert HEADER not in text
    assert FOOTER not in text
    assert COPYRIGHT not in text


def test_page_numbers_never_appear_as_standalone_lines():
    text, _ = pdf_to_notes(cloud_deck())
    for raw in text.split("\n"):
        stripped = raw.strip().lstrip("#- ").strip()
        assert not stripped.isdigit(), f"a bare page number survived: {raw!r}"


def test_boilerplate_only_page_is_dropped_entirely():
    text, report = pdf_to_notes(cloud_deck())
    # slide 8 of the fixture carries nothing but decoration
    assert 8 in report["empty_pages"]
    assert "<!-- page 8 -->" not in text


def test_repeated_educational_heading_is_preserved():
    """Repetition alone must not delete content.

    "Cloud Computing" is the title of three different slides in the fixture
    and IS the lecture's central idea. The cleanup removes repeated page
    DECORATION, not repeated teaching.
    """
    text, _ = pdf_to_notes(cloud_deck())
    assert text.lower().count("cloud computing") >= 2
    doc = parse_lecture(text)
    assert any(s["heading"] == "Cloud Computing" for s in doc["sections"])


def test_letter_spaced_and_hyphenated_artifacts_are_repaired():
    pdf = build_pdf([spaced_line("Cloud Computing", 60) + [
        line("Virtualisation lets one physical machine run several inde-", 140, 14),
        line("pendent virtual machines at the same time.", 160, 14),
        line("Each virtual machine behaves like a separate computer with its own", 190, 14),
        line("operating system, memory and storage allocation.", 210, 14),
    ]])
    text, _ = pdf_to_notes(pdf)
    assert "Cloud Computing" in text
    assert "independent virtual machines" in text
    assert "inde- pendent" not in text


# --------------------------------------------------------------- provenance

def test_page_provenance_survives_into_concepts():
    doc = parse_lecture(_notes(cloud_deck()))
    assert doc["has_pages"]
    assert any(s.get("page") for s in doc["sections"])

    prep = _prep(cloud_deck, title="Cloud Computing")
    assert prep["concepts"], "no concepts extracted from the deck"
    for c in prep["concepts"]:
        assert c["source_page"] or c["source_pages"], f"{c['name']} has no page provenance"
        assert "page" in c["confidence_reason"].lower()


def test_page_markers_are_structure_not_content():
    text = _notes(cloud_deck())
    assert "<!-- page" in text          # provenance is carried in the notes...
    doc = parse_lecture(text)           # ...and never leaks into the content
    body = " ".join(s["heading"] + " ".join(s["sentences"]) for s in doc["sections"])
    assert "<!--" not in body


# -------------------------------------------------- the end-to-end guarantee

@pytest.mark.parametrize("name,deck", DECKS)
def test_no_boilerplate_becomes_a_concept_meaning_or_fact(name, deck):
    prep = _prep(deck)
    banned = ("copyright", "all rights reserved", "©", "(c) 20", "(c) 19")
    for c in prep["concepts"]:
        blob = " ".join([c["name"], c["description"]] + c["facts"] + c["examples"]).lower()
        for bad in banned:
            assert bad not in blob, f"{bad!r} survived into concept {c['name']!r}"
        assert not c["name"].strip().isdigit()
        assert c["description"] or c["facts"], f"{c['name']} has no supporting evidence at all"


def test_structural_headings_do_not_become_concepts():
    names = {c["name"].lower() for c in _prep(cloud_deck)["concepts"]}
    for structural in ("lesson: cloud computing overview", "thank you",
                       "essential cloud characteristics"):
        assert structural not in names, f"{structural!r} became a concept"


def test_real_taught_ideas_survive_in_both_decks():
    cloud = {c["name"].lower() for c in _prep(cloud_deck)["concepts"]}
    assert "on-demand self-service" in cloud
    assert "cloud computing" in cloud
    sorting = {c["name"].lower() for c in _prep(sorting_deck)["concepts"]}
    assert {"bubble sort", "merge sort"} <= sorting


def test_example_is_kept_as_an_example():
    concepts = {c["name"]: c for c in _prep(sorting_deck)["concepts"]}
    bubble = concepts["Bubble Sort"]
    assert any("[3, 1, 2]" in e for e in bubble["examples"])
    assert not any("[3, 1, 2]" in f for f in bubble["facts"])


# ----------------------------------------------------------- scanned / edge

def test_scanned_pdf_is_reported_not_faked():
    with pytest.raises(ScannedPdfError) as exc:
        extract_material("scan.pdf", scanned_deck())
    message = str(exc.value).lower()
    assert "scanned" in message or "image" in message
    assert "ocr" in message
    assert exc.value.report["scanned"] is True


def test_extract_endpoint_returns_cleaned_text_and_report():
    payload = base64.b64encode(cloud_deck()).decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pdf", "content_base64": payload})
    assert r.status_code == 200
    body = r.json()
    assert "copyright" not in body["text"].lower()
    assert body["report"]["page_count"] == 10
    assert body["report"]["removed_total"] > 0
    # the untouched extraction stays available for transparency/debugging
    assert "Copyright" in body["raw_text"]


def test_extract_endpoint_rejects_scanned_pdf_with_a_useful_message():
    payload = base64.b64encode(scanned_deck()).decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "scan.pdf", "content_base64": payload})
    assert r.status_code == 400
    assert "ocr" in r.json()["detail"].lower()


def test_txt_and_md_ingestion_is_unchanged():
    notes = "# Strings\n\nA string is text between quotes.\n"
    for filename in ("notes.txt", "notes.md"):
        text, report = extract_material(filename, notes.encode())
        assert text == notes
        assert report["kind"] == "text"
        assert report["removed_total"] == 0


def test_empty_and_metadata_only_documents_produce_no_concepts():
    only_chrome = build_pdf([
        [line("Course Handbook", 16, 9), line("(c) Copyright 2020 Acme. All rights reserved.", 516, 8),
         line(str(i), 500, 8, x=670)]
        for i in range(1, 5)
    ])
    text, report = pdf_to_notes(only_chrome)
    prep = prepare_lecture(text)
    assert prep["concepts"] == []
    assert prep["structure"]["notes"], "the teacher must be told extraction found nothing"


def test_cleanup_report_counts_match_the_blocks_removed():
    doc = extract_document(cloud_deck())
    report = clean_document(doc)
    dropped = [b for p in doc["pages"] for b in p["blocks"] if b["dropped"]]
    assert len(dropped) == report["removed_total"]
    assert report["blocks_kept"] + report["removed_total"] == report["blocks_total"]
    assert all(b["drop_reason"] for b in dropped)
    # the rebuild only ever uses surviving blocks
    markdown = document_to_markdown(doc)
    for block in dropped:
        if len(block["text"]) > 12:
            assert block["text"] not in markdown
