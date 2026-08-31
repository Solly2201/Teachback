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
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_decks import COPYRIGHT, FOOTER, HEADER, cloud_deck, scanned_deck, sorting_deck
from pdf_fixture import build_pdf, line, spaced_line

from app.main import app
from app.nlp.lecture_parser import parse_lecture
from app.nlp.lecture_prep import ScannedPdfError, extract_material, prepare_lecture
from app.nlp.pdf_clean import clean_document, document_to_markdown, pdf_to_notes
from app.nlp.pdf_extract import MIN_CHARS_PER_PAGE, extract_document

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


# ==========================================================================
# Page-aware extractability
# ==========================================================================
# A document total hides the case that matters. The real 22-page "Regular
# expression" lecture has 15 pages that are photographs of the board and 7
# typed pages; the typed pages carry thousands of characters, so a
# document-level average declares the file a normal text PDF and two thirds of
# the lecture is dropped without anyone being told. Extractability is therefore
# judged page by page.

from pdf_decks import REGEX_IMAGE_PAGES, mixed_deck, sparse_but_readable_deck  # noqa: E402

from app.nlp.pdf_extract import assess_text_coverage, format_page_ranges  # noqa: E402


# --- A. a normal text PDF ---------------------------------------------------

@pytest.mark.parametrize("name,deck", DECKS + [("sparse", sparse_but_readable_deck)])
def test_all_text_pdfs_are_classified_as_text(name, deck):
    doc = extract_document(deck())
    assert doc["text_quality"] == "text"
    assert doc["scanned"] is False
    assert doc["image_page_count"] == 0
    assert doc["text_page_count"] == doc["page_count"]
    text, report = pdf_to_notes(deck())
    assert text.strip(), "a text PDF must still produce notes"
    assert report["text_quality"] == "text"


def test_a_short_section_divider_is_sparse_not_missing():
    """"Part One" on its own is a divider, not a lost page. Sparse and absent
    are different problems and must not share a threshold."""
    doc = extract_document(sparse_but_readable_deck())
    assert doc["text_quality"] == "text"
    assert doc["image_page_count"] == 0
    prep = prepare_lecture(pdf_to_notes(sparse_but_readable_deck())[0])
    assert prep["concepts"], "a readable deck with dividers must still extract"


# --- B. a fully scanned PDF -------------------------------------------------

def test_a_fully_scanned_pdf_is_classified_and_refused():
    doc = extract_document(scanned_deck())
    assert doc["text_quality"] == "scanned"
    assert doc["scanned"] is True
    assert doc["image_page_count"] == doc["page_count"]
    assert doc["text_page_count"] == 0

    with pytest.raises(ScannedPdfError) as exc:
        extract_material("scan.pdf", scanned_deck())
    message = str(exc.value).lower()
    assert "scanned" in message or "image" in message
    assert "ocr" in message
    assert exc.value.report["text_quality"] == "scanned"


# --- C/D. the mixed PDF, which is the regression ----------------------------

def test_a_mixed_pdf_is_not_mistaken_for_a_text_pdf():
    """15 image pages + 7 text pages: the exact shape of the real lecture."""
    doc = extract_document(mixed_deck())
    assert doc["page_count"] == 22
    assert doc["text_quality"] == "mixed", doc["text_quality"]
    assert doc["scanned"] is True
    assert doc["image_page_count"] == 15
    assert doc["text_page_count"] == 7
    assert doc["image_pages"] == list(REGEX_IMAGE_PAGES)
    # the typed pages carry enough text that a document total would pass
    assert doc["char_count"] > MIN_CHARS_PER_PAGE * doc["page_count"], (
        "the fixture must reproduce the trap: a document average would say 'text'")


def test_a_mixed_pdf_is_refused_and_never_partly_ingested():
    with pytest.raises(ScannedPdfError) as exc:
        extract_material("lecture.pdf", mixed_deck())
    report = exc.value.report
    assert report["text_quality"] == "mixed"
    assert report["image_page_count"] == 15
    message = str(exc.value)
    assert "1-8, 10-16" in message, message
    assert "15 of 22" in message
    assert "ocr" in message.lower()

    # and nothing partial escapes towards the parser
    text, _ = pdf_to_notes(mixed_deck())
    assert text == "", "a mixed PDF must yield no notes at all"
    assert prepare_lecture(text)["concepts"] == []


def test_a_text_page_in_the_middle_does_not_rescue_a_mixed_pdf():
    """Page 9 of the real lecture is typed, sitting between two runs of
    images. Interleaving must not change the verdict."""
    interleaved = extract_document(mixed_deck())
    front_loaded = extract_document(mixed_deck(image_pages=15, page_count=22))
    assert interleaved["text_quality"] == front_loaded["text_quality"] == "mixed"


@pytest.mark.parametrize("image_pages,page_count,expected", [
    (15, 22, "mixed"),     # the real lecture
    (3, 10, "mixed"),      # a third of a short lecture lost
    (2, 8, "mixed"),       # the absolute floor
    (1, 20, "text"),       # one diagram slide in a long deck: tolerated
    (0, 10, "text"),
    (10, 10, "scanned"),   # nothing readable at all
])
def test_the_boundary_between_text_mixed_and_scanned(image_pages, page_count, expected):
    doc = extract_document(mixed_deck(image_pages=image_pages, page_count=page_count))
    assert doc["text_quality"] == expected, (
        f"{image_pages}/{page_count} image pages -> {doc['text_quality']}")


# --- the report has to be able to say WHICH pages ---------------------------

def test_the_report_identifies_the_affected_pages():
    _, report = pdf_to_notes(mixed_deck())
    for key in ("text_quality", "image_pages", "image_page_count",
                "low_text_pages", "low_text_page_count", "text_page_count",
                "page_count"):
        assert key in report, f"the report must expose {key}"
    assert report["image_pages"] == list(REGEX_IMAGE_PAGES)
    assert report["low_text_page_count"] >= report["image_page_count"]
    # the raw extraction stays available even for a refused document
    assert "raw_text" in report


def test_page_ranges_are_readable():
    assert format_page_ranges([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]) \
        == "1-8, 10-16"
    assert format_page_ranges([3]) == "3"
    assert format_page_ranges([]) == ""
    assert format_page_ranges([1, 3, 5, 7, 9, 11, 13, 15]).endswith("more")


def test_coverage_assessment_is_page_wise_not_total():
    """One enormous page cannot vouch for nine empty ones."""
    pages = [{"number": 1, "blocks": [{"text": "x" * 5000}]}]
    pages += [{"number": n, "blocks": []} for n in range(2, 11)]
    coverage = assess_text_coverage(pages)
    assert coverage["text_quality"] == "scanned"
    assert coverage["image_page_count"] == 9
    assert coverage["text_page_count"] == 1


# --- the upload endpoint tells the teacher the same thing -------------------

def test_the_extract_endpoint_refuses_a_mixed_pdf_with_the_page_list():
    payload = base64.b64encode(mixed_deck()).decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "lecture.pdf", "content_base64": payload})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "1-8, 10-16" in detail
    assert "ocr" in detail.lower()


def test_the_extract_endpoint_still_accepts_a_normal_deck():
    payload = base64.b64encode(cloud_deck()).decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pdf", "content_base64": payload})
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["text_quality"] == "text"
    assert body["report"]["image_page_count"] == 0
    assert body["text"].strip()


# ==========================================================================
# The real lecture PDFs, when they are available locally
# ==========================================================================
# These are the two files that motivated the page-aware detection. They are
# not committed (they are copyrighted course material and large), so the tests
# skip when the folder is absent — the synthetic fixtures above reproduce both
# shapes and are what CI relies on.

REAL_PDF_DIR = Path(__file__).resolve().parent.parent / "test data"
CLOUD_PDF = REAL_PDF_DIR / "cc_m1_merged.pdf"
REGEX_PDF = REAL_PDF_DIR / "Regular expression.pdf"


@pytest.mark.skipif(not CLOUD_PDF.exists(), reason="real Cloud Computing PDF not present")
def test_real_cloud_deck_is_still_a_normal_text_pdf():
    doc = extract_document(CLOUD_PDF.read_bytes())
    assert doc["page_count"] == 108
    assert doc["text_quality"] == "text", doc["image_pages"][:20]
    assert doc["image_page_count"] == 0
    text, report = pdf_to_notes(CLOUD_PDF.read_bytes())
    assert text.strip()
    assert report["removed_total"] > 0, "its running header/footer should still be removed"
    lowered = text.lower()
    assert "all rights reserved" not in lowered
    assert "copyright" not in lowered


@pytest.mark.skipif(not REGEX_PDF.exists(), reason="real Regular expression PDF not present")
def test_real_regex_lecture_is_identified_as_mixed_and_refused():
    doc = extract_document(REGEX_PDF.read_bytes())
    assert doc["page_count"] == 22
    assert doc["text_quality"] == "mixed", doc["text_quality"]
    assert doc["image_page_count"] == 15
    assert doc["text_page_count"] == 7
    # a document total would have called this a normal text PDF
    assert doc["char_count"] > MIN_CHARS_PER_PAGE * doc["page_count"]
    with pytest.raises(ScannedPdfError) as exc:
        extract_material("Regular expression.pdf", REGEX_PDF.read_bytes())
    assert "15 of 22" in str(exc.value)


# ==========================================================================
# Pages that read fine but still hide content
# ==========================================================================
# A page can be perfectly extractable and STILL be missing the lesson: the EMC
# deck has slides whose labels are baked into a diagram while the surrounding
# prose extracts normally. Without OCR we cannot know whether the picture
# contains words — only that a large diagram is there. So this is a WARNING on
# an ACCEPTED page, and is kept strictly separate from the image-only pages
# that drive the mixed/scanned refusal.

from pdf_decks import diagram_deck  # noqa: E402
from pdf_fixture import image  # noqa: E402

from app.nlp.pdf_extract import assess_image_content  # noqa: E402


# --- A. an ordinary text page raises nothing --------------------------------

@pytest.mark.parametrize("name,deck", DECKS)
def test_a_plain_text_deck_reports_no_image_heavy_pages(name, deck):
    doc = extract_document(deck())
    assert doc["image_heavy_page_count"] == 0
    assert doc["image_heavy_pages"] == []


# --- C/D. text + a large diagram is accepted, and flagged -------------------

def test_a_page_with_text_and_a_large_diagram_is_accepted_and_flagged():
    doc = extract_document(diagram_deck())
    # accepted: every page has real text, so nothing is image-only
    assert doc["text_quality"] == "text"
    assert doc["image_page_count"] == 0
    # ...but the diagram slide is reported as possibly incomplete
    assert doc["image_heavy_pages"] == [3], doc["image_heavy_pages"]

    text, report = pdf_to_notes(diagram_deck())
    assert text.strip(), "an image-heavy deck must still ingest"
    assert report["image_heavy_page_count"] == 1
    assert report["image_heavy_pages_label"] == "3"
    # the two categories never collapse into one another
    assert report["image_page_count"] == 0
    assert report["image_pages_label"] == ""


def test_an_image_heavy_page_is_not_an_image_only_page():
    """#3 (missing content, refused) and #4 (possibly incomplete, accepted) are
    different things and must not be merged."""
    doc = extract_document(diagram_deck())
    assert set(doc["image_pages"]).isdisjoint(doc["image_heavy_pages"])
    with_text_and_diagram = extract_material("deck.pdf", diagram_deck())
    assert with_text_and_diagram[0].strip(), "an image-heavy deck is never refused"


def test_repeated_chrome_and_full_bleed_backgrounds_are_not_diagrams():
    """A logo in the same corner of every slide, and a full-page background,
    are template furniture. Flagging them would train the teacher to ignore the
    warning."""
    doc = extract_document(diagram_deck())
    # page 1 carries a full-page background, pages 1-4 all carry the same logo
    assert 1 not in doc["image_heavy_pages"]
    assert doc["pages"][0]["images"][0]["fraction"] >= 0.9, "fixture must be full-bleed"
    assert 4 not in doc["image_heavy_pages"], "a small banner is decoration"


def test_an_image_only_page_is_not_double_reported_as_image_heavy():
    """A page with a big image and NO text is missing content, not merely
    incomplete — it belongs to the refusal path alone."""
    pdf = build_pdf([
        [line("Regular Expressions", 50, 26, bold=True),
         line("A pattern describes the shape of the text you want to match.", 120, 14),
         line("Patterns combine literal characters with metacharacters.", 145, 14)],
        [image(60, 60, 600, 420)],  # a photographed slide: image, no text
    ])
    doc = extract_document(pdf)
    assert doc["image_pages"] == [2]
    assert 2 not in doc["image_heavy_pages"], "an image-only page is not 'image-heavy'"


def test_image_assessment_needs_no_text_extraction_to_be_safe():
    """Called with pages that carry no image data at all (the pypdf fallback),
    it must simply report nothing rather than fail."""
    pages = [{"number": 1, "blocks": [{"text": "x" * 200}]},
             {"number": 2, "blocks": [{"text": "y" * 200}], "images": []}]
    assert assess_image_content(pages) == {"image_heavy_pages": [],
                                           "image_heavy_page_count": 0}
    assert assess_image_content([]) == {"image_heavy_pages": [],
                                        "image_heavy_page_count": 0}


# --- E. the mixed PDF is untouched by any of this ---------------------------

def test_the_mixed_pdf_verdict_is_unchanged_by_image_heavy_detection():
    doc = extract_document(mixed_deck())
    assert doc["text_quality"] == "mixed"
    assert doc["image_page_count"] == 15
    assert doc["image_pages"] == list(REGEX_IMAGE_PAGES)
    with pytest.raises(ScannedPdfError):
        extract_material("lecture.pdf", mixed_deck())


# --- the report keeps the page numbers, for both categories -----------------

def test_the_report_labels_both_kinds_of_page_for_the_teacher():
    _, mixed_report = pdf_to_notes(mixed_deck())
    assert mixed_report["image_pages_label"] == "1-8, 10-16"
    _, diagram_report = pdf_to_notes(diagram_deck())
    assert diagram_report["image_heavy_pages_label"] == "3"
    for report in (mixed_report, diagram_report):
        for key in ("image_pages_label", "image_heavy_pages_label",
                    "image_heavy_pages", "image_heavy_page_count"):
            assert key in report


def test_the_extract_endpoint_returns_the_image_heavy_pages():
    payload = base64.b64encode(diagram_deck()).decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pdf", "content_base64": payload})
    assert r.status_code == 200
    report = r.json()["report"]
    assert report["image_heavy_page_count"] == 1
    assert report["image_heavy_pages_label"] == "3"
    assert report["image_page_count"] == 0


@pytest.mark.skipif(not CLOUD_PDF.exists(), reason="real Cloud Computing PDF not present")
def test_real_cloud_deck_flags_its_diagram_slides_without_refusing():
    doc = extract_document(CLOUD_PDF.read_bytes())
    assert doc["text_quality"] == "text", "it must still ingest"
    assert doc["image_page_count"] == 0
    # the deck is diagram-heavy, but only a minority of its pages
    assert 5 <= doc["image_heavy_page_count"] <= 40, doc["image_heavy_page_count"]
    assert 5 in doc["image_heavy_pages"], "the infrastructure diagram slide"
    # the logo on all 108 pages and the 26 full-bleed backgrounds are excluded
    assert doc["image_heavy_page_count"] < doc["page_count"] / 2
