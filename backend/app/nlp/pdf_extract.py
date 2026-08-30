"""Layout-aware PDF extraction (deterministic, no LLM, no cloud service).

Real lecture PDFs — especially slide decks — are not linear documents. A flat
``page.extract_text()`` collapses a deck into one undifferentiated stream in
which a running footer, a slide number and a slide title are indistinguishable
from a definition. That is exactly how a copyright notice ends up as a
concept's meaning.

This module keeps the structure instead:

    LectureDocument
      page_count, body_size, scanned, extractor
      pages[]
        number, width, height
        blocks[]
          text, page_number, order, bbox (x0, top, x1, bottom),
          size (font size), bold, bullet

One block is one visual line, which is the right granularity for slides and
still fine for prose (the Markdown rebuild in ``pdf_clean`` re-joins wrapped
prose lines into paragraphs).

Extractor choice: ``pdfplumber`` is already installed with the project (it
ships on top of pdfminer.six) and gives word-level bounding boxes AND font
name/size — everything the cleanup and heading detection need — so no new
heavy dependency is introduced. ``pypdf`` remains a graceful fallback when
pdfplumber cannot open a file; in that mode blocks carry text and order but no
geometry, and the cleanup falls back to its geometry-free signals.
"""
from __future__ import annotations

import re
from io import BytesIO
from statistics import median

# A page with almost no selectable text is an image/scan, not a text PDF.
# Judged per page (a 200-page scan and a 3-page scan are both scans), with an
# absolute floor that only applies to multi-page documents so a single short
# slide is not mistaken for an image.
MIN_CHARS_PER_PAGE = 40
MIN_TOTAL_CHARS = 200
SCAN_FLOOR_PAGES = 3

# words on the same visual line differ by at most this many points vertically
LINE_TOLERANCE = 2.5

_BULLET_GLYPHS = "•▪◦‣●○·–—*"
_BULLET_RE = re.compile(rf"^\s*(?:[{re.escape(_BULLET_GLYPHS)}])\s*")
_LETTER_SPACED_RE = re.compile(r"^(?:\S\s){5,}\S?$")


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot be turned into usable lecture material."""


def _normalise_line(text: str) -> str:
    """Collapse whitespace and repair the two common extraction artifacts."""
    text = text.replace("\xa0", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"\s+", " ", text).strip()
    # "C l o u d C o m p u t i n g" -> "Cloud Computing": letter-spaced titles
    # are a real slide-deck artifact and are otherwise unusable as headings.
    if _LETTER_SPACED_RE.match(text):
        tokens = text.split(" ")
        if sum(1 for t in tokens if len(t) == 1) >= 0.7 * len(tokens):
            joined, buf = [], []
            for t in tokens:
                if len(t) == 1:
                    buf.append(t)
                else:
                    if buf:
                        joined.append("".join(buf))
                        buf = []
                    joined.append(t)
            if buf:
                joined.append("".join(buf))
            text = " ".join(joined)
    return text


def _make_block(text: str, page_number: int, order: int, bbox=None,
                size: float | None = None, bold: bool = False) -> dict | None:
    raw = _normalise_line(text)
    bullet = bool(_BULLET_RE.match(raw))
    if bullet:
        raw = _BULLET_RE.sub("", raw).strip()
    if len(raw) < 2 or not re.search(r"[A-Za-z0-9]", raw):
        return None
    return {"text": raw, "page_number": page_number, "order": order, "bbox": bbox,
            "size": size, "bold": bold, "bullet": bullet}


def _lines_from_words(words: list[dict]) -> list[list[dict]]:
    """Group pdfplumber words into visual lines, in reading order."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _join_line_words(words: list[dict]) -> str:
    """Join one line's words, repairing letter-spaced text.

    Slide titles are often typeset with character spacing, which extractors
    report as a run of single-character "words" ("C l o u d  C o m p u t i n g").
    The word boundaries are still visible in the geometry — the gap between
    words is wider than the gap between letters — so the repair uses the gaps
    rather than guessing, and does nothing at all to ordinary lines.
    """
    tokens = [w["text"] for w in words]
    if len(tokens) < 6 or sum(1 for t in tokens if len(t) == 1) < 0.7 * len(tokens):
        return " ".join(tokens)
    gaps = [words[i + 1]["x0"] - words[i]["x1"] for i in range(len(words) - 1)]
    if not gaps or max(gaps) <= 0:
        return "".join(tokens)
    letter_gap = median(gaps)
    parts = [tokens[0]]
    for i, gap in enumerate(gaps):
        parts.append(" " if gap > max(letter_gap * 1.8, letter_gap + 0.5) else "")
        parts.append(tokens[i + 1])
    return "".join(parts)


def _extract_with_pdfplumber(data: bytes) -> dict:
    import pdfplumber

    pages = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                words = page.extract_words(extra_attrs=["fontname", "size"]) or []
            except Exception:  # a single damaged page must not kill the upload
                words = []
            blocks = []
            for order, line in enumerate(_lines_from_words(words)):
                text = _join_line_words(line)
                sizes = [float(w.get("size") or 0) for w in line if w.get("size")]
                fonts = " ".join(str(w.get("fontname") or "") for w in line).lower()
                bbox = (min(w["x0"] for w in line), min(w["top"] for w in line),
                        max(w["x1"] for w in line), max(w["bottom"] for w in line))
                block = _make_block(
                    text, pno, order, bbox=tuple(round(float(v), 2) for v in bbox),
                    size=round(median(sizes), 2) if sizes else None,
                    bold=("bold" in fonts or "black" in fonts or "heavy" in fonts),
                )
                if block:
                    blocks.append(block)
            pages.append({"number": pno, "width": float(page.width), "height": float(page.height),
                          "blocks": blocks})
    return {"pages": pages, "extractor": "pdfplumber"}


def _extract_with_pypdf(data: bytes) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = []
    for pno, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        blocks = []
        for order, raw in enumerate(text.split("\n")):
            block = _make_block(raw, pno, order)
            if block:
                blocks.append(block)
        pages.append({"number": pno, "width": 0.0, "height": 0.0, "blocks": blocks})
    return {"pages": pages, "extractor": "pypdf"}


def body_size_of(blocks_with_pages) -> float | None:
    """Median font size weighted by text length — the document's body size.

    Headings are judged relative to this, so a deck typeset at 32pt and a
    paper typeset at 10pt are handled by the same rules.
    """
    sizes: list[float] = []
    for block in blocks_with_pages:
        if block.get("size"):
            sizes.extend([block["size"]] * max(1, len(block["text"])))
    return round(median(sizes), 2) if sizes else None


def _in_page_margin(block: dict, page_height: float) -> bool:
    """True for blocks sitting in the very top/bottom band of the page.

    Running headers and footers live there and are usually set in a small
    font; including them would drag the "body size" down until every real line
    looked like a heading.
    """
    bbox = block.get("bbox")
    if not bbox or not page_height:
        return False
    return (bbox[1] / page_height) <= 0.12 or (bbox[3] / page_height) >= 0.86


def _body_size(pages: list[dict]) -> float | None:
    inner = [b for p in pages for b in p["blocks"]
             if not _in_page_margin(b, p.get("height") or 0.0)]
    return body_size_of(inner) or body_size_of(b for p in pages for b in p["blocks"])


def extract_document(data: bytes) -> dict:
    """Extract a structured LectureDocument from PDF bytes.

    Never raises for an unreadable *page*; raises PdfExtractionError only when
    the file cannot be opened at all. Image-only PDFs come back with
    ``scanned=True`` and (almost) no blocks — the caller reports that honestly
    rather than pretending the extraction worked.
    """
    result = None
    errors = []
    for extractor in (_extract_with_pdfplumber, _extract_with_pypdf):
        try:
            result = extractor(data)
            break
        except Exception as exc:  # pragma: no cover - depends on the PDF
            errors.append(f"{extractor.__name__}: {exc}")
    if result is None:
        raise PdfExtractionError(
            "The PDF could not be read. It may be encrypted or damaged. "
            f"({'; '.join(errors)[:200]})"
        )

    pages = result["pages"]
    total_chars = sum(len(b["text"]) for p in pages for b in p["blocks"])
    page_count = len(pages)
    scanned = page_count > 0 and (
        total_chars < MIN_CHARS_PER_PAGE * page_count
        or (page_count >= SCAN_FLOOR_PAGES and total_chars < MIN_TOTAL_CHARS)
    )
    return {
        "pages": pages,
        "page_count": page_count,
        "block_count": sum(len(p["blocks"]) for p in pages),
        "char_count": total_chars,
        "body_size": _body_size(pages),
        "scanned": scanned,
        "extractor": result["extractor"],
    }


def raw_text(doc: dict) -> str:
    """The unmodified extraction, for the "View raw extraction" transparency view."""
    out = []
    for page in doc["pages"]:
        out.append(f"--- page {page['number']} ---")
        out.extend(b["text"] for b in page["blocks"])
    return "\n".join(out)
