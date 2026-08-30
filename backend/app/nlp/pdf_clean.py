"""Deterministic cleanup of an extracted PDF, and rebuild into lecture notes.

Two jobs, both explainable and both free of any hardcoded subject vocabulary:

1. ``clean_document`` decides which extracted lines are PAGE DECORATION rather
   than lecture content, and records WHY. The decision never rests on a single
   signal, because every single signal has a legitimate counter-example:

     * repetition alone is wrong — "Cloud Computing" may genuinely be taught on
       twelve slides, and a recurring section heading is real content;
     * position alone is wrong — the first line of a slide is usually its title;
     * shortness alone is wrong — "Immutability" is a real concept.

   So repetition-based removal requires repetition AND a consistent header /
   footer position AND a font size no larger than the body text AND that the
   line never behaves like a heading (never followed by real prose on any
   page). Copyright / legal notices and page-number patterns are removed on
   their own because those patterns are unambiguous, and even then only when
   the line is short and standalone.

2. ``document_to_markdown`` rebuilds the surviving blocks as ordinary lecture
   notes in the format the existing deterministic parser already understands:
   headings (from relative font size / weight, falling back to shape), bullets,
   prose paragraphs with wrapped lines re-joined and hyphenation repaired.
   Page provenance is preserved with ``<!-- page N -->`` markers, which the
   parser reads and strips, so a suggestion can still say "found on page 7"
   even after the teacher edits the extracted text by hand.

Nothing here is subject-specific and nothing is generated: every surviving
character came from the teacher's own file.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median

# --- removal thresholds (documented so the behaviour is explainable) --------
HEADER_ZONE = 0.12        # top 12% of the page height
FOOTER_ZONE = 0.86        # below 86% of the page height
REPEAT_FRACTION = 0.5     # "repeated" = on at least this share of pages...
REPEAT_MIN_PAGES = 3      # ...and on at least this many pages
POSITION_CONSISTENCY = 0.6  # share of occurrences that must sit in one zone
BRANDING_FRACTION = 0.8   # near-every-page short line = running branding
HEADING_SIZE_RATIO = 1.12  # font this much larger than body text = heading
PROSE_WORDS = 12          # a line this long counts as explanatory prose

_COPYRIGHT_RE = re.compile(
    r"(©|\(c\)\s*(19|20)\d{2}|\bcopyright\b|\ball rights reserved\b|\bproprietary\b"
    r"|\bconfidential\b|\btrademarks?\b|\bregistered trademark\b|™|®"
    r"|\breproduction .{0,30}prohibited\b|\bno part of this\b)", re.I)
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/|\|)\s*\d{1,4})?$", re.I)
_DASHED_PAGE_RE = re.compile(r"^[-–—|\[\(]?\s*\d{1,4}\s*[-–—|\]\)]?$")
_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.I)
_SEPARATOR_RE = re.compile(r"^[\s\-=_~*·•.…—–|<>/\\]+$")
# a page number fused onto the following text: "5© Copyright 2014 EMC"
_LEADING_PAGENO_RE = re.compile(r"^\s*\d{1,3}\s*(?=[©(]|copyright\b)", re.I)
_TRAILING_PAGENO_RE = re.compile(r"\s+\d{1,4}$")

REASONS = {
    "copyright": "copyright / legal notice",
    "page_number": "page or slide number",
    "running_header": "repeated header",
    "running_footer": "repeated footer",
    "branding": "repeated course / module branding",
    "url": "standalone link in the page margin",
    "separator": "layout separator or stray fragment",
}


def _norm_key(text: str) -> str:
    """Normalised identity of a line: what makes two lines "the same" line.

    Digits are dropped so "Module 3 — Cloud" and "Module 7 — Cloud" count as
    the same running header, which is how slide decks number their sections.
    """
    t = re.sub(r"\d+", "#", text.lower())
    t = re.sub(r"[^a-z#]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _zone(block: dict, page_height: float) -> str | None:
    bbox = block.get("bbox")
    if not bbox or not page_height:
        return None
    top, bottom = bbox[1] / page_height, bbox[3] / page_height
    if top <= HEADER_ZONE:
        return "header"
    if bottom >= FOOTER_ZONE:
        return "footer"
    return "body"


def _edge_position(block: dict, n_blocks: int) -> str | None:
    """Geometry-free fallback: first/last line of the page."""
    if block["order"] == 0:
        return "header"
    if block["order"] >= n_blocks - 2:
        return "footer"
    return "body"


def _is_heading_like(block: dict, body_size: float | None) -> bool:
    if block.get("size") and body_size:
        return block["size"] >= body_size * HEADING_SIZE_RATIO
    return False


def clean_document(doc: dict) -> dict:
    """Mark page decoration on a LectureDocument. Returns a cleanup report.

    Mutates each block with ``dropped`` (bool) and ``drop_reason`` (str) so the
    raw extraction stays inspectable — nothing is thrown away silently.
    """
    pages = doc["pages"]
    page_count = len(pages) or 1
    body_size = doc.get("body_size")

    # --- pass 1: where does each distinct line appear, and how? ------------
    occurrences: dict[str, list[dict]] = defaultdict(list)
    heading_like_keys: set[str] = set()
    for page in pages:
        n = len(page["blocks"])
        height = page.get("height") or 0.0
        for i, block in enumerate(page["blocks"]):
            key = _norm_key(block["text"])
            if not key:
                continue
            zone = _zone(block, height) or _edge_position(block, n)
            occurrences[key].append({"page": page["number"], "zone": zone,
                                     "size": block.get("size")})
            # a line that is followed by real prose on the SAME page is acting
            # as a section heading — never treat it as running decoration
            follower = page["blocks"][i + 1] if i + 1 < n else None
            if (_is_heading_like(block, body_size)
                    or (follower and len(follower["text"].split()) >= PROSE_WORDS
                        and zone != "footer")):
                heading_like_keys.add(key)

    pages_with = {k: len({o["page"] for o in v}) for k, v in occurrences.items()}
    repeat_floor = max(REPEAT_MIN_PAGES, int(REPEAT_FRACTION * page_count))
    branding_floor = max(REPEAT_MIN_PAGES, int(BRANDING_FRACTION * page_count))

    def zone_share(key: str, zone: str) -> float:
        occ = occurrences[key]
        return sum(1 for o in occ if o["zone"] == zone) / len(occ) if occ else 0.0

    def not_larger_than_body(key: str) -> bool:
        sizes = [o["size"] for o in occurrences[key] if o["size"]]
        if not sizes or not body_size:
            return True  # no font data: fall back to the other signals
        return median(sizes) <= body_size * 1.05

    # --- pass 2: decide, block by block -----------------------------------
    removed = Counter()
    samples: dict[str, list[str]] = defaultdict(list)

    def drop(block: dict, reason: str) -> None:
        block["dropped"] = True
        block["drop_reason"] = reason
        removed[reason] += 1
        if len(samples[reason]) < 3 and block["text"] not in samples[reason]:
            samples[reason].append(block["text"])

    for page in pages:
        n = len(page["blocks"])
        height = page.get("height") or 0.0
        for i, block in enumerate(page["blocks"]):
            block.setdefault("dropped", False)
            text = block["text"]
            stripped = _LEADING_PAGENO_RE.sub("", text).strip()
            if stripped != text:
                # "5© Copyright 2014 EMC Corporation." — the slide number was
                # fused onto the notice; keep the notice for the rule below
                block["text"] = text = stripped
            key = _norm_key(text)
            zone = _zone(block, height) or _edge_position(block, n)
            words = text.split()

            if _SEPARATOR_RE.match(text) or len(text) < 3:
                drop(block, "separator")
                continue
            # unambiguous legal boilerplate, when it is a standalone short line
            if _COPYRIGHT_RE.search(text) and len(words) <= 20:
                drop(block, "copyright")
                continue
            if _PAGE_NUMBER_RE.match(text) or _DASHED_PAGE_RE.match(text):
                drop(block, "page_number")
                continue
            if _URL_RE.match(text) and zone in ("header", "footer"):
                drop(block, "url")
                continue
            if key in heading_like_keys:
                continue  # behaves like a section heading somewhere: protected
            repeats = pages_with.get(key, 0)
            if repeats >= repeat_floor and not_larger_than_body(key):
                if zone == "header" and zone_share(key, "header") >= POSITION_CONSISTENCY:
                    drop(block, "running_header")
                    continue
                if zone == "footer" and zone_share(key, "footer") >= POSITION_CONSISTENCY:
                    drop(block, "running_footer")
                    continue
                # course/module branding repeated almost everywhere, short, and
                # never acting as a heading
                if repeats >= branding_floor and len(words) <= 8:
                    drop(block, "branding")
                    continue

    kept = [b for p in pages for b in p["blocks"] if not b["dropped"]]
    # with the decoration gone, the body font size can be measured honestly:
    # running footers no longer drag it down, so heading detection is stable
    from .pdf_extract import body_size_of

    refined = body_size_of(kept)
    if refined:
        doc["body_size"] = refined
    return {
        "page_count": page_count,
        "blocks_total": sum(len(p["blocks"]) for p in pages),
        "blocks_kept": len(kept),
        "removed_total": sum(removed.values()),
        "removed_by_reason": [
            {"reason": r, "label": REASONS.get(r, r), "count": c, "examples": samples[r]}
            for r, c in removed.most_common()
        ],
        "empty_pages": [p["number"] for p in pages
                        if not any(not b["dropped"] for b in p["blocks"])],
    }


# ---------------------------------------------------------------------------
# rebuild: surviving blocks -> ordinary Markdown lecture notes
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.!?:;]$")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+[.)]|[IVXLC]+[.)])\s+\S")
_CODEY_RE = re.compile(r"(>>>|[\w\)\]]\s*=\s*\S|\w+\([^)]*\)|\w+\[[^\]]*\]|#\s|->|→)")


def _heading_level(block: dict, body_size: float | None, sizes_desc: list[float]) -> int | None:
    """Heading level for a block, or None when it is body text."""
    words = block["text"].split()
    if block.get("bullet") or len(words) > 14:
        return None
    if _SENTENCE_END_RE.search(block["text"]) and not block["text"].endswith(":"):
        return None
    size = block.get("size")
    if size and body_size:
        if size >= body_size * HEADING_SIZE_RATIO:
            # deeper heading levels for progressively smaller heading sizes
            rank = next((i for i, s in enumerate(sizes_desc) if abs(size - s) < 0.51), len(sizes_desc))
            return min(1 + rank, 4)
        if block.get("bold") and size >= body_size and len(words) <= 10:
            return 3
        return None
    # no font information (pypdf fallback): shape-based detection only
    if _CODEY_RE.search(block["text"]):
        return None
    if len(words) <= 10 and (block["text"].isupper() or _NUMBERED_HEADING_RE.match(block["text"])
                             or block["text"][:1].isupper()) and block["order"] == 0:
        return 2
    return None


def _heading_sizes(doc: dict) -> list[float]:
    body_size = doc.get("body_size")
    if not body_size:
        return []
    sizes = {b["size"] for p in doc["pages"] for b in p["blocks"]
             if not b.get("dropped") and b.get("size")
             and b["size"] >= body_size * HEADING_SIZE_RATIO}
    return sorted(sizes, reverse=True)


def document_to_markdown(doc: dict) -> str:
    """Rebuild cleaned blocks as lecture notes the existing parser understands."""
    body_size = doc.get("body_size")
    sizes_desc = _heading_sizes(doc)
    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            text = " ".join(paragraph)
            text = re.sub(r"(\w)-\s+(?=[a-z])", r"\1", text)  # de-hyphenate
            out.append(re.sub(r"\s{2,}", " ", text).strip())
            out.append("")
            paragraph.clear()

    seen_title = False
    for page in doc["pages"]:
        blocks = [b for b in page["blocks"] if not b.get("dropped")]
        if not blocks:
            continue
        flush()
        out.append(f"<!-- page {page['number']} -->")
        for block in blocks:
            level = _heading_level(block, body_size, sizes_desc)
            text = block["text"].rstrip()
            if block.get("bullet"):
                flush()
                out.append(f"- {text}")
                continue
            if level is not None:
                flush()
                if not seen_title and level == 1:
                    seen_title = True
                    out.append(f"# {text.rstrip(':')}")
                else:
                    out.append(f"{'#' * max(2, level)} {text.rstrip(':')}")
                out.append("")
                continue
            # body text: join wrapped lines back into one paragraph
            paragraph.append(text)
            if _SENTENCE_END_RE.search(text):
                flush()
        flush()
    # collapse runs of blank lines
    md = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def pdf_to_notes(data: bytes) -> tuple[str, dict]:
    """Full PDF ingestion: extract -> clean -> rebuild as notes.

    Returns ``(markdown_text, report)``. The report carries the page count, the
    cleanup counts/examples and the raw extraction, so the review screen can
    show what was removed and offer a raw view.
    """
    from .pdf_extract import extract_document, raw_text

    doc = extract_document(data)
    raw = raw_text(doc)
    if doc["scanned"]:
        return "", {"scanned": True, "page_count": doc["page_count"],
                    "extractor": doc["extractor"], "raw_text": raw,
                    "removed_total": 0, "removed_by_reason": [], "empty_pages": []}
    report = clean_document(doc)
    report.update({"scanned": False, "extractor": doc["extractor"], "raw_text": raw})
    return document_to_markdown(doc), report
