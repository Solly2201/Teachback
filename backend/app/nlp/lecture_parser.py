"""Structured lecture-note parser (deterministic, no LLM).

Lecture material is NOT a bag of words: headings, bullets, code blocks and
examples carry structure that tells us what the teacher considers important.
This module turns raw notes (Markdown or plain text) into a document tree:

    parse_lecture(text) -> {
        "title": str | None,
        "objectives": [str],          # from a "Learning Objectives" section
        "sections": [                 # one per content heading
            {"heading", "level", "sentences", "bullets", "examples", "page"}
        ],
        "connections": [str],         # raw lines of an "Important Connections" section
        "mistakes": [str],            # raw lines of a "Common Mistakes" section
        "summary": str,
        "has_structure": bool,        # True when real headings were found
    }

Recognised structure:
* Markdown headings (#, ##, ...), setext underlines (=== / ---)
* short ALL-CAPS or title-like standalone lines as headings in plain text
* bullets (-, *, •) and numbered items
* fenced code blocks and "Example:"-style example lines (kept as examples,
  never treated as prose — code tokens must not become concepts)
* special sections by heading keyword: objectives / connections / common
  mistakes / summary
* ``<!-- page N -->`` provenance markers, written by the PDF ingestion step
  (nlp/pdf_clean.py) and never shown as content — they let a suggestion say
  "found on page 7" and survive the teacher editing the extracted text

Everything is heuristic; the faculty review step remains the authority.
"""
import re

from .analyzer import split_sentences

_BULLET_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+(.*)$")
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+[.)]\s*")

_OBJECTIVE_HEADINGS = ("learning objective", "objectives", "objective", "goals", "goal")
_CONNECTION_HEADINGS = ("important connection", "connections", "connection",
                        "relationships", "relationship", "how ideas connect")
_MISTAKE_HEADINGS = ("common mistake", "common confusion", "misconception",
                     "common error", "common mix-up", "common mixups")
_SUMMARY_HEADINGS = ("summary", "recap", "key takeaway", "wrap-up", "wrap up")
# Headings that are DOCUMENT STRUCTURE, not teachable concepts. A slide deck
# is full of these ("Lesson: ... Overview", "Agenda", "Thank You"): they
# organise the material without being something a student can explain.
_GENERIC_HEADINGS = ("introduction", "overview", "agenda", "outline", "homework",
                     "references", "reading", "exercises", "practice", "examples",
                     "example", "notes", "today", "lecture", "module", "lesson",
                     "chapter", "slide", "contents",
                     "table of contents", "objectives", "learning objectives",
                     "learning outcomes", "goals", "summary", "recap", "conclusion",
                     "conclusions", "thank you", "thanks", "questions", "any questions",
                     "q&a", "acknowledgements", "acknowledgments", "disclaimer",
                     "copyright", "welcome", "roadmap", "syllabus",
                     "revision", "key takeaways", "further reading", "bibliography",
                     "appendix", "glossary", "topics covered", "in this lesson",
                     "in this module", "what we will cover", "next steps")
# suffixes that turn any heading into a structural one ("Cloud Computing Overview")
_STRUCTURAL_SUFFIXES = (" overview", " introduction", " agenda", " outline",
                        " contents", " roadmap", " summary", " recap", " objectives",
                        " outcomes", " takeaways")
# "Module 3:", "Lesson -", "Chapter 4 —", "Slide 12", "Week 2:", "Unit II."
_STRUCTURAL_PREFIX_RE = re.compile(
    r"^\s*(module|lesson|chapter|unit|section|slide|part|topic|week|lecture|day)"
    r"\s*(?:\d{1,3}|[ivxlc]{1,6})?\s*[:.\-–—]\s*", re.I)
# a heading that is only numbering ("4.", "(3)") or a bare roman numeral
_NUMBER_ONLY_RE = re.compile(r"^(?:[\s\d.,:;()\[\]/|-]+|[IVXLC]{1,6}[.)]?)$")
_LEGAL_RE = re.compile(r"(©|\(c\)\s*(19|20)\d{2}|copyright|all rights reserved|™|®"
                       r"|confidential|proprietary)", re.I)
_PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->$", re.I)

_EXAMPLE_PREFIX_RE = re.compile(r"^\s*(example|e\.g\.|eg)\s*[:\-]", re.I)


def _is_code_like(line: str) -> bool:
    """Lines that look like code/REPL output rather than prose."""
    s = line.strip()
    if not s:
        return False
    if s.startswith((">>>", "$ ", "` ")) or (s.startswith("`") and s.endswith("`")):
        return True
    words = re.findall(r"[A-Za-z]+", s)
    symbolish = sum(s.count(ch) for ch in "=[](){}\"'#<>")
    if "→" in s or "->" in s:
        return True
    # short line dominated by symbols, e.g.  s[0]  or  greeting = "Hello"
    if len(words) <= 6 and symbolish >= 2:
        return True
    if re.search(r"\w+\s*=\s*\S", s) and len(words) <= 8:
        return True
    if re.search(r"\w+\.\w+\(", s) or re.search(r"\w+\[[^\]]*\]", s):
        return True
    return False


def _heading_kind(text: str) -> str:
    t = text.strip().lower().rstrip(":")
    for key, kind in ((_OBJECTIVE_HEADINGS, "objectives"), (_CONNECTION_HEADINGS, "connections"),
                      (_MISTAKE_HEADINGS, "mistakes"), (_SUMMARY_HEADINGS, "summary")):
        if any(t.startswith(k) or t == k for k in key):
            return kind
    return "content"


def _clean_heading(text: str) -> str:
    text = _NUMBER_PREFIX_RE.sub("", text.strip())
    return text.strip().rstrip(":").strip()


def _plain_heading(line: str, next_line: str | None) -> bool:
    """Standalone short line that acts as a heading in unmarked plain text."""
    s = line.strip()
    if not s or len(s.split()) > 6 or _is_code_like(s):
        return False
    if s.endswith((".", "!", "?", ",", ";")):
        return False
    if _BULLET_RE.match(line):
        return False
    if s.isupper():
        return True
    # Title-case short phrase followed by explanatory content
    words = s.split()
    return len(words) <= 4 and s[0].isupper() and next_line is not None and bool(next_line.strip())


def parse_lecture(text: str) -> dict:
    lines = (text or "").replace("\r\n", "\n").split("\n")
    doc = {"title": None, "objectives": [], "sections": [], "connections": [],
           "mistakes": [], "summary": "", "has_structure": False, "has_pages": False}

    current = None          # current content section dict
    mode = "content"        # content | objectives | connections | mistakes | summary
    in_fence = False
    example_run = False     # inside an "Example:" block
    summary_parts: list[str] = []
    md_heading_count = 0
    page = None             # current page number, from <!-- page N --> markers

    def start_section(heading: str, level: int):
        nonlocal current, mode, example_run
        example_run = False
        kind = _heading_kind(heading)
        mode = kind
        if kind == "content":
            current = {"heading": _clean_heading(heading), "level": level,
                       "sentences": [], "bullets": [], "examples": [], "page": page}
            doc["sections"].append(current)

    def add_prose(textline: str):
        nonlocal current
        if mode == "objectives":
            doc["objectives"].append(textline.strip())
        elif mode == "connections":
            doc["connections"].append(textline.strip())
        elif mode == "mistakes":
            doc["mistakes"].append(textline.strip())
        elif mode == "summary":
            summary_parts.append(textline.strip())
        else:
            if current is None:
                start_section(doc["title"] or "Notes", 1)
            current["sentences"].extend(split_sentences(textline))

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()

        marker = _PAGE_MARKER_RE.match(stripped)
        if marker:
            page = int(marker.group(1))
            doc["has_pages"] = True
            # a page break also ends the current paragraph/example run
            example_run = False
            continue

        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            if stripped and mode == "content":
                if current is None:
                    start_section(doc["title"] or "Notes", 1)
                current["examples"].append(stripped)
            continue

        if not stripped:
            example_run = False
            continue

        m = _MD_HEADING_RE.match(stripped)
        if m:
            md_heading_count += 1
            level = len(m.group(1))
            heading = m.group(2)
            if level == 1 and doc["title"] is None:
                doc["title"] = _clean_heading(heading)
                # a level-1 title also opens a section: its prose often defines
                # the topic itself ("# Strings" followed by "A string is ...")
            start_section(heading, level)
            continue

        # setext underline: previous line was the heading (already consumed as
        # prose — cheap approach: detect and promote is complex, so we accept
        # markdown/plain headings as the main paths)
        if re.fullmatch(r"=+|-{3,}", stripped) and i > 0 and lines[i - 1].strip():
            continue

        bm = _BULLET_RE.match(line)
        if bm:
            item = bm.group(2).strip()
            example_run = False
            if mode == "objectives":
                doc["objectives"].append(item)
            elif mode == "connections":
                doc["connections"].append(item)
            elif mode == "mistakes":
                doc["mistakes"].append(item)
            elif mode == "summary":
                summary_parts.append(item)
            else:
                if current is None:
                    start_section(doc["title"] or "Notes", 1)
                (current["examples"] if _is_code_like(item) else current["bullets"]).append(item)
            continue

        if _EXAMPLE_PREFIX_RE.match(stripped):
            example_run = True
            rest = _EXAMPLE_PREFIX_RE.sub("", stripped).strip()
            if rest and mode == "content":
                if current is None:
                    start_section(doc["title"] or "Notes", 1)
                current["examples"].append(rest)
            continue

        if _is_code_like(stripped) or example_run:
            if mode == "content":
                if current is None:
                    start_section(doc["title"] or "Notes", 1)
                current["examples"].append(stripped)
            continue

        # plain-text heading detection (only when markdown headings are absent)
        if md_heading_count == 0 and mode == "content":
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if _plain_heading(line, nxt):
                if doc["title"] is None and not doc["sections"]:
                    doc["title"] = _clean_heading(stripped)
                start_section(stripped, 2)
                continue

        add_prose(stripped)

    doc["summary"] = " ".join(summary_parts)
    content_sections = [s for s in doc["sections"]
                        if s["sentences"] or s["bullets"] or s["examples"]]
    doc["has_structure"] = md_heading_count >= 2 or len(content_sections) >= 2
    return doc


def strip_structural_prefix(heading: str) -> str:
    """Remove a "Module 3:" / "Lesson —" style navigation prefix."""
    return _STRUCTURAL_PREFIX_RE.sub("", heading or "", count=1).strip()


def is_generic_heading(heading: str) -> bool:
    """True when a heading is document STRUCTURE rather than a teachable idea.

    A heading is evidence that a concept *might* be here; it is not proof.
    "Lesson: Cloud Computing Overview", "Agenda", "Thank You" and "Module 4"
    organise a deck without naming anything a student could explain back.
    """
    raw = (heading or "").strip().rstrip(":.").strip()
    if not raw:
        return True
    if _LEGAL_RE.search(raw) or _NUMBER_ONLY_RE.match(raw):
        return True
    inner = strip_structural_prefix(raw).rstrip(":.").strip()
    if not inner:
        return True  # the heading was nothing but navigation ("Module 3:")
    for candidate in {raw.lower(), inner.lower()}:
        if any(candidate == g or candidate.startswith(g + " ") for g in _GENERIC_HEADINGS):
            return True
        if any(candidate.endswith(sfx) for sfx in _STRUCTURAL_SUFFIXES):
            return True
    return False


def parse_connection_line(line: str) -> dict | None:
    """Parse an "A → label → B" style connection bullet.

    Accepts →, ->, or " - " as separators with 3 parts; returns None when the
    line is a plain sentence (handled by the co-mention fallback instead).
    """
    for sep in ("→", "->"):
        if line.count(sep) >= 2:
            parts = [p.strip() for p in line.split(sep)]
            if len(parts) >= 3 and parts[0] and parts[-1]:
                return {"source": parts[0], "label": " ".join(parts[1:-1]).strip() or "relates to",
                        "target": parts[-1], "description": line}
    return None
