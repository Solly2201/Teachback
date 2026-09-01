"""Deterministic NLP preparation of lecture material — no LLM involved.

Pipeline (structure first, semantics second):

  lecture text (pasted, or extracted+cleaned from a PDF by nlp/pdf_clean.py)
      -> structured parse (nlp/lecture_parser.py): headings, bullets, code
         blocks, examples, page provenance, and the special sections
         (objectives / connections / common mistakes / summary)
      -> STRUCTURED path (real headings found): each content section becomes a
         CANDIDATE concept, not a concept. A heading is evidence that a concept
         might be here; it is not proof. Every candidate must earn its place:
            * it is not document structure ("Agenda", "Lesson: X Overview")
            * the lecture actually explains it — a definition-like sentence, or
              supporting facts plus prose
            * its meaning is CHOSEN, not taken blindly from the first line
            * its facts are ranked, not the first N leftovers
         Candidates are then scored (definition, facts, examples, objective
         match, semantic centrality) and the best ones kept, with an honest
         "strong / moderate / weak evidence" label for the review screen.
      -> UNSTRUCTURED fallback (plain prose): content-word n-grams scored by
         frequency x embedding centrality, kept only when the lecture actually
         explains them; meanings and facts go through the same selectors
      -> relationships: explicit "A -> label -> B" lines from an Important
         Connections section are authoritative. Prose-inferred relationships
         are deliberately conservative: two concepts sharing a sentence is NOT
         a pedagogical relationship, so an explicit relational cue (uses,
         enables, consists of, produces, ...) must sit between them. A false
         relationship is worse than a missing one because it later shapes how
         a student's answer is judged.
      -> misconception suggestions: lines from a Common Mistakes section, plus
         misconceptions already authored in the system
      -> deterministic question drafts per concept (main / easier / probe /
         application), grounded in that concept's own facts and examples
      -> suggested activities built from the lecture's own concepts

Every concept keeps its provenance (source_section, source_page,
source_sentences, examples) so the review screen can show WHERE a suggestion
came from and how strong the evidence was. Everything produced here is a
SUGGESTION for the faculty review screen — nothing becomes authoritative
until the teacher publishes it.
"""
import math
import re

import numpy as np

from .analyzer import _STOPWORDS, content_words
from .embedder import cosine_matrix, embed
from .lecture_parser import (_is_code_like, is_generic_heading,
                             parse_connection_line, parse_lecture,
                             strip_structural_prefix)

MAX_CONCEPTS = 8
MAX_RELATIONSHIPS = 6
MAX_MISCON_SUGGESTIONS = 4
MAX_CANDIDATES = 40          # embedding-batch cap for candidate phrases
DEDUPE_T = 0.80              # candidates closer than this are duplicates
MIN_CENTRALITY = 0.15        # phrase must be at least loosely about the lecture
MISCON_SUGGEST_T = 0.45      # configured misconception close enough to suggest
MAX_FACTS = 4
MAX_EXAMPLES = 3

MIN_MEANING_SCORE = 2        # below this, no sentence is a usable "meaning"
MIN_FACT_SCORE = 2           # below this, a sentence is not an "important fact"
MIN_CANDIDATE_SCORE = 1.5    # below this, a heading is not a teachable concept
OBJECTIVE_MATCH_T = 0.45     # objective <-> candidate similarity that counts
MAX_REL_SPAN = 12            # words allowed between two linked concepts

# words that are frequent in lecture notes but are never useful concepts on
# their own (they can still appear inside a longer phrase or a heading)
_LECTURE_NOISE = {"example", "examples", "lecture", "today", "slide", "slides",
                  "chapter", "note", "notes", "topic", "student", "students",
                  "letter", "letters", "value", "values", "thing", "things",
                  "word", "words", "number", "numbers", "way", "ways", "part",
                  "parts", "item", "items", "output", "result", "results",
                  "line", "lines", "code", "position", "positions",
                  # generic nouns that a slide deck repeats without teaching
                  "organization", "organizations", "organisation", "organisations",
                  "system", "systems", "information", "service", "services",
                  "resource", "resources", "customer", "customers", "user",
                  "users", "company", "companies", "business", "businesses",
                  "provider", "providers", "decade", "year", "years", "manner",
                  "module", "lesson", "course", "copyright", "corporation"}

# common lecture verbs: never a concept on their own, and trimmed from the
# edges of candidate phrases ("Assignment uses" -> "Assignment")
_COMMON_VERBS = {"use", "uses", "used", "using", "support", "supports", "combine", "combines",
                 "combined", "produce", "produces", "give", "gives", "given", "refer", "refers",
                 "store", "stores", "stored", "make", "makes", "made", "provide", "provides",
                 "apply", "applies", "call", "called", "calls", "mean", "means", "take", "takes",
                 "show", "shows", "shown", "write", "writes", "written", "perform", "performs",
                 "allow", "allows", "contain", "contains", "assign", "assigns", "assigned",
                 "multiply", "multiplies", "multiplied", "learn", "learns", "learned", "help",
                 "helps", "work", "works", "come", "comes", "get", "gets", "compute", "computes",
                 "computed", "tell", "tells", "told", "propagate", "propagates", "propagated",
                 "propagating", "measure", "measures", "measured", "update", "updates", "updated",
                 "updating", "decrease", "decreases", "decreased", "increase", "increases",
                 "increased", "change", "changes", "changed", "changing",
                 "map", "maps", "mapped", "happen", "happens", "happened", "land",
                 "lands", "landed", "go", "goes", "went", "put", "puts", "keep",
                 "keeps", "kept", "hold", "holds", "held", "need", "needs", "needed",
                 "find", "finds", "start", "starts", "started", "begin", "begins",
                 "add", "adds", "added", "remove", "removes", "removed", "return",
                 "returns", "returned", "create", "creates", "created", "describe",
                 "describes", "described", "represent", "represents", "represented",
                 "mention", "mentions", "mentioned", "talk", "talks", "talked",
                 "discuss", "discusses", "discussed", "look", "looks", "looked",
                 "cover", "covers", "covered", "explain", "explains", "explained",
                 "define", "defines", "defined", "consist", "consists", "enable",
                 "enables", "enabled", "require", "requires", "required"}

# Discourse markers that introduce lecture ADMIN rather than lecture content
# ("we also mentioned the canteen timings"). A candidate whose only support is
# a narration sentence is not a taught idea. Structural, not subject-specific.
_NARRATION_RE = re.compile(
    r"\b(we (also |then |just |briefly )?(talked|spoke|mentioned|discussed|looked|covered|"
    r"went over|will (look|talk|cover))|today we|last (class|week|time)|next (class|week|time)|"
    r"don't forget|any questions|as i (said|mentioned)|see you|reminder that)\b", re.I)

# ---------------------------------------------------------------------------
# sentence-level quality signals (used for meanings AND facts)
# ---------------------------------------------------------------------------

# predicates that mark a sentence as explaining/defining something
_DEFINITION_RE = re.compile(
    r"\b(is|are|was|were|means|meaning|refers to|is called|are called|is known as|"
    r"is defined as|describes?|represents?|consists? of|contains?|includes?|"
    r"is made up of|made up of|allows?|enables?|lets|provides?|supplies|offers?|"
    r"is used (?:to|for)|are used (?:to|for)|helps?|happens? when|occurs? when|"
    r"stands for|can be|cannot be|builds?|creates?|produces?|works? by|"
    r"gives?|returns?|stores?|holds?|starts?|breaks?)\b", re.I)

# a much broader "this sentence asserts something" test
_VERBISH_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|can|cannot|could|may|might|will|"
    r"would|should|must|does|do|did|not)\b|\b\w{3,}(?:s|es|ed|ing)\b", re.I)

# unambiguous document metadata that must never become a meaning or a fact
_METADATA_RE = re.compile(
    r"(©|\(c\)\s*(19|20)\d{2}|\bcopyright\b|\ball rights reserved\b|™|®|"
    r"\bconfidential\b|\bproprietary\b|\btrademarks?\b|"
    r"^\s*(module|lesson|chapter|slide|unit|section|week|page)\s*\d+\b|"
    r"^\s*(figure|fig\.|table|exhibit)\s*\d+\b|"
    r"^\s*(source|reference|references|see also|adapted from)\s*:|"
    r"^\s*(https?://|www\.)|@\w+\.\w+)", re.I)

_TRAILING_FRAGMENT_RE = re.compile(r"[,;:]\s*$")


def _sentence_signals(text: str) -> dict:
    """Deterministic quality signals for one candidate sentence."""
    s = (text or "").strip()
    words = s.split()
    cw = content_words(s)
    return {
        "text": s,
        "n_words": len(words),
        "n_content": len(set(cw)),
        "metadata": bool(_METADATA_RE.search(s)),
        "code": _is_code_like(s),
        "definition": bool(_DEFINITION_RE.search(s)),
        "declarative": bool(_VERBISH_RE.search(s)),
        "fragment": bool(_TRAILING_FRAGMENT_RE.search(s)) or len(words) < 4,
        "has_number": bool(re.search(r"(?<![\w.])\d+(?![\w.])", s)),
    }


def _name_tokens(name: str) -> list[str]:
    return [_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", name or "")]


def _mentions(name: str, text: str) -> bool:
    targets = _name_tokens(name)
    if not targets:
        return False
    words = set(_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", text or ""))
    return all(t in words for t in targets)


def _meaning_score(name: str, sig: dict, first: bool) -> float:
    """How well this sentence works as "what a student should be able to say"."""
    if sig["metadata"]:
        return -10.0
    if sig["code"]:
        return -6.0
    score = 0.0
    if _mentions(name, sig["text"]):
        score += 3.0
        if sig["text"].lower().lstrip("a ").lstrip("an ").lstrip("the ").startswith(
                (name or "").lower()[:12]):
            score += 1.0
    if sig["definition"]:
        score += 3.0
    if 6 <= sig["n_words"] <= 40:
        score += 2.0
    elif 4 <= sig["n_words"] < 6:
        score += 1.0
    else:
        score -= 2.0
    if not sig["declarative"]:
        score -= 3.0
    if sig["fragment"]:
        score -= 2.0
    if sig["text"].rstrip().endswith(":"):
        score -= 2.0
    if first:
        score += 1.0  # the lead sentence of a section IS a mild prior
    return score


def best_meaning(name: str, candidates: list[str]) -> tuple[str, float]:
    """Pick the best explanation sentence for a concept, or ("", score).

    Deliberately NOT "the first sentence under the heading": a slide's first
    line is as likely to be a lead-in, a fragment or a leftover artifact as it
    is to be the definition.
    """
    best, best_score = "", MIN_MEANING_SCORE - 0.001
    for i, text in enumerate(candidates):
        sig = _sentence_signals(text)
        score = _meaning_score(name, sig, first=(i == 0))
        if score > best_score:
            best, best_score = sig["text"], score
    return best, best_score


def _overlap_ratio(a: str, b: str) -> float:
    wa, wb = set(content_words(a)), set(content_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _fact_score(name: str, sig: dict) -> float:
    """How well this sentence works as a specific, checkable lecture claim."""
    if sig["metadata"]:
        return -10.0
    if sig["code"]:
        return -6.0
    score = 0.0
    if 5 <= sig["n_words"] <= 30:
        score += 2.0
    elif sig["n_words"] < 5:
        score -= 3.0
    else:
        score -= 1.0
    score += 2.0 if sig["declarative"] else -4.0
    if sig["fragment"]:
        score -= 2.0
    if _mentions(name, sig["text"]):
        score += 2.0
    if sig["has_number"]:
        score += 1.0  # "Indexes start at 0" — specific and checkable
    if sig["n_content"] >= 3:
        score += 1.0
    return score


def rank_facts(name: str, meaning: str, candidates: list[str],
               limit: int = MAX_FACTS) -> list[str]:
    """Rank supporting claims instead of taking the first N leftovers.

    A fact must say something about the concept, must be a complete claim
    rather than a fragment ("organizations", "the coming decade"), and must
    not simply restate the meaning.
    """
    scored = []
    seen = set()
    for i, text in enumerate(candidates):
        sig = _sentence_signals(text)
        key = " ".join(sorted(set(content_words(text))))
        if not key or key in seen:
            continue
        if meaning and _overlap_ratio(text, meaning) >= 0.8:
            continue
        score = _fact_score(name, sig)
        if score < MIN_FACT_SCORE:
            continue
        seen.add(key)
        scored.append((-score, i, sig["text"]))
    scored.sort()
    return [t for _, _, t in scored[:limit]]


def extract_text(filename: str, data: bytes) -> str:
    """Text from an uploaded lecture file (see extract_material for the report)."""
    return extract_material(filename, data)[0]


def extract_material(filename: str, data: bytes) -> tuple[str, dict]:
    """Lecture text plus an ingestion report, from an uploaded file.

    .txt/.md keep their existing behaviour untouched — they already carry the
    structure the parser wants. .pdf goes through the layout-aware extractor
    and the deterministic cleanup, and comes back as ordinary Markdown notes
    with ``<!-- page N -->`` provenance markers, so BOTH paths feed exactly
    the same downstream pipeline.
    """
    name = (filename or "").lower()
    if name.endswith((".txt", ".md")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return text, {"kind": "text", "scanned": False, "page_count": None,
                      "removed_total": 0, "removed_by_reason": []}
    if name.endswith(".pdf"):
        from .pdf_clean import pdf_to_notes
        from .pdf_extract import PdfExtractionError

        try:
            text, report = pdf_to_notes(data)
        except PdfExtractionError as exc:
            raise ValueError(str(exc)) from exc
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ValueError("PDF support needs the 'pdfplumber' or 'pypdf' package.") from exc
        report["kind"] = "pdf"
        if report.get("text_quality", "text") != "text":
            raise ScannedPdfError(_coverage_message(report), report)
        return text, report
    raise ValueError("Unsupported file type — upload .txt, .md or .pdf, or paste the notes as text.")


class ScannedPdfError(ValueError):
    """A PDF that could not be fully extracted, reported instead of half-ingested.

    Covers both a wholly scanned document and — the case that actually bites —
    a MIXED one, where some pages are photographs of the board and the rest are
    typed. There is no OCR here, so those pages carry no text at all; accepting
    the remainder would hand the teacher a fraction of the lecture presented as
    the whole of it.
    """

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _coverage_message(report: dict) -> str:
    """Say exactly which pages could not be read, and what to do instead."""
    from .pdf_extract import format_page_ranges

    pages = format_page_ranges(report.get("image_pages") or [])
    total = report.get("page_count") or 0
    n_image = report.get("image_page_count") or 0
    advice = ("Upload an OCR-enabled PDF, paste the lecture text directly, or use the "
              "prepared-note workflow.")
    if report.get("text_quality") == "scanned":
        return ("This PDF appears to be scanned or image-based: none of its "
                f"{total} page(s) contain selectable text, so nothing could be "
                f"extracted. {advice}")
    return (f"This PDF contains image-based pages with little or no selectable text "
            f"(page{'s' if n_image != 1 else ''} {pages}). {n_image} of {total} pages "
            "could not be read, so the lecture would only be partly imported — "
            f"TeachBack will not import part of a lecture as if it were all of it. {advice}")


def _norm_word(w: str) -> str:
    """Naive singular/plural + case normalisation for matching."""
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _norm_phrase(phrase: str) -> str:
    return " ".join(_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", phrase))


def _display_name(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    first = words[0] if words[0][:1].isupper() else words[0].capitalize()
    return " ".join([first] + words[1:])


# ---------------------------------------------------------------------------
# question and activity drafting (templates filled with the lecture's own
# facts/examples — deterministic, no generation)
# ---------------------------------------------------------------------------

def _questions_for(name: str, meaning: str, facts: list[str], examples: list[str]) -> dict:
    probe = f"What is the key idea of {name}?"
    for fact in facts:
        short = fact.strip().rstrip(".")
        if 3 <= len(short.split()) <= 18:
            probe = f'The lecture mentioned: "{short}." What does that mean?'
            break
    application = ""
    if examples:
        application = (f"Here is an example from the lecture: {examples[0]} — "
                       "can you explain what it shows?")
    return {
        "main_question": f'What did you understand about "{name}"?',
        "easier_question": f'In simple words — how would you describe {name} to a friend?',
        "probe_question": probe,
        "application_question": application,  # optional extension, never required
    }


def _suggested_activities(title: str, concepts: list[dict], relationships: list[dict]) -> list[dict]:
    """Draft activities grounded in the lecture's own concepts and examples."""
    if not concepts:
        return []
    acts = [{
        "target_state": "not_trying", "kind": "re_engagement",
        "title": f"One-line warm-up: {title or concepts[0]['name']}",
        "description": "One very short question to get moving again.",
        "content": "No pressure — one honest sentence is enough.",
        "question": f"In one sentence: what was the lecture on {title or concepts[0]['name']} about?",
    }]
    c0 = concepts[0]
    review_body = c0.get("description", "")
    if c0.get("facts"):
        review_body = (review_body + " " + " ".join(c0["facts"][:2])).strip()
    acts.append({
        "target_state": "unclear", "kind": "concept_review",
        "title": f"{c0['name']} in plain words",
        "description": f"Re-read the core idea of {c0['name']} and say it back simply.",
        "content": f"From the lecture: {review_body}" if review_body else f"Revisit {c0['name']} in your notes.",
        "question": f"Now explain {c0['name']} in one sentence, in your own words.",
    })
    example_concept = next((c for c in concepts if c.get("examples")), None)
    if example_concept:
        acts.append({
            "target_state": "struggling", "kind": "guided_practice",
            "title": f"Work through an example of {example_concept['name']}",
            "description": "Step through one example from the lecture.",
            "content": f"Example from the lecture: {example_concept['examples'][0]}",
            "question": "What is the result of this example, and why?",
        })
    else:
        acts.append({
            "target_state": "struggling", "kind": "guided_practice",
            "title": f"An everyday example of {c0['name']}",
            "description": f"Connect {c0['name']} to something familiar.",
            "content": f"As a reminder: {review_body}" if review_body else "",
            "question": f"Give one simple real-world example or analogy for {c0['name']}.",
        })
    apply_concept = concepts[1] if len(concepts) > 1 else c0
    acts.append({
        "target_state": "understanding", "kind": "application",
        "title": f"Try using {apply_concept['name']}",
        "description": f"Apply {apply_concept['name']} to a small case of your own.",
        "content": (f"Example from the lecture: {apply_concept['examples'][0]}"
                    if apply_concept.get("examples") else
                    f"Reminder: {apply_concept.get('description', '')}"),
        "question": f"Make up one small example of your own that uses {apply_concept['name']}, and explain it.",
    })
    if relationships:
        r = relationships[0]
        acts.append({
            "target_state": "confident", "kind": "challenge",
            "title": f"Connect the ideas: {r['source']} & {r['target']}",
            "description": "Optional extension — connect two ideas from this lecture.",
            "content": "The strongest test of understanding is connecting ideas rather than repeating them.",
            "question": f"How does {r['source']} relate to {r['target']}? Explain the connection in one or two sentences.",
        })
    elif len(concepts) >= 2:
        acts.append({
            "target_state": "confident", "kind": "challenge",
            "title": f"Connect the ideas: {concepts[0]['name']} & {concepts[1]['name']}",
            "description": "Optional extension — connect two ideas from this lecture.",
            "content": "The strongest test of understanding is connecting ideas rather than repeating them.",
            "question": f"How does {concepts[0]['name']} relate to {concepts[1]['name']}?",
        })
    return acts


# ---------------------------------------------------------------------------
# structured path: scored candidate concepts from document sections
# ---------------------------------------------------------------------------

CONFIDENCE_LABELS = {
    "strong": "Strongly supported",
    "moderate": "Moderately supported",
    "weak": "Weak evidence — review carefully",
}


def _merge_key(heading: str) -> str:
    """Identity of a heading for merging. Keeps digits, because "Layer 1" and
    "Layer 2" are two concepts even though "Module 3" and "Module 7" are one
    navigation label (those are filtered as structure before we get here)."""
    return re.sub(r"[^a-z0-9]+", " ", (heading or "").lower()).strip()


def _merge_sections(sections: list[dict]) -> list[dict]:
    """One candidate per distinct heading, evidence merged across its slides.

    A slide deck often returns to the same title ("Cloud Computing" on slides
    1, 2 and 9). That is one concept taught in several places, not three.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for sec in sections:
        heading = _clean_candidate_name(sec["heading"])
        key = _merge_key(heading)
        if not key:
            continue
        if key not in merged:
            merged[key] = {"heading": heading, "level": sec["level"],
                           "sentences": [], "bullets": [], "examples": [],
                           "page": sec.get("page"), "pages": [], "occurrences": 0}
            order.append(key)
        entry = merged[key]
        entry["occurrences"] += 1
        entry["sentences"] += sec["sentences"]
        entry["bullets"] += sec["bullets"]
        entry["examples"] += sec["examples"]
        if sec.get("page") and sec["page"] not in entry["pages"]:
            entry["pages"].append(sec["page"])
        if entry["page"] is None:
            entry["page"] = sec.get("page")
    return [merged[k] for k in order]


def _clean_candidate_name(heading: str) -> str:
    """Strip navigation noise from a heading before it becomes a concept name."""
    name = strip_structural_prefix(heading or "").strip()
    return (name or (heading or "").strip()).strip(" .:-–—")


def _candidate_from_section(sec: dict) -> dict | None:
    """Turn one merged section into a scored concept candidate, or None."""
    name = _display_name(sec["heading"])
    if not name or is_generic_heading(name):
        return None
    prose = list(sec["sentences"]) + [b for b in sec["bullets"] if len(b.split()) >= 3]
    if not prose:
        return None  # a bare heading with no content is not a teachable concept

    meaning, meaning_score = best_meaning(name, prose)
    remaining = [p for p in prose if p != meaning]
    facts = rank_facts(name, meaning, remaining)
    examples = [e for e in sec["examples"] if not _METADATA_RE.search(e)][:MAX_EXAMPLES]

    signals = []
    if meaning:
        signals.append("a definition-like sentence")
    if facts:
        signals.append(f"{len(facts)} supporting statement{'s' if len(facts) > 1 else ''}")
    if examples:
        signals.append("a worked example")
    if sec["occurrences"] > 1:
        signals.append(f"taught on {sec['occurrences']} slides")

    score = 0.0
    score += 3.0 if meaning else 0.0
    score += min(len(facts), 3) * 1.0
    score += 1.5 if examples else 0.0
    score += 0.5 if sec["occurrences"] > 1 else 0.0
    if not meaning and len(facts) < 1:
        score -= 3.0

    return {
        "name": name,
        "description": meaning,
        "facts": facts,
        "examples": examples,
        "source_section": sec["heading"],
        "source_page": sec.get("page"),
        "source_pages": sec.get("pages") or ([sec["page"]] if sec.get("page") else []),
        "source_sentences": [x for x in ([meaning] + facts) if x],
        "count": sec["occurrences"],
        "score": round(score, 3),
        "_base_score": score,
        "_signals": signals,
        "_has_meaning": bool(meaning),
        "_meaning_score": round(meaning_score, 2),
    }


def _objective_similarity(candidates: list[dict], objectives: list[str]) -> list[float]:
    """Semantic match between each candidate and the teacher's own objectives.

    Teacher objectives are authoritative about what matters in this lecture,
    so they RANK candidates. They never invent one: a concept still has to be
    supported by the material itself.
    """
    if not candidates or not objectives:
        return [0.0] * len(candidates)
    texts = [f"{c['name']}. {c.get('description', '')}".strip() for c in candidates]
    emb = embed(texts + objectives)
    cand_emb, obj_emb = emb[:len(texts)], emb[len(texts):]
    sims = cosine_matrix(cand_emb, obj_emb)
    return [float(np.max(sims[i])) for i in range(len(texts))]


def _centrality(candidates: list[dict], document_text: str) -> list[float]:
    if not candidates or not document_text.strip():
        return [0.0] * len(candidates)
    texts = [f"{c['name']}. {c.get('description', '')}".strip() for c in candidates]
    emb = embed(texts + [document_text])
    sims = cosine_matrix(emb[:len(texts)], emb[len(texts):])
    return [float(sims[i, 0]) for i in range(len(texts))]


def _confidence_of(cand: dict, objective_match: float) -> tuple[str, str]:
    strong_support = cand["_has_meaning"] and (cand["facts"] or cand["examples"])
    if strong_support:
        level = "strong"
    elif cand["_has_meaning"] or len(cand["facts"]) >= 2:
        level = "moderate"
    else:
        level = "weak"
    signals = list(cand["_signals"])
    if objective_match >= OBJECTIVE_MATCH_T:
        signals.append("matches one of your learning objectives")
        if level == "weak":
            level = "moderate"
    where = ""
    if cand.get("source_pages"):
        pages = cand["source_pages"]
        where = (f"page {pages[0]}" if len(pages) == 1
                 else "pages " + ", ".join(str(p) for p in pages[:3]))
    parts = []
    if cand.get("source_section"):
        parts.append(f'Found under "{cand["source_section"]}"' + (f" on {where}" if where else ""))
    elif where:
        parts.append(f"Found on {where}")
    if signals:
        parts.append("supported by " + ", ".join(signals))
    reason = "; ".join(parts) or "Suggested from repeated mentions in the material."
    return level, reason


def _concepts_from_sections(doc: dict, objectives: list[str], max_concepts: int) -> list[dict]:
    merged = _merge_sections(doc["sections"])
    candidates = [c for c in (_candidate_from_section(s) for s in merged) if c]
    if not candidates:
        return []

    document_text = " ".join(
        s for sec in doc["sections"] for s in (sec["sentences"] + sec["bullets"])
    )[:6000]
    obj_sims = _objective_similarity(candidates, objectives)
    centralities = _centrality(candidates, document_text)

    kept = []
    for i, cand in enumerate(candidates):
        obj = obj_sims[i]
        total = cand["_base_score"] + 2.0 * obj + 2.0 * centralities[i]
        prioritised = obj >= OBJECTIVE_MATCH_T
        # a heading only becomes a concept when the lecture explains it — or
        # when the teacher's own objective names it AND there is some prose
        supported = cand["_has_meaning"] or len(cand["facts"]) >= 1
        if not supported and not prioritised:
            continue
        if total < MIN_CANDIDATE_SCORE and not prioritised:
            continue
        level, reason = _confidence_of(cand, obj)
        cand.update({"score": round(total, 3), "confidence": level,
                     "confidence_label": CONFIDENCE_LABELS[level],
                     "confidence_reason": reason,
                     "objective_match": round(obj, 3)})
        kept.append((total, i, cand))

    # rank by evidence, keep the best, then restore document order so the
    # review screen reads in the order the lecture taught things
    kept.sort(key=lambda t: (-t[0], t[1]))
    selected = sorted(kept[:max_concepts], key=lambda t: t[1])
    out = []
    for _, _, cand in selected:
        out.append({k: v for k, v in cand.items() if not k.startswith("_")})
    return out


# ---------------------------------------------------------------------------
# unstructured fallback: scored n-grams with explanatory-support filter
# ---------------------------------------------------------------------------

def _candidate_phrases(sentences: list[str]) -> list[dict]:
    """Contiguous content-word n-grams (1-3 words) with occurrence counts."""
    counts: dict[str, dict] = {}

    def add(gram: list[str]) -> None:
        if any(w.lower() in _COMMON_VERBS for w in gram):
            return
        display = " ".join(gram)
        key = _norm_phrase(display)
        if not key or any(_norm_word(w) in _LECTURE_NOISE for w in gram):
            return
        entry = counts.setdefault(key, {"display": display, "count": 0, "words": len(gram)})
        entry["count"] += 1

    for s in sentences:
        run: list[str] = []
        for w in re.findall(r"[A-Za-z][A-Za-z\-']*", s) + [""]:
            if w and w.lower() not in _STOPWORDS and len(w) > 2:
                run.append(w)
                continue
            for n in (1, 2, 3):
                for i in range(len(run) - n + 1):
                    add(run[i:i + n])
            run = []
    return list(counts.values())


def _has_explanatory_support(name: str, sentences: list[str]) -> str | None:
    """The sentence that explains this candidate, or None.

    A concept is only worth suggesting when the lecture actually says
    something ABOUT it: it appears in the first half of a sentence that then
    elaborates (>= 6 words). Bare mentions inside longer clauses don't count.
    """
    targets = [_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", name)]
    if not targets:
        return None
    for s in sentences:
        words = [_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", s)]
        if len(words) < 6 or not all(t in words for t in targets):
            continue
        if _METADATA_RE.search(s) or _is_code_like(s) or _NARRATION_RE.search(s):
            continue
        first_pos = words.index(targets[0])
        if first_pos <= max(2, len(words) // 2 - 1):
            return s
    return None


def _concepts_from_prose(doc: dict, title: str, objectives: list[str],
                         max_concepts: int) -> list[dict]:
    sentences = [s for sec in doc["sections"] for s in sec["sentences"]
                 if not _METADATA_RE.search(s)]
    if not sentences:
        return []
    candidates = sorted(_candidate_phrases(sentences), key=lambda c: -c["count"])[:MAX_CANDIDATES]
    if not candidates:
        return []

    texts = [c["display"] for c in candidates] + sentences + [" ".join(sentences)]
    emb = embed(texts)
    n_c, n_s = len(candidates), len(sentences)
    cand_emb, sent_emb, doc_emb = emb[:n_c], emb[n_c:n_c + n_s], emb[n_c + n_s:]

    centrality = cosine_matrix(cand_emb, doc_emb)[:, 0]
    priority_norm = _norm_phrase(title + " " + " ".join(objectives))
    priority_words = set(priority_norm.split())

    scored = []
    for i, c in enumerate(candidates):
        cent = float(centrality[i])
        if cent < MIN_CENTRALITY:
            continue
        support = _has_explanatory_support(c["display"], sentences)
        key_words = set(_norm_phrase(c["display"]).split())
        prioritized = bool(key_words & priority_words)
        # no explanatory sentence and not named by the teacher -> not a concept
        if support is None and not prioritized:
            continue
        boost = 1.5 if prioritized else 1.0
        score = (1 + math.log(c["count"])) * cent * boost * (1 + 0.1 * (c["words"] - 1))
        scored.append((score, i, c, support))
    scored.sort(key=lambda t: -t[0])

    kept: list[tuple[int, dict, str | None]] = []
    for score, i, c, support in scored:
        dup = False
        for j, k, _ in kept:
            if float(cosine_matrix(cand_emb[i:i + 1], cand_emb[j:j + 1])[0, 0]) >= DEDUPE_T:
                dup = True
                break
            a = set(_norm_phrase(c["display"]).split())
            b = set(_norm_phrase(k["display"]).split())
            if a <= b or b <= a:
                dup = True
                break
        if not dup:
            kept.append((i, c, support))
        if len(kept) >= max_concepts:
            break

    concepts = []
    for i, c, support in kept:
        name = _display_name(c["display"])
        sims = cosine_matrix(cand_emb[i:i + 1], sent_emb)[0]
        best_j = int(np.argmax(sims))
        # the same deliberate meaning selection the structured path uses, over
        # the sentences that actually mention the candidate
        mentioning = [s for s in sentences if _mentions(name, s)]
        pool = ([support] if support else []) + mentioning
        if float(sims[best_j]) >= 0.3 and sentences[best_j] not in pool:
            pool.append(sentences[best_j])
        meaning, _ = best_meaning(name, pool)
        facts = rank_facts(name, meaning, [s for s in mentioning if s != meaning])
        has_support = bool(meaning or facts)
        level = "strong" if (meaning and facts) else ("moderate" if has_support else "weak")
        concepts.append({
            "name": name,
            "description": meaning,
            "facts": facts,
            "examples": [],
            "source_section": "",
            "source_page": None,
            "source_pages": [],
            "source_sentences": [x for x in ([meaning] + facts) if x],
            "count": c["count"],
            "score": round(float(sims[best_j]), 3),
            "confidence": level,
            "confidence_label": CONFIDENCE_LABELS[level],
            "confidence_reason": (
                f"Mentioned {c['count']} times in the notes"
                + ("; explained in a full sentence" if meaning else "")
                + (f"; {len(facts)} supporting statements" if facts else "")
            ),
            "objective_match": 0.0,
        })
    return concepts


# ---------------------------------------------------------------------------
# relationships and misconceptions
# ---------------------------------------------------------------------------

def _mention_index(concept_name: str, sentence_words: list[str]) -> int | None:
    targets = [_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", concept_name)]
    if not targets:
        return None
    norm_sentence = [_norm_word(w) for w in sentence_words]
    if not all(t in norm_sentence for t in targets):
        return None
    return norm_sentence.index(targets[0])


# A relationship claims something pedagogical about two ideas. Two concepts
# merely appearing in one sentence claims nothing, so an explicit relational
# cue must sit BETWEEN them. Ordered: the first match wins, so the specific
# patterns come before the generic ones.
_REL_CUE_PATTERNS = [
    (r"\bis made up of\b|\bmade up of\b|\bconsists? of\b|\bcomposed of\b", "consists of"),
    (r"\bis part of\b|\bpart of\b|\bbelongs? to\b", "is part of"),
    (r"\bcontains?\b|\bincludes?\b|\bholds?\b", "contains"),
    (r"\bconverts?\b|\btransforms?\b|\bturns?\b", "converts into"),
    (r"\bproduces?\b|\bresults? in\b|\breturns?\b|\bcreates?\b|\bgenerates?\b", "produces"),
    (r"\bcauses?\b|\bleads? to\b", "leads to"),
    (r"\brequires?\b|\bdepends? on\b|\bneeds?\b", "requires"),
    (r"\benables?\b|\ballows?\b|\blets\b|\bmakes? it possible\b", "enables"),
    (r"\bprovides?\b|\bsupplies\b|\boffers?\b|\bdelivers?\b", "provides"),
    (r"\bis based on\b|\bbased on\b|\bbuilt on\b|\brelies on\b", "is based on"),
    (r"\baccesses?\b|\breads?\b|\bextracts?\b|\bselects?\b|\bretrieves?\b", "accesses"),
    (r"\bcomputes?\b|\bcalculates?\b|\bmeasures?\b", "computes"),
    (r"\bupdates?\b|\bmodifies?\b|\badjusts?\b", "updates"),
    (r"\bstores?\b|\brepresents?\b|\bdescribes?\b", "describes"),
    (r"\bunlike\b|\bcompared (?:to|with)\b|\bwhereas\b|\brather than\b|\binstead of\b",
     "contrasts with"),
    (r"\buses?\b|\busing\b|\bapplies\b|\bapply\b", "uses"),
]


def _relational_cue(words: list[str]) -> str | None:
    """The relational label expressed by the words between two concepts."""
    if not words or len(words) > MAX_REL_SPAN:
        return None
    span = " ".join(words).lower()
    for pattern, label in _REL_CUE_PATTERNS:
        if re.search(pattern, span):
            return label
    return None


def _relationships_from_doc(doc: dict, concepts: list[dict]) -> list[dict]:
    relationships = []
    seen_pairs = set()

    # 1) explicit "A -> label -> B" lines from an Important Connections
    #    section. This is the teacher's own statement and stays authoritative.
    for line in doc["connections"]:
        parsed = parse_connection_line(line)
        if parsed and (parsed["source"], parsed["target"]) not in seen_pairs:
            relationships.append({**parsed, "source_sentence": line, "origin": "explicit"})
            seen_pairs.add((parsed["source"], parsed["target"]))
        elif not parsed:
            # a plain connection sentence in an explicit connections section:
            # the teacher put it there on purpose, so anchoring it to two
            # concepts is enough
            words = re.findall(r"[A-Za-z][A-Za-z\-']*", line)
            mentioned = sorted(
                (idx, c["name"]) for c in concepts
                if (idx := _mention_index(c["name"], words)) is not None
            )
            if len(mentioned) >= 2:
                (ia, a), (ib, b) = mentioned[0], mentioned[1]
                label = _relational_cue(words[ia + 1:ib]) or "relates to"
                if (a, b) not in seen_pairs:
                    relationships.append({"source": a, "label": label, "target": b,
                                          "description": line, "source_sentence": line,
                                          "origin": "explicit"})
                    seen_pairs.add((a, b))
        if len(relationships) >= MAX_RELATIONSHIPS:
            return relationships

    # 2) prose sentences. CONSERVATIVE by design: co-mention alone is not a
    #    relationship — a relational cue must connect the two concepts, and
    #    they must be close enough for the cue to be about them.
    sentences = [s for sec in doc["sections"] for s in sec["sentences"]
                 if not _METADATA_RE.search(s) and len(s.split()) <= 45]
    for s in sentences:
        sentence_words = re.findall(r"[A-Za-z][A-Za-z\-']*", s)
        mentioned = sorted(
            (idx, c["name"]) for c in concepts
            if (idx := _mention_index(c["name"], sentence_words)) is not None
        )
        for (ia, a), (ib, b) in zip(mentioned, mentioned[1:]):
            if (a, b) in seen_pairs or a == b:
                continue
            label = _relational_cue(sentence_words[ia + 1:ib])
            if label is None:
                continue  # no relational evidence: do NOT invent a connection
            relationships.append({"source": a, "label": label, "target": b,
                                  "description": s, "source_sentence": s,
                                  "origin": "prose"})
            seen_pairs.add((a, b))
            if len(relationships) >= MAX_RELATIONSHIPS:
                return relationships
    return relationships


_MISTAKE_PREFIX_RE = re.compile(
    r"^(students?\s+(may|might|often|sometimes)\s+(think|believe|assume|say)\s+(that\s+)?"
    r"|a\s+common\s+(confusion|mistake|error)\s+is\s+(that\s+|thinking\s+(that\s+)?)?)", re.I)
_MISTAKE_SPLIT_RE = re.compile(
    r"\s*[—–\-;,]?\s*\b(but\s+actually|actually|in\s+fact|but\s+really|whereas|but)\b\s*[,:]?\s*", re.I)


def _misconceptions_from_doc(doc: dict) -> list[dict]:
    """Common Mistakes lines -> misconception suggestions with provenance."""
    out = []
    for line in doc["mistakes"]:
        claim = _MISTAKE_PREFIX_RE.sub("", line.strip()).strip()
        clarification = ""
        m = _MISTAKE_SPLIT_RE.search(claim)
        if m:
            clarification = claim[m.end():].strip().rstrip(".")
            claim = claim[:m.start()].strip().rstrip(".")
        if not claim or _METADATA_RE.search(claim):
            continue
        claim = claim[0].upper() + claim[1:]
        if clarification:
            clarification = clarification[0].upper() + clarification[1:] + "."
        name = claim if len(claim) <= 80 else claim[:77] + "…"
        out.append({
            "name": name,
            "description": claim if claim.endswith(".") else claim + ".",
            "clarification": clarification,
            "probe_question": "Can you explain that part again in your own words?",
            "source": "lecture",
            "source_sentence": line,
        })
    return out[:MAX_MISCON_SUGGESTIONS]


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def prepare_lecture(material: str, title: str = "", description: str = "",
                    objectives: list[str] | None = None,
                    known_misconceptions: list[dict] | None = None,
                    max_concepts: int = MAX_CONCEPTS) -> dict:
    """Suggest a knowledge structure from lecture material.

    Returns {"concepts", "relationships", "objectives",
             "misconception_suggestions", "activities", "structure"}.
    """
    objectives = [o.strip() for o in (objectives or []) if o.strip()]
    doc = parse_lecture(material)

    empty = {"concepts": [], "relationships": [], "objectives": objectives,
             "misconception_suggestions": [], "activities": [],
             "structure": {"has_structure": False, "title": doc["title"],
                           "section_count": 0, "page_count": 0,
                           "candidate_count": 0, "strong_count": 0, "notes": []}}
    if not any(sec["sentences"] or sec["bullets"] or sec["examples"] for sec in doc["sections"]):
        empty["structure"]["notes"] = [
            "No explanatory text could be found in this material, so no concepts were "
            "suggested. Paste the lecture notes or add a short explanation per topic."
        ]
        return empty

    effective_objectives = objectives or doc["objectives"]
    if doc["has_structure"]:
        concepts = _concepts_from_sections(doc, effective_objectives, max_concepts)
        # structure gave nothing usable (e.g. headings without content) ->
        # fall back to prose mining
        if not concepts:
            concepts = _concepts_from_prose(doc, title or (doc["title"] or ""),
                                            effective_objectives, max_concepts)
    else:
        concepts = _concepts_from_prose(doc, title or (doc["title"] or ""),
                                        effective_objectives, max_concepts)

    for c in concepts:
        c.update(_questions_for(c["name"], c["description"], c["facts"], c["examples"]))

    relationships = _relationships_from_doc(doc, concepts)

    # learning objectives: the teacher's own objectives always win, then the
    # notes' own Objectives section, then a template from the concepts
    if not objectives:
        objectives = doc["objectives"] or [
            f"Explain the idea of {c['name']} in your own words." for c in concepts[:3]
        ]

    # misconceptions: lecture "Common Mistakes" lines first, then previously
    # authored misconceptions that this lecture is semantically about
    miscon_suggestions = _misconceptions_from_doc(doc)
    known = [m for m in (known_misconceptions or []) if m.get("description")]
    sentences = [s for sec in doc["sections"] for s in sec["sentences"]]
    if known and sentences and len(miscon_suggestions) < MAX_MISCON_SUGGESTIONS:
        m_emb = embed([m["description"] for m in known])
        s_emb = embed(sentences)
        sims_m = cosine_matrix(m_emb, s_emb)
        catalog = []
        for mi, m in enumerate(known):
            best = float(np.max(sims_m[mi]))
            if best >= MISCON_SUGGEST_T:
                catalog.append({
                    "name": m["name"], "description": m["description"],
                    "clarification": m.get("clarification", ""),
                    "probe_question": m.get("probe_question", ""),
                    "similarity": round(best, 3), "source": "catalog",
                })
        catalog.sort(key=lambda m: -m["similarity"])
        existing = {m["name"] for m in miscon_suggestions}
        for m in catalog:
            if m["name"] not in existing and len(miscon_suggestions) < MAX_MISCON_SUGGESTIONS:
                miscon_suggestions.append(m)

    activities = _suggested_activities(title or doc["title"] or "", concepts, relationships)

    # honest reporting: how much of this draft is actually well supported?
    strong = [c for c in concepts if c.get("confidence") == "strong"]
    weak = [c for c in concepts if c.get("confidence") == "weak"]
    notes = []
    if not concepts:
        notes.append("No concept in this material had enough explanatory support to suggest. "
                     "Add a sentence or two explaining each idea, or enter the concepts yourself.")
    elif len(strong) < len(concepts):
        notes.append(
            f"{len(strong)} of {len(concepts)} suggested concepts have strong supporting evidence."
        )
    if weak:
        notes.append(f"{len(weak)} suggestion(s) rest on weak evidence — review them before publishing.")

    return {
        "concepts": concepts,
        "relationships": relationships,
        "objectives": objectives,
        "misconception_suggestions": miscon_suggestions,
        "activities": activities,
        "structure": {"has_structure": doc["has_structure"], "title": doc["title"],
                      "section_count": len(doc["sections"]),
                      "page_count": len({sec.get("page") for sec in doc["sections"]
                                         if sec.get("page")}),
                      "candidate_count": len(doc["sections"]),
                      "strong_count": len(strong),
                      "notes": notes},
    }
