"""Deterministic NLP preparation of lecture material — no LLM involved.

Pipeline:

  lecture text (pasted / extracted from a file)
      -> sentence segmentation (rule-based splitter from analyzer.py)
      -> candidate phrase extraction: contiguous content-word n-grams,
         scored by frequency x embedding centrality (similarity of the
         phrase to the whole document), boosted when the phrase appears
         in the title or the teacher's learning objectives
      -> embedding-based deduplication
      -> per-concept "possible explanation": the lecture sentence closest
         to the phrase, quoted verbatim
      -> relationship suggestions: sentences mentioning two concepts, with
         the connecting word used as the link label where possible
      -> learning-objective templates (teacher-provided objectives win)
      -> optional misconception suggestions matched against misconceptions
         already configured in the system

Everything produced here is a SUGGESTION for the faculty review screen.
The teacher can rename, remove, add and edit before publishing — the
review step exists precisely because this extraction is heuristic, not
semantic understanding of the lecture.
"""
import math
import re

import numpy as np

from .analyzer import _STOPWORDS, content_words, split_sentences
from .embedder import cosine_matrix, embed

MAX_CONCEPTS = 6
MAX_RELATIONSHIPS = 5
MAX_MISCON_SUGGESTIONS = 3
MAX_CANDIDATES = 40          # embedding-batch cap for candidate phrases
DEDUPE_T = 0.80              # candidates closer than this are duplicates
MIN_CENTRALITY = 0.15        # phrase must be at least loosely about the lecture
MISCON_SUGGEST_T = 0.45      # configured misconception close enough to suggest

# words that are frequent in lecture notes but are never useful concepts
_LECTURE_NOISE = {"example", "examples", "lecture", "today", "slide", "slides",
                  "chapter", "note", "notes", "topic", "student", "students"}

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
                 "increased", "change", "changes", "changed", "changing"}


def extract_text(filename: str, data: bytes) -> str:
    """Text from an uploaded lecture file. Supports .txt/.md and .pdf."""
    name = (filename or "").lower()
    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    if name.endswith(".pdf"):
        try:
            from io import BytesIO

            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ValueError("PDF support needs the 'pypdf' package (pip install pypdf).") from exc
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Unsupported file type — upload .txt, .md or .pdf, or paste the notes as text.")


def _norm_word(w: str) -> str:
    """Naive singular/plural + case normalisation for matching."""
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _norm_phrase(phrase: str) -> str:
    return " ".join(_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", phrase))


def _candidate_phrases(sentences: list[str]) -> list[dict]:
    """Contiguous content-word n-grams (1-3 words) with occurrence counts."""
    counts: dict[str, dict] = {}

    def add(gram: list[str]) -> None:
        # concepts are noun phrases: reject any candidate containing a common
        # verb ("Backpropagation computes gradients" is a clause, not a
        # concept — its nouns still appear as their own candidates)
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


def _display_name(phrase: str) -> str:
    words = phrase.split()
    return " ".join(w if w[:1].isupper() else w.capitalize() for w in words[:1]) + \
        ("" if len(words) == 1 else " " + " ".join(words[1:]))


def _mention_index(concept_name: str, sentence_words: list[str]) -> int | None:
    """Position of the concept's first word in the sentence, or None."""
    targets = [_norm_word(w) for w in re.findall(r"[A-Za-z][A-Za-z\-']*", concept_name)]
    if not targets:
        return None
    norm_sentence = [_norm_word(w) for w in sentence_words]
    if not all(t in norm_sentence for t in targets):
        return None
    return norm_sentence.index(targets[0])


def prepare_lecture(material: str, title: str = "", description: str = "",
                    objectives: list[str] | None = None,
                    known_misconceptions: list[dict] | None = None,
                    max_concepts: int = MAX_CONCEPTS) -> dict:
    """Suggest concepts / relationships / objectives from lecture material.

    Returns {"concepts", "relationships", "objectives", "misconception_suggestions"}.
    """
    objectives = [o.strip() for o in (objectives or []) if o.strip()]
    sentences = split_sentences(material)
    if not sentences:
        return {"concepts": [], "relationships": [],
                "objectives": objectives, "misconception_suggestions": []}

    candidates = sorted(_candidate_phrases(sentences), key=lambda c: -c["count"])[:MAX_CANDIDATES]
    if not candidates:
        return {"concepts": [], "relationships": [],
                "objectives": objectives, "misconception_suggestions": []}

    # one embedding batch: candidates + sentences + full document
    texts = [c["display"] for c in candidates] + sentences + [material]
    emb = embed(texts)
    n_c, n_s = len(candidates), len(sentences)
    cand_emb, sent_emb, doc_emb = emb[:n_c], emb[n_c:n_c + n_s], emb[n_c + n_s:]

    centrality = cosine_matrix(cand_emb, doc_emb)[:, 0]
    # phrases named in the title/objectives are what the teacher cares about
    priority_norm = _norm_phrase(title + " " + " ".join(objectives))
    priority_words = set(priority_norm.split())

    scored = []
    for i, c in enumerate(candidates):
        cent = float(centrality[i])
        if cent < MIN_CENTRALITY:
            continue
        key_words = set(_norm_phrase(c["display"]).split())
        boost = 1.5 if key_words & priority_words else 1.0
        score = (1 + math.log(c["count"])) * cent * boost * (1 + 0.1 * (c["words"] - 1))
        scored.append((score, i, c))
    scored.sort(key=lambda t: -t[0])

    # dedupe near-identical phrases (embeddings + shared normalised words)
    kept: list[tuple[int, dict]] = []
    for score, i, c in scored:
        dup = False
        for j, k in kept:
            if float(cosine_matrix(cand_emb[i:i + 1], cand_emb[j:j + 1])[0, 0]) >= DEDUPE_T:
                dup = True
                break
            a = set(_norm_phrase(c["display"]).split())
            b = set(_norm_phrase(k["display"]).split())
            if a <= b or b <= a:
                dup = True
                break
        if not dup:
            kept.append((i, c))
        if len(kept) >= max_concepts:
            break

    # per-concept description suggestion: the closest lecture sentence
    concepts = []
    for i, c in kept:
        sims = cosine_matrix(cand_emb[i:i + 1], sent_emb)[0]
        best_j = int(np.argmax(sims))
        concepts.append({
            "name": _display_name(c["display"]),
            "description": sentences[best_j] if float(sims[best_j]) >= 0.3 else "",
            "source_sentence": sentences[best_j],
            "count": c["count"],
            "score": round(float(sims[best_j]), 3),
        })

    # relationship suggestions: a sentence mentioning two concepts in order
    relationships = []
    seen_pairs = set()
    for s in sentences:
        sentence_words = re.findall(r"[A-Za-z][A-Za-z\-']*", s)
        mentioned = []
        for concept in concepts:
            idx = _mention_index(concept["name"], sentence_words)
            if idx is not None:
                mentioned.append((idx, concept["name"]))
        mentioned.sort()
        for (ia, a), (ib, b) in zip(mentioned, mentioned[1:]):
            if (a, b) in seen_pairs or a == b:
                continue
            between = [w for w in sentence_words[ia + 1:ib]]
            link = next((w.lower() for w in between if w.lower() not in _STOPWORDS), None)
            label = link if link and len(between) <= 6 else "relates to"
            relationships.append({"source": a, "label": label, "target": b, "description": s})
            seen_pairs.add((a, b))
            if len(relationships) >= MAX_RELATIONSHIPS:
                break
        if len(relationships) >= MAX_RELATIONSHIPS:
            break

    # learning objectives: the teacher's own objectives always win
    if not objectives:
        objectives = [f"Explain the idea of {c['name']} in your own words." for c in concepts[:3]]

    # bounded misconception suggestions: only misconceptions a teacher has
    # already authored somewhere, and only if the lecture is actually about them
    miscon_suggestions = []
    known = [m for m in (known_misconceptions or []) if m.get("description")]
    if known:
        m_emb = embed([m["description"] for m in known])
        sims_m = cosine_matrix(m_emb, sent_emb)
        for mi, m in enumerate(known):
            best = float(np.max(sims_m[mi]))
            if best >= MISCON_SUGGEST_T:
                miscon_suggestions.append({
                    "name": m["name"], "description": m["description"],
                    "clarification": m.get("clarification", ""),
                    "probe_question": m.get("probe_question", ""),
                    "similarity": round(best, 3),
                })
        miscon_suggestions.sort(key=lambda m: -m["similarity"])
        miscon_suggestions = miscon_suggestions[:MAX_MISCON_SUGGESTIONS]

    return {
        "concepts": concepts,
        "relationships": relationships,
        "objectives": objectives,
        "misconception_suggestions": miscon_suggestions,
    }
