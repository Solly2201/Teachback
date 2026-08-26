"""Deterministic NLP preparation of lecture material — no LLM involved.

Pipeline (structure first, semantics second):

  lecture text (pasted / extracted from a file)
      -> structured parse (nlp/lecture_parser.py): headings, bullets, code
         blocks, examples, and special sections (objectives / connections /
         common mistakes / summary)
      -> STRUCTURED path (real headings found): each content section becomes a
         candidate concept — name from the heading, meaning from the section's
         explanatory prose, supporting facts from the remaining short
         sentences/bullets, examples kept as examples (never mined as words)
      -> UNSTRUCTURED fallback (plain prose): content-word n-grams scored by
         frequency x embedding centrality, but a candidate is only kept when
         the lecture actually explains it (a sentence where it appears early
         and gets elaborated) — generic nouns without explanatory support are
         rejected
      -> relationships: explicit "A → label → B" lines from an Important
         Connections section first, then sentences mentioning two concepts
      -> misconception suggestions: lines from a Common Mistakes section
         ("Students may think X ... actually Y" splits into claim +
         clarification), plus misconceptions already authored in the system
      -> deterministic question drafts per concept (main / easier / probe /
         application), grounded in the concept's own facts and examples
      -> suggested activities built from the lecture's own concepts

Every concept keeps its provenance (source_section, source_sentences,
examples) so the review screen can show WHERE a suggestion came from.
Everything produced here is a SUGGESTION for the faculty review screen —
nothing becomes authoritative until the teacher publishes it.
"""
import math
import re

import numpy as np

from .analyzer import _STOPWORDS, content_words, split_sentences
from .embedder import cosine_matrix, embed
from .lecture_parser import is_generic_heading, parse_connection_line, parse_lecture

MAX_CONCEPTS = 8
MAX_RELATIONSHIPS = 6
MAX_MISCON_SUGGESTIONS = 4
MAX_CANDIDATES = 40          # embedding-batch cap for candidate phrases
DEDUPE_T = 0.80              # candidates closer than this are duplicates
MIN_CENTRALITY = 0.15        # phrase must be at least loosely about the lecture
MISCON_SUGGEST_T = 0.45      # configured misconception close enough to suggest
MAX_FACTS = 4
MAX_EXAMPLES = 3

# words that are frequent in lecture notes but are never useful concepts on
# their own (they can still appear inside a longer phrase or a heading)
_LECTURE_NOISE = {"example", "examples", "lecture", "today", "slide", "slides",
                  "chapter", "note", "notes", "topic", "student", "students",
                  "letter", "letters", "value", "values", "thing", "things",
                  "word", "words", "number", "numbers", "way", "ways", "part",
                  "parts", "item", "items", "output", "result", "results",
                  "line", "lines", "code", "position", "positions"}

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
# structured path: concepts from document sections
# ---------------------------------------------------------------------------

def _concepts_from_sections(doc: dict, max_concepts: int) -> list[dict]:
    concepts = []
    seen_names = set()
    for sec in doc["sections"]:
        name = _display_name(sec["heading"].strip())
        key = _norm_phrase(name)
        if not key or key in seen_names or is_generic_heading(name):
            continue
        prose = list(sec["sentences"]) + [b for b in sec["bullets"] if len(b.split()) >= 3]
        if not prose and not sec["examples"]:
            continue  # a bare heading with no content is not a teachable concept
        meaning = prose[0] if prose else ""
        facts = [p for p in prose[1:] if len(p.split()) <= 30][:MAX_FACTS]
        concepts.append({
            "name": name,
            "description": meaning,
            "facts": facts,
            "examples": sec["examples"][:MAX_EXAMPLES],
            "source_section": sec["heading"],
            "source_sentences": prose[: 1 + MAX_FACTS],
            "count": 1,
            "score": 1.0,
        })
        seen_names.add(key)
        if len(concepts) >= max_concepts:
            break
    return concepts


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
        first_pos = words.index(targets[0])
        if first_pos <= max(2, len(words) // 2 - 1):
            return s
    return None


def _concepts_from_prose(doc: dict, title: str, objectives: list[str],
                         max_concepts: int) -> list[dict]:
    sentences = [s for sec in doc["sections"] for s in sec["sentences"]]
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
        sims = cosine_matrix(cand_emb[i:i + 1], sent_emb)[0]
        best_j = int(np.argmax(sims))
        meaning = support or (sentences[best_j] if float(sims[best_j]) >= 0.3 else "")
        # other sentences that mention the concept become supporting facts
        name_words = set(_norm_phrase(c["display"]).split())
        facts = [s for s in sentences
                 if s != meaning and name_words <= set(_norm_phrase(s).split())][:MAX_FACTS]
        concepts.append({
            "name": _display_name(c["display"]),
            "description": meaning,
            "facts": facts,
            "examples": [],
            "source_section": "",
            "source_sentences": [x for x in ([meaning] + facts) if x],
            "count": c["count"],
            "score": round(float(sims[best_j]), 3),
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


def _relationships_from_doc(doc: dict, concepts: list[dict]) -> list[dict]:
    relationships = []
    seen_pairs = set()

    # 1) explicit "A → label → B" lines from an Important Connections section
    for line in doc["connections"]:
        parsed = parse_connection_line(line)
        if parsed and (parsed["source"], parsed["target"]) not in seen_pairs:
            relationships.append({**parsed, "source_sentence": line})
            seen_pairs.add((parsed["source"], parsed["target"]))
        elif not parsed:
            # a plain connection sentence: try to anchor it to two concepts
            words = re.findall(r"[A-Za-z][A-Za-z\-']*", line)
            mentioned = sorted(
                (idx, c["name"]) for c in concepts
                if (idx := _mention_index(c["name"], words)) is not None
            )
            if len(mentioned) >= 2:
                a, b = mentioned[0][1], mentioned[1][1]
                if (a, b) not in seen_pairs:
                    relationships.append({"source": a, "label": "relates to", "target": b,
                                          "description": line, "source_sentence": line})
                    seen_pairs.add((a, b))
        if len(relationships) >= MAX_RELATIONSHIPS:
            return relationships

    # 2) prose sentences that mention two concepts in order
    sentences = [s for sec in doc["sections"] for s in sec["sentences"]]
    for s in sentences:
        sentence_words = re.findall(r"[A-Za-z][A-Za-z\-']*", s)
        mentioned = sorted(
            (idx, c["name"]) for c in concepts
            if (idx := _mention_index(c["name"], sentence_words)) is not None
        )
        for (ia, a), (ib, b) in zip(mentioned, mentioned[1:]):
            if (a, b) in seen_pairs or a == b:
                continue
            between = sentence_words[ia + 1:ib]
            link = next((w.lower() for w in between if w.lower() not in _STOPWORDS), None)
            label = link if link and len(between) <= 6 else "relates to"
            relationships.append({"source": a, "label": label, "target": b,
                                  "description": s, "source_sentence": s})
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
        if not claim:
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
                           "section_count": 0}}
    if not any(sec["sentences"] or sec["bullets"] or sec["examples"] for sec in doc["sections"]):
        return empty

    if doc["has_structure"]:
        concepts = _concepts_from_sections(doc, max_concepts)
        # structure gave nothing usable (e.g. headings without content) ->
        # fall back to prose mining
        if not concepts:
            concepts = _concepts_from_prose(doc, title or (doc["title"] or ""),
                                            objectives or doc["objectives"], max_concepts)
    else:
        concepts = _concepts_from_prose(doc, title or (doc["title"] or ""),
                                        objectives or doc["objectives"], max_concepts)

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

    return {
        "concepts": concepts,
        "relationships": relationships,
        "objectives": objectives,
        "misconception_suggestions": miscon_suggestions,
        "activities": activities,
        "structure": {"has_structure": doc["has_structure"], "title": doc["title"],
                      "section_count": len(doc["sections"])},
    }
