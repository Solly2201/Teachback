"""NLP analysis of a student's teach-back explanation.

Approach (bounded, explainable - not open-domain "AI understands everything"):

The teacher defines a topic with required concepts, known misconceptions and a
short reference explanation. The student's response is split into sentences and
embedded with a pretrained sentence-transformer. We then compute:

* concept_coverage      - for each required concept, the best cosine similarity
                          between any student sentence and the concept's
                          reference texts. A concept is represented by SEVERAL
                          texts — its meaning plus each teacher-reviewed
                          "important fact" from the lecture — so a student who
                          says "the first position is zero" matches the fact
                          "Indexes start at 0" even with no shared textbook
                          wording. >= COVERED_T counts as demonstrated,
                          a band below counts as partially demonstrated.
* misconception_score   - each known misconception is stored as the wrong claim
                          plus a correct "clarification". A sentence is flagged
                          only if it is similar enough to the wrong claim AND
                          closer to the wrong claim than to the correction.
* semantic_correctness  - cosine similarity between the whole response and the
                          topic's reference explanation, rescaled to 0-1.
* explanation_depth     - structural richness (sentences + distinct content words).
* response_effort       - length-based engagement measure.

All features are heuristic 0-1 scores meant to feed the HMM as observations;
they are NOT claimed to be objective measurements of human understanding.
"""
import re

import numpy as np

from .embedder import cosine_matrix, embed

# Thresholds tuned on the labelled evaluation set (data/nlp_eval); see
# evaluation/evaluate.py for the resulting precision/recall.
CONCEPT_COVERED_T = 0.62   # similarity at which a concept counts as demonstrated
CONCEPT_PARTIAL_T = 0.56   # partial credit band
FACT_MATCH_T = 0.60        # a specific lecture fact counts as mentioned
FACT_LEX_T = 0.30          # ...or this similarity plus a shared content word
MISCONCEPTION_T = 0.65     # minimum similarity to the wrong claim to flag it
MISCONCEPTION_MARGIN = 0.08  # must beat similarity to the correction by this

# Concept relationships. Embeddings tolerate rephrasing but are nearly blind to
# polarity flips ("reduces the loss" vs "increases the loss" differ by ~0.002
# cosine), so contradictions are detected with cue words derived automatically
# from the teacher-authored pair: content words that appear in the wrong
# version but not in the correct one (e.g. {"increases"}). A sentence is
# contradicted only if it is semantically ABOUT the relationship AND contains
# such a cue — never from a cue word alone.
RELATIONSHIP_T = 0.68      # similarity at which a relationship counts as demonstrated
RELATIONSHIP_ABOUT_T = 0.60  # sentence is on-topic enough to check for contradiction

_STOPWORDS = set(
    """a an the and or but if then else of in on at to for from by with about as is are was
    were be been being do does did have has had it its this that these those i you he she we
    they them my your our their so not no very can could would should will just also there
    what which who when where how because into over under again more most some such only own
    same than too s t don now""".split()
)

_ABBREVIATIONS = ("e.g", "i.e", "etc", "vs", "dr", "mr", "mrs")


def split_sentences(text: str) -> list[str]:
    """Small rule-based sentence splitter (good enough for short explanations)."""
    text = text.strip()
    if not text:
        return []
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(abbr)}\.", abbr.replace(".", "<dot>") + "<dot>", protected, flags=re.I)
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    sentences = [p.replace("<dot>", ".").strip() for p in parts if p.strip()]
    return [s for s in sentences if len(s.split()) >= 2] or ([text] if text else [])


_DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def content_words(text: str) -> list[str]:
    """Content words, with lone digits normalised to their word form so that
    "starts at 0" and "the first position is zero" share the word "zero"."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    digits = [_DIGIT_WORDS[d] for d in re.findall(r"(?<![\w.])(\d)(?![\w.])", text)]
    return [w for w in words if w not in _STOPWORDS and len(w) > 2] + digits


def _word_match(a: str, b: str) -> bool:
    """Loose inflection-tolerant match: 'increase'/'increases'/'increasing'."""
    return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def contradiction_cues(description: str, contradiction: str) -> set[str]:
    """Content words that appear only in the wrong version of a relationship."""
    if not contradiction:
        return set()
    desc_words = set(content_words(description))
    return {w for w in content_words(contradiction)
            if not any(_word_match(w, d) for d in desc_words)}


def analyze_response(text: str, topic_def: dict) -> dict:
    """Analyze one student response against a structured topic definition.

    topic_def = {
      "name": str, "reference_explanation": str,
      "concepts": [{"id", "name", "description"}, ...],
      "misconceptions": [{"id", "name", "description", "clarification"}, ...],
    }
    """
    sentences = split_sentences(text)
    words = text.split()
    n_words = len(words)

    concepts = topic_def.get("concepts", [])
    misconceptions = topic_def.get("misconceptions", [])
    relationships = topic_def.get("relationships", [])

    # Build one embedding batch for everything to keep this fast. Each concept
    # contributes several reference texts: its meaning, plus each reviewed
    # "important fact" from the lecture (so simple, fact-level explanations
    # still match), all prefixed with the concept name for context.
    concept_refs: list[list[str]] = []
    for c in concepts:
        refs = [f"{c['name']}: {c['description']}"]
        refs += [f"{c['name']}: {f}" for f in (c.get("facts") or [])[:4]]
        concept_refs.append(refs)
    flat_concept_texts = [t for refs in concept_refs for t in refs]
    miscon_texts = [m["description"] for m in misconceptions]
    clar_texts = [m.get("clarification", "") or m["description"] for m in misconceptions]
    rel_texts = [r["description"] for r in relationships]
    ref_text = topic_def.get("reference_explanation", "") or topic_def.get("name", "")

    to_embed = sentences + [text] + flat_concept_texts + miscon_texts + clar_texts + rel_texts + [ref_text]
    emb = embed(to_embed)

    n_s = len(sentences)
    sent_emb = emb[:n_s]
    full_emb = emb[n_s : n_s + 1]
    i = n_s + 1
    concept_emb = emb[i : i + len(flat_concept_texts)]; i += len(flat_concept_texts)
    miscon_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    clar_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    rel_emb = emb[i : i + len(relationships)]; i += len(relationships)
    ref_emb = emb[i : i + 1]

    # --- concept coverage ---
    concept_results = []
    coverage_points = 0.0
    if concepts and n_s:
        sims = cosine_matrix(concept_emb, sent_emb)  # (all concept refs) x sentences
        row = 0
        for ci, c in enumerate(concepts):
            n_refs = len(concept_refs[ci])
            block = sims[row : row + n_refs]  # this concept's refs x sentences
            row += n_refs
            flat_best = int(np.argmax(block))
            best_ref, best_j = divmod(flat_best, n_s)
            best = float(block[best_ref, best_j])
            if best >= CONCEPT_COVERED_T:
                status, pts = "covered", 1.0
            elif best >= CONCEPT_PARTIAL_T:
                status, pts = "partial", 0.5
            else:
                status, pts = "missing", 0.0
            coverage_points += pts
            # fact-level evidence: which reviewed lecture facts did the
            # student actually express? (rows 1.. are the facts) — either a
            # clear semantic match, or a loose one anchored by a shared
            # content word (embeddings alone under-rate terse fact echoes
            # like "the first position is zero" vs "indexes start at 0")
            facts = (c.get("facts") or [])[:4]
            answer_words = set(content_words(text))
            facts_matched = []
            for k, fact in enumerate(facts):
                fact_sim = float(np.max(block[k + 1]))
                shared = answer_words & set(content_words(fact))
                if fact_sim >= FACT_MATCH_T or (fact_sim >= FACT_LEX_T and shared):
                    facts_matched.append(fact)
            concept_results.append(
                {
                    "id": c.get("id"),
                    "name": c["name"],
                    "status": status,
                    "similarity": round(best, 3),
                    "best_sentence": sentences[best_j] if status != "missing" else None,
                    "facts_matched": facts_matched,
                    "facts_missing": [f for f in facts if f not in facts_matched],
                }
            )
    else:
        concept_results = [
            {"id": c.get("id"), "name": c["name"], "status": "missing", "similarity": 0.0,
             "best_sentence": None, "facts_matched": [],
             "facts_missing": (c.get("facts") or [])[:4]}
            for c in concepts
        ]
    concept_coverage = coverage_points / len(concepts) if concepts else 0.0

    # --- misconception detection ---
    # A sentence is only flagged when it is genuinely closer to the wrong
    # claim than to the correct account of the material. Two paths:
    #
    # 1) semantic margin — closer to the wrong claim than to BOTH the
    #    clarification and the concepts' own reference texts. The concept
    #    comparison stops the system inventing a misconception out of a
    #    correct answer ("slicing takes part of the string using start and end
    #    positions" can sit near a wrong claim without being wrong).
    # 2) cue words — embeddings are nearly blind to polarity/number flips
    #    ("index 1" vs "index 0"), so a sentence that IS about the wrong
    #    claim, uses a word unique to the wrong claim (e.g. "one") and none
    #    unique to the clarification (e.g. "zero"), is flagged even when the
    #    correct concept text is embedding-close.
    sent_concept_best = np.zeros(n_s)
    if concepts and n_s:
        sent_concept_best = np.max(cosine_matrix(concept_emb, sent_emb), axis=0)
    miscon_results = []
    detected = []
    if misconceptions and n_s:
        sims_m = cosine_matrix(miscon_emb, sent_emb)
        sims_c = cosine_matrix(clar_emb, sent_emb)
        sent_words = [content_words(s) for s in sentences]
        for mi, m in enumerate(misconceptions):
            best_j = int(np.argmax(sims_m[mi]))
            sim_wrong = float(sims_m[mi, best_j])
            sim_clar = float(sims_c[mi, best_j])
            sim_right = max(sim_clar, float(sent_concept_best[best_j]))
            clarification = m.get("clarification", "")
            wrong_cues = contradiction_cues(clarification, m["description"])
            clar_cues = contradiction_cues(m["description"], clarification)
            words = sent_words[best_j]
            cue_hit = bool(wrong_cues) and \
                any(_word_match(c, w) for c in wrong_cues for w in words) and \
                not any(_word_match(c, w) for c in clar_cues for w in words)
            # the cue path only applies when the sentence is at least as much
            # about the wrong claim as about any correct concept text (small
            # tolerance for the polarity-blindness of embeddings) — otherwise
            # ordinary topic vocabulary in the cue set would cause false
            # accusations on perfectly correct sentences
            near_concept = sim_wrong >= float(sent_concept_best[best_j]) - 0.05
            hit = sim_wrong >= MISCONCEPTION_T and (
                sim_wrong > sim_right + MISCONCEPTION_MARGIN
                or (cue_hit and near_concept and sim_wrong > sim_clar))
            miscon_results.append(
                {
                    "id": m.get("id"),
                    "name": m["name"],
                    "detected": hit,
                    "similarity": round(sim_wrong, 3),
                    "matched_sentence": sentences[best_j] if hit else None,
                }
            )
            if hit:
                detected.append(m["name"])
    misconception_score = 0.0
    if misconceptions:
        hit_sims = [r["similarity"] for r in miscon_results if r["detected"]]
        if hit_sims:
            # scale: one strong hit ~0.6-0.8, several hits saturate towards 1
            misconception_score = float(np.clip(max(hit_sims) * 0.6 + 0.2 * (len(hit_sims) - 1) + 0.2, 0, 1))

    # --- concept relationships (demonstrated / contradicted / not shown) ---
    relationship_results = []
    if relationships and n_s:
        sims_r = cosine_matrix(rel_emb, sent_emb)  # relationships x sentences
        sent_words = [content_words(s) for s in sentences]
        for ri, r in enumerate(relationships):
            best_j = int(np.argmax(sims_r[ri]))
            best = float(sims_r[ri, best_j])
            cues = contradiction_cues(r["description"], r.get("contradiction", ""))
            # a sentence contradicts only if it is about this relationship AND
            # uses a cue word unique to the teacher-authored wrong version
            contradicted_j = next(
                (j for j in range(n_s)
                 if float(sims_r[ri, j]) >= RELATIONSHIP_ABOUT_T
                 and any(_word_match(c, w) for c in cues for w in sent_words[j])),
                None,
            ) if cues else None
            if contradicted_j is not None:
                status, match_j = "contradicted", contradicted_j
            elif best >= RELATIONSHIP_T:
                status, match_j = "demonstrated", best_j
            else:
                status, match_j = "not_shown", None
            relationship_results.append(
                {
                    "id": r.get("id"),
                    "source": r["source"],
                    "label": r.get("label", "relates to"),
                    "target": r["target"],
                    "status": status,
                    "similarity": round(best, 3),
                    "matched_sentence": sentences[match_j] if match_j is not None else None,
                }
            )
    else:
        relationship_results = [
            {"id": r.get("id"), "source": r["source"], "label": r.get("label", "relates to"),
             "target": r["target"], "status": "not_shown", "similarity": 0.0, "matched_sentence": None}
            for r in relationships
        ]

    # --- semantic correctness ---
    raw = float(cosine_matrix(full_emb, ref_emb)[0, 0]) if n_words else 0.0
    # cosine values for on-topic explanations live roughly in [0.2, 0.8]; rescale
    semantic_correctness = float(np.clip((raw - 0.15) / 0.6, 0, 1))

    # --- depth & effort ---
    cw = content_words(text)
    explanation_depth = float(
        np.clip(0.5 * min(1.0, n_s / 4.0) + 0.5 * min(1.0, len(set(cw)) / 40.0), 0, 1)
    )
    response_effort = float(np.clip(n_words / 80.0, 0, 1))

    return {
        "sentences": sentences,
        "word_count": n_words,
        "concepts": concept_results,
        "misconceptions": miscon_results,
        "detected_misconceptions": detected,
        "relationships": relationship_results,
        "features": {
            "concept_coverage": round(concept_coverage, 3),
            "semantic_correctness": round(semantic_correctness, 3),
            "misconception_score": round(misconception_score, 3),
            "explanation_depth": round(explanation_depth, 3),
            "response_effort": round(response_effort, 3),
        },
    }


def targeted_concept_check(text: str, concept: dict, topic_name: str = "") -> dict:
    """Evaluate a short answer against ONE concept, using the question context.

    Short conversational answers ("It uses gradients.") often rely on the
    question for context, so their plain similarity to the concept description
    is low. Prefixing the concept name to the answer restores that context.
    Because the shared prefix inflates similarity for any text, the contextual
    score is only trusted when the answer also shares at least one content
    word with the concept text (checked by the caller via "overlap").
    """
    name = concept["name"]
    # the concept is represented by its meaning AND each important lecture
    # fact — a short answer that expresses any one of them is on-point
    refs = [f"{name}: {concept.get('description', '')}"]
    refs += [f"{name}: {f}" for f in (concept.get("facts") or [])[:4]]
    emb = embed([text, f"{name}: {text}"] + refs)
    ref_emb = emb[2:]
    plain = float(np.max(cosine_matrix(emb[0:1], ref_emb)))
    contextual = float(np.max(cosine_matrix(emb[1:2], ref_emb)))
    # topic-title words ("Python", "Strings" for a lecture called "Strings in
    # Python") appear all over the reference texts without being evidence of
    # anything — "It's something in Python" must not pass the overlap gate
    ref_words = set().union(*(content_words(r) for r in refs))
    ref_words -= set(content_words(topic_name))
    overlap = len(set(content_words(text)) & ref_words)
    return {"plain": round(plain, 3), "contextual": round(contextual, 3), "overlap": overlap}


def merge_session_analyses(analyses: list[dict]) -> dict:
    """Combine per-response analyses into session-level NLP features.

    Concept coverage accumulates across the conversation (a concept explained in
    any exchange counts). Misconceptions count if still present in the latest
    mention. Other features are averaged, weighted towards later responses.
    """
    if not analyses:
        return {
            "concept_coverage": 0.0,
            "semantic_correctness": 0.0,
            "misconception_score": 0.0,
            "explanation_depth": 0.0,
            "response_effort": 0.0,
        }

    # cumulative best status per concept id
    best: dict = {}
    for a in analyses:
        for c in a["concepts"]:
            pts = {"covered": 1.0, "partial": 0.5, "missing": 0.0}[c["status"]]
            key = c["id"] if c["id"] is not None else c["name"]
            best[key] = max(best.get(key, 0.0), pts)
    coverage = sum(best.values()) / len(best) if best else 0.0

    weights = np.linspace(1.0, 1.5, len(analyses))
    weights /= weights.sum()

    def wavg(key):
        return float(sum(w * a["features"][key] for w, a in zip(weights, analyses)))

    return {
        "concept_coverage": round(coverage, 3),
        "semantic_correctness": round(wavg("semantic_correctness"), 3),
        "misconception_score": round(max(a["features"]["misconception_score"] for a in analyses), 3),
        "explanation_depth": round(wavg("explanation_depth"), 3),
        "response_effort": round(wavg("response_effort"), 3),
    }
