"""NLP analysis of a student's teach-back explanation.

Approach (bounded, explainable - not open-domain "AI understands everything"):

The teacher defines a topic with required concepts, known misconceptions and a
short reference explanation. The student's response is split into sentences and
embedded with a pretrained sentence-transformer. We then compute:

* concept_coverage      - for each required concept, the best cosine similarity
                          between any student sentence and the concept's
                          description. >= COVERED_T counts as demonstrated,
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
MISCONCEPTION_T = 0.65     # minimum similarity to the wrong claim to flag it
MISCONCEPTION_MARGIN = 0.08  # must beat similarity to the correction by this

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


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


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

    # Build one embedding batch for everything to keep this fast.
    concept_texts = [f"{c['name']}: {c['description']}" for c in concepts]
    miscon_texts = [m["description"] for m in misconceptions]
    clar_texts = [m.get("clarification", "") or m["description"] for m in misconceptions]
    ref_text = topic_def.get("reference_explanation", "") or topic_def.get("name", "")

    to_embed = sentences + [text] + concept_texts + miscon_texts + clar_texts + [ref_text]
    emb = embed(to_embed)

    n_s = len(sentences)
    sent_emb = emb[:n_s]
    full_emb = emb[n_s : n_s + 1]
    i = n_s + 1
    concept_emb = emb[i : i + len(concepts)]; i += len(concepts)
    miscon_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    clar_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    ref_emb = emb[i : i + 1]

    # --- concept coverage ---
    concept_results = []
    coverage_points = 0.0
    if concepts and n_s:
        sims = cosine_matrix(concept_emb, sent_emb)  # concepts x sentences
        for ci, c in enumerate(concepts):
            best_j = int(np.argmax(sims[ci]))
            best = float(sims[ci, best_j])
            if best >= CONCEPT_COVERED_T:
                status, pts = "covered", 1.0
            elif best >= CONCEPT_PARTIAL_T:
                status, pts = "partial", 0.5
            else:
                status, pts = "missing", 0.0
            coverage_points += pts
            concept_results.append(
                {
                    "id": c.get("id"),
                    "name": c["name"],
                    "status": status,
                    "similarity": round(best, 3),
                    "best_sentence": sentences[best_j] if status != "missing" else None,
                }
            )
    else:
        concept_results = [
            {"id": c.get("id"), "name": c["name"], "status": "missing", "similarity": 0.0, "best_sentence": None}
            for c in concepts
        ]
    concept_coverage = coverage_points / len(concepts) if concepts else 0.0

    # --- misconception detection ---
    miscon_results = []
    detected = []
    if misconceptions and n_s:
        sims_m = cosine_matrix(miscon_emb, sent_emb)
        sims_c = cosine_matrix(clar_emb, sent_emb)
        for mi, m in enumerate(misconceptions):
            best_j = int(np.argmax(sims_m[mi]))
            sim_wrong = float(sims_m[mi, best_j])
            sim_right = float(sims_c[mi, best_j])
            hit = sim_wrong >= MISCONCEPTION_T and sim_wrong > sim_right + MISCONCEPTION_MARGIN
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
        "features": {
            "concept_coverage": round(concept_coverage, 3),
            "semantic_correctness": round(semantic_correctness, 3),
            "misconception_score": round(misconception_score, 3),
            "explanation_depth": round(explanation_depth, 3),
            "response_effort": round(response_effort, 3),
        },
    }


def targeted_concept_check(text: str, concept: dict) -> dict:
    """Evaluate a short answer against ONE concept, using the question context.

    Short conversational answers ("It uses gradients.") often rely on the
    question for context, so their plain similarity to the concept description
    is low. Prefixing the concept name to the answer restores that context.
    Because the shared prefix inflates similarity for any text, the contextual
    score is only trusted when the answer also shares at least one content
    word with the concept text (checked by the caller via "overlap").
    """
    name = concept["name"]
    ref = f"{name}: {concept.get('description', '')}"
    emb = embed([text, f"{name}: {text}", ref])
    plain = float(cosine_matrix(emb[0:1], emb[2:3])[0, 0])
    contextual = float(cosine_matrix(emb[1:2], emb[2:3])[0, 0])
    overlap = len(set(content_words(text)) & set(content_words(ref)))
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
