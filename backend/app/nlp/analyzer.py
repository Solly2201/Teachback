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
                          wording. The lecture's own examples are reference
                          texts too, so a student who answers with the example
                          they were shown is matched against it rather than
                          scored as if they had said nothing.
                          >= COVERED_T counts as demonstrated,
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
MAX_EXAMPLE_REFS = 2       # lecture examples used as concept reference texts
FACT_MATCH_T = 0.60        # a specific lecture fact counts as mentioned
FACT_LEX_T = 0.30          # ...or this similarity plus a shared content word
MISCONCEPTION_T = 0.65     # minimum similarity to the wrong claim to flag it
MISCONCEPTION_MARGIN = 0.08  # must beat similarity to the correction by this
# An answer can be clearly closer to a known wrong claim than to the concept's
# own explanation without meeting the (deliberately strict) bar for accusing
# the student of holding that misconception. Full credit is withheld in that
# case — "not enough evidence yet", never "you are wrong".
MISCONCEPTION_SHADOW = 0.10
# Both similarity scores have a floor the answer never has to earn: the
# concept name alone scores ~0.54-0.82 against its own reference texts,
# depending on how self-descriptive that name is. An absolute bar therefore
# means something different for every concept, and the self-naming ones clear
# it by saying nothing at all. Each score is instead compared against its own
# floor — the same measurement with the answer removed — so what is left is
# what the answer contributed. A correction to the measurement, not a change
# of threshold: the similarity bars themselves are untouched. Requiring a
# MARGIN above the floor was swept over 0.00-0.10 on the calibration split
# of data/nlp/labeled_answers.json and bought nothing the floor did not
# already buy, while costing real answers.
NAME_ONLY_LIFT = 0.0  # credit requires beating the floor, not merely reaching it
# How many informative words an answer must carry before a prefix-inflated
# similarity is trusted on its own. One word is a fragment, not an
# explanation — unless that word is one of the teacher's own key terms.
MIN_EVIDENCE_TERMS = 2

# Concept relationships. A relationship has THREE meaningful outcomes, and the
# difference between them is the difference between "no evidence" and "wrong":
#
#   demonstrated  - the answer contains evidence for the teacher's connection
#   contradicted  - the answer expresses the connection incorrectly
#   partial       - the answer is about the connection but stops short of it
#   not_shown     - nothing either way (NOT a mistake, and never treated as one)
#
# Demonstration has two paths, mirroring how concepts are scored:
#   (a) a strong direct match against the teacher's own wording, or
#   (b) a weaker match that is corroborated by the sentence ALSO carrying
#       evidence for BOTH endpoints of the link — semantically (each endpoint's
#       concept texts) and lexically (a word distinctive to each endpoint).
#       Path (b) exists because a single reference sentence under-rates ordinary
#       phrasings: "they're individual letters or symbols inside the string"
#       expresses "a string is made of characters" at cosine 0.674, just under
#       the direct bar, and the endpoint evidence is what makes it safe to
#       accept without lowering the bar for everything.
#
# Embeddings tolerate rephrasing but are nearly blind to polarity flips
# ("reduces the loss" vs "increases the loss" differ by ~0.002 cosine), so
# contradictions come from explicit cues, never from similarity alone:
#   1. cue words derived from the teacher-authored wrong version of the pair
#      (content words in the wrong version but not the correct one), and
#   2. an explicit negation of one of the endpoints ("...not characters"),
#      which only applies when the sentence is about the relationship but is
#      not itself a strong direct match.
RELATIONSHIP_T = 0.68        # direct match: demonstrated on its own
RELATIONSHIP_LINK_T = 0.55   # weaker match, demonstrated only with endpoint evidence
RELATIONSHIP_ABOUT_T = 0.60  # sentence is on-topic enough to be an attempt at the link
RELATIONSHIP_ENDPOINT_T = 0.56  # endpoint evidence inside the same sentence
NEGATION_WINDOW = 3          # tokens after a negation that count as negated
# Words that are too weak to prove a relationship was stated backwards. They
# are ordinary connective adverbs, not the substance of any connection.
_WEAK_CUES = {"back", "again", "then", "still", "just", "also", "here", "there",
              "now", "well", "even", "much", "many", "some", "any", "way", "ways"}

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


# Words that carry no information ABOUT a concept: evaluative verdicts on it,
# and talk about the lecture rather than the idea. Deliberately small and
# closed — it exists to catch "X is important" / "X was covered today", not to
# blacklist subject vocabulary.
_VACUOUS_TERMS = {
    "bad", "good", "important", "useful", "useless", "interesting", "boring",
    "fundamental", "essential", "key", "crucial", "basic",
    "confusing", "hard", "easy", "difficult", "nice", "great", "fine", "okay",
    "mentioned", "covered", "discussed", "taught", "exists", "exist",
    "matters", "matter", "involved", "related", "lecture", "class", "today",
    "yesterday", "slide", "slides", "notes", "chapter", "topic", "subject",
    "called", "part", "thing", "things", "stuff", "something", "anything",
    "used", "uses", "use", "using", "lot", "here", "there", "name", "sure",
    "know", "remember", "forgot", "idea", "guess", "maybe", "think",
    # bare agreement and evaluation
    "yep", "yeah", "yup", "exactly", "okay", "alright", "love", "hate",
    "enjoy", "enjoyed", "like", "liked", "course", "general", "generally",
    "overall", "many", "popular",
    # intensifiers and hedges
    "really", "very", "quite", "pretty", "actually", "essentially", "simply",
    "totally", "definitely", "probably", "obviously", "sort", "kind",
    # filler and bare size/verdict words. "Overfitting is a big problem" and
    # "umm well you know how it is" both cleared the two-informative-words bar
    # on these alone, while saying nothing about any concept.
    "umm", "uhh", "uh", "erm", "hmm", "huh", "basically", "big", "small",
    "huge", "problem", "problems", "concept", "concepts",
}


# Openers that make a sentence a request to be taught rather than an
# explanation, even without a question mark. Deliberately short and closed:
# no real explanation begins with any of them.
_ASKING_OPENERS = (
    "can you", "could you", "can u", "would you", "will you", "can we",
    "please explain", "explain ", "tell me", "help me", "remind me",
    "i don't understand", "i dont understand", "i don't get", "i dont get",
    "i'm not sure what", "im not sure what", "no idea what",
)


def is_question(sentence: str) -> bool:
    """True when the student is ASKING rather than explaining.

    A teach-back is evidence of understanding because the student produced the
    explanation. A question produces none: "what is a gradient?", "sorry what
    does gradient descent mean?" and "I don't understand the gradient, could
    you go over it?" all sit close to the concept's reference text — they are
    about it, in its vocabulary — so they were scored as demonstrating it. The
    last one credited the concept to a student who had just said outright that
    they could not explain it.

    Asking is still engagement, and response_effort keeps counting it; it just
    cannot be evidence of understanding, and equally cannot be held against
    the student as a misconception ("is the gradient the same as the loss?" is
    a question, not a claim).
    """
    s = sentence.strip().lower()
    return bool(s) and (s.endswith("?") or s.startswith(_ASKING_OPENERS))


def answering_sentences(sentences: list[str]) -> list[str]:
    """The sentences that assert something, questions removed."""
    return [s for s in sentences if not is_question(s)]


def is_term_list(text: str) -> bool:
    """True for a bare list of terms rather than a statement.

    Any real English clause of four or more words contains at least one
    function word ("text IN quotes", "the first position IS zero"). A takeaway
    like "backpropagation gradient weight loss optimization" has none: it names
    the lecture's topics without saying anything about them.

    Used only for the free-form takeaway summary — a per-question ANSWER may
    legitimately be a terse noun phrase ("square brackets pick one letter"),
    so applying this there would reject real explanations.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']*", text.lower())
    return len(tokens) >= 4 and not any(t in _STOPWORDS for t in tokens)


def concept_evidence(text: str, concept: dict, topic_name: str = "",
                     sibling_names: list[str] | None = None) -> dict:
    """How much the answer says about ONE concept.

    Two independent signals, neither of which naming the concept can satisfy:
    the informative words (name, lecture title and filler removed), and how
    many of those land on the teacher's own explanatory vocabulary. Shared by
    the whole-response analysis and the targeted per-question check so both
    apply the same rule.

    ``sibling_names`` extends the same principle to the lecture's OTHER
    concept names, so a list of the lecture's own labels ("markov hidden
    states observations transitions") is recognised as naming things rather
    than explaining any of them. A word is only discounted as a sibling name
    when it is not also part of THIS concept's explanation.
    """
    name = concept.get("name", "")
    refs = [concept.get("description", "")]
    refs += list((concept.get("facts") or [])[:4])
    refs += list((concept.get("examples") or [])[:MAX_EXAMPLE_REFS])
    ref_words = set()
    for r in refs:
        ref_words |= set(content_words(r))
    ref_words -= set(content_words(name)) | set(content_words(topic_name))
    labels = set()
    for other in sibling_names or []:
        if (other or "").strip().lower() == name.strip().lower():
            continue
        labels |= {w for w in content_words(other) if w not in ref_words}
    terms = {w for w in informative_terms(text, name, topic_name)
             if not any(_word_match(w, l) for l in labels)}
    # key terms are drawn from what SURVIVES: a label that was filtered out as
    # another concept's name must not come back as corroboration here
    key_terms = sorted(terms & ref_words)
    return {"informative_terms": sorted(terms), "key_terms": key_terms,
            "corroborated": bool(terms)
            and (len(terms) >= MIN_EVIDENCE_TERMS or bool(key_terms))}


def informative_terms(text: str, concept_name: str, topic_name: str = "") -> set[str]:
    """Content words that say something ABOUT the concept.

    Naming the concept is not evidence about it, and neither is naming the
    lecture or passing judgement on it. What is left after removing those is
    the only thing that can support a claim of understanding. Matching is
    inflection-tolerant so "gradients" still counts as the name "Gradient".
    """
    drop = set(content_words(concept_name)) | set(content_words(topic_name))
    return {w for w in content_words(text)
            if w not in _VACUOUS_TERMS and not any(_word_match(w, d) for d in drop)}


def contradiction_cues(description: str, contradiction: str) -> set[str]:
    """Content words that appear only in the wrong version of a relationship."""
    if not contradiction:
        return set()
    desc_words = set(content_words(description))
    return {w for w in content_words(contradiction)
            if not any(_word_match(w, d) for d in desc_words)}


def _has_any(words, vocabulary) -> bool:
    """True if any of `words` inflection-matches something in `vocabulary`."""
    return any(_word_match(w, v) for w in words for v in vocabulary)


def _distinctive(words: set[str], other: set[str]) -> set[str]:
    """`words` minus anything that also (loosely) appears in `other`."""
    return {w for w in words if not any(_word_match(w, o) for o in other)}


def endpoint_refs(name: str, concepts: list[dict]) -> list[str]:
    """Reference texts describing one end of a relationship.

    An endpoint usually names a concept the teacher already defined, in which
    case it inherits that concept's meaning and reviewed facts. Endpoints with
    no matching concept ("Substring", "List") are represented by their name.
    """
    key = (name or "").strip().lower()
    for c in concepts:
        if (c.get("name") or "").strip().lower() == key:
            refs = [f"{c['name']}: {c.get('description', '')}"]
            return refs + [f"{c['name']}: {f}" for f in (c.get("facts") or [])[:4]]
    return [name]


_NEGATION_TOKENS = {
    "not", "no", "never", "none", "nor", "cannot", "without", "neither",
    "isnt", "arent", "wasnt", "werent", "doesnt", "dont", "didnt", "cant",
    "wont", "hasnt", "havent", "aint",
}


def negated_terms(sentence: str) -> list[str]:
    """Content words that fall inside the scope of an explicit negation.

    Deliberately shallow: the words in the NEGATION_WINDOW tokens after a
    negation marker. Embeddings cannot see polarity, so "a string is a
    collection of variables, not characters" needs this to be told apart from
    "a string is a collection of characters".
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']*", sentence.lower())
    tokens = [t.replace("'", "") for t in tokens]
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in _NEGATION_TOKENS:
            out += [t for t in tokens[i + 1 : i + 1 + NEGATION_WINDOW]
                    if t not in _STOPWORDS and len(t) > 2]
    return out


# Some concepts are defined by what they EXCLUDE: "the next state depends only
# on the current state, not on the whole earlier history". Flipping that
# exclusion produces a sentence that shares nearly every content word with the
# teacher's own wording, so the embedding puts it CLOSER to the concept than to
# the taught misconception it actually restates (0.91 vs 0.80 for the Markov
# property). No similarity rule can separate them; the difference is one word.
#
# So it is read lexically, and narrowly: an answer that asserts the totality
# the concept explicitly narrows, while carrying neither the narrowing itself
# nor a negation of its own, has inverted the concept. "Each state emits an
# observation" must NOT trip this — "each" quantifies the states, not the
# scope — which is why the marker set is restricted to totality words.
#
# This only ever withholds full credit. It never names a misconception and
# never tells the student they are wrong.
EXCLUSIVITY_MARKERS = {"only", "just", "solely", "alone", "merely", "purely"}
TOTALITY_MARKERS = {"everything", "entire", "whole", "all", "always",
                    "complete", "completely", "everywhere"}


def inverts_exclusivity(description: str, text: str) -> bool:
    """True when the answer asserts the totality the concept rules out."""
    desc = set(content_words(description)) | set(
        re.findall(r"[a-z]+", (description or "").lower()))
    if not desc & EXCLUSIVITY_MARKERS:
        return False
    words = {t.replace("'", "") for t in
             re.findall(r"[a-zA-Z][a-zA-Z\-']*", (text or "").lower())}
    return bool(words & TOTALITY_MARKERS) and not (
        words & EXCLUSIVITY_MARKERS) and not (words & _NEGATION_TOKENS)


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

    # Questions are requests, not demonstrations (see is_question). They are
    # excluded from every similarity that decides evidence — for the student's
    # benefit in both directions: no credit for asking, and no accusation for
    # asking either. When the answer contains no question at all, `asking` is
    # empty and everything below is byte-for-byte what it always was.
    asking = [is_question(s) for s in sentences]
    answering = [s for s, q in zip(sentences, asking) if not q]
    # what the student actually asserted; the untouched text when they asked
    # nothing, so ordinary answers are unaffected by the join
    said = " ".join(answering) if any(asking) else text

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
        # the lecture's own examples are reference texts too: students often
        # explain by reproducing the example they were shown. Kept AFTER the
        # facts so the fact-level indexing below is unaffected.
        refs += [f"{c['name']}: {e}" for e in (c.get("examples") or [])[:MAX_EXAMPLE_REFS]]
        concept_refs.append(refs)
    flat_concept_texts = [t for refs in concept_refs for t in refs]
    miscon_texts = [m["description"] for m in misconceptions]
    clar_texts = [m.get("clarification", "") or m["description"] for m in misconceptions]
    rel_texts = [r["description"] for r in relationships]
    # each relationship also carries reference texts for its two endpoints, so
    # a sentence can be checked for evidence of BOTH ideas it connects
    endpoint_refs_per_rel = [
        (endpoint_refs(r["source"], concepts), endpoint_refs(r["target"], concepts))
        for r in relationships
    ]
    flat_endpoint_texts = [t for pair in endpoint_refs_per_rel for side in pair for t in side]
    ref_text = topic_def.get("reference_explanation", "") or topic_def.get("name", "")

    to_embed = (sentences + [said] + flat_concept_texts + miscon_texts + clar_texts
                + rel_texts + flat_endpoint_texts + [ref_text])
    emb = embed(to_embed)

    n_s = len(sentences)
    sent_emb = emb[:n_s]
    ask_cols = np.array(asking, dtype=bool)

    def sent_sims(x_emb):
        """Similarity of each reference text against the ANSWERING sentences.

        Question columns are pushed below every threshold rather than dropped,
        so an all-question response simply matches nothing instead of needing
        a separate empty-array path through the whole function.
        """
        m = cosine_matrix(x_emb, sent_emb)
        if ask_cols.any():
            m = m.copy()
            m[:, ask_cols] = -1.0
        return m
    full_emb = emb[n_s : n_s + 1]
    i = n_s + 1
    concept_emb = emb[i : i + len(flat_concept_texts)]; i += len(flat_concept_texts)
    miscon_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    clar_emb = emb[i : i + len(misconceptions)]; i += len(misconceptions)
    rel_emb = emb[i : i + len(relationships)]; i += len(relationships)
    endpoint_emb = emb[i : i + len(flat_endpoint_texts)]; i += len(flat_endpoint_texts)
    ref_emb = emb[i : i + 1]

    # how close each sentence sits to any known wrong claim; used only to
    # WITHHOLD credit, never to accuse (see MISCONCEPTION_SHADOW)
    sent_miscon_best = np.zeros(n_s)
    if misconceptions and n_s:
        sent_miscon_best = np.max(sent_sims(miscon_emb), axis=0)

    # --- concept coverage ---
    concept_results = []
    coverage_points = 0.0
    if concepts and n_s:
        sims = sent_sims(concept_emb)  # (all concept refs) x answering sentences
        row = 0
        for ci, c in enumerate(concepts):
            n_refs = len(concept_refs[ci])
            block = sims[row : row + n_refs]  # this concept's refs x sentences
            row += n_refs
            flat_best = int(np.argmax(block))
            best_ref, best_j = divmod(flat_best, n_s)
            best = float(block[best_ref, best_j])
            # Naming a concept is not explaining it. The reference texts carry
            # the concept name, so "python uses indexing" sits very close to
            # them without saying anything about indexing; an answer that adds
            # nothing beyond the name cannot demonstrate the concept.
            says_something = concept_evidence(
                said, c, topic_def.get("name", ""),
                sibling_names=[x["name"] for x in concepts])["corroborated"]
            # a sentence that sits closer to a taught wrong claim than to this
            # concept's own explanation does not demonstrate the concept
            shadowed = float(sent_miscon_best[best_j]) > best + MISCONCEPTION_SHADOW
            if not says_something:
                status, pts = "missing", 0.0
            elif best >= CONCEPT_COVERED_T and not shadowed:
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
            answer_words = set(content_words(said))
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
        sent_concept_best = np.max(sent_sims(concept_emb), axis=0)
    miscon_results = []
    detected = []
    if misconceptions and n_s:
        sims_m = sent_sims(miscon_emb)
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

    # --- concept relationships ---
    # demonstrated / contradicted / partial / not_shown. "not_shown" means the
    # answer said nothing either way about this connection: it is an absence of
    # evidence, never a mistake, and nothing downstream may treat it as one.
    relationship_results = []
    if relationships and n_s:
        sims_r = sent_sims(rel_emb)  # relationships x answering sentences
        sims_e = sent_sims(endpoint_emb)  # endpoint refs x answering sentences
        sent_words = [content_words(s) for s in sentences]
        sent_negated = [negated_terms(s) for s in sentences]
        title_words = set(content_words(topic_def.get("name", "")))
        row = 0
        for ri, r in enumerate(relationships):
            src_refs, tgt_refs = endpoint_refs_per_rel[ri]
            src_block = sims_e[row : row + len(src_refs)]; row += len(src_refs)
            tgt_block = sims_e[row : row + len(tgt_refs)]; row += len(tgt_refs)
            src_sim = np.max(src_block, axis=0)
            tgt_sim = np.max(tgt_block, axis=0)

            # vocabulary that is distinctive to each end of the link: words the
            # OTHER end also uses ("string" for Strings -> Characters) prove
            # nothing about the connection, and neither do the topic's title
            # words, so both are removed
            src_vocab = set(content_words(r["source"]))
            src_vocab.update(*(set(content_words(t)) for t in src_refs))
            tgt_vocab = set(content_words(r["target"]))
            tgt_vocab.update(*(set(content_words(t)) for t in tgt_refs))
            src_only = _distinctive(src_vocab - title_words, tgt_vocab)
            tgt_only = _distinctive(tgt_vocab - title_words, src_vocab)

            best_j = int(np.argmax(sims_r[ri]))
            best = float(sims_r[ri, best_j])

            # (1) contradiction from the teacher's own wrong version of the pair:
            #     the sentence is about the link, uses a word unique to the wrong
            #     version, and uses NO word unique to the correct version (the
            #     same two-sided cue test the misconception detector applies).
            #     Without the second guard a shared incidental word is enough to
            #     accuse a correct answer:
            #     "split breaks the text into pieces and join puts the pieces
            #     back together" shares "back" with the wrong version of split()
            #     while stating the right one with "breaks".
            #     _WEAK_CUES is dropped for relationships only; misconception
            #     cues legitimately turn on small words like "one" vs "zero".
            cues = contradiction_cues(r["description"], r.get("contradiction", "")) - _WEAK_CUES
            right_cues = (contradiction_cues(r.get("contradiction", ""), r["description"])
                           - _WEAK_CUES if r.get("contradiction") else set())
            contradicted_j = next(
                (j for j in range(n_s)
                 if float(sims_r[ri, j]) >= RELATIONSHIP_ABOUT_T
                 and _has_any(sent_words[j], cues)
                 and not _has_any(sent_words[j], right_cues)),
                None,
            ) if cues else None
            # (2) contradiction from an explicit negation of one of the ends.
            #     Only for sentences that are about the link but fall short of a
            #     strong direct match — "a string is not a number, it is a
            #     sequence of characters" states the link and must not be flagged.
            if contradicted_j is None:
                endpoint_terms = (set(content_words(r["source"]))
                                  | set(content_words(r["target"])) | src_only | tgt_only)
                contradicted_j = next(
                    (j for j in range(n_s)
                     if RELATIONSHIP_ABOUT_T <= float(sims_r[ri, j]) < RELATIONSHIP_T
                     and _has_any(sent_negated[j], endpoint_terms)),
                    None,
                )

            # (3) demonstration: a strong direct match, or a weaker one where the
            #     same sentence also carries evidence for both ends of the link
            linked_j = next(
                (j for j in range(n_s)
                 if float(sims_r[ri, j]) >= RELATIONSHIP_LINK_T
                 and float(src_sim[j]) >= RELATIONSHIP_ENDPOINT_T
                 and float(tgt_sim[j]) >= RELATIONSHIP_ENDPOINT_T
                 and _has_any(sent_words[j], src_only)
                 and _has_any(sent_words[j], tgt_only)),
                None,
            )

            if contradicted_j is not None:
                status, match_j = "contradicted", contradicted_j
            elif best >= RELATIONSHIP_T:
                status, match_j = "demonstrated", best_j
            elif linked_j is not None:
                status, match_j = "demonstrated", linked_j
            elif best >= RELATIONSHIP_ABOUT_T:
                # about this connection, but stops short of establishing it —
                # on its own still NOT a mistake; only a direct probe of this
                # relationship reads it as an incomplete attempt
                status, match_j = "partial", best_j
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
    raw = float(cosine_matrix(full_emb, ref_emb)[0, 0]) if said.strip() else 0.0
    # cosine values for on-topic explanations live roughly in [0.2, 0.8]; rescale
    semantic_correctness = float(np.clip((raw - 0.15) / 0.6, 0, 1))

    # --- depth & effort ---
    cw = content_words(said)
    n_said = len(answering) if any(asking) else n_s
    explanation_depth = float(
        np.clip(0.5 * min(1.0, n_said / 4.0) + 0.5 * min(1.0, len(set(cw)) / 40.0), 0, 1)
    )
    # effort is deliberately measured on everything the student wrote: asking a
    # question is engagement, it is only not an explanation
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


def targeted_concept_check(text: str, concept: dict, topic_name: str = "",
                           misconceptions: list[dict] | None = None,
                           sibling_names: list[str] | None = None) -> dict:
    """Evaluate a short answer against ONE concept, using the question context.

    Short conversational answers ("It uses gradients.") often rely on the
    question for context, so their plain similarity to the concept description
    is low. Prefixing the concept name to the answer restores that context.
    Because the shared prefix inflates similarity for any text, neither score
    is trusted unless the answer actually says something about the concept —
    reported as "informative" (see informative_terms). The older "overlap"
    count is still returned for reference.
    """
    # the same rule as the whole-response analysis: a question asked back is
    # not an answer to the question that was asked
    text = " ".join(answering_sentences(split_sentences(text)))
    name = concept["name"]
    if not text.strip():
        return {"plain": 0.0, "contextual": 0.0, "plain_lift": 0.0, "contextual_lift": 0.0,
                "overlap": 0, "informative": False, "shadowed": False,
                "inverts_exclusivity": False, **concept_evidence("", concept, topic_name)}
    # the concept is represented by its meaning AND each important lecture
    # fact — a short answer that expresses any one of them is on-point
    refs = [f"{name}: {concept.get('description', '')}"]
    refs += [f"{name}: {f}" for f in (concept.get("facts") or [])[:4]]
    refs += [f"{name}: {e}" for e in (concept.get("examples") or [])[:MAX_EXAMPLE_REFS]]
    # Prefixed the SAME way as the concept references above. Without this the
    # comparison is rigged: the concept side gets a similarity boost from the
    # shared "Name: " prefix that the wrong-claim side never sees, so an answer
    # that restates a taught misconception almost verbatim still looks closer
    # to the concept than to the misconception.
    wrong = [f"{name}: {m['description']}" for m in (misconceptions or [])
             if m.get("description")]
    # The last two probes are the answer removed: the concept's name on its own
    # (the floor for `plain`) and the bare context prefix (the floor for
    # `contextual`). Both are embedded in the same batch, so this costs one
    # encode call either way. See NAME_ONLY_LIFT.
    emb = embed([text, f"{name}: {text}"] + refs + wrong + [name, f"{name}:"])
    ref_emb = emb[2:2 + len(refs)]
    plain = float(np.max(cosine_matrix(emb[0:1], ref_emb)))
    contextual = float(np.max(cosine_matrix(emb[1:2], ref_emb)))
    plain_floor = float(np.max(cosine_matrix(emb[-2:-1], ref_emb)))
    contextual_floor = float(np.max(cosine_matrix(emb[-1:], ref_emb)))
    # closer to a taught wrong claim than to the concept itself: withhold full
    # credit without accusing the student of anything (MISCONCEPTION_SHADOW)
    shadowed = False
    if wrong:
        wrong_emb = emb[2 + len(refs):2 + len(refs) + len(wrong)]
        worst = float(np.max(cosine_matrix(emb[0:1], wrong_emb)))
        shadowed = worst > plain + MISCONCEPTION_SHADOW
    # topic-title words ("Python", "Strings" for a lecture called "Strings in
    # Python") appear all over the reference texts without being evidence of
    # anything — "It's something in Python" must not pass the overlap gate
    ref_words = set().union(*(content_words(r) for r in refs))
    ref_words -= set(content_words(topic_name))
    overlap = len(set(content_words(text)) & ref_words)
    # what the answer actually says about this concept, name and lecture title
    # removed: an empty set means the student named it and stopped
    evidence = concept_evidence(text, concept, topic_name,
                                sibling_names=sibling_names)
    # the concept's own exclusion, asserted the other way round
    inverted = inverts_exclusivity(concept.get("description", ""), text)
    return {"plain": round(plain, 3), "contextual": round(contextual, 3),
            # how much of each score the answer itself accounts for
            "plain_lift": round(plain - plain_floor, 3),
            "contextual_lift": round(contextual - contextual_floor, 3),
            "overlap": overlap, "informative": bool(evidence["informative_terms"]),
            "shadowed": shadowed or inverted,
            "inverts_exclusivity": inverted, **evidence}


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
