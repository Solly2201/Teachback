"""Deterministic MCQ generation for the optional "Quick knowledge check".

Every question is built ONLY from the teacher-reviewed knowledge structure —
concept meanings, reviewed facts, lecture examples, relationships and
teacher-authored misconceptions. No external knowledge, no randomness, no LLM.
The teacher reviews/edits the generated questions on the lecture page before
publishing, exactly like concepts and activities.

Question kinds (difficulty mix, spec: 4 basic / 3 application /
2 misconception / 1 relationship, filled from what the material supports):

* basic          — recognise a concept's meaning ("Which of these best
                   describes X?" / "Which idea does this describe?")
* application    — apply a taught fact or example ("What does s[0] give?",
                   "Which statement about X is correct?")
* misconception  — spot the false statement among taught facts
* relationship   — complete a taught connection ("split() converts a string
                   into ...?")

All distractors come from the same lecture (other concepts' meanings/names,
other examples' results, the teacher's wrong claims), so they are plausible
without requiring outside knowledge. Selection is rotation-based (indexing,
not random) so generation is reproducible.
"""
import hashlib
import re

QUIZ_SIZE = 10
KIND_QUOTA = [("basic", 4), ("application", 3), ("misconception", 2), ("relationship", 1)]


def _normalise_option(text: str) -> str:
    """Comparison form of an option: case, punctuation and ellipsis removed."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower().rstrip("…")).strip()


def validate_question(q: dict) -> bool:
    """Structural validation every generated (or teacher-edited) MCQ must pass.

    The checks exist to stop the two ways a deterministic generator produces
    an unfair question: an ambiguous option set (duplicates, near-duplicates)
    and an answerable-without-knowing question (the answer is echoed in the
    stem, or is the only long/plausible option).
    """
    options = q.get("options") or []
    if len(options) != 4:
        return False
    cleaned = [str(o).strip() for o in options]
    if any(not o for o in cleaned):
        return False
    if len({o.lower() for o in cleaned}) != 4:  # duplicate options are ambiguous
        return False
    normalised = [_normalise_option(o) for o in cleaned]
    if len(set(normalised)) != 4:  # near-duplicates differing only in punctuation
        return False
    ci = q.get("correct_index")
    if not isinstance(ci, int) or not 0 <= ci < 4:
        return False
    question = str(q.get("question", "")).strip()
    if len(question.split()) < 4 or not str(q.get("explanation", "")).strip():
        return False
    if not str(q.get("concept_name", "")).strip():
        return False
    # the correct option must not literally appear inside the question text
    # (an accidental answer clue)
    if cleaned[ci].lower() in question.lower():
        return False
    if _normalise_option(cleaned[ci]) and _normalise_option(cleaned[ci]) in _normalise_option(question):
        return False
    # length as an elimination cue: if the correct option is far longer than
    # every distractor, the question can be answered by shape alone
    distractor_lengths = [len(o) for i, o in enumerate(cleaned) if i != ci]
    if len(cleaned[ci]) > 2.5 * max(distractor_lengths):
        return False
    return True


def _short(text: str, limit: int = 160) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rotate_pick(pool: list, start: int, n: int, exclude: set | None = None) -> list:
    """Pick n distinct items from pool, starting at index `start` (wrap-around)."""
    exclude = {e.strip().lower() for e in (exclude or set())}
    out = []
    for k in range(len(pool)):
        item = pool[(start + k) % len(pool)]
        key = str(item).strip().lower()
        if key in exclude or any(str(o).strip().lower() == key for o in out):
            continue
        out.append(item)
        if len(out) == n:
            break
    return out


_RESULT_SPLIT_RE = re.compile(r"\s*(?:#|→|->|\bgives\b|\breturns\b)\s*", re.I)


def _example_results(concepts: list[dict]) -> list[tuple[str, str, str]]:
    """(concept_name, expression, result) triples parsed from lecture examples."""
    triples = []
    for c in concepts:
        for ex in c.get("examples") or []:
            parts = _RESULT_SPLIT_RE.split(str(ex), maxsplit=1)
            if len(parts) == 2:
                expr, result = parts[0].strip(), parts[1].strip()
                if expr and result and len(result) <= 40 and "=" not in expr.split("(")[0].split("[")[0]:
                    triples.append((c["name"], expr, result))
    return triples


def generate_quiz_candidates(topic_def: dict) -> list[dict]:
    """All valid question candidates, in deterministic order, kinds mixed."""
    concepts = [c for c in topic_def.get("concepts", []) if c.get("name")]
    described = [c for c in concepts if (c.get("description") or "").strip()]
    misconceptions = [m for m in topic_def.get("misconceptions", [])
                      if (m.get("description") or "").strip()]
    relationships = topic_def.get("relationships", [])

    facts = [(c["name"], f) for c in concepts for f in (c.get("facts") or []) if f.strip()]
    true_statements = [f for _, f in facts] + [c["description"] for c in described]
    false_statements = [m["description"] for m in misconceptions] + \
        [r["contradiction"] for r in relationships if (r.get("contradiction") or "").strip()]

    candidates: list[dict] = []

    # --- basic: recognise the meaning -------------------------------------
    if len(described) >= 4:
        descriptions = [c["description"] for c in described]
        for i, c in enumerate(described):
            if i % 2 == 0:
                distractors = _rotate_pick(descriptions, i + 1, 3, exclude={c["description"]})
                options = [_short(c["description"])] + [_short(d) for d in distractors]
                candidates.append({
                    "kind": "basic", "concept_name": c["name"],
                    "question": f'Which of these best describes "{c["name"]}", as taught in the lecture?',
                    "options": options, "correct_index": 0,
                    "explanation": f'{c["name"]}: {c["description"]}',
                })
            else:
                names = [x["name"] for x in described]
                distractors = _rotate_pick(names, i + 1, 3, exclude={c["name"]})
                candidates.append({
                    "kind": "basic", "concept_name": c["name"],
                    "question": f'Which idea from the lecture does this describe: "{_short(c["description"])}"?',
                    "options": [c["name"]] + distractors, "correct_index": 0,
                    "explanation": f'That is the meaning of {c["name"]}.',
                })

    # --- application: example results -------------------------------------
    triples = _example_results(concepts)
    results = [r for _, _, r in triples]
    for i, (cname, expr, result) in enumerate(triples):
        distractors = _rotate_pick(results, i + 1, 3, exclude={result})
        if len(distractors) < 3:
            continue
        concept = next(c for c in concepts if c["name"] == cname)
        candidates.append({
            "kind": "application", "concept_name": cname,
            "question": f"From the lecture example: what does {expr} give?",
            "options": [result] + distractors, "correct_index": 0,
            "explanation": _short(concept.get("description", "") or f"See the {cname} example in the lecture."),
        })

    # --- application: which statement is correct --------------------------
    for i, (cname, fact) in enumerate(facts):
        distractors = _rotate_pick(false_statements, i, 3, exclude={fact})
        if len(distractors) < 3:
            continue
        candidates.append({
            "kind": "application", "concept_name": cname,
            "question": f'Which of these statements about {cname} is correct, according to the lecture?',
            "options": [_short(fact)] + [_short(d) for d in distractors], "correct_index": 0,
            "explanation": _short(fact),
        })

    # --- misconception: spot the false statement --------------------------
    false_stems = [
        "One of these statements is NOT what the lecture taught. Which one?",
        "Which of these statements is FALSE, according to the lecture?",
        "Three of these come straight from the lecture — which one does not?",
        "Which statement contradicts what the lecture taught?",
    ]
    for i, m in enumerate(misconceptions):
        distractors = _rotate_pick(true_statements, i * 3, 3, exclude={m["description"]})
        if len(distractors) < 3:
            continue
        concept_name = _closest_concept_name(m, concepts) or (concepts[0]["name"] if concepts else "")
        candidates.append({
            "kind": "misconception", "concept_name": concept_name,
            "question": false_stems[i % len(false_stems)],
            "options": [_short(m["description"])] + [_short(d) for d in distractors],
            "correct_index": 0,
            "explanation": _short(m.get("clarification") or
                                  f'The lecture taught the opposite of: {m["description"]}'),
        })

    # --- relationship: complete the connection ----------------------------
    other_names = [c["name"] for c in concepts]
    for i, r in enumerate(relationships):
        target = (r.get("target") or "").strip()
        if not target:
            continue
        pool = [t for t in ([x.get("target", "") for x in relationships] + other_names) if t.strip()]
        distractors = _rotate_pick(pool, i + 1, 3, exclude={target, r.get("source", "")})
        if len(distractors) < 3:
            continue
        candidates.append({
            "kind": "relationship", "concept_name": r.get("source", ""),
            "question": f'Complete the connection from the lecture: {r["source"]} {r.get("label", "relates to")} …?',
            "options": [target] + distractors, "correct_index": 0,
            "explanation": _short(r.get("description", "") or f'{r["source"]} {r.get("label", "")} {target}.'),
        })

    # de-duplicate by question text, keep only structurally valid ones
    seen = set()
    unique = []
    for q in candidates:
        key = q["question"].strip().lower()
        if key in seen or not validate_question(q):
            continue
        seen.add(key)
        unique.append(q)
    return unique


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _closest_concept_name(miscon: dict, concepts: list[dict]) -> str | None:
    """Cheap lexical association of a misconception with a concept.

    Scored by the FRACTION of the concept-name words present in the
    misconception text, so "Indexing" (fully contained) beats
    "String assignment" (half contained) for "indexing starts at 1".
    """
    words = {_stem(w) for w in re.findall(
        r"[a-z]+", (miscon.get("name", "") + " " + miscon.get("description", "")).lower())}
    best, best_score = None, 0.0
    for c in concepts:
        cw = {_stem(w) for w in re.findall(r"[a-z]+", c["name"].lower())}
        if not cw:
            continue
        score = len(words & cw) / len(cw)
        if score > best_score:
            best, best_score = c["name"], score
    return best


def _spread_correct_index(questions: list[dict]) -> list[dict]:
    """Rotate each question's options so the correct answer isn't always A.

    The rotation is derived from the question's own text, not from its position
    in the quiz. Rotating by the index produced a perfectly predictable
    A, B, C, D, A, B ... cycle in every generated quiz — a student who noticed
    it could score full marks without reading a single question. Hashing the
    stem keeps generation reproducible (the same material always yields the
    same quiz) while removing the pattern.
    """
    out = []
    for q in questions:
        digest = hashlib.sha1(q["question"].strip().lower().encode("utf-8")).digest()
        shift = digest[0] % 4
        options = q["options"][-shift:] + q["options"][:-shift] if shift else list(q["options"])
        out.append({**q, "options": options,
                    "correct_index": (q["correct_index"] + shift) % 4})
    return out


def generate_quiz_questions(topic_def: dict, target: int = QUIZ_SIZE) -> list[dict]:
    """The suggested quiz: quota per kind, then fill remaining slots."""
    candidates = generate_quiz_candidates(topic_def)
    by_kind: dict[str, list[dict]] = {}
    for q in candidates:
        by_kind.setdefault(q["kind"], []).append(q)

    chosen: list[dict] = []
    used = set()
    for kind, quota in KIND_QUOTA:
        for q in by_kind.get(kind, [])[:quota]:
            chosen.append(q)
            used.add(q["question"])
    for q in candidates:  # top up to the target with the leftovers
        if len(chosen) >= target:
            break
        if q["question"] not in used:
            chosen.append(q)
            used.add(q["question"])
    return _spread_correct_index(chosen[:target])
