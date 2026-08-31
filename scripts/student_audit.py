"""Large-scale audit: does TeachBack recognise what ordinary students say?

This is an AUDIT, not a calibration set. Nothing here is used to tune a
threshold. The question it answers is:

    If 150+ ordinary students explain what they learned in their own
    imperfect words, does TeachBack reasonably recognise what they
    understood — without crediting people who said nothing?

Design notes that make the numbers meaningful:

* Answers are generated from hand-written per-concept phrasings that a real
  student might produce, then passed through deterministic surface
  transformations (shorten, informalise, add filler, add a typo). The
  transformations never change what the answer MEANS, so the gold label
  travels with the phrasing, not with the wording.
* Deliberately NOT keyword-friendly: most correct answers never say the
  concept's name. "you use the number to get the letter" has to be recognised
  as Indexing on meaning alone.
* Both failure directions are measured. A correct simple answer scored as
  "unclear" is a false negative; "Python uses indexing" scored as demonstrated
  is a false positive. The second is the more damaging of the two.
* Everything runs through the REAL pipeline — analyze_response +
  targeted_concept_check + the conversation engine's own verdict — so this
  audits the system, not a reimplementation of it.

Usage:
    python scripts/student_audit.py [--seed 20260831] [--out data/nlp/student_simulation.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.nlp import conversation  # noqa: E402
from app.nlp.analyzer import analyze_response, targeted_concept_check  # noqa: E402
from app.seed_content import PYTHON_LECTURE, TOPICS  # noqa: E402

# gold labels, in the vocabulary a teacher would use
DEMO, PARTIAL, INSUFFICIENT, MISCON = "demonstrated", "partial", "insufficient", "misconception"
LABELS = [DEMO, PARTIAL, INSUFFICIENT, MISCON]

VERDICT_TO_LABEL = {"correct": DEMO, "partial": PARTIAL, "analogy": PARTIAL,
                    "affirm": INSUFFICIENT, "unclear": INSUFFICIENT}


# ---------------------------------------------------------------------------
# topic definitions, exactly as the seeded/published material defines them
# ---------------------------------------------------------------------------

def build_topic_defs() -> dict:
    by_name = {t["name"]: t for t in TOPICS}
    strings = {
        "name": PYTHON_LECTURE["title"],
        "reference_explanation": " ".join(c["description"]
                                          for c in PYTHON_LECTURE["reviewed_concepts"]),
        "concepts": [dict(c, id=i + 1) for i, c in enumerate(PYTHON_LECTURE["reviewed_concepts"])],
        "misconceptions": [dict(m, id=i + 1)
                           for i, m in enumerate(PYTHON_LECTURE["reviewed_misconceptions"])],
        "relationships": [dict(r, id=i + 1)
                          for i, r in enumerate(PYTHON_LECTURE["reviewed_relationships"])],
    }
    return {
        "strings": strings,
        "backprop": by_name["Backpropagation"],
        "hmm": by_name["Hidden Markov Models"],
        "overfitting": by_name["Overfitting and Regularization"],
    }


# ---------------------------------------------------------------------------
# what a student might actually say
# ---------------------------------------------------------------------------
# Keys: (topic, concept). Each bucket carries its own gold label:
#   correct / terminology_free / short / example / informal -> demonstrated
#   analogy / partial / vague                               -> partial
#   keyword_only / unrelated                                -> insufficient
#   misconception                                           -> misconception
# The concept name is deliberately absent from most correct phrasings.

BANK: dict[tuple[str, str], dict[str, list[str]]] = {
    ("strings", "Strings"): {
        "correct": ["a string is text stored between quotes",
                    "text that you write inside quotation marks is a string"],
        "terminology_free": ["it's basically words you put inside speech marks",
                             "anything you type between two quote symbols"],
        "short": ["text in quotes", "words inside quotes"],
        "example": ['like when you write "Hello" with the quotes around it'],
        "analogy": ["it's like a sentence written on a label"],
        "partial": ["something that holds letters"],
        "keyword_only": ["python is useful for strings", "strings come up a lot in python"],
    },
    ("strings", "String assignment"): {
        "correct": ["you save the text in a variable so you can use it later",
                    "giving the text a name lets you reuse it"],
        "terminology_free": ["you put it in a box with a name and use the name after that"],
        "short": ["you store it under a name"],
        "example": ['like name = "Python" and then you use name'],
        "partial": ["you store it somewhere"],
        "keyword_only": ["assignment was in the lecture"],
    },
    ("strings", "Characters"): {
        "correct": ["a string is made of single letters and symbols in a fixed order",
                    "it's a sequence of individual letters one after another"],
        "terminology_free": ["it's built out of the separate letters that sit in order"],
        "short": ["single letters in order"],
        "analogy": ["it's like beads on a thread, each bead is one letter"],
        "partial": ["it has letters in it"],
        "keyword_only": ["characters were mentioned in class"],
    },
    ("strings", "Indexing"): {
        "correct": ["you use the position number to get one particular letter out",
                    "each letter has a position and you use that position to reach it"],
        "terminology_free": ["you say which number you want and it gives you that letter",
                             "you use the number to get the letter"],
        "short": ["the first position is zero", "position zero is the first one"],
        "example": ["like s[0] gives you the P in Python"],
        "analogy": ["it's like seat numbers, you ask for seat 0 and get the first person"],
        "partial": ["you use square brackets"],
        "misconception": ["counting starts at one so the first letter is at position 1",
                          "the first letter is number one and the second is number two"],
        "keyword_only": ["python uses indexing", "indexing is important in python"],
    },
    ("strings", "Slicing"): {
        "correct": ["you take a part of the text between a start and an end position",
                    "it pulls out a chunk of the string from one position to another"],
        "terminology_free": ["you can cut out a piece of the words from here to there"],
        "short": ["it takes part of the text"],
        "example": ["like s[0:3] on Python gives you Pyt"],
        "analogy": ["it's like cutting a slice out of a loaf of bread"],
        "partial": ["the end number is left out"],
        "keyword_only": ["slicing was covered today"],
    },
    ("strings", "split() and join()"): {
        "correct": ["split breaks the text into a list of pieces and join sticks them back together",
                    "one cuts a sentence into parts, the other glues the parts into one string"],
        "terminology_free": ["you can chop it up into separate bits, or stick the bits back"],
        "short": ["split makes a list of pieces"],
        "example": ['like "a,b,c" split on the comma gives you a, b and c separately'],
        "partial": ["they cut and paste text i think"],
        "keyword_only": ["we did string methods"],
    },
    ("backprop", "Loss / error"): {
        "correct": ["it's a number that says how far the prediction was from the right answer",
                    "it measures how wrong the guess was"],
        "terminology_free": ["a score for how badly it got the answer wrong"],
        "short": ["how wrong the guess was"],
        "analogy": ["it's like the distance between where the dart landed and the bullseye"],
        "partial": ["you compare the guess with the right answer"],
        "keyword_only": ["loss is part of backpropagation"],
    },
    ("backprop", "Gradient"): {
        "correct": ["it tells you how the error changes when you nudge one weight a little",
                    "it shows which direction to move a weight to make the error smaller"],
        "terminology_free": ["it says which way to push each number to get less wrong"],
        "short": ["it's a slope of the error"],
        "analogy": ["it's like feeling which way is downhill when you cannot see"],
        "partial": ["it's a slope"],
        "misconception": ["the gradient changes the weights all by itself without an optimizer"],
        "keyword_only": ["gradients are used in neural networks"],
    },
    ("backprop", "Backward propagation of error"): {
        "correct": ["the error starts at the output and travels back through the layers",
                    "the mistake signal is passed backwards layer by layer using the chain rule"],
        "terminology_free": ["it goes back through the network to see what caused the mistake"],
        "short": ["the error travels backwards"],
        "partial": ["something moves backwards"],
        "misconception": ["it works by changing the input data until the answer is right"],
        "keyword_only": ["backpropagation is the topic"],
    },
    ("backprop", "Weight update"): {
        "correct": ["each weight moves a small step in the direction that lowers the error",
                    "the optimizer nudges the weights a little bit using the learning rate"],
        "terminology_free": ["you adjust the knobs slightly so next time it is less wrong"],
        "short": ["the weights change a little"],
        "partial": ["the weights get changed"],
        "keyword_only": ["weights matter here"],
    },
    ("backprop", "Optimization / iteration"): {
        "correct": ["the whole cycle repeats many times so the error gets smaller gradually",
                    "it happens over and over, improving a bit each round"],
        "terminology_free": ["you keep doing it again and again until it stops being wrong"],
        "short": ["it repeats many times"],
        "analogy": ["it's like practising free throws until you stop missing"],
        "partial": ["it happens more than once"],
        "keyword_only": ["training takes a while"],
    },
    ("hmm", "Hidden states"): {
        "correct": ["there is a real situation underneath that you never observe directly",
                    "the actual state is hidden and you only see what it produces"],
        "terminology_free": ["you can't see what's really going on, you only see the clues"],
        "short": ["the real situation you cannot see"],
        "analogy": ["like guessing someone's mood from the messages they send"],
        "partial": ["something is hidden"],
        "misconception": ["you can just read the states straight off the data"],
        "keyword_only": ["hidden markov models have states"],
    },
    ("hmm", "Observations / emissions"): {
        "correct": ["the things you can actually see, produced by whatever state you are in",
                    "each hidden state gives off visible outputs with certain probabilities"],
        "terminology_free": ["the stuff you can actually notice that depends on the situation"],
        "short": ["what you can actually see"],
        "example": ["umbrellas are what you see, the weather is what you don't"],
        "partial": ["you see some outputs"],
        "keyword_only": ["emissions were on the slide"],
    },
    ("hmm", "Transition probabilities"): {
        "correct": ["they say how likely you are to move from one situation to another",
                    "the chance of switching from the current state to each next state"],
        "terminology_free": ["how likely it is to change from one thing to the next"],
        "short": ["how likely it switches"],
        "partial": ["there are some probabilities"],
        "keyword_only": ["probabilities are involved"],
    },
    ("hmm", "Markov property"): {
        "correct": ["only the current state matters for what happens next, not the whole history",
                    "what comes next depends on where you are now, not on how you got there"],
        "terminology_free": ["it only cares about right now, not everything before"],
        "short": ["only the present matters"],
        "partial": ["it forgets the past"],
        "misconception": ["the next state depends on everything that happened before it"],
        "keyword_only": ["markov is in the name"],
    },
    ("hmm", "State inference / decoding"): {
        "correct": ["you work out the most likely sequence of hidden situations from what you saw",
                    "given the observations you figure out which states were most likely"],
        "terminology_free": ["you guess backwards what was probably going on from the clues"],
        "short": ["you infer the likely states"],
        "partial": ["there is an algorithm for it"],
        "keyword_only": ["viterbi was mentioned"],
    },
    ("overfitting", "Overfitting"): {
        "correct": ["the model learns the training examples too closely including the noise",
                    "it memorises the training data instead of learning the general pattern"],
        "terminology_free": ["it just remembers the practice questions instead of learning"],
        "short": ["it memorises the training data"],
        "analogy": ["like a student who memorises past papers but cannot do a new question"],
        "partial": ["it learns too much"],
        "keyword_only": ["overfitting is bad"],
    },
    ("overfitting", "Generalisation gap"): {
        "correct": ["it does well on the data it trained on but much worse on new data",
                    "training accuracy is high while test accuracy is much lower"],
        "terminology_free": ["great on the stuff it has seen, bad on anything new"],
        "short": ["good on training, bad on new data"],
        "partial": ["there is a difference between the two scores"],
        "keyword_only": ["there is a gap"],
    },
    ("overfitting", "Model complexity"): {
        "correct": ["models with lots of parameters can bend to fit anything, so they overfit more",
                    "the more flexible the model, the easier it is for it to memorise"],
        "terminology_free": ["if it has too many knobs it can twist itself around every point"],
        "short": ["more parameters means more overfitting"],
        "partial": ["complex models are risky"],
        "keyword_only": ["complexity matters"],
    },
    ("overfitting", "Regularization penalty"): {
        "correct": ["you add a penalty on large weights so the model is pushed to stay simpler",
                    "an extra cost on big weights discourages the model from getting too complex"],
        "terminology_free": ["you charge it for using big numbers so it keeps things simple"],
        "short": ["a penalty on big weights"],
        "partial": ["there is a penalty term"],
        "keyword_only": ["l1 and l2 exist"],
    },
    ("overfitting", "Validation-based control"): {
        "correct": ["you watch performance on held-out data and stop when it stops improving",
                    "a separate validation set tells you when to stop training"],
        "terminology_free": ["you keep some questions aside to check when to stop"],
        "short": ["you stop early using held-out data"],
        "partial": ["you use a validation set"],
        "keyword_only": ["dropout was mentioned"],
    },
}

# answers that carry no concept-specific evidence at all
NOISE = {
    "dont_know": ["i don't know", "no idea sorry", "not sure about this one", "idk",
                  "i forgot", "i don't really remember this one", "can't remember"],
    "confirmation": ["yeah", "yes", "i think so", "correct", "yep exactly", "right"],
    "unrelated": ["the canteen was closed today", "we also talked about the sports day schedule",
                  "python is a programming language", "the lecture was quite fast today",
                  "i liked the slides", "can we get the notes by email"],
    "vague_noise": ["maybe it means the position?", "i remember something about zero",
                    "it's basically text", "not really sure", "something to do with the lecture"],
}
NOISE_GOLD = {"dont_know": INSUFFICIENT, "confirmation": INSUFFICIENT,
              "unrelated": INSUFFICIENT, "vague_noise": INSUFFICIENT}

BUCKET_GOLD = {
    "correct": DEMO, "terminology_free": DEMO, "short": DEMO, "example": DEMO,
    "informal": DEMO, "analogy": PARTIAL, "partial": PARTIAL,
    "keyword_only": INSUFFICIENT, "misconception": MISCON,
}

# realistic proportions: most students who answer do try to explain
BUCKET_WEIGHTS = {
    "correct": 5, "terminology_free": 5, "short": 4, "example": 3, "analogy": 2,
    "partial": 4, "keyword_only": 3, "misconception": 2,
}


# ---------------------------------------------------------------------------
# deterministic surface transformations (never change the meaning)
# ---------------------------------------------------------------------------

FILLERS = ["um ", "i think ", "basically ", "like ", "so ", "kind of "]
TRAILERS = [" i think", " or something", " right?", " if i remember"]
CONTRACTIONS = {"you are": "ur", "you": "u", "and": "n", "because": "cuz",
                "it is": "its", "cannot": "cant", "does not": "doesnt"}


def informalise(text: str, rng: random.Random) -> str:
    for src, dst in CONTRACTIONS.items():
        if src in text and rng.random() < 0.5:
            text = text.replace(src, dst, 1)
    return text


def add_filler(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        text = rng.choice(FILLERS) + text
    if rng.random() < 0.4:
        text = text + rng.choice(TRAILERS)
    return text


def add_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def drop_punctuation(text: str, rng: random.Random) -> str:
    return text.replace(".", "").replace(",", "") if rng.random() < 0.6 else text


TRANSFORMS = [informalise, add_filler, add_typo, drop_punctuation]


def vary(text: str, rng: random.Random) -> tuple[str, list[str]]:
    """Apply 0-2 meaning-preserving surface changes."""
    applied = []
    for fn in rng.sample(TRANSFORMS, k=rng.choice([0, 1, 1, 2])):
        new = fn(text, rng)
        if new != text:
            text, _ = new, applied.append(fn.__name__)
    return text, applied


# ---------------------------------------------------------------------------
# case generation
# ---------------------------------------------------------------------------

def generate_cases(seed: int, target: int = 170) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    keys = sorted(BANK)

    # one case from every bucket of every concept: guarantees coverage
    for key in keys:
        topic, concept = key
        for bucket, phrasings in sorted(BANK[key].items()):
            text = rng.choice(phrasings)
            varied, applied = vary(text, rng)
            cases.append({"topic": topic, "concept": concept, "text": varied,
                          "category": bucket, "gold": BUCKET_GOLD[bucket],
                          "transforms": applied, "base": text})

    # noise, spread over concepts
    for bucket, phrasings in sorted(NOISE.items()):
        for text in phrasings:
            topic, concept = keys[rng.randrange(len(keys))]
            cases.append({"topic": topic, "concept": concept, "text": text,
                          "category": bucket, "gold": NOISE_GOLD[bucket],
                          "transforms": [], "base": text})

    # top up with weighted random draws to reach the target volume
    weighted = [b for b, w in sorted(BUCKET_WEIGHTS.items()) for _ in range(w)]
    guard = 0
    while len(cases) < target and guard < 5000:
        guard += 1
        topic, concept = keys[rng.randrange(len(keys))]
        bucket = rng.choice(weighted)
        phrasings = BANK[(topic, concept)].get(bucket)
        if not phrasings:
            continue
        varied, applied = vary(rng.choice(phrasings), rng)
        if any(c["text"] == varied and c["concept"] == concept for c in cases):
            continue
        cases.append({"topic": topic, "concept": concept, "text": varied,
                      "category": bucket, "gold": BUCKET_GOLD[bucket],
                      "transforms": applied, "base": rng.choice(phrasings)})
    return cases


# ---------------------------------------------------------------------------
# evaluation through the real pipeline
# ---------------------------------------------------------------------------

def judge(case: dict, tdef: dict) -> tuple[str, dict]:
    """Exactly what a live TeachBack turn does with this answer."""
    analysis = analyze_response(case["text"], tdef)
    concept = next(c for c in tdef["concepts"] if c["name"] == case["concept"])
    analysis["target_check"] = targeted_concept_check(
        case["text"], concept, topic_name=tdef.get("name", ""),
        misconceptions=tdef.get("misconceptions"))
    entry = {"id": concept.get("id"), "name": concept["name"],
             "status": "pending", "attempts": 0}
    verdict = conversation._verdict(analysis, entry)
    detected = analysis.get("detected_misconceptions", [])
    label = MISCON if detected else VERDICT_TO_LABEL[verdict]
    return label, {"verdict": verdict, "detected": detected,
                   "target": analysis["target_check"]}


def prf(confusion: Counter, label: str) -> dict:
    tp = confusion.get((label, label), 0)
    fp = sum(v for (t, p), v in confusion.items() if p == label and t != label)
    fn = sum(v for (t, p), v in confusion.items() if t == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "support": tp + fn}


def evaluate_answers(cases: list[dict], topic_defs: dict) -> dict:
    confusion: Counter = Counter()
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    results = []
    for case in cases:
        pred, info = judge(case, topic_defs[case["topic"]])
        gold = case["gold"]
        exact = pred == gold
        acceptable = exact or {pred, gold} == {DEMO, PARTIAL}
        confusion[(gold, pred)] += 1
        bucket = by_category[case["category"]]
        bucket[1] += 1
        bucket[0] += int(acceptable)
        results.append({**case, "predicted": pred, "exact": exact,
                       "acceptable": acceptable, "verdict": info["verdict"],
                        "detected": info["detected"]})

    n = len(cases)
    strict = sum(v for (t, p), v in confusion.items() if t == p) / n
    evidence = sum(v for (t, p), v in confusion.items()
                   if t == p or {t, p} == {DEMO, PARTIAL}) / n

    # the two failure directions that matter most
    credited = lambda r: r["predicted"] == DEMO  # noqa: E731
    no_evidence = [r for r in results if r["category"] in
                   ("keyword_only", "unrelated", "dont_know", "confirmation", "vague_noise")]
    simple = [r for r in results if r["category"] in ("terminology_free", "short", "example")]
    return {
        "n": n,
        "strict_accuracy": round(strict, 3),
        "evidence_accuracy": round(evidence, 3),
        "per_label": {lbl: prf(confusion, lbl) for lbl in LABELS},
        "confusion_labels": LABELS,
        "confusion_matrix": [[confusion.get((t, p), 0) for p in LABELS] for t in LABELS],
        "per_category": {c: {"accuracy": round(ok / total, 3), "n": total}
                         for c, (ok, total) in sorted(by_category.items())},
        "false_credit_rate_no_evidence": round(
            sum(1 for r in no_evidence if credited(r)) / max(len(no_evidence), 1), 3),
        "false_credit_rate_dont_know": round(
            sum(1 for r in results if r["category"] == "dont_know" and credited(r))
            / max(sum(1 for r in results if r["category"] == "dont_know"), 1), 3),
        "false_credit_rate_unrelated": round(
            sum(1 for r in results if r["category"] == "unrelated" and credited(r))
            / max(sum(1 for r in results if r["category"] == "unrelated"), 1), 3),
        "missed_rate_simple_language": round(
            sum(1 for r in simple if r["predicted"] == INSUFFICIENT) / max(len(simple), 1), 3),
        "results": results,
    }


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------

REL_CASES = [
    ("strings", ("Strings", "Characters"),
     "a string is just a row of separate letters", "demonstrated"),
    ("strings", ("Strings", "Characters"),
     "we got the notes by email afterwards", "not_shown"),
    ("strings", ("Indexing", "Characters"),
     "you use the number to pull out one letter", "demonstrated"),
    ("strings", ("Slicing", "Substring"),
     "it hands you back a smaller piece of the text", "demonstrated"),
    ("strings", ("Slicing", "Substring"),
     "i think the lecture was on tuesday", "not_shown"),
    ("strings", ("split()", "List"),
     "split gives you back the separate pieces in a list", "demonstrated"),
    ("strings", ("split()", "List"),
     "split takes a list and glues it into one string", "contradicted"),
    ("backprop", ("Gradient descent", "Weight"),
     "it nudges each weight so the error goes down", "demonstrated"),
    ("backprop", ("Gradient descent", "Weight"),
     "it changes the weights so that the loss goes up", "contradicted"),
    ("backprop", ("Weight update", "Loss"),
     "we change the numbers so the error gets smaller each round", "demonstrated"),
    ("backprop", ("Weight update", "Loss"),
     "the room was quite cold", "not_shown"),
    ("hmm", ("Hidden state", "Observation"),
     "what you can see comes from the situation you cannot see", "demonstrated"),
]


def evaluate_relationships(topic_defs: dict) -> dict:
    ok, rows = 0, []
    by_expectation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for topic, (src, tgt), text, expected in REL_CASES:
        tdef = topic_defs[topic]
        analysis = analyze_response(text, tdef)
        res = next((r for r in analysis["relationships"]
                    if r["source"] == src and r["target"] == tgt), None)
        status = res["status"] if res else "no_such_relationship"
        if expected == "demonstrated":
            good = status == "demonstrated"
        elif expected == "not_shown":
            good = status == "not_shown"
        else:
            good = status in ("contradicted", "not_shown")
        by_expectation[expected][1] += 1
        by_expectation[expected][0] += int(good)
        ok += int(good)
        rows.append({"text": text, "pair": f"{src} -> {tgt}",
                     "expected": expected, "got": status, "ok": good})
    return {"n": len(REL_CASES), "ok": ok,
            "by_expectation": {k: {"ok": v[0], "n": v[1]} for k, v in by_expectation.items()},
            "rows": rows}


# ---------------------------------------------------------------------------
# takeaways: upgrade-only, and never fabricated from repeated vocabulary
# ---------------------------------------------------------------------------

TAKEAWAYS = [
    ("strings", "i learned that strings are text stored in quotes and you can use positions "
                "to get individual characters", ["Strings", "Indexing"]),
    ("strings", "basically indexing lets you get single things out and slicing gets a range",
     ["Indexing", "Slicing"]),
    ("strings", "i understand strings and variables but slicing is still confusing", ["Strings"]),
    ("strings", "ok", []),
    ("strings", "it was about strings", []),
    ("strings", "strings strings strings python python indexing slicing", []),
    ("strings", "you can chop text into pieces and stick them back together again",
     ["split() and join()"]),
    ("strings", "the letters sit in order and each one has its own spot", ["Characters"]),
    ("backprop", "the error goes backwards through the layers and the weights get nudged a bit",
     ["Backward propagation of error", "Weight update"]),
    ("backprop", "i got that it measures how wrong it was and then keeps improving",
     ["Loss / error", "Optimization / iteration"]),
    ("backprop", "not much honestly", []),
    ("backprop", "backpropagation gradient weight loss optimization", []),
    ("hmm", "you cannot see the real situation, you only see what it produces",
     ["Hidden states", "Observations / emissions"]),
    ("hmm", "only where you are now matters for where you go next", ["Markov property"]),
    ("hmm", "i think i followed most of it", []),
    ("hmm", "markov hidden states observations transitions", []),
    ("overfitting", "the model can just memorise the practice data and then fail on new data",
     ["Overfitting", "Generalisation gap"]),
    ("overfitting", "you add a cost for big weights to keep it simple", ["Regularization penalty"]),
    ("overfitting", "it was fine", []),
    ("overfitting", "overfitting regularization complexity validation", []),
]


def evaluate_takeaways(topic_defs: dict) -> dict:
    from app.api.teachback import _apply_summary_to_plan
    from app.nlp.analyzer import is_term_list

    rows, upgrade_ok, downgrade_ok, fabrication_ok = [], 0, 0, 0
    for topic, text, expected in TAKEAWAYS:
        tdef = topic_defs[topic]
        # a takeaway that is a bare term list adds nothing, exactly as the
        # finish endpoint treats it
        analysis = ({"concepts": [], "relationships": []} if is_term_list(text)
                    else analyze_response(text, tdef))
        # a plan where everything is already partial, so a downgrade would show
        plan = {"concepts": [{"id": c.get("id"), "name": c["name"], "status": "partial"}
                             for c in tdef["concepts"]], "relationships": []}
        before = {c["name"]: c["status"] for c in plan["concepts"]}
        upgraded, mentioned = _apply_summary_to_plan(plan, analysis)
        after = {c["name"]: c["status"] for c in plan["concepts"]}
        rank = {"pending": 0, "unclear": 0, "partial": 1, "covered": 2}
        never_down = all(rank[after[k]] >= rank[before[k]] for k in before)
        downgrade_ok += int(never_down)
        # keyword-salad takeaways must not manufacture understanding
        is_salad = not expected and len(set(text.split())) <= 6
        fabricated = is_salad and bool(upgraded)
        fabrication_ok += int(not fabricated)
        hit = set(upgraded) & set(expected)
        if expected:
            upgrade_ok += int(bool(hit))
        rows.append({"topic": topic, "text": text, "expected": expected,
                     "upgraded": upgraded, "mentioned": mentioned,
                     "never_downgraded": never_down, "fabricated": fabricated})
    substantive = [r for r in rows if r["expected"]]
    return {
        "n": len(rows),
        "never_downgraded": downgrade_ok == len(rows),
        "no_fabrication_from_keyword_salad": fabrication_ok == len(rows),
        "upgrade_hit_rate": round(upgrade_ok / max(len(substantive), 1), 3),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# full conversation sessions with distinct student personalities
# ---------------------------------------------------------------------------

PERSONAS = {
    "A concise but knowledgeable": ["correct", "short"],
    "B informal": ["terminology_free", "informal", "correct"],
    "C uncertain": ["partial", "vague_noise", "terminology_free"],
    "D different terminology": ["terminology_free"],
    "E misconception then correction": ["misconception", "correct"],
    "F vague": ["partial", "vague_noise"],
    "G example-driven": ["example", "correct"],
    "H overconfident but wrong": ["keyword_only", "misconception"],
    "I relationships not terminology": ["terminology_free", "correct"],
    "J genuinely struggling": ["dont_know", "vague_noise", "partial"],
}


def run_sessions(seed: int) -> dict:
    """Ten real sessions through the HTTP API, one per personality."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.seed import seed_db

    seed_db()
    client = TestClient(app)
    students = client.get("/api/students").json()
    topics = client.get("/api/topics").json()
    topic = next((t for t in topics if "Strings" in t["name"]), topics[0])
    tdef = client.get(f"/api/topics/{topic['id']}").json()
    rng = random.Random(seed)
    rows = []

    for i, (persona, buckets) in enumerate(sorted(PERSONAS.items())):
        student = students[i % len(students)]
        start = client.post("/api/sessions/start",
                            json={"student_id": student["id"], "topic_id": topic["id"]}).json()
        sid = start["session_id"]
        question = start["question"]
        turns, guard = [], 0
        while question is not None and guard < conversation.MAX_QUESTIONS + 2:
            guard += 1
            concept_name = (question.get("concept") or "").split(" → ")[0]
            key = ("strings", concept_name)
            bucket_pool = [b for b in buckets if BANK.get(key, {}).get(b) or b in NOISE]
            answer = None
            for bucket in bucket_pool:
                options = BANK.get(key, {}).get(bucket) or NOISE.get(bucket)
                if options:
                    answer = rng.choice(options)
                    break
            if answer is None:
                answer = rng.choice(NOISE["vague_noise"])
            # persona E repairs its misconception on the follow-up
            if persona.startswith("E") and question.get("kind") == "misconception":
                answer = "no sorry, the first letter is at position zero not one"
            step = client.post(f"/api/sessions/{sid}/respond", json={"text": answer}).json()
            turns.append({"q": question["text"], "a": answer,
                          "feedback": step["feedback"]})
            if step["awaiting_self_report"]:
                break
            question = step.get("followup")

        result = client.post(f"/api/sessions/{sid}/finish", json={
            "attention": 7, "confidence": 6, "difficulty": 5,
            "summary": "i think i got most of it, strings are text in quotes"}).json()
        demonstrated = [c["name"] for c in result["concept_summary"] if c["status"] == "covered"]
        rows.append({
            "persona": persona, "student": student["name"], "turns": len(turns),
            "questions_asked": len(turns),
            "within_cap": len(turns) <= conversation.MAX_QUESTIONS,
            "demonstrated": demonstrated,
            "needs_clarification": [c["name"] for c in result["concept_summary"]
                                    if c["status"] in ("partial", "unclear")],
            "not_discussed": [c["name"] for c in result["concept_summary"]
                              if c["status"] == "missing"],
            "misconceptions_open": result["detected_misconceptions"],
            "misconceptions_resolved": result["resolved_misconceptions"],
            "state": result["state"]["label"],
            "student_state": result["state"]["student_label"],
            "transcript": turns,
        })
    return {"n": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

REPRESENTATIVE_ORDER = [
    "correct", "terminology_free", "short", "example", "analogy", "partial",
    "keyword_only", "unrelated", "dont_know", "confirmation", "vague_noise",
    "misconception",
]


def print_report(answers: dict, rels: dict, takeaways: dict, sessions: dict) -> None:
    print("\n" + "=" * 78)
    print("STUDENT UNDERSTANDING AUDIT")
    print("=" * 78)
    print(f"answers evaluated : {answers['n']}")
    print(f"full sessions     : {sessions['n']}")
    print(f"strict accuracy   : {answers['strict_accuracy']:.3f}")
    print(f"evidence accuracy : {answers['evidence_accuracy']:.3f}"
          "   (demonstrated/partial confusion counted as acceptable)")
    print()
    for lbl in LABELS:
        m = answers["per_label"][lbl]
        print(f"  {lbl:14} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} (n={m['support']})")
    print("\nConfusion matrix (rows=gold, cols=TeachBack):")
    print(f"{'':16}" + "".join(f"{l[:11]:>13}" for l in LABELS))
    for lbl, row in zip(LABELS, answers["confusion_matrix"]):
        print(f"{lbl:16}" + "".join(f"{v:>13}" for v in row))

    print("\nThe two failure directions that matter:")
    print(f"  false credit on answers with no evidence : "
          f"{answers['false_credit_rate_no_evidence']:.3f}")
    print(f"    ... on \"I don't know\"                  : "
          f"{answers['false_credit_rate_dont_know']:.3f}")
    print(f"    ... on unrelated answers               : "
          f"{answers['false_credit_rate_unrelated']:.3f}")
    print(f"  simple/terminology-free correct answers scored as no-evidence : "
          f"{answers['missed_rate_simple_language']:.3f}")

    print("\nPer-category accuracy:")
    for cat, m in answers["per_category"].items():
        print(f"  {cat:20} {m['accuracy']:.3f} (n={m['n']})")

    print(f"\nRelationships: {rels['ok']}/{rels['n']}")
    for k, v in sorted(rels["by_expectation"].items()):
        print(f"  {k:14} {v['ok']}/{v['n']}")
    for row in rels["rows"]:
        if not row["ok"]:
            print(f"  MISS  expected {row['expected']:<13} got {row['got']:<13} "
                  f"{row['pair']}: {row['text'][:52]}")

    print(f"\nTakeaways ({takeaways['n']}):")
    print(f"  never downgraded existing evidence     : {takeaways['never_downgraded']}")
    print(f"  no fabrication from keyword-salad text : "
          f"{takeaways['no_fabrication_from_keyword_salad']}")
    print(f"  upgraded at least one expected concept : {takeaways['upgrade_hit_rate']:.3f}")

    print("\n" + "-" * 78)
    print("REPRESENTATIVE EXAMPLES (student answer -> what TeachBack concluded)")
    print("-" * 78)
    shown = 0
    for category in REPRESENTATIVE_ORDER:
        pool = [r for r in answers["results"] if r["category"] == category]
        for r in pool[:3]:
            mark = "ok  " if r["acceptable"] else "FAIL"
            print(f"[{mark}] {category:16} concept={r['concept']}")
            print(f"        answer   : {r['text']}")
            print(f"        teacher  : {r['gold']}")
            print(f"        TeachBack: {r['predicted']} (verdict={r['verdict']}"
                  + (f", misconception={r['detected']}" if r["detected"] else "") + ")")
            shown += 1
    print(f"\n({shown} examples shown)")

    print("\n" + "-" * 78)
    print("FULL SESSIONS (student-level outcome)")
    print("-" * 78)
    for row in sessions["rows"]:
        print(f"\n  {row['persona']}  ({row['student']}) — {row['questions_asked']} questions, "
              f"within cap: {row['within_cap']}")
        print(f"    demonstrated      : {row['demonstrated']}")
        print(f"    needs clarifying  : {row['needs_clarification']}")
        print(f"    not discussed     : {row['not_discussed']}")
        print(f"    misconceptions    : open={row['misconceptions_open']} "
              f"resolved={row['misconceptions_resolved']}")
        print(f"    state             : {row['state']} (student sees: {row['student_state']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--out", default="data/nlp/student_simulation.json")
    ap.add_argument("--target", type=int, default=170)
    ap.add_argument("--skip-sessions", action="store_true")
    args = ap.parse_args()

    topic_defs = build_topic_defs()
    cases = generate_cases(args.seed, args.target)
    print(f"generated {len(cases)} student answers (seed {args.seed})")

    answers = evaluate_answers(cases, topic_defs)
    rels = evaluate_relationships(topic_defs)
    takeaways = evaluate_takeaways(topic_defs)
    sessions = {"n": 0, "rows": []} if args.skip_sessions else run_sessions(args.seed)

    print_report(answers, rels, takeaways, sessions)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Deterministic audit of how TeachBack interprets ordinary student answers. "
            "Generated from hand-written per-concept phrasings plus meaning-preserving "
            "surface transformations, with a fixed seed. This is an AUDIT set: it is "
            "never used to tune thresholds."),
        "seed": args.seed,
        "cases": [{k: v for k, v in c.items() if k != "base"} for c in cases],
        "metrics": {k: v for k, v in answers.items() if k != "results"},
        "relationships": {k: v for k, v in rels.items() if k != "rows"},
        "takeaways": {k: v for k, v in takeaways.items() if k != "rows"},
        "sessions": [{k: v for k, v in r.items() if k != "transcript"}
                     for r in sessions["rows"]],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAudit data written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
