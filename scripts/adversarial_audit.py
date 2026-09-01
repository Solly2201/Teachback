"""Adversarial audit: what dangerous things can TeachBack be made to say?

`student_audit.py` asks whether ordinary student wording is recognised. This
asks the opposite question — where does the system produce a *dangerous* output?
Two failures matter far more than accuracy:

    FALSE CREDIT      an answer with no understanding in it is reported as
                      demonstrated. The student is told they know something
                      they do not.
    FALSE ACCUSATION  a correct or merely absent answer is reported as a
                      misconception, or turned into a remediation task. The
                      student is corrected for something they did not do.

Everything here runs the REAL pipeline (analyze_response +
targeted_concept_check + the conversation engine's verdict + the recommender),
across four topics, from a fixed seed. It is an AUDIT: nothing in it is ever
used to tune a threshold.

    python scripts/adversarial_audit.py [--seed 7] [--out data/nlp/adversarial_audit.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.nlp import conversation  # noqa: E402
from app.nlp.analyzer import analyze_response, targeted_concept_check  # noqa: E402
from app.recommend.rules import recommend  # noqa: E402
from student_audit import build_topic_defs  # noqa: E402

# What a teacher would conclude. Deliberately coarse — the audit is about
# safety, not about splitting hairs between "good" and "very good".
CREDIT = "credit"            # this answer demonstrates the concept
PARTIAL = "partial"          # some of the idea is there
NO_EVIDENCE = "no_evidence"  # nothing to go on; NOT a mistake
WRONG = "wrong"              # actively contradicts what was taught

VERDICT_TO_OUTCOME = {"correct": CREDIT, "partial": PARTIAL, "analogy": PARTIAL,
                      "affirm": NO_EVIDENCE, "unclear": NO_EVIDENCE}


# ---------------------------------------------------------------------------
# the cases, grouped by the 8 families the brief asks for
# ---------------------------------------------------------------------------
# (topic, concept, text, expected_outcome, category)
# `expect_misconception` marks the cases where naming a misconception is the
# CORRECT behaviour; everywhere else, naming one is a false accusation.

S, B, H, O = "strings", "backprop", "hmm", "overfitting"

CASES: list[tuple] = [
    # --- A. clearly correct ------------------------------------------------
    (S, "Strings", "A string is a sequence of characters enclosed in quotation marks.", CREDIT, "textbook"),
    (S, "Strings", "It is text that you keep between quote marks.", CREDIT, "paraphrase"),
    (S, "Strings", "it's basically text inside quotes", CREDIT, "informal"),
    (S, "Strings", "just words you shove between quote marks yeah", CREDIT, "colloquial"),
    (S, "Strings", "text in quotes", CREDIT, "very_short"),
    (S, "Strings", "A string holds text. You write it between quote marks, and Python then "
                   "treats everything inside those marks as one piece of text.", CREDIT, "longer"),
    (S, "Characters", "it is built from the separate letters and symbols sitting in a set order",
     CREDIT, "own_terminology"),
    (S, "Indexing", "you use the position number to pull out one particular letter", CREDIT, "paraphrase"),
    (S, "Indexing", "the first position is zero", CREDIT, "very_short"),
    (S, "Indexing", "like s[0] gives you the P in Python", CREDIT, "correct_example"),
    (S, "Slicing", "you take a part of the text between a start and an end position",
     CREDIT, "paraphrase"),
    (S, "Slicing", "honestly the lecture was long but anyway you cut out a piece of the text "
                   "from one position up to another", CREDIT, "irrelevant_extra"),
    (S, "split() and join()", "split breaks the text into a list of pieces and join sticks them "
                              "back together", CREDIT, "paraphrase"),
    (S, "String assignment", "you save the text under a name so you can use it again later",
     CREDIT, "paraphrase"),
    (B, "Loss / error", "it measures how far the prediction was from the right answer", CREDIT, "paraphrase"),
    (B, "Gradient", "it tells you how the error changes when you nudge one weight a little",
     CREDIT, "paraphrase"),
    (B, "Weight update", "each weight moves a small step in the direction that lowers the error",
     CREDIT, "paraphrase"),
    (B, "Backward propagation of error", "the mistake signal starts at the output and travels "
                                         "back through the layers", CREDIT, "paraphrase"),
    (H, "Hidden states", "there is a real situation underneath that you never observe directly",
     CREDIT, "paraphrase"),
    (H, "Markov property", "only the current state matters for what happens next, not the whole "
                           "history", CREDIT, "paraphrase"),
    (O, "Overfitting", "the model learns the training examples too closely, including the noise",
     CREDIT, "textbook"),
    (O, "Generalisation gap", "it does well on the data it trained on but much worse on new data",
     CREDIT, "paraphrase"),

    # --- B. partially correct ---------------------------------------------
    (S, "Indexing", "you use square brackets", PARTIAL, "missing_fact"),
    (S, "Slicing", "the end number is not included", PARTIAL, "fact_without_meaning"),
    (S, "String assignment", "you store it somewhere", PARTIAL, "vague_direction"),
    (S, "Strings", "something that holds letters", PARTIAL, "vague_direction"),
    (B, "Gradient", "it's a slope", PARTIAL, "fact_without_meaning"),
    (B, "Optimization / iteration", "it happens more than once", PARTIAL, "missing_fact"),
    (H, "Transition probabilities", "there are some probabilities involved", PARTIAL, "vague_direction"),
    (O, "Model complexity", "complex models are risky", PARTIAL, "vague_direction"),
    (S, "Characters", "a string is text in quotes and it has letters in it", PARTIAL, "mixed_concepts"),

    # --- C. incorrect (but NOT a taught misconception) ---------------------
    (S, "Slicing", "slicing sorts the characters of the string alphabetically", WRONG, "plausible_wrong"),
    (S, "split() and join()", "split removes every space from the text", WRONG, "plausible_wrong"),
    (S, "Indexing", "indexing tells you how many characters the string has", WRONG, "neighbouring_concept"),
    (B, "Gradient", "the gradient is the final prediction the network makes", WRONG, "plausible_wrong"),
    (H, "Hidden states", "the hidden states are the numbers you feed into the model", WRONG, "plausible_wrong"),

    # --- D. taught misconceptions (naming one here is CORRECT) ------------
    (S, "Indexing", "the first letter is at index 1 and the second is at index 2",
     WRONG, "misconception_index_one", True),
    (S, "Indexing", "you start counting the positions from one", WRONG, "misconception_index_one", True),
    (S, "Strings", "you can change a letter inside the string whenever you want",
     WRONG, "misconception_mutable", True),
    (B, "Gradient", "the gradient changes the weights by itself with no optimizer",
     WRONG, "misconception_gradient", True),
    (B, "Backward propagation of error", "backpropagation works by changing the input data",
     WRONG, "misconception_input", True),
    (H, "Hidden states", "you can read the states straight off the data", WRONG,
     "misconception_observable", True),
    (H, "Markov property", "the next state depends on everything that happened before it",
     WRONG, "misconception_history", True),

    # --- E. no evidence ----------------------------------------------------
    (S, "Indexing", "the canteen was closed today", NO_EVIDENCE, "unrelated"),
    (S, "Strings", "can we get the slides by email", NO_EVIDENCE, "unrelated"),
    (S, "Slicing", "yeah so basically like you know the thing", NO_EVIDENCE, "filler"),
    (S, "Indexing", "What did you understand about Indexing?", NO_EVIDENCE, "question_echo"),
    (B, "Gradient", "What did you understand about Gradient?", NO_EVIDENCE, "question_echo"),
    (S, "Characters", "table chair window bottle", NO_EVIDENCE, "random_nouns"),
    (B, "Weight update", "the parameters are optimised through a stochastic process over the "
                         "manifold", NO_EVIDENCE, "technical_sounding"),
    (H, "Observations / emissions", "the system leverages a probabilistic framework for robust "
                                    "inference", NO_EVIDENCE, "technical_sounding"),
    (O, "Regularization penalty", "it is an important part of machine learning", NO_EVIDENCE,
     "technical_sounding"),

    # --- F. uncertainty ----------------------------------------------------
    (S, "Indexing", "I don't know", NO_EVIDENCE, "dont_know"),
    (S, "Slicing", "not sure", NO_EVIDENCE, "dont_know"),
    (S, "Strings", "I have no idea", NO_EVIDENCE, "dont_know"),
    (B, "Gradient", "I don't remember this one", NO_EVIDENCE, "dont_know"),
    (H, "Markov property", "hmm", NO_EVIDENCE, "near_blank"),
    (O, "Overfitting", "...", NO_EVIDENCE, "near_blank"),

    # --- G. natural language variation ------------------------------------
    (S, "Indexing", "u use teh postion number to get one caracter", CREDIT, "spelling"),
    (S, "Strings", "its text inside quotes", CREDIT, "no_punctuation"),
    (S, "Slicing", "you grab a chunk of the text between two spots", CREDIT, "lowercase"),
    (S, "split() and join()", "split chops it up n join glues it back", CREDIT, "slang"),
    (S, "Characters", "so basically what happens is that the string is made up of the individual "
                      "letters, one after the other, in a fixed order", CREDIT, "conversational"),
    (S, "String assignment", "we are storing the text in one variable only na, so that we can "
                             "use it after", CREDIT, "indian_english"),
    (B, "Loss / error", "how wrong", NO_EVIDENCE, "overly_terse"),
    (B, "Optimization / iteration", "so the thing is that it keeps going round and round doing "
                                    "the same steps again and again and each time it gets a "
                                    "little bit better than before which is the point",
     CREDIT, "rambling"),

    # --- H. adversarial ----------------------------------------------------
    (S, "Indexing", "you use the position to get a character out of the string", CREDIT,
     "correct_with_other_concept_words"),
    (S, "Slicing", "slicing indexing characters strings quotes positions", NO_EVIDENCE,
     "wrong_with_many_right_words"),
    (S, "Characters", "characters are the quotes that surround a string", WRONG,
     "wrong_with_many_right_words"),
    (S, "Slicing", "you can pull out a smaller bit from somewhere in the middle", CREDIT,
     "no_reference_phrase"),
    (S, "Indexing", "python uses indexing", NO_EVIDENCE, "names_concept_only"),
    (S, "Slicing", "slicing is important", NO_EVIDENCE, "names_concept_only"),
    (B, "Gradient", "gradients matter here", NO_EVIDENCE, "names_concept_only"),
    (S, "Indexing", "i thought the first letter was at one, but actually it is at zero",
     CREDIT, "misconception_then_correction"),
    (S, "Strings", "a string is not a number, it is text kept between quote marks", CREDIT,
     "negation"),
    (S, "Indexing", "indexing does not start at one, it starts at zero", CREDIT, "negation"),
    (S, "Slicing", "slicing takes a part of the string, and it also sorts the letters",
     PARTIAL, "mixed_right_and_wrong"),
    (S, "Indexing", "the position gives you a letter, and the first one is at index one",
     PARTIAL, "mixed_right_and_wrong"),

    # --- A. clearly correct, across the remaining topics -------------------
    (B, "Optimization / iteration", "the whole cycle repeats many times so the error shrinks "
                                    "gradually", CREDIT, "paraphrase"),
    (B, "Weight update", "you adjust the knobs slightly so next time it is less wrong",
     CREDIT, "own_terminology"),
    (B, "Loss / error", "a score for how badly it got the answer wrong", CREDIT, "informal"),
    (H, "Observations / emissions", "the things you can actually see, produced by whatever "
                                    "state you are in", CREDIT, "paraphrase"),
    (H, "Transition probabilities", "they say how likely you are to move from one situation "
                                    "to another", CREDIT, "paraphrase"),
    (H, "State inference / decoding", "you work out the most likely run of hidden situations "
                                      "from what you saw", CREDIT, "paraphrase"),
    (O, "Model complexity", "models with lots of parameters can bend to fit anything, so they "
                            "overfit more", CREDIT, "paraphrase"),
    (O, "Regularization penalty", "you add a penalty on large weights so the model is pushed "
                                  "to stay simpler", CREDIT, "paraphrase"),
    (O, "Validation-based control", "you watch performance on held-out data and stop when it "
                                    "stops improving", CREDIT, "paraphrase"),
    (S, "Characters", "it's like beads on a thread, each bead is one letter", CREDIT,
     "correct_analogy"),
    (O, "Overfitting", "like a student who memorises past papers but cannot do a new question",
     CREDIT, "correct_analogy"),
    (S, "Slicing", "s[0:3] on Python gives you Pyt", CREDIT, "correct_example"),
    (S, "split() and join()", '"a,b,c" split on the comma gives you a, b and c separately',
     CREDIT, "correct_example"),
    (B, "Gradient", "it says which way to push each number to get less wrong", CREDIT,
     "own_terminology"),
    (H, "Hidden states", "you cannot see what is really going on, you only see the clues",
     CREDIT, "own_terminology"),
    (S, "Indexing", "negative numbers count backwards from the end", CREDIT, "very_short"),

    # --- B. partially correct ----------------------------------------------
    (B, "Loss / error", "you compare the guess with the right answer", PARTIAL, "missing_fact"),
    (H, "Markov property", "it forgets the past", PARTIAL, "vague_direction"),
    (H, "Hidden states", "something is hidden", PARTIAL, "vague_direction"),
    (O, "Generalisation gap", "there is a difference between the two scores", PARTIAL,
     "vague_direction"),
    (S, "Characters", "it has letters in it", PARTIAL, "vague_direction"),
    (B, "Backward propagation of error", "something moves backwards", PARTIAL, "vague_direction"),
    (O, "Validation-based control", "you use a validation set", PARTIAL, "fact_without_meaning"),

    # --- C. incorrect -------------------------------------------------------
    (S, "String assignment", "assigning a string makes a copy of every character in memory",
     WRONG, "plausible_wrong"),
    (B, "Optimization / iteration", "training runs exactly once through the data and then stops",
     WRONG, "plausible_wrong"),
    (H, "Transition probabilities", "the transition probabilities say how likely each "
                                    "observation is", WRONG, "neighbouring_concept"),
    (O, "Regularization penalty", "regularization increases the size of the weights", WRONG,
     "plausible_wrong"),
    (S, "Characters", "characters are the variables that hold a string", WRONG,
     "neighbouring_concept"),

    # --- D. taught misconceptions -------------------------------------------
    (O, "Validation-based control", "training a model for more epochs always makes it better "
                                    "on new data", WRONG, "misconception_more_training", True),
    (O, "Generalisation gap", "if the model gets very high accuracy on the training set it is "
                              "a good model", WRONG, "misconception_train_accuracy", True),
    (H, "Observations / emissions", "the observations are the states - each output tells you "
                                    "exactly which state produced it", WRONG,
     "misconception_obs_equal_states", True),
    (B, "Weight update", "the network reduces error by editing its output prediction after "
                         "seeing the answer", WRONG, "misconception_edit_prediction", True),

    # --- E. no evidence ------------------------------------------------------
    (O, "Overfitting", "sir said this is very important for the exam", NO_EVIDENCE, "unrelated"),
    (H, "Markov property", "is this going to be in the test", NO_EVIDENCE, "unrelated"),
    (B, "Loss / error", "i attended the lecture yesterday", NO_EVIDENCE, "unrelated"),
    (S, "String assignment", "umm well you know how it is", NO_EVIDENCE, "filler"),
    (H, "State inference / decoding", "What did you understand about State inference?",
     NO_EVIDENCE, "question_echo"),
    (O, "Model complexity", "paper pencil bottle cupboard", NO_EVIDENCE, "random_nouns"),
    (B, "Backward propagation of error", "the architecture leverages an end-to-end "
                                         "differentiable paradigm", NO_EVIDENCE,
     "technical_sounding"),
    (S, "Slicing", "it is a fundamental concept in computer science", NO_EVIDENCE,
     "technical_sounding"),

    # --- F. uncertainty ------------------------------------------------------
    (O, "Regularization penalty", "no clue", NO_EVIDENCE, "dont_know"),
    (B, "Optimization / iteration", "i cant remember this one", NO_EVIDENCE, "dont_know"),
    (H, "Transition probabilities", "   ", NO_EVIDENCE, "near_blank"),
    (S, "Characters", "?", NO_EVIDENCE, "near_blank"),

    # --- G. natural language variation --------------------------------------
    (B, "Weight update", "u chnage teh wieghts a litle bit so the eror goes down", CREDIT,
     "spelling"),
    (H, "Markov property", "only now matters not the whole past", CREDIT, "overly_terse"),
    (O, "Overfitting", "it just remembers the practice questions instead of learning", CREDIT,
     "informal"),
    (S, "Indexing", "the position it is telling which character we want na", CREDIT,
     "indian_english"),
    (B, "Loss / error", "the bigger the mistake the bigger that number gets", CREDIT,
     "conversational"),

    # --- H. adversarial ------------------------------------------------------
    (O, "Overfitting", "overfitting regularization complexity validation generalisation",
     NO_EVIDENCE, "wrong_with_many_right_words"),
    (H, "Hidden states", "hidden states observations transitions markov", NO_EVIDENCE,
     "wrong_with_many_right_words"),
    (B, "Gradient", "gradient descent backpropagation loss weights", NO_EVIDENCE,
     "wrong_with_many_right_words"),
    (O, "Overfitting", "overfitting is a big problem", NO_EVIDENCE, "names_concept_only"),
    (H, "Markov property", "the markov property is in the name", NO_EVIDENCE, "names_concept_only"),
    (B, "Loss / error", "the loss is part of backpropagation", NO_EVIDENCE, "names_concept_only"),
    (S, "Indexing", "people think indexing starts at one but it actually starts at zero",
     CREDIT, "misconception_then_correction"),
    (H, "Markov property", "the next state does not depend on the whole history, only on the "
                           "current one", CREDIT, "negation"),
    (B, "Weight update", "the weights are updated so the loss decreases, not increases",
     CREDIT, "negation"),
    (O, "Overfitting", "it memorises the training data, and it also always improves on new data",
     PARTIAL, "mixed_right_and_wrong"),
    (H, "Hidden states", "the states are hidden, and you can also read them straight off the "
                         "data", PARTIAL, "mixed_right_and_wrong"),
    (S, "Slicing", "you take a part of the text using positions and characters and indexes",
     CREDIT, "correct_with_other_concept_words"),
]

# relationship probes: (topic, (source, target), text, expected_status)
REL_CASES = [
    (S, ("Strings", "Characters"), "a string is just a row of separate letters", "demonstrated"),
    (S, ("split()", "List"), "split gives you back the separate pieces in a list", "demonstrated"),
    (S, ("split()", "List"), "split takes a list and glues it into one string", "contradicted"),
    (S, ("Strings", "Characters"), "I don't know", "not_shown"),
    (S, ("Strings", "Characters"), "", "not_shown"),
    (S, ("Slicing", "Substring"), "the canteen was closed today", "not_shown"),
    (S, ("Indexing", "Characters"), "we did not really cover that part", "not_shown"),
    (B, ("Gradient descent", "Weight"), "it nudges each weight so the error goes down", "demonstrated"),
    (B, ("Gradient descent", "Weight"), "gradient descent updates the weights so the loss "
                                        "increases", "contradicted"),
    (B, ("Weight update", "Loss"), "not sure about this one", "not_shown"),
    (H, ("Hidden state", "Observation"), "what you can see is produced by the situation you "
                                         "cannot see", "demonstrated"),
    (O, ("Model complexity", "Overfitting"), "i think it was on the last slide", "not_shown"),
]


def judge(topic_defs: dict, topic: str, concept_name: str, text: str) -> dict:
    """Exactly what a live TeachBack turn concludes from this answer."""
    tdef = topic_defs[topic]
    concept = next(c for c in tdef["concepts"] if c["name"] == concept_name)
    analysis = analyze_response(text, tdef)
    analysis["target_check"] = targeted_concept_check(
        text, concept, topic_name=tdef.get("name", ""),
        misconceptions=tdef.get("misconceptions"),
        sibling_names=[c["name"] for c in tdef["concepts"]])
    entry = {"id": concept.get("id"), "name": concept_name,
             "status": "pending", "attempts": 0}
    verdict = conversation._verdict(analysis, entry)
    detected = analysis.get("detected_misconceptions", [])
    return {"verdict": verdict, "outcome": VERDICT_TO_OUTCOME[verdict],
            "misconceptions": detected}


def run_answers(topic_defs: dict) -> dict:
    rows = []
    for case in CASES:
        topic, concept, text, expected, category = case[:5]
        expect_miscon = len(case) > 5 and case[5]
        result = judge(topic_defs, topic, concept, text)
        credited = result["outcome"] == CREDIT
        accused = bool(result["misconceptions"]) and not expect_miscon
        rows.append({
            "topic": topic, "concept": concept, "text": text,
            "category": category, "expected": expected,
            "outcome": result["outcome"], "verdict": result["verdict"],
            "misconceptions": result["misconceptions"],
            "expect_misconception": bool(expect_miscon),
            # the two dangerous outcomes
            "false_credit": credited and expected in (NO_EVIDENCE, WRONG),
            "false_accusation": accused,
            "missed_misconception": bool(expect_miscon) and not result["misconceptions"],
            "credited_a_misconception": bool(expect_miscon) and credited,
            # What "handled correctly" means depends on the family:
            #   a taught misconception  -> named, or at least not credited
            #   any other wrong answer  -> not credited
            #   everything else         -> the expected outcome (credit and
            #                              partial are interchangeable)
            "ok": (bool(result["misconceptions"]) or not credited)
                   if expect_miscon else
                  (not credited if expected == WRONG else
                   (result["outcome"] == expected
                    or {result["outcome"], expected} == {CREDIT, PARTIAL})),
        })
    return {"rows": rows}


def run_relationships(topic_defs: dict) -> dict:
    rows = []
    for topic, (src, tgt), text, expected in REL_CASES:
        analysis = analyze_response(text, topic_defs[topic])
        res = next((r for r in analysis["relationships"]
                    if r["source"] == src and r["target"] == tgt), None)
        status = res["status"] if res else "no_such_relationship"
        if expected == "demonstrated":
            ok = status == "demonstrated"
        elif expected == "not_shown":
            ok = status in ("not_shown", "partial")
        else:
            ok = status in ("contradicted", "not_shown", "partial")
        rows.append({"pair": f"{src} -> {tgt}", "text": text, "expected": expected,
                     "got": status, "ok": ok,
                     # the dangerous one: silence read as a misunderstanding
                     "silence_called_wrong": expected == "not_shown"
                     and status == "contradicted"})
    return {"rows": rows}


def run_safety_invariants(topic_defs: dict) -> list[dict]:
    """The properties that must hold regardless of any accuracy number."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # 1. a concept that never came up must not become a remediation task
    rec = recommend(1, [], evidence={"demonstrated": ["Indexing"], "unclear": [],
                                     "not_discussed": ["Slicing"]},
                    topic_def=topic_defs["strings"])
    text = (rec["activity"]["title"] + rec["activity"]["question"] + rec["why"]).lower()
    add("not_discussed never becomes a remediation target", "slicing" not in text, text[:70])

    # 2. ...and is never worded as a mistake
    note = " ".join(rec["notes"]).lower()
    add("not_discussed is never worded as a mistake",
        all(w not in note for w in ("wrong", "misunderstood", "failed")), note[:70])

    # 3. confidence alone must not manufacture understanding
    high_conf_no_evidence = recommend(
        1, [], evidence={"demonstrated": [], "unclear": ["Indexing"], "not_discussed": []},
        signals={"understanding": 0.1, "confidence": 0.95, "difficulty": 0.2},
        topic_def=topic_defs["strings"])
    add("high confidence with no evidence does not claim understanding",
        high_conf_no_evidence["state_key"] == "unclear"
        and "double-check" in " ".join(high_conf_no_evidence["notes"]).lower(),
        str(high_conf_no_evidence["notes"])[:80])

    # 4. an empty answer earns nothing anywhere
    for topic in topic_defs:
        concept = topic_defs[topic]["concepts"][0]["name"]
        out = judge(topic_defs, topic, concept, "")
        add(f"empty answer earns nothing ({topic})", out["outcome"] == NO_EVIDENCE,
            out["verdict"])
    return checks


def report(answers: dict, rels: dict, invariants: list[dict]) -> dict:
    rows = answers["rows"]
    n = len(rows)
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        b = by_cat[r["category"]]
        b[1] += 1
        b[0] += int(r["ok"])

    no_evidence_rows = [r for r in rows if r["expected"] in (NO_EVIDENCE, WRONG)]
    correct_rows = [r for r in rows if r["expected"] == CREDIT]
    miscon_rows = [r for r in rows if r["expect_misconception"]]
    false_credits = [r for r in rows if r["false_credit"]]
    false_accusations = [r for r in rows if r["false_accusation"]]

    detected_any = [r for r in rows if r["misconceptions"]]
    miscon_precision = (len(miscon_rows and [r for r in detected_any if r["expect_misconception"]])
                        / len(detected_any)) if detected_any else 1.0
    miscon_recall = (sum(1 for r in miscon_rows if r["misconceptions"]) / len(miscon_rows)
                     if miscon_rows else 0.0)

    summary = {
        "n_answers": n,
        "n_relationships": len(rels["rows"]),
        "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
        "false_credit_rate": round(len(false_credits) / max(len(no_evidence_rows), 1), 3),
        "false_accusation_rate": round(len(false_accusations) / n, 3),
        "correct_answers_recognised": round(
            sum(1 for r in correct_rows if r["outcome"] in (CREDIT, PARTIAL))
            / max(len(correct_rows), 1), 3),
        "misconception_precision": round(miscon_precision, 3),
        "misconception_recall": round(miscon_recall, 3),
        "misconceptions_credited": sum(1 for r in rows if r["credited_a_misconception"]),
        "relationships_ok": sum(1 for r in rels["rows"] if r["ok"]),
        "silence_called_wrong": sum(1 for r in rels["rows"] if r["silence_called_wrong"]),
        "invariants_passed": sum(1 for c in invariants if c["ok"]),
        "invariants_total": len(invariants),
        "per_category": {c: {"ok": v[0], "n": v[1]} for c, v in sorted(by_cat.items())},
    }

    print("=" * 78)
    print("ADVERSARIAL AUDIT")
    print("=" * 78)
    print(f"answers            : {n}   relationships: {len(rels['rows'])}")
    print(f"overall accuracy   : {summary['accuracy']:.3f}   (coarse: credit/partial/none/wrong)")
    print()
    print("DANGEROUS OUTCOMES")
    print(f"  false credit  (no-evidence or wrong answer credited) : "
          f"{len(false_credits)}/{len(no_evidence_rows)} = {summary['false_credit_rate']:.3f}")
    print(f"  false accusation (misconception named wrongly)       : "
          f"{len(false_accusations)}/{n} = {summary['false_accusation_rate']:.3f}")
    print(f"  a taught misconception credited as understanding     : "
          f"{summary['misconceptions_credited']}")
    print(f"  silence reported as a wrong relationship             : "
          f"{summary['silence_called_wrong']}")
    print()
    print(f"correct answers recognised : {summary['correct_answers_recognised']:.3f}")
    print(f"misconception precision    : {summary['misconception_precision']:.3f}   "
          f"recall: {summary['misconception_recall']:.3f}")
    print(f"relationships as expected  : {summary['relationships_ok']}/{len(rels['rows'])}")
    print(f"safety invariants          : {summary['invariants_passed']}/{summary['invariants_total']}")

    print("\nPER CATEGORY")
    for cat, v in summary["per_category"].items():
        flag = "" if v["ok"] == v["n"] else "   <-- "
        print(f"  {cat:<34} {v['ok']}/{v['n']}{flag}")

    if false_credits:
        print("\nFALSE CREDIT (the dangerous direction):")
        for r in false_credits:
            print(f"  [{r['category']}] {r['concept']}: {r['text'][:64]}")
    if false_accusations:
        print("\nFALSE ACCUSATION:")
        for r in false_accusations:
            print(f"  [{r['category']}] {r['concept']}: {r['text'][:52]} -> {r['misconceptions']}")
    bad_rel = [r for r in rels["rows"] if not r["ok"]]
    if bad_rel:
        print("\nRELATIONSHIP MISSES:")
        for r in bad_rel:
            print(f"  expected {r['expected']:<13} got {r['got']:<13} {r['pair']}: {r['text'][:44]}")
    for c in invariants:
        if not c["ok"]:
            print(f"\nINVARIANT FAILED: {c['name']} — {c['detail']}")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/nlp/adversarial_audit.json")
    args = ap.parse_args()

    topic_defs = build_topic_defs()
    answers = run_answers(topic_defs)
    rels = run_relationships(topic_defs)
    invariants = run_safety_invariants(topic_defs)
    summary = report(answers, rels, invariants)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": ("Adversarial audit of TeachBack's answer evaluation. Fixed cases, four "
                        "topics, real pipeline. Measures dangerous outcomes (false credit, "
                        "false accusation) rather than accuracy. Never used for tuning."),
        "summary": summary,
        "answers": answers["rows"],
        "relationships": rels["rows"],
        "invariants": invariants,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
