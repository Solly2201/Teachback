"""Evaluation of the two ML components.

NLP: the hand-labelled responses in data/nlp_eval/labeled_responses.json are
run through the real analyzer; concept detection and misconception detection
are scored as binary classification per (response, concept/misconception) pair.

HMM: students in the synthetic dataset are split 80/20; a model trained on the
training students Viterbi-decodes the held-out students' sequences, and the
decoded canonical states are compared against the generator's true states.

Nothing here is hard-coded: numbers are whatever the models actually achieve.
"""
import json

import numpy as np

from ..config import DATA_DIR, EVAL_RESULTS_PATH
from ..nlp.analyzer import analyze_response
from ..seed_content import TOPICS
from ..states import STATE_NAMES

NLP_EVAL_PATH = DATA_DIR / "nlp_eval" / "labeled_responses.json"


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_nlp() -> dict:
    with open(NLP_EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    topic_defs = {t["name"]: t for t in TOPICS}

    c_tp = c_fp = c_fn = c_tn = 0
    m_tp = m_fp = m_fn = m_tn = 0
    errors = []

    for item in eval_set["items"]:
        tdef = topic_defs[item["topic"]]
        analysis = analyze_response(item["text"], tdef)

        true_c = set(item["concepts_present"])
        for c in analysis["concepts"]:
            pred = c["status"] in ("covered", "partial")
            actual = c["name"] in true_c
            if pred and actual:
                c_tp += 1
            elif pred and not actual:
                c_fp += 1
                errors.append({"type": "concept_fp", "topic": item["topic"], "concept": c["name"],
                               "text": item["text"][:80]})
            elif not pred and actual:
                c_fn += 1
                errors.append({"type": "concept_fn", "topic": item["topic"], "concept": c["name"],
                               "text": item["text"][:80]})
            else:
                c_tn += 1

        true_m = set(item["misconceptions_present"])
        for m in analysis["misconceptions"]:
            pred = m["detected"]
            actual = m["name"] in true_m
            if pred and actual:
                m_tp += 1
            elif pred and not actual:
                m_fp += 1
                errors.append({"type": "miscon_fp", "topic": item["topic"], "misconception": m["name"],
                               "text": item["text"][:80]})
            elif not pred and actual:
                m_fn += 1
                errors.append({"type": "miscon_fn", "topic": item["topic"], "misconception": m["name"],
                               "text": item["text"][:80]})
            else:
                m_tn += 1

    c_total = c_tp + c_fp + c_fn + c_tn
    m_total = m_tp + m_fp + m_fn + m_tn
    return {
        "n_labelled_responses": len(eval_set["items"]),
        "concept_detection": {
            **_prf(c_tp, c_fp, c_fn),
            "accuracy": round((c_tp + c_tn) / c_total, 3) if c_total else 0.0,
            "n_pairs": c_total,
        },
        "misconception_detection": {
            **_prf(m_tp, m_fp, m_fn),
            "accuracy": round((m_tp + m_tn) / m_total, 3) if m_total else 0.0,
            "n_pairs": m_total,
        },
        "errors": errors,
    }


def evaluate_hmm(dataset: dict, test_frac: float = 0.2, seed: int = 5) -> dict:
    from hmmlearn.hmm import GaussianHMM
    from scipy.optimize import linear_sum_assignment

    from ..states import STATE_PROFILES

    rng = np.random.default_rng(seed)
    students = dataset["students"]
    idx = rng.permutation(len(students))
    n_test = max(1, int(len(students) * test_frac))
    test_ids = set(idx[:n_test].tolist())
    train = [s for i, s in enumerate(students) if i not in test_ids]
    test = [s for i, s in enumerate(students) if i in test_ids]

    X_train = np.vstack([[sess["features"] for sess in s["sessions"]] for s in train])
    lengths = [len(s["sessions"]) for s in train]
    model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=200, random_state=7)
    model.fit(X_train, lengths)

    profiles = np.array([STATE_PROFILES[i] for i in range(5)])
    cost = np.linalg.norm(model.means_[:, None, :] - profiles[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    to_canon = {int(r): int(c) for r, c in zip(rows, cols)}

    y_true, y_pred = [], []
    for s in test:
        X = np.array([sess["features"] for sess in s["sessions"]])
        pred_raw = model.predict(X)
        y_pred.extend(to_canon[int(p)] for p in pred_raw)
        y_true.extend(sess["true_state"] for sess in s["sessions"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = float((y_true == y_pred).mean())

    confusion = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_state = {}
    for i in range(5):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        per_state[STATE_NAMES[i]] = _prf(tp, fp, fn)

    # adjacent-state tolerance: predictions off by one ordinal state
    adjacent_acc = float((np.abs(y_true - y_pred) <= 1).mean())

    return {
        "n_train_students": len(train),
        "n_test_students": len(test),
        "n_test_sessions": int(len(y_true)),
        "state_accuracy": round(accuracy, 3),
        "adjacent_state_accuracy": round(adjacent_acc, 3),
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": STATE_NAMES,
        "per_state": per_state,
    }


def save_results(nlp_results: dict, hmm_results: dict):
    payload = {"nlp": nlp_results, "hmm": hmm_results}
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
