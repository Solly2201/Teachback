"""Evaluate the TeachBack answer evaluator on the curated labelled dataset.

Usage (from the repository root):

    python scripts/evaluate_nlp.py            # evaluate with current thresholds
    python scripts/evaluate_nlp.py --tune     # also sweep evaluator thresholds

The dataset (data/nlp/labeled_answers.json) contains hand-written student
answers labelled strong / partial / unclear / misconception, spread over
textbook wording, paraphrases, simple/informal language, short answers,
analogies, noise and different-terminology categories. It does NOT contain
real student data; it exists to tune deterministic thresholds and to prevent
regressions.

Each item is scored exactly the way a live TeachBack turn scores it:
analyze_response() + targeted_concept_check() + the conversation _verdict().
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import ARTIFACTS_DIR, DATA_DIR  # noqa: E402
from app.nlp import conversation  # noqa: E402
from app.nlp.analyzer import analyze_response, targeted_concept_check  # noqa: E402
from app.seed_content import PYTHON_LECTURE, TOPICS  # noqa: E402

DATASET_PATH = DATA_DIR / "nlp" / "labeled_answers.json"
RESULTS_PATH = ARTIFACTS_DIR / "nlp_answer_eval.json"

LABELS = ["strong", "partial", "unclear", "misconception"]
VERDICT_TO_LABEL = {"correct": "strong", "partial": "partial",
                    "analogy": "partial", "affirm": "unclear", "unclear": "unclear"}


def split_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic calibration/held-out split (every 3rd item is held out).

    Thresholds are tuned ONLY on the calibration portion; the held-out portion
    is reported separately and never used for tuning. The interleaved split
    keeps both portions balanced across topics/concepts/categories. With ~200
    items total the held-out set (~66) gives coarse estimates only — treat its
    numbers as a sanity check, not precise statistics.
    """
    calibration = [it for i, it in enumerate(items) if i % 3 != 2]
    heldout = [it for i, it in enumerate(items) if i % 3 == 2]
    return calibration, heldout


def build_topic_defs() -> dict:
    """The same reviewed knowledge structures the seeded demo uses."""
    defs = {t["name"]: t for t in TOPICS}
    strings = {
        "name": PYTHON_LECTURE["title"],
        "reference_explanation": " ".join(
            c["description"] for c in PYTHON_LECTURE["reviewed_concepts"]),
        "concepts": [dict(c, id=i + 1) for i, c in enumerate(PYTHON_LECTURE["reviewed_concepts"])],
        "misconceptions": [dict(m, id=i + 1) for i, m in enumerate(PYTHON_LECTURE["reviewed_misconceptions"])],
        "relationships": [dict(r, id=i + 1) for i, r in enumerate(PYTHON_LECTURE["reviewed_relationships"])],
    }
    return {"strings": strings,
            "backprop": defs["Backpropagation"],
            "hmm": defs["Hidden Markov Models"]}


_analysis_cache: dict = {}


def analyze_item(item: dict, tdef: dict) -> dict:
    """Cached NLP analysis (embeddings don't depend on verdict thresholds)."""
    key = (item["topic"], item["concept"], item["text"])
    if key not in _analysis_cache:
        analysis = analyze_response(item["text"], tdef)
        concept = next(c for c in tdef["concepts"] if c["name"] == item["concept"])
        analysis["target_check"] = targeted_concept_check(
            item["text"], concept, topic_name=tdef.get("name", ""))
        _analysis_cache[key] = analysis
    return _analysis_cache[key]


def judge_item(item: dict, tdef: dict) -> tuple[str, list[str]]:
    """Predicted label + detected misconception names for one answer."""
    analysis = analyze_item(item, tdef)
    concept = next(c for c in tdef["concepts"] if c["name"] == item["concept"])
    entry = {"id": concept.get("id"), "name": concept["name"], "status": "pending", "attempts": 0}
    verdict = conversation._verdict(analysis, entry)
    detected = analysis.get("detected_misconceptions", [])
    label = "misconception" if detected else VERDICT_TO_LABEL[verdict]
    return label, detected


def prf(confusion: dict, label: str) -> dict:
    tp = confusion.get((label, label), 0)
    fp = sum(v for (t, p), v in confusion.items() if p == label and t != label)
    fn = sum(v for (t, p), v in confusion.items() if t == label and p != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "support": tp + fn}


def evaluate(items: list[dict], topic_defs: dict, verbose: bool = True) -> dict:
    confusion: Counter = Counter()
    by_category = defaultdict(lambda: [0, 0])   # category -> [ok, total]
    errors = []
    for item in items:
        pred, detected = judge_item(item, topic_defs[item["topic"]])
        truth = item["label"]
        confusion[(truth, pred)] += 1
        # "acceptable": strong<->partial confusion still lands in the right
        # bucket educationally (evidence vs no evidence vs contradiction)
        ok = pred == truth or {pred, truth} == {"strong", "partial"}
        cat = item.get("category", "other")
        by_category[cat][1] += 1
        if ok:
            by_category[cat][0] += 1
        else:
            errors.append({"topic": item["topic"], "concept": item["concept"],
                           "text": item["text"], "truth": truth, "pred": pred})
        if truth == "misconception" and pred == "misconception":
            expected = item.get("misconception")
            if expected and expected not in detected:
                errors.append({"topic": item["topic"], "concept": item["concept"],
                               "text": item["text"], "truth": f"miscon:{expected}",
                               "pred": f"miscon:{detected}"})

    n = sum(confusion.values())
    strict_acc = sum(v for (t, p), v in confusion.items() if t == p) / n
    grouped_acc = sum(v for (t, p), v in confusion.items()
                      if t == p or {t, p} == {"strong", "partial"}) / n
    per_label = {lbl: prf(confusion, lbl) for lbl in LABELS}
    matrix = [[confusion.get((t, p), 0) for p in LABELS] for t in LABELS]

    result = {
        "n_items": n,
        "strict_accuracy": round(strict_acc, 3),
        "evidence_accuracy": round(grouped_acc, 3),
        "per_label": per_label,
        "confusion_labels": LABELS,
        "confusion_matrix": matrix,
        "per_category": {cat: {"accuracy": round(ok / total, 3), "n": total}
                         for cat, (ok, total) in sorted(by_category.items())},
        "errors": errors,
    }
    if verbose:
        print(f"Examples: {n}")
        print(f"Strict accuracy:   {strict_acc:.3f}")
        print(f"Evidence accuracy: {grouped_acc:.3f}   "
              "(strong/partial confusion counted as acceptable)")
        print()
        for lbl in LABELS:
            m = per_label[lbl]
            print(f"{lbl:14} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1']:.3f} (n={m['support']})")
        print("\nConfusion matrix (rows=truth, cols=predicted):")
        print(f"{'':16}" + "".join(f"{l[:10]:>12}" for l in LABELS))
        for t, row in zip(LABELS, matrix):
            print(f"{t:16}" + "".join(f"{v:>12}" for v in row))
        print("\nPer-category accuracy (evidence-level):")
        for cat, m in result["per_category"].items():
            print(f"  {cat:20} {m['accuracy']:.3f} (n={m['n']})")
        if errors:
            print(f"\n{len(errors)} disagreements:")
            for e in errors[:15]:
                print(f"  [{e['topic']}/{e['concept']}] {e['truth']} -> {e['pred']}: {e['text'][:60]}")
    return result


def evaluate_relationships(rel_items: list[dict], topic_defs: dict) -> dict:
    ok = 0
    errors = []
    for item in rel_items:
        tdef = topic_defs[item["topic"]]
        analysis = analyze_response(item["text"], tdef)
        src, tgt = item["relationship"]
        res = next((r for r in analysis["relationships"]
                    if r["source"] == src and r["target"] == tgt), None)
        status = res["status"] if res else "missing_definition"
        if item["expected"] == "demonstrated":
            good = status == "demonstrated"
        else:  # contradicted: must not be demonstrated; contradicted preferred
            good = status in ("contradicted", "not_shown") and status != "demonstrated"
        if good:
            ok += 1
        else:
            errors.append({"text": item["text"], "expected": item["expected"], "got": status})
    print(f"\nRelationship checks: {ok}/{len(rel_items)} as expected")
    for e in errors:
        print(f"  expected {e['expected']}, got {e['got']}: {e['text'][:60]}")
    return {"n": len(rel_items), "ok": ok, "errors": errors}


def tune(items: list[dict], topic_defs: dict):
    """Small deterministic grid sweep over the conversation thresholds."""
    import itertools
    best = None
    grid = itertools.product([0.54, 0.58, 0.62], [0.42, 0.45, 0.48],
                             [0.62, 0.66, 0.70], [0.55, 0.58, 0.61])
    original = (conversation.TARGET_COVERED_T, conversation.TARGET_PARTIAL_T,
                conversation.CTX_COVERED_T, conversation.CTX_PARTIAL_T)
    print("\nTuning sweep (this runs the evaluator many times — slow):")
    for tc, tp, cc, cp in grid:
        conversation.TARGET_COVERED_T, conversation.TARGET_PARTIAL_T = tc, tp
        conversation.CTX_COVERED_T, conversation.CTX_PARTIAL_T = cc, cp
        res = evaluate(items, topic_defs, verbose=False)
        score = res["evidence_accuracy"]
        marker = ""
        if best is None or score > best[0]:
            best = (score, (tc, tp, cc, cp), res["strict_accuracy"])
            marker = "  <-- best so far"
        print(f"  covered={tc} partial={tp} ctx_cov={cc} ctx_part={cp} "
              f"-> evidence_acc={score:.3f} strict={res['strict_accuracy']:.3f}{marker}")
    (conversation.TARGET_COVERED_T, conversation.TARGET_PARTIAL_T,
     conversation.CTX_COVERED_T, conversation.CTX_PARTIAL_T) = original
    print(f"\nBest: thresholds={best[1]} evidence_acc={best[0]:.3f} strict={best[2]:.3f}")
    print("(Thresholds are applied by editing backend/app/nlp/conversation.py "
          "explicitly — nothing is changed automatically.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="sweep evaluator thresholds (on the calibration portion only)")
    args = parser.parse_args()

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    topic_defs = build_topic_defs()
    calibration, heldout = split_items(data["items"])

    print(f"=== CALIBRATION portion ({len(calibration)} items — thresholds were tuned on these) ===")
    cal_result = evaluate(calibration, topic_defs)
    print(f"\n=== HELD-OUT portion ({len(heldout)} items — never used for tuning) ===")
    held_result = evaluate(heldout, topic_defs, verbose=True)
    rel_result = evaluate_relationships(data.get("relationship_items", []), topic_defs)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"calibration": {k: v for k, v in cal_result.items() if k != "errors"},
                   "heldout": {k: v for k, v in held_result.items() if k != "errors"},
                   "heldout_errors": held_result["errors"],
                   "relationships": rel_result}, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")

    if args.tune:
        tune(calibration, topic_defs)


if __name__ == "__main__":
    main()
