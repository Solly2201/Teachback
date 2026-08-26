"""One-shot build: dataset -> HMM evaluation -> final HMM -> seeded DB -> NLP evaluation.

Run from the backend/ directory:  python scripts/build_all.py [--force-seed]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.evaluate import evaluate_hmm, evaluate_nlp, save_results
from app.hmm.model import train_hmm
from app.hmm.synthetic import generate_dataset, save_dataset
from app.seed import seed_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-seed", action="store_true", help="wipe and reseed the database")
    parser.add_argument("--students", type=int, default=200)
    args = parser.parse_args()

    print("[1/5] Generating synthetic dataset...")
    dataset = generate_dataset(n_students=args.students, seed=42)
    path = save_dataset(dataset)
    n_sessions = sum(len(s["sessions"]) for s in dataset["students"])
    print(f"      {args.students} students, {n_sessions} sessions -> {path}")

    print("[2/5] Evaluating HMM (80/20 student split)...")
    hmm_results = evaluate_hmm(dataset)
    print(f"      state accuracy: {hmm_results['state_accuracy']}, "
          f"adjacent-state accuracy: {hmm_results['adjacent_state_accuracy']}")

    print("[3/5] Training final HMM on the full dataset...")
    model, mapping = train_hmm(dataset)
    print(f"      learned-state labels: {mapping['labels']}")

    print("[4/5] Seeding database...")
    seeded = seed_db(force=args.force_seed)
    print("      seeded" if seeded else "      already seeded (use --force-seed to reset)")

    print("[5/5] Evaluating NLP analyzer on labelled responses...")
    nlp_results = evaluate_nlp()
    c = nlp_results["concept_detection"]
    m = nlp_results["misconception_detection"]
    print(f"      concept detection    P={c['precision']} R={c['recall']} F1={c['f1']}")
    print(f"      misconception detect P={m['precision']} R={m['recall']} F1={m['f1']}")

    save_results(nlp_results, hmm_results)
    print("Done. Evaluation results saved to data/artifacts/evaluation_results.json")


if __name__ == "__main__":
    main()
