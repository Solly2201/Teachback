"""Synthetic student dataset generator.

Generates ~200 students with 5-10 sequential TeachBack sessions each. Instead
of independent random rows, each student follows a learner archetype defined by
an initial state distribution and a Markov transition matrix over the five
canonical learning states. Observations are drawn from the per-state emission
profiles in states.py with Gaussian noise, so an "Understanding" student can
still occasionally produce a weak response or report low confidence.

The true hidden state of every session is kept, which later lets us evaluate
how well the (unsupervised) HMM recovers the states.
"""
import json

import numpy as np

from ..config import SYNTHETIC_DIR
from ..states import FEATURE_NAMES, STATE_NAMES, STATE_NOISE, STATE_PROFILES

# state order: [Not Trying, Unclear, Struggling, Understanding, Confident]
ARCHETYPES = {
    "fast_learner": {
        "weight": 0.20,
        "start": [0.02, 0.55, 0.18, 0.20, 0.05],
        "trans": [
            [0.30, 0.40, 0.20, 0.10, 0.00],
            [0.02, 0.28, 0.25, 0.40, 0.05],
            [0.01, 0.09, 0.25, 0.55, 0.10],
            [0.01, 0.04, 0.05, 0.45, 0.45],
            [0.00, 0.02, 0.03, 0.15, 0.80],
        ],
    },
    "hardworking_struggler": {
        "weight": 0.25,
        "start": [0.05, 0.55, 0.35, 0.05, 0.00],
        "trans": [
            [0.25, 0.50, 0.20, 0.05, 0.00],
            [0.03, 0.35, 0.50, 0.11, 0.01],
            [0.02, 0.13, 0.55, 0.27, 0.03],
            [0.01, 0.09, 0.20, 0.55, 0.15],
            [0.00, 0.05, 0.10, 0.35, 0.50],
        ],
    },
    "disengaged": {
        "weight": 0.15,
        "start": [0.60, 0.30, 0.08, 0.02, 0.00],
        "trans": [
            [0.60, 0.30, 0.08, 0.02, 0.00],
            [0.30, 0.45, 0.20, 0.05, 0.00],
            [0.15, 0.30, 0.40, 0.14, 0.01],
            [0.10, 0.20, 0.20, 0.45, 0.05],
            [0.05, 0.10, 0.10, 0.35, 0.40],
        ],
    },
    "inconsistent": {
        "weight": 0.20,
        "start": [0.05, 0.30, 0.25, 0.35, 0.05],
        "trans": [
            [0.30, 0.40, 0.20, 0.10, 0.00],
            [0.05, 0.30, 0.35, 0.28, 0.02],
            [0.04, 0.26, 0.35, 0.30, 0.05],
            [0.02, 0.28, 0.20, 0.35, 0.15],
            [0.01, 0.14, 0.15, 0.40, 0.30],
        ],
    },
    "strong": {
        "weight": 0.20,
        "start": [0.00, 0.05, 0.10, 0.55, 0.30],
        "trans": [
            [0.20, 0.40, 0.25, 0.15, 0.00],
            [0.01, 0.20, 0.24, 0.45, 0.10],
            [0.01, 0.09, 0.25, 0.50, 0.15],
            [0.00, 0.04, 0.06, 0.40, 0.50],
            [0.00, 0.01, 0.02, 0.12, 0.85],
        ],
    },
}


def _sample_observation(state: int, rng: np.random.Generator) -> list[float]:
    mean = np.array(STATE_PROFILES[state])
    noise = rng.normal(0, STATE_NOISE, size=len(mean))
    return list(np.clip(mean + noise, 0.0, 1.0).round(4))


def generate_dataset(n_students: int = 200, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    names = list(ARCHETYPES.keys())
    weights = np.array([ARCHETYPES[n]["weight"] for n in names])
    weights = weights / weights.sum()

    students = []
    for sid in range(1, n_students + 1):
        arch = rng.choice(names, p=weights)
        spec = ARCHETYPES[arch]
        n_sessions = int(rng.integers(5, 11))
        start = np.array(spec["start"]) / np.sum(spec["start"])
        trans = np.array(spec["trans"])
        trans = trans / trans.sum(axis=1, keepdims=True)

        state = int(rng.choice(5, p=start))
        sessions = []
        for t in range(n_sessions):
            obs = _sample_observation(state, rng)
            sessions.append({"t": t, "true_state": state, "features": obs})
            state = int(rng.choice(5, p=trans[state]))
        students.append({"student_id": sid, "archetype": arch, "sessions": sessions})

    return {
        "seed": seed,
        "n_students": n_students,
        "feature_names": FEATURE_NAMES,
        "state_names": STATE_NAMES,
        "students": students,
    }


def save_dataset(dataset: dict, name: str = "students.json") -> str:
    path = SYNTHETIC_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    # also a flat CSV for easy inspection
    import csv

    csv_path = SYNTHETIC_DIR / name.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "archetype", "t", "true_state", "state_name"] + FEATURE_NAMES)
        for s in dataset["students"]:
            for sess in s["sessions"]:
                w.writerow(
                    [s["student_id"], s["archetype"], sess["t"], sess["true_state"],
                     STATE_NAMES[sess["true_state"]]] + sess["features"]
                )
    return str(path)


def load_dataset(name: str = "students.json") -> dict:
    with open(SYNTHETIC_DIR / name, encoding="utf-8") as f:
        return json.load(f)
