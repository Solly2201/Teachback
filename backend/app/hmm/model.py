"""Hidden Markov Model over student learning states.

A GaussianHMM (hmmlearn) with 5 hidden states is trained *unsupervised* on the
synthetic observation sequences (8 continuous features per session).

State mapping
-------------
Unsupervised HMM state IDs are arbitrary: learned state "2" means nothing by
itself. To attach labels honestly, each learned state's emission mean is
matched to the closest canonical state profile (states.STATE_PROFILES) using
the Hungarian algorithm (scipy.optimize.linear_sum_assignment) on Euclidean
distance, giving a one-to-one mapping learned-state -> canonical label. The
mapping and per-state distances are saved next to the model so the choice is
auditable.

Inference for a live student runs Viterbi over that student's full observation
sequence, so the estimated current state depends on their history, not just the
latest session - which is the entire reason an HMM is used instead of a
per-session classifier.
"""
import json

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

from ..config import HMM_MAPPING_PATH, HMM_MODEL_PATH
from ..states import STATE_NAMES, STATE_PROFILES

_cache: dict = {}


def train_hmm(dataset: dict, n_iter: int = 200, seed: int = 7) -> tuple[GaussianHMM, dict]:
    sequences = [np.array([s["features"] for s in st["sessions"]]) for st in dataset["students"]]
    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]

    model = GaussianHMM(
        n_components=5,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        init_params="stmc",
    )
    model.fit(X, lengths)

    mapping = _map_states(model)
    joblib.dump(model, HMM_MODEL_PATH)
    with open(HMM_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    _cache.clear()
    return model, mapping


def _map_states(model: GaussianHMM) -> dict:
    """One-to-one match learned states to canonical profiles (Hungarian)."""
    profiles = np.array([STATE_PROFILES[i] for i in range(5)])
    cost = np.linalg.norm(model.means_[:, None, :] - profiles[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    state_to_canonical = {int(r): int(c) for r, c in zip(rows, cols)}
    return {
        "state_to_canonical": {str(k): v for k, v in state_to_canonical.items()},
        "labels": {str(k): STATE_NAMES[v] for k, v in state_to_canonical.items()},
        "distances": {str(r): round(float(cost[r, c]), 4) for r, c in zip(rows, cols)},
        "learned_means": [[round(float(x), 4) for x in row] for row in model.means_],
    }


def load_hmm() -> tuple[GaussianHMM, dict]:
    if "model" not in _cache:
        _cache["model"] = joblib.load(HMM_MODEL_PATH)
        with open(HMM_MAPPING_PATH, encoding="utf-8") as f:
            _cache["mapping"] = json.load(f)
    return _cache["model"], _cache["mapping"]


def hmm_available() -> bool:
    return HMM_MODEL_PATH.exists() and HMM_MAPPING_PATH.exists()


def infer_sequence(features_seq: list[list[float]]) -> dict:
    """Viterbi-decode a student's observation sequence into canonical states.

    Returns per-session canonical state indices/labels plus the posterior
    distribution over states for the most recent session.

    Works for a single session (the HMM then falls back to the initial-state
    distribution and one emission) as well as a long history, which is the
    case that actually motivates the HMM: Viterbi re-reads the whole
    trajectory, so new evidence can revise the reading of earlier sessions.
    Malformed input is rejected loudly rather than being coerced into a
    silently wrong state.
    """
    model, mapping = load_hmm()
    to_canon = {int(k): v for k, v in mapping["state_to_canonical"].items()}
    if not features_seq:
        raise ValueError("Cannot infer a learning state from an empty observation sequence.")
    X = np.array(features_seq, dtype=float)
    if X.ndim != 2 or X.shape[1] != model.means_.shape[1]:
        raise ValueError(
            f"Observation vectors must be {model.means_.shape[1]}-dimensional; "
            f"got array of shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Observation vectors contain NaN or infinite values.")
    raw_states = model.predict(X)
    canon = [to_canon[int(s)] for s in raw_states]

    posteriors = model.predict_proba(X)[-1]
    canon_post = [0.0] * 5
    for raw_idx, p in enumerate(posteriors):
        canon_post[to_canon[raw_idx]] += float(p)

    return {
        "states": canon,
        "labels": [STATE_NAMES[c] for c in canon],
        "current_state": canon[-1],
        "current_label": STATE_NAMES[canon[-1]],
        "current_posterior": [round(p, 4) for p in canon_post],
    }


def validate_model(tolerance: float = 1e-6) -> dict:
    """Read-only sanity report on the preserved HMM artifacts.

    The model is a preserved artifact: this NEVER retrains, refits or writes
    anything. It only checks the properties the rest of the system relies on,
    so a corrupted or swapped artifact is caught explicitly instead of quietly
    producing plausible-looking states:

      * the transition matrix and the start distribution are proper
        distributions (non-negative, rows summing to 1)
      * emission means sit in the 0-1 range the features are defined on
      * covariances are positive
      * the learned-state -> canonical-profile mapping is a bijection, and its
        Hungarian assignment distances are recorded

    Returns a report dict; ``report["ok"]`` is True when every check passes.
    """
    model, mapping = load_hmm()
    n_states, n_features = model.means_.shape
    problems = []

    row_sums = model.transmat_.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        problems.append(f"transition rows do not sum to 1: {row_sums.tolist()}")
    if float(model.transmat_.min()) < -tolerance:
        problems.append("transition matrix has negative entries")
    if not np.isclose(float(model.startprob_.sum()), 1.0, atol=1e-6):
        problems.append("start probabilities do not sum to 1")
    if float(model.means_.min()) < -0.5 or float(model.means_.max()) > 1.5:
        problems.append("emission means fall far outside the 0-1 feature range")
    # covariance_type="diag": covars_ is returned as full diagonal matrices,
    # so only the diagonal carries the actual per-feature variances
    variances = np.diagonal(np.atleast_3d(model.covars_), axis1=1, axis2=2)
    if float(np.min(variances)) <= 0:
        problems.append("non-positive emission variance")

    to_canon = {int(k): int(v) for k, v in mapping["state_to_canonical"].items()}
    if sorted(to_canon) != list(range(n_states)):
        problems.append("state mapping does not cover every learned state")
    if sorted(to_canon.values()) != list(range(n_states)):
        problems.append("state mapping is not a one-to-one assignment")

    distances = {k: float(v) for k, v in mapping.get("distances", {}).items()}
    return {
        "ok": not problems,
        "problems": problems,
        "n_states": int(n_states),
        "n_features": int(n_features),
        "transition_row_sums": [round(float(v), 6) for v in row_sums],
        "self_transition": [round(float(model.transmat_[i, i]), 4) for i in range(n_states)],
        "mapping": {STATE_NAMES[v]: int(k) for k, v in to_canon.items()},
        "min_variance": round(float(np.min(variances)), 6),
        "mapping_distances": distances,
        "max_mapping_distance": round(max(distances.values()), 4) if distances else None,
    }
