"""Tests for the synthetic generator and the HMM training/inference pipeline."""
import numpy as np
import pytest

from app.hmm.synthetic import ARCHETYPES, generate_dataset
from app.states import STATE_NAMES, STATE_PROFILES


def test_dataset_shape():
    ds = generate_dataset(n_students=30, seed=1)
    assert len(ds["students"]) == 30
    for s in ds["students"]:
        assert 5 <= len(s["sessions"]) <= 10
        for sess in s["sessions"]:
            assert len(sess["features"]) == 8
            assert all(0.0 <= f <= 1.0 for f in sess["features"])
            assert 0 <= sess["true_state"] <= 4


def test_dataset_is_reproducible():
    a = generate_dataset(n_students=10, seed=3)
    b = generate_dataset(n_students=10, seed=3)
    assert a == b


def test_archetype_rows_are_distributions():
    for spec in ARCHETYPES.values():
        assert abs(sum(spec["start"]) - 1.0) < 1e-6
        for row in spec["trans"]:
            assert abs(sum(row) - 1.0) < 1e-6


def test_observations_are_noisy_not_deterministic():
    ds = generate_dataset(n_students=50, seed=2)
    by_state = {i: [] for i in range(5)}
    for s in ds["students"]:
        for sess in s["sessions"]:
            by_state[sess["true_state"]].append(sess["features"])
    for state, rows in by_state.items():
        if len(rows) > 5:
            stds = np.array(rows).std(axis=0)
            assert stds.mean() > 0.03, f"state {state} observations look deterministic"


def test_hmm_train_and_state_mapping(tmp_path, monkeypatch):
    import app.hmm.model as hmm_model

    monkeypatch.setattr(hmm_model, "HMM_MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(hmm_model, "HMM_MAPPING_PATH", tmp_path / "map.json")
    hmm_model._cache.clear()

    ds = generate_dataset(n_students=60, seed=4)
    model, mapping = hmm_model.train_hmm(ds, n_iter=50)

    canon = sorted(int(v) for v in mapping["state_to_canonical"].values())
    assert canon == [0, 1, 2, 3, 4], "mapping must be a one-to-one assignment"

    # a clearly 'Confident'-profile sequence should decode to a high state
    conf_seq = [STATE_PROFILES[4]] * 5
    result = hmm_model.infer_sequence(conf_seq)
    assert result["current_label"] in ("Confident", "Understanding")

    # a clearly disengaged sequence should decode to a low state
    low_seq = [STATE_PROFILES[0]] * 5
    result = hmm_model.infer_sequence(low_seq)
    assert result["current_label"] in ("Not Trying", "Unclear")

    # posterior sums to ~1
    assert abs(sum(result["current_posterior"]) - 1.0) < 1e-3
    hmm_model._cache.clear()


def test_hmm_tracks_state_change(tmp_path, monkeypatch):
    """A trajectory moving from Unclear-like to Confident-like observations
    should end in a higher state than it started - the temporal point of the HMM."""
    import app.hmm.model as hmm_model

    monkeypatch.setattr(hmm_model, "HMM_MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(hmm_model, "HMM_MAPPING_PATH", tmp_path / "map.json")
    hmm_model._cache.clear()

    ds = generate_dataset(n_students=60, seed=4)
    hmm_model.train_hmm(ds, n_iter=50)

    seq = [STATE_PROFILES[1], STATE_PROFILES[1], STATE_PROFILES[2],
           STATE_PROFILES[3], STATE_PROFILES[4], STATE_PROFILES[4]]
    result = hmm_model.infer_sequence(seq)
    assert result["states"][-1] > result["states"][0]
    hmm_model._cache.clear()
