"""Validation of the preserved HMM, and of what we are allowed to say about it.

The model itself is untouched by this pass (tests/test_hmm_integrity.py pins
its SHA256). These tests check the properties the rest of the system relies
on, the behaviour at the edges (one session, many sessions, malformed input),
and the honesty of the wording: the posterior is the model's confidence in a
learning CONDITION, never "the probability the student understands".
"""
import pytest

from app.hmm.model import hmm_available, infer_sequence, validate_model
from app.states import (STATE_NAMES, STATE_PROFILES, STATE_STUDENT_DESCRIPTIONS,
                        STATE_STUDENT_NAMES)

pytestmark = pytest.mark.skipif(not hmm_available(), reason="HMM not trained")


def _profile(state_index, jitter=0.0):
    return [min(1.0, max(0.0, v + jitter)) for v in STATE_PROFILES[state_index]]


# ------------------------------------------------------------- model sanity

def test_model_passes_every_structural_check():
    report = validate_model()
    assert report["ok"], report["problems"]
    assert report["n_states"] == 5
    assert report["n_features"] == 8


def test_transition_matrix_rows_are_distributions():
    report = validate_model()
    for row_sum in report["transition_row_sums"]:
        assert row_sum == pytest.approx(1.0, abs=1e-6)


def test_emission_means_stay_in_the_feature_range_and_variance_is_positive():
    report = validate_model()
    assert report["min_variance"] > 0
    assert not any("means" in p for p in report["problems"])


def test_canonical_mapping_is_a_bijection_and_close_to_its_profiles():
    report = validate_model()
    assert sorted(report["mapping"]) == sorted(STATE_NAMES)
    assert len(set(report["mapping"].values())) == 5
    # each learned state sits near the canonical profile it was matched to;
    # a large distance would mean the label is not warranted by the model
    assert report["max_mapping_distance"] < 0.35, report["mapping_distances"]


# ---------------------------------------------------------------- inference

def test_inference_with_a_single_session():
    result = infer_sequence([_profile(3)])
    assert len(result["states"]) == 1
    assert result["current_label"] in STATE_NAMES
    assert result["current_state"] == result["states"][-1]
    assert sum(result["current_posterior"]) == pytest.approx(1.0, abs=1e-3)


def test_inference_with_many_sessions_uses_the_whole_history():
    """Viterbi decodes the full sequence, so earlier sessions can be re-read
    in the light of later ones — the reason an HMM is used at all."""
    sequence = [_profile(1), _profile(1), _profile(2), _profile(3), _profile(4)]
    result = infer_sequence(sequence)
    assert len(result["states"]) == len(sequence)
    assert len(result["labels"]) == len(sequence)
    # a clear improvement trajectory should not end below where it started
    assert result["states"][-1] >= result["states"][0]
    # decoding is deterministic
    assert infer_sequence(sequence)["states"] == result["states"]


def test_prefix_of_a_sequence_is_not_required_to_match_its_decoding():
    """A later session may change how an earlier one reads; that is expected
    and must not crash or produce out-of-range states."""
    full = infer_sequence([_profile(0), _profile(1), _profile(4), _profile(4)])
    assert all(0 <= s < 5 for s in full["states"])


def test_posterior_is_a_distribution_over_the_five_states():
    result = infer_sequence([_profile(4), _profile(4)])
    assert len(result["current_posterior"]) == 5
    assert all(0.0 <= p <= 1.0 for p in result["current_posterior"])
    assert sum(result["current_posterior"]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("bad", [[], [[]], [[0.1, 0.2]], [[0.1] * 9]])
def test_malformed_sequences_are_rejected_loudly(bad):
    with pytest.raises(ValueError):
        infer_sequence(bad)


def test_non_finite_observations_are_rejected():
    with pytest.raises(ValueError):
        infer_sequence([[float("nan")] * 8])


# ----------------------------------------------------------- what we claim

def test_student_facing_wording_describes_evidence_not_motivation():
    assert len(STATE_STUDENT_NAMES) == len(STATE_NAMES) == 5
    assert len(STATE_STUDENT_DESCRIPTIONS) == 5
    # the internal state model is unchanged...
    assert STATE_NAMES[0] == "Not Trying"
    # ...but a student is never told what the system cannot observe
    joined = " ".join(STATE_STUDENT_NAMES + STATE_STUDENT_DESCRIPTIONS).lower()
    for judgement in ("not trying", "lazy", "careless", "gave up", "didn't try"):
        assert judgement not in joined
    assert "evidence" in STATE_STUDENT_NAMES[0].lower()


def test_meta_endpoint_exposes_validation_and_student_wording():
    from fastapi.testclient import TestClient

    from app.main import app

    meta = TestClient(app).get("/api/meta/states").json()
    assert meta["state_names"] == STATE_NAMES
    assert meta["state_student_names"] == STATE_STUDENT_NAMES
    assert meta["hmm_validation"]["ok"] is True
