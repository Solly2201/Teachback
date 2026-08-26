"""HMM artifact integrity regression tests.

The trained HMM (data/artifacts/hmm_model.joblib) and its state mapping are
preserved artifacts: nothing in the quality/architecture pass may retrain or
silently modify them. These tests pin the exact SHA256 hashes recorded before
the pass began, plus the 8-dimensional observation contract.
"""
import hashlib

import pytest

from app.config import HMM_MAPPING_PATH, HMM_MODEL_PATH
from app.hmm.model import hmm_available, load_hmm
from app.states import FEATURE_NAMES

# Recorded 2026-08-26, before the NLP/architecture quality pass.
EXPECTED_MODEL_SHA256 = "e854818f2ea315b78aabe43c0187b4c5a25d08b032a14c1027f7f2589b59c5b6"
EXPECTED_MAPPING_SHA256 = "b9eba277fed578360f9a53229d0184d0f05ca18385cabf4023fadb3baa8045bb"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_hmm_model_artifact_unchanged():
    assert _sha256(HMM_MODEL_PATH) == EXPECTED_MODEL_SHA256, \
        "hmm_model.joblib changed — the HMM must not be retrained or modified"


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_hmm_state_mapping_unchanged():
    assert _sha256(HMM_MAPPING_PATH) == EXPECTED_MAPPING_SHA256, \
        "hmm_state_mapping.json changed — the state mapping must be preserved"


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_observation_vector_is_8_dimensional():
    model, mapping = load_hmm()
    assert model.means_.shape == (5, 8)
    assert len(FEATURE_NAMES) == 8
    assert len(mapping["state_to_canonical"]) == 5
