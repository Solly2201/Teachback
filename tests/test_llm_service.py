"""Provider abstraction and failover behavior, entirely against fakes.

The scenarios mirror the failover contract in app/llm/service.py:
transient/quota/auth errors move to the fallback provider (once, immediately
— never a retry storm), malformed output is retried once on the same
provider, our own malformed requests are not retried anywhere, and when
everything fails LLMUnavailable tells the caller to stay deterministic.
"""
import pytest

from app.llm.errors import (LLMAuthConfigError, LLMOutputError, LLMRequestError,
                            LLMTransientError, LLMUnavailable, classify_http, sanitize)
from app.llm.service import LLMService
from app.llm.settings import llm_settings

from llm_fakes import FAKE_KEY, FakeProvider, fake_settings, probe_json

DECISION = {
    "action": "ASK_PROBE",
    "target_type": "concept",
    "target_id": 7,
    "target_name": "Gradient",
    "difficulty": "easy",
}
ITEMS = [{"id": "concept:7", "kind": "concept_explanation", "text": "Gradient: how the loss changes."},
         {"id": "concept_fact:7:0", "kind": "teacher_fact", "text": "The gradient points uphill."}]
ANSWER = "It tells us something about the loss."


def make_service(gemini_script, groq_script, **settings_overrides):
    gemini = FakeProvider("gemini", gemini_script)
    groq = FakeProvider("groq", groq_script)
    service = LLMService(settings=fake_settings(**settings_overrides),
                         providers=[gemini, groq])
    return service, gemini, groq


def good():
    return probe_json(DECISION, ["concept:7"])


# --- success and failover ---------------------------------------------------

def test_gemini_success_uses_gemini():
    service, gemini, groq = make_service([good()], [])
    probe, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "gemini"
    assert meta["fallback_used"] is False
    assert meta["model_used"] == "fake-gemini-model"
    assert meta["prompt_version"]
    assert "latency_ms" in meta
    assert gemini.calls == 1 and groq.calls == 0
    assert "?" in probe.question


def test_gemini_429_falls_over_to_groq():
    service, gemini, groq = make_service(
        [LLMTransientError("gemini", "quota/rate limit (HTTP 429)", 429)], [good()])
    probe, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"
    assert meta["fallback_used"] is True
    assert meta["model_used"] == "fake-groq-model"
    # no retry storm: the rate-limited provider is not hammered again
    assert gemini.calls == 1 and groq.calls == 1


def test_gemini_quota_exhausted_falls_over():
    err = classify_http("gemini", 403, '{"error": {"status": "RESOURCE_EXHAUSTED"}}')
    assert isinstance(err, LLMTransientError)
    service, gemini, groq = make_service([err], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq" and meta["fallback_used"] is True


def test_gemini_timeout_falls_over():
    service, gemini, groq = make_service(
        [LLMTransientError("gemini", "timed out after 15s")], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"
    assert gemini.calls == 1


def test_gemini_5xx_falls_over():
    err = classify_http("gemini", 503, "service unavailable")
    assert isinstance(err, LLMTransientError)
    service, _, _ = make_service([err], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"


def test_auth_error_still_tries_the_other_provider():
    # a bad Gemini key says nothing about the Groq key
    err = classify_http("gemini", 401, "API key not valid")
    assert isinstance(err, LLMAuthConfigError)
    service, gemini, groq = make_service([err], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"
    assert gemini.calls == 1 and groq.calls == 1


def test_malformed_request_is_not_retried_anywhere():
    # a payload WE built wrong would be rejected by both providers alike
    err = classify_http("gemini", 400, "invalid request payload")
    assert isinstance(err, LLMRequestError)
    service, gemini, groq = make_service([err], [good()])
    with pytest.raises(LLMRequestError):
        service.generate_probe(DECISION, ITEMS, ANSWER)
    assert gemini.calls == 1 and groq.calls == 0


# --- malformed output -------------------------------------------------------

def test_malformed_json_retries_once_then_falls_over():
    service, gemini, groq = make_service(["this is not json", "{broken"], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert gemini.calls == 2  # one retry, as configured (max_retries=1)
    assert meta["provider_used"] == "groq" and meta["fallback_used"] is True


def test_malformed_json_recovers_on_same_provider_retry():
    service, gemini, groq = make_service(["not json", good()], [])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert gemini.calls == 2 and groq.calls == 0
    assert meta["provider_used"] == "gemini" and meta["fallback_used"] is False


def test_schema_valid_but_retargeted_output_is_rejected():
    # the LLM may not redirect the probe at a different target...
    retargeted = probe_json(DECISION, ["concept:7"], target_id=99)
    # ...nor change the difficulty the controller chose
    harder = probe_json(DECISION, ["concept:7"], difficulty="standard")
    service, gemini, groq = make_service([retargeted, harder], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert gemini.calls == 2 and meta["provider_used"] == "groq"


def test_uncited_grounding_ids_are_rejected():
    ungrounded = probe_json(DECISION, ["concept:7", "lecture:everything"])
    service, gemini, _ = make_service([ungrounded, ungrounded], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"


def test_response_without_a_question_is_rejected():
    no_question = probe_json(DECISION, ["concept:7"],
                             question="A string is a sequence of characters.")
    service, _, _ = make_service([no_question, no_question], [good()])
    _, meta = service.generate_probe(DECISION, ITEMS, ANSWER)
    assert meta["provider_used"] == "groq"


# --- nothing available ------------------------------------------------------

def test_both_providers_unavailable_raises_llm_unavailable():
    service, gemini, groq = make_service(
        [LLMTransientError("gemini", "quota", 429)],
        [LLMTransientError("groq", "quota", 429)])
    with pytest.raises(LLMUnavailable) as exc:
        service.generate_probe(DECISION, ITEMS, ANSWER)
    kinds = [a["error_kind"] for a in exc.value.attempts]
    assert kinds == ["LLMTransientError", "LLMTransientError"]
    assert gemini.calls == 1 and groq.calls == 1


def test_disabled_service_makes_zero_provider_calls():
    service, gemini, groq = make_service([good()], [good()], enabled=False)
    with pytest.raises(LLMUnavailable):
        service.generate_probe(DECISION, ITEMS, ANSWER)
    assert gemini.calls == 0 and groq.calls == 0


def test_no_configured_providers_raises():
    service = LLMService(settings=fake_settings(), providers=[])
    with pytest.raises(LLMUnavailable):
        service.generate_probe(DECISION, ITEMS, ANSWER)


# --- settings / key handling ------------------------------------------------

def test_missing_gemini_key_leaves_groq_as_the_chain(monkeypatch):
    monkeypatch.setenv("TEACHBACK_LLM_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    order = [p.name for p in llm_settings().provider_order()]
    assert order == ["groq"]


def test_missing_both_keys_leaves_no_providers(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert llm_settings().provider_order() == []


def test_flags_default_off_in_tests():
    s = llm_settings()
    assert s.enabled is False and s.generative_probes is False


# --- secret hygiene ---------------------------------------------------------

def test_provider_errors_do_not_leak_api_keys(monkeypatch):
    secret = "sk-test-abcdefghijklmnopqrstuvwxyz123456"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    err = classify_http("gemini", 400, f"bad request; key={secret} rejected")
    assert secret not in err.message and secret not in str(err)
    assert secret not in sanitize(f"Authorization: Bearer {secret}")
    assert secret not in sanitize(f"x-goog-api-key: {secret}")


def test_error_attempt_records_hold_no_payloads():
    service, _, _ = make_service(
        [LLMTransientError("gemini", "quota", 429)],
        [LLMAuthConfigError("groq", "auth/model configuration rejected (HTTP 401)", 401)])
    with pytest.raises(LLMUnavailable) as exc:
        service.generate_probe(DECISION, ITEMS, ANSWER)
    for attempt in exc.value.attempts:
        assert set(attempt) == {"provider", "error_kind", "status_code", "message"}
        # the message is the short sanitized classification, never a payload
        assert len(attempt["message"]) < 120
