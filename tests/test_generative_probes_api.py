"""The generative-probe path through the real API, with every LLM faked.

The invariants under test are the ones the research design depends on:
with the flags off nothing changes and no provider is touched; with them on,
only the WORDING of a follow-up changes — the plan, the NLP evidence and the
concept outcomes are identical; and every failure degrades silently to the
deterministic v1 question.
"""
import json

import pytest
from fastapi.testclient import TestClient

import app.llm.providers as providers_mod
import app.probe.generate as generate_mod
from app.llm.errors import LLMUnavailable
from app.llm.prompts import PROMPT_VERSION
from app.main import app

from llm_fakes import FakeService

client = TestClient(app)

PARTIAL_ANSWER = "It has something to do with how wrong the prediction is."
STRONG_ANSWER = (
    "The loss function measures how wrong the network's prediction is compared to "
    "the true label, and training tries to make that error smaller."
)


def start_session():
    student = client.get("/api/students").json()[0]
    topic = client.get("/api/topics").json()[0]
    r = client.post("/api/sessions/start",
                    json={"student_id": student["id"], "topic_id": topic["id"]})
    assert r.status_code == 200
    return r.json()


def block_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to make a real HTTP call to an LLM provider")
    monkeypatch.setattr(providers_mod.httpx, "post", refuse)


def enable_flags(monkeypatch):
    monkeypatch.setenv("TEACHBACK_LLM_ENABLED", "true")
    monkeypatch.setenv("TEACHBACK_GENERATIVE_PROBES", "true")


# --- flags off: v1 exactly, zero LLM calls ----------------------------------

def test_disabled_by_default_makes_zero_llm_calls(monkeypatch):
    block_network(monkeypatch)
    constructed = []
    monkeypatch.setattr(generate_mod, "LLMService",
                        lambda *a, **k: constructed.append(1))
    start = start_session()
    r = client.post(f"/api/sessions/{start['session_id']}/respond",
                    json={"text": PARTIAL_ANSWER})
    assert r.status_code == 200
    followup = r.json()["followup"]
    assert followup and "generated" not in followup
    assert constructed == []  # the service was never even constructed


def test_llm_enabled_but_probes_disabled_stays_v1(monkeypatch):
    block_network(monkeypatch)
    monkeypatch.setenv("TEACHBACK_LLM_ENABLED", "true")  # probes flag still false
    start = start_session()
    r = client.post(f"/api/sessions/{start['session_id']}/respond",
                    json={"text": PARTIAL_ANSWER})
    assert "generated" not in (r.json()["followup"] or {})


# --- flags on: wording swapped, everything else identical -------------------

def test_generated_probe_replaces_wording_with_audit_metadata(monkeypatch):
    block_network(monkeypatch)
    enable_flags(monkeypatch)
    fake = FakeService(question="Fake wording: how would you explain that idea to a friend?")
    monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: fake)

    start = start_session()
    r = client.post(f"/api/sessions/{start['session_id']}/respond",
                    json={"text": PARTIAL_ANSWER})
    followup = r.json()["followup"]
    assert followup["text"] == fake.question
    meta = followup["generated"]
    assert meta["provider_used"] == "gemini"
    assert meta["fallback_used"] is False
    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["target_type"] in ("concept", "relationship", "misconception")
    assert meta["target_id"] is not None
    assert meta["action"]
    assert meta["difficulty"] in ("easy", "standard")
    assert meta["grounding_ids"]
    assert meta["reason"]
    # the decision the fake service received targeted the same thing
    decision = fake.calls[0]["decision"]
    assert decision["target_type"] == meta["target_type"]
    assert decision["target_id"] == meta["target_id"]


def test_generated_probe_context_is_targeted_teacher_material(monkeypatch):
    block_network(monkeypatch)
    enable_flags(monkeypatch)
    fake = FakeService()
    monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: fake)

    start = start_session()
    client.post(f"/api/sessions/{start['session_id']}/respond", json={"text": PARTIAL_ANSWER})
    call = fake.calls[0]
    items = call["items"]
    assert items, "the LLM must receive teacher-grounded context"
    tid = str(call["decision"]["target_id"])
    # every item is either the target's own material or a concept explanation
    # anchoring it — never a dump of the whole topic
    topic = client.get(f"/api/topics/{start['topic']['id']}").json()
    total_material = (len(topic["concepts"]) * 3 + len(topic["misconceptions"]) * 3
                      + len(topic["relationships"]) * 3)
    assert len(items) < total_material
    assert any(tid in i["id"] for i in items)
    for i in items:
        assert set(i) == {"id", "kind", "text"}


def test_generated_metadata_never_copies_the_student_answer(monkeypatch):
    block_network(monkeypatch)
    enable_flags(monkeypatch)
    fake = FakeService()
    monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: fake)

    start = start_session()
    distinctive = "my very distinctive personal answer about mislabeled gradients"
    r = client.post(f"/api/sessions/{start['session_id']}/respond", json={"text": distinctive})
    meta = r.json()["followup"]["generated"]
    assert distinctive not in json.dumps(meta)


def test_next_exchange_stores_generated_question_as_its_prompt(monkeypatch):
    block_network(monkeypatch)
    enable_flags(monkeypatch)
    fake = FakeService(question="Generated: what would you tell a friend this means?")
    monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: fake)

    start = start_session()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": PARTIAL_ANSWER})
    client.post(f"/api/sessions/{sid}/respond", json={"text": STRONG_ANSWER})

    # the teacher's evidence view shows the generated wording as the question
    # asked, plus the audit metadata on the exchange that produced it
    subject_id = start["topic"]["subject_id"]
    evidence = client.get(f"/api/teacher/sessions/{sid}/evidence",
                          params={"subject_id": subject_id})
    assert evidence.status_code == 200
    rows = evidence.json()["responses"]
    assert rows[0]["generated_probe"]["provider_used"] == "gemini"
    assert rows[1]["question"] == fake.question
    assert rows[1]["generated_probe"] is None or rows[1]["generated_probe"]  # key present


def test_llm_failure_keeps_the_deterministic_question(monkeypatch):
    block_network(monkeypatch)
    enable_flags(monkeypatch)
    failing = FakeService(fail=LLMUnavailable([{"provider": "gemini", "error_kind": "LLMTransientError",
                                               "status_code": 429}]))
    monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: failing)

    baseline = start_session()
    rb = client.post(f"/api/sessions/{baseline['session_id']}/respond",
                     json={"text": PARTIAL_ANSWER})
    with_llm_down = rb.json()["followup"]

    monkeypatch.setenv("TEACHBACK_GENERATIVE_PROBES", "false")
    v1 = start_session()
    rv = client.post(f"/api/sessions/{v1['session_id']}/respond",
                     json={"text": PARTIAL_ANSWER})
    v1_followup = rv.json()["followup"]

    assert failing.calls, "the LLM path was attempted"
    assert with_llm_down["text"] == v1_followup["text"]
    assert "generated" not in with_llm_down


def test_generated_wording_changes_no_evidence_or_outcome(monkeypatch):
    """Asking a generated question must not count as student evidence: with
    identical student answers, flag on and flag off produce identical concept
    evidence, statuses and progress — only the question wording differs."""
    block_network(monkeypatch)

    def run_session(enabled: bool):
        if enabled:
            enable_flags(monkeypatch)
            fake = FakeService(question="Totally different generated wording, right?")
            monkeypatch.setattr(generate_mod, "LLMService", lambda *a, **k: fake)
        else:
            monkeypatch.setenv("TEACHBACK_LLM_ENABLED", "false")
            monkeypatch.setenv("TEACHBACK_GENERATIVE_PROBES", "false")
        start = start_session()
        sid = start["session_id"]
        outcomes = []
        for text in (PARTIAL_ANSWER, STRONG_ANSWER, PARTIAL_ANSWER):
            r = client.post(f"/api/sessions/{sid}/respond", json={"text": text}).json()
            outcomes.append({"analysis": r["analysis"], "timeline": r["timeline"],
                             "concept_no": r["concept_no"]})
            if r["awaiting_self_report"]:
                break
        return outcomes

    with_generation = run_session(True)
    v1_only = run_session(False)
    assert with_generation == v1_only


def test_no_provider_keys_configured_is_not_an_error(monkeypatch):
    # flags on but neither key present: the service finds no providers,
    # raises LLMUnavailable internally, and the student sees a v1 question
    block_network(monkeypatch)
    enable_flags(monkeypatch)  # conftest already blanked both keys
    start = start_session()
    r = client.post(f"/api/sessions/{start['session_id']}/respond",
                    json={"text": PARTIAL_ANSWER})
    assert r.status_code == 200
    assert "generated" not in (r.json()["followup"] or {})
