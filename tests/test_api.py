"""API integration tests against the real app (requires build_all.py to have run,
so the HMM artifacts exist; the app seeds its own SQLite DB on startup)."""
import pytest
from fastapi.testclient import TestClient

from app.hmm.model import hmm_available
from app.main import app

client = TestClient(app)

GOOD_ANSWER = (
    "The network makes a prediction and a loss function measures how wrong it is. "
    "The error is propagated backwards through the layers using the chain rule to compute "
    "the gradient of the loss with respect to each weight. An optimizer like gradient descent "
    "then updates the weights step by step, and repeating this many times minimises the loss."
)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_students_listed():
    r = client.get("/api/students")
    assert r.status_code == 200
    students = r.json()
    assert len(students) >= 8
    assert all("name" in s for s in students)


def test_topics_and_detail():
    r = client.get("/api/topics")
    assert r.status_code == 200
    topics = r.json()
    assert len(topics) >= 3
    detail = client.get(f"/api/topics/{topics[0]['id']}").json()
    assert detail["concepts"] and detail["misconceptions"] and detail["activities"]


def test_teacher_overview():
    r = client.get("/api/teacher/overview")
    assert r.status_code == 200
    data = r.json()
    assert sum(d["count"] for d in data["distribution"]) > 0
    assert len(data["topic_stats"]) >= 3


def test_topic_crud():
    payload = {
        "name": "Test Topic",
        "description": "d",
        "reference_explanation": "ref",
        "concepts": [{"name": "c1", "description": "concept one"}],
        "misconceptions": [{"name": "m1", "description": "wrong claim", "clarification": "right claim"}],
        "activities": [{"title": "a1", "description": "act", "target_state": "unclear"}],
    }
    r = client.post("/api/topics", json=payload)
    assert r.status_code == 200
    tid = r.json()["id"]
    payload["name"] = "Test Topic v2"
    r = client.put(f"/api/topics/{tid}", json=payload)
    assert r.status_code == 200
    assert r.json()["name"] == "Test Topic v2"


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained (run scripts/build_all.py)")
def test_full_teachback_flow():
    students = client.get("/api/students").json()
    topics = client.get("/api/topics").json()
    backprop = next(t for t in topics if t["name"] == "Backpropagation")

    r = client.post("/api/sessions/start", json={"student_id": students[0]["id"], "topic_id": backprop["id"]})
    assert r.status_code == 200
    start = r.json()
    sid = start["session_id"]
    # the session opens with a short concept question and a concept timeline
    assert start["prompt"]
    assert start["total_concepts"] == 5
    assert len(start["timeline"]) == 5
    assert start["timeline"][0]["status"] == "current"

    # answer each short question with a comprehensive explanation: coverage
    # accumulates and the guided session finishes without needing every question
    done = False
    for _ in range(14):
        r = client.post(f"/api/sessions/{sid}/respond", json={"text": GOOD_ANSWER})
        assert r.status_code == 200
        step = r.json()
        assert step["feedback"]
        if step["awaiting_self_report"]:
            done = True
            break
        assert step["followup"]["text"]
    assert done, "guided session did not terminate"

    r = client.post(f"/api/sessions/{sid}/finish", json={"attention": 8, "confidence": 7, "difficulty": 4})
    assert r.status_code == 200
    result = r.json()
    assert result["state"]["label"] in (
        "Not Trying", "Unclear", "Struggling but Trying", "Understanding", "Confident")
    assert result["recommendation"]["activity"]["title"]
    assert result["recommendation"]["why"]
    assert {c["status"] for c in result["concept_summary"]} <= {"covered", "partial", "unclear", "missing"}
    assert abs(sum(result["state"]["posterior"].values()) - 1.0) < 1e-2

    progress = client.get(f"/api/students/{students[0]['id']}/progress").json()
    assert any(o["source"] == "live" for o in progress["timeline"])
    assert all("evidence" in o for o in progress["timeline"])


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained (run scripts/build_all.py)")
def test_misconception_detour_and_resolution():
    students = client.get("/api/students").json()
    topics = client.get("/api/topics").json()
    backprop = next(t for t in topics if t["name"] == "Backpropagation")

    r = client.post("/api/sessions/start", json={"student_id": students[1]["id"], "topic_id": backprop["id"]})
    sid = r.json()["session_id"]

    # first answer states a known misconception -> the system should probe it
    r = client.post(f"/api/sessions/{sid}/respond", json={"text": (
        "The loss measures the error. Backpropagation and gradient descent are the same thing."
    )})
    step = r.json()
    assert step["misconception"] is not None
    assert step["misconception"]["clarification"]
    assert step["followup"]["kind"] == "misconception"

    # the corrected answer resolves it
    r = client.post(f"/api/sessions/{sid}/respond", json={"text": (
        "Backpropagation only computes the gradients, while gradient descent is the separate "
        "optimizer that uses those gradients to update the weights."
    )})
    step = r.json()
    assert step["resolved_misconception"] == "Backpropagation is the same as gradient descent"
