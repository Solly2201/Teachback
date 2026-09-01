"""Recommendation → activity → completion flow, plus HMM-artifact regression.

Recommended activities must be real, openable and completable: a stored
Activity row (with content and a question) is preferred and referenced by id;
generic fallbacks carry their own content so they are still performable.
"""
import hashlib

from fastapi.testclient import TestClient

from app.config import HMM_MODEL_PATH
from app.main import app
from app.recommend.rules import GENERIC_ACTIVITIES, recommend
from app.states import FEATURE_NAMES

client = TestClient(app)

# The HMM artifact is pinned: no change to the app may retrain or replace it.
# Pinned again here (see tests/test_hmm_integrity.py) so a change to the
# recommendation flow cannot quietly move the model underneath it.
HMM_ARTIFACT_SHA256 = "e854818f2ea315b78aabe43c0187b4c5a25d08b032a14c1027f7f2589b59c5b6"


def _backprop_topic():
    topics = client.get("/api/topics").json()
    t = next(t for t in topics if t["name"] == "Backpropagation")
    return client.get(f"/api/topics/{t['id']}").json()


# 8. A recommendation references a real stored activity (by id) with content
def test_recommendation_references_real_activity():
    detail = _backprop_topic()
    rec = recommend(1, detail["activities"])  # state "unclear"
    activity = rec["activity"]
    assert activity["id"] is not None
    stored = next(a for a in detail["activities"] if a["id"] == activity["id"])
    assert stored["title"] == activity["title"]
    assert activity["content"] and activity["question"]


# 9-11. The recommended activity can be opened by id and its content renders
def test_recommended_activity_can_be_opened():
    detail = _backprop_topic()
    rec = recommend(1, detail["activities"])
    r = client.get(f"/api/activities/{rec['activity']['id']}")
    assert r.status_code == 200
    a = r.json()
    assert a["title"] == "Hiker-on-a-hill analogy"
    assert "foggy hill" in a["content"]
    assert a["question"]
    assert a["topic_name"] == "Backpropagation"


SEEDED_TOPICS = {"Backpropagation", "Overfitting and Regularization", "Hidden Markov Models"}


def test_every_seeded_activity_has_content_and_question():
    for t in client.get("/api/topics").json():
        if t["name"] not in SEEDED_TOPICS:
            continue  # teacher-created test topics may leave content empty
        detail = client.get(f"/api/topics/{t['id']}").json()
        for a in detail["activities"]:
            assert a["content"], f"{t['name']} / {a['title']} has no content"
            assert a["question"], f"{t['name']} / {a['title']} has no question"


def test_generic_fallback_activities_are_performable():
    rec = recommend(2)  # no topic activities -> generic fallback
    assert rec["activity"]["id"] is None
    for g in GENERIC_ACTIVITIES.values():
        assert g["content"] and g["question"]


# 12-13. A student can complete an activity and the completion is recorded
def test_student_can_complete_activity_and_it_is_recorded():
    detail = _backprop_topic()
    students = client.get("/api/students").json()
    student = students[0]
    activity_id = detail["activities"][0]["id"]

    r = client.post("/api/activities/complete", json={
        "student_id": student["id"], "activity_id": activity_id,
        "answer": "The slope represents how the loss changes when a weight changes.",
    })
    assert r.status_code == 200
    result = r.json()
    assert result["completed"] is True
    assert "You completed" in result["message"]

    progress = client.get(f"/api/students/{student['id']}/progress").json()
    assert any(c["id"] == result["id"] for c in progress["completions"])


def test_generic_activity_completion_recorded_by_title():
    students = client.get("/api/students").json()
    r = client.post("/api/activities/complete", json={
        "student_id": students[1]["id"], "activity_id": None,
        "title": "Quick warm-up question", "kind": "re_engagement",
        "answer": "It is about how networks learn.",
    })
    assert r.status_code == 200
    assert r.json()["completed"] is True


def test_unknown_activity_404():
    students = client.get("/api/students").json()
    assert client.get("/api/activities/999999").status_code == 404
    r = client.post("/api/activities/complete", json={
        "student_id": students[0]["id"], "activity_id": 999999, "answer": "x"})
    assert r.status_code == 404


# 16-17. Regression: HMM artifact unchanged, observation vector stays 8-dim
def test_hmm_artifact_unchanged():
    digest = hashlib.sha256(HMM_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == HMM_ARTIFACT_SHA256


def test_observation_vector_is_8_dimensional():
    assert len(FEATURE_NAMES) == 8


# 21. The named demo student for the faculty demo still exists
def test_shreshtha_bindal_demo_student_present():
    students = client.get("/api/students").json()
    s = next((x for x in students if x["name"] == "Shreshtha Bindal"), None)
    assert s is not None
    assert s["program"] == "B.Tech CE"
    assert s["roll_no"] == "B023"
