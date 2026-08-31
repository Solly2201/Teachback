"""Regression tests for Problem A: teacher/subject scoping.

The Python subject dashboard must never show Neural Networks data (and vice
versa). Scoping must happen in the backend queries, not by frontend filtering.
"""
import pytest
from fastapi.testclient import TestClient

from app.hmm.model import hmm_available
from app.main import app

client = TestClient(app)


def _subject_ids():
    teachers = client.get("/api/teachers").json()
    subjects = {s["name"]: s["id"] for t in teachers for s in t["subjects"]}
    return subjects["Neural Networks"], subjects["Python Programming"]


def _topic_names(subject_id):
    return {t["name"] for t in client.get(f"/api/topics?subject_id={subject_id}").json()}


def test_overview_topic_stats_scoped():
    nn_id, py_id = _subject_ids()
    nn_names = _topic_names(nn_id)
    py_names = _topic_names(py_id)
    assert nn_names and py_names and not (nn_names & py_names)

    nn = client.get(f"/api/teacher/overview?subject_id={nn_id}").json()
    py = client.get(f"/api/teacher/overview?subject_id={py_id}").json()

    assert {t["name"] for t in nn["topic_stats"]} <= nn_names
    assert {t["name"] for t in py["topic_stats"]} <= py_names
    # the specific reported bug: NN topics visible under the Python subject
    assert not any(t["name"] in nn_names for t in py["topic_stats"])


def test_overview_recent_interactions_scoped():
    nn_id, py_id = _subject_ids()
    nn_names = _topic_names(nn_id)
    py_names = _topic_names(py_id)
    py = client.get(f"/api/teacher/overview?subject_id={py_id}").json()
    nn = client.get(f"/api/teacher/overview?subject_id={nn_id}").json()
    assert all(o["topic_name"] in py_names for o in py["recent_interactions"])
    assert all(o["topic_name"] in nn_names for o in nn["recent_interactions"])


def test_overview_misconceptions_scoped():
    """Misconception aggregates only come from the subject's own topics."""
    nn_id, py_id = _subject_ids()
    nn_topics = client.get(f"/api/topics?subject_id={nn_id}").json()
    nn_miscons = set()
    for t in nn_topics:
        detail = client.get(f"/api/topics/{t['id']}").json()
        nn_miscons |= {m["name"] for m in detail["misconceptions"]}
    py = client.get(f"/api/teacher/overview?subject_id={py_id}").json()
    assert not any(m["name"] in nn_miscons for m in py["common_misconceptions"])


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_live_session_appears_only_in_its_subject():
    """A completed session on a Python topic shows up on the Python dashboard
    and never on the Neural Networks dashboard."""
    nn_id, py_id = _subject_ids()
    py_topics = client.get(f"/api/topics?subject_id={py_id}").json()
    assert py_topics, "Python subject has no topics seeded"
    topic_id = py_topics[0]["id"]
    students = client.get("/api/students").json()
    student = next(s for s in students if s["name"] == "Shreshtha Bindal")

    before_nn = client.get(f"/api/teacher/overview?subject_id={nn_id}").json()
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic_id}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond",
                json={"text": "Strings are basically text stored between quotes."})
    client.post(f"/api/sessions/{sid}/finish",
                json={"attention": 7, "confidence": 6, "difficulty": 4})

    after_py = client.get(f"/api/teacher/overview?subject_id={py_id}").json()
    after_nn = client.get(f"/api/teacher/overview?subject_id={nn_id}").json()

    py_names = _topic_names(py_id)
    assert any(o["topic_name"] in py_names and o["source"] == "live"
               for o in after_py["recent_interactions"])
    # the NN dashboard is untouched by the Python session
    assert after_nn["live_session_count"] == before_nn["live_session_count"]
    assert all(o["topic_name"] in _topic_names(nn_id) for o in after_nn["recent_interactions"])


def test_a_topic_cannot_be_created_outside_a_subject():
    """A subject-less topic is invisible to every subject-scoped list — the
    only way the UI reaches topics — yet still reachable by id and startable,
    so it escapes subject isolation entirely. verify_persistence.py found six
    of them left behind by unvalidated requests."""
    payload = {"name": "Homeless topic", "description": "d",
               "reference_explanation": "ref", "concepts": [], "relationships": [],
               "misconceptions": [], "activities": []}
    assert client.post("/api/topics", json=payload).status_code == 400
    assert client.post("/api/topics", json={**payload, "subject_id": 999999}
                       ).status_code == 404

    subject_id = client.get("/api/teachers").json()[0]["subjects"][0]["id"]
    created = client.post("/api/topics", json={**payload, "subject_id": subject_id})
    assert created.status_code == 200
    topic_id = created.json()["id"]
    # and it must actually appear in that subject's list, not just be accepted
    listed = client.get(f"/api/topics?subject_id={subject_id}").json()
    assert any(t["id"] == topic_id for t in listed)


def test_an_edit_cannot_move_a_topic_out_of_every_subject():
    subject_id = client.get("/api/teachers").json()[0]["subjects"][0]["id"]
    payload = {"name": "Kept topic", "subject_id": subject_id, "description": "d",
               "reference_explanation": "ref", "concepts": [], "relationships": [],
               "misconceptions": [], "activities": []}
    topic_id = client.post("/api/topics", json=payload).json()["id"]
    # omitting subject_id on an edit keeps the existing one rather than clearing it
    assert client.put(f"/api/topics/{topic_id}",
                      json={**payload, "subject_id": None}).status_code == 200
    assert client.get(f"/api/topics/{topic_id}").json()["id"] == topic_id
    listed = client.get(f"/api/topics?subject_id={subject_id}").json()
    assert any(t["id"] == topic_id for t in listed)


# ---------------------------------------------------------------------------
# the lecture preparation step is faculty-facing too
# ---------------------------------------------------------------------------

# Material for a Python lecture that talks about error, weights and updates —
# the vocabulary the Neural Networks misconceptions are written in. Before the
# catalog was scoped, preparing this offered Prof. Krishnan's authored
# misconceptions to Prof. Rao, ready to publish into a Python topic.
CROSS_SUBJECT_MATERIAL = """# Model Training Loops in Python

## The training loop

A training loop repeats over the dataset many times.
Each pass computes the error between the prediction and the target.
The weights are then updated so the loss goes down.

## Learning rate

The learning rate controls how big each update step is.
A rate that is too large makes the loss jump around instead of settling.
"""


def _misconception_names(subject_id):
    """Every misconception authored under a subject's topics."""
    names = set()
    for t in client.get(f"/api/topics?subject_id={subject_id}&include_archived=true").json():
        names |= {m["name"] for m in client.get(f"/api/topics/{t['id']}").json()["misconceptions"]}
    return names


def test_lecture_preparation_never_suggests_another_subjects_misconceptions():
    nn_id, py_id = _subject_ids()
    nn_miscons = _misconception_names(nn_id)
    assert nn_miscons, "the Neural Networks topics should have authored misconceptions"

    lec = client.post("/api/lectures", json={
        "subject_id": py_id, "title": "Model Training Loops",
        "material_text": CROSS_SUBJECT_MATERIAL}).json()
    suggested = {m["name"] for m in lec["suggestions"]["misconception_suggestions"]}
    leaked = suggested & nn_miscons
    assert not leaked, f"another subject's misconceptions were suggested: {sorted(leaked)}"


def test_a_subjects_own_misconceptions_are_still_reusable():
    """Scoping must not silently disable the catalog for the subject that owns it."""
    nn_id, _ = _subject_ids()
    lec = client.post("/api/lectures", json={
        "subject_id": nn_id, "title": "Backpropagation recap",
        "material_text": CROSS_SUBJECT_MATERIAL}).json()
    suggested = {m["name"] for m in lec["suggestions"]["misconception_suggestions"]}
    assert suggested & _misconception_names(nn_id), (
        "the subject's own authored misconceptions should still be offered")
