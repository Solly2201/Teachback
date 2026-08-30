"""Lecture lifecycle: create -> publish -> update -> delete/archive.

The rule these tests exist to protect: a teacher tidying their lecture list
must never destroy student learning records. A lecture nobody has used is
deleted outright; a lecture with student history is archived, disappearing
from the active lists while every session, observation, quiz attempt and
activity completion stays intact and readable.
"""
import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.hmm.model import hmm_available
from app.main import app
from app.models import (ActivityCompletion, Lecture, Observation, Quiz,
                        QuizAttempt, TeachSession, Topic)

client = TestClient(app)

MATERIAL = """# Queues

## Enqueue

Enqueue adds a new element at the back of the queue.
The queue keeps the order in which elements arrived.

## Dequeue

Dequeue removes the element at the front of the queue.
The element that waited longest is served first.
"""


def _subject(name="Python Programming"):
    teachers = client.get("/api/teachers").json()
    return next(s for t in teachers for s in t["subjects"] if s["name"] == name)


def _student():
    students = client.get("/api/students").json()
    return students[0]


def _make_lecture(title, publish=False):
    lec = client.post("/api/lectures", json={
        "subject_id": _subject()["id"], "title": title, "material_text": MATERIAL,
    }).json()
    if publish:
        published = client.post(f"/api/lectures/{lec['id']}/publish").json()
        return client.get(f"/api/lectures/{lec['id']}").json(), published["topic"]
    return lec, None


def _run_session(topic_id, student_id):
    start = client.post("/api/sessions/start",
                        json={"student_id": student_id, "topic_id": topic_id}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond",
                json={"text": "Enqueue puts a new item at the back and the queue keeps the order."})
    fin = client.post(f"/api/sessions/{sid}/finish",
                      json={"attention": 7, "confidence": 6, "difficulty": 4,
                            "summary": "Items join at the back and leave from the front."})
    return sid, fin.json()


# ------------------------------------------------------------- create/edit

def test_material_that_is_too_short_is_rejected_clearly():
    r = client.post("/api/lectures", json={
        "subject_id": _subject()["id"], "title": "Nothing", "material_text": "Queues."})
    assert r.status_code == 400
    assert "short" in r.json()["detail"].lower()


def test_missing_subject_and_lecture_are_404():
    assert client.post("/api/lectures", json={
        "subject_id": 999999, "title": "X", "material_text": MATERIAL}).status_code == 404
    assert client.get("/api/lectures/999999").status_code == 404
    assert client.delete("/api/lectures/999999").status_code == 404
    assert client.post("/api/lectures/999999/publish").status_code == 404


def test_publish_requires_a_named_concept_with_a_meaning():
    lec, _ = _make_lecture("Queues draft only")
    client.put(f"/api/lectures/{lec['id']}", json={"concepts": []})
    r = client.post(f"/api/lectures/{lec['id']}/publish")
    assert r.status_code == 400 and "concept" in r.json()["detail"].lower()

    client.put(f"/api/lectures/{lec['id']}",
               json={"concepts": [{"name": "Enqueue", "description": ""}]})
    r = client.post(f"/api/lectures/{lec['id']}/publish")
    assert r.status_code == 400 and "meaning" in r.json()["detail"].lower()


def test_republishing_updates_the_same_topic():
    lec, topic = _make_lecture("Queues republish", publish=True)
    client.put(f"/api/lectures/{lec['id']}", json={
        "concepts": [{"name": "Enqueue", "description": "Enqueue adds an element at the back."}]})
    again = client.post(f"/api/lectures/{lec['id']}/publish").json()
    assert again["topic"]["id"] == topic["id"]


def test_publishing_never_overwrites_another_lectures_topic():
    """One lecture owns one published topic.

    If two lecture rows ever pointed at the same topic, publishing one would
    silently replace the other's concepts. Publishing detects that and gives
    the second lecture its own topic instead.
    """
    first, topic = _make_lecture("Queues owner", publish=True)
    second, _ = _make_lecture("Queues intruder")
    db = SessionLocal()
    try:  # force the dangerous state directly in the database
        row = db.get(Lecture, second["id"])
        row.topic_id = topic["id"]
        db.commit()
    finally:
        db.close()

    published = client.post(f"/api/lectures/{second['id']}/publish").json()
    assert published["topic"]["id"] != topic["id"]
    survivor = client.get(f"/api/topics/{topic['id']}").json()
    assert survivor["name"] == "Queues owner"
    assert survivor["concepts"], "the first lecture's published structure was destroyed"


# ------------------------------------------------------- delete (no history)

def test_delete_unused_draft_lecture_removes_it():
    lec, _ = _make_lecture("Queues unused draft")
    preview = client.get(f"/api/lectures/{lec['id']}/delete-preview").json()
    assert preview["mode"] == "delete" and preview["history"]["total"] == 0

    r = client.delete(f"/api/lectures/{lec['id']}").json()
    assert r["mode"] == "deleted"
    assert client.get(f"/api/lectures/{lec['id']}").status_code == 404
    listed = client.get(f"/api/lectures?subject_id={_subject()['id']}").json()
    assert lec["id"] not in [x["id"] for x in listed]


def test_delete_published_lecture_without_sessions_removes_its_topic():
    lec, topic = _make_lecture("Queues published unused", publish=True)
    r = client.delete(f"/api/lectures/{lec['id']}").json()
    assert r["mode"] == "deleted"
    assert client.get(f"/api/topics/{topic['id']}").status_code == 404
    db = SessionLocal()
    try:  # no orphaned owned rows left behind
        assert db.get(Topic, topic["id"]) is None
        assert db.query(Quiz).filter(Quiz.topic_id == topic["id"]).first() is None
    finally:
        db.close()


# ---------------------------------------------------- archive (with history)

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_lecture_with_student_history_is_archived_not_erased():
    lec, topic = _make_lecture("Queues with history", publish=True)
    student = _student()
    session_id, result = _run_session(topic["id"], student["id"])

    quiz = client.get(f"/api/topics/{topic['id']}/quiz").json()
    if quiz.get("available"):
        answers = [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]
        client.post(f"/api/quiz/{quiz['quiz_id']}/submit",
                    json={"student_id": student["id"], "session_id": session_id,
                          "answers": answers})
    client.post("/api/activities/complete", json={
        "student_id": student["id"], "topic_id": topic["id"],
        "title": "Practice queues", "kind": "practice", "answer": "Done."})

    preview = client.get(f"/api/lectures/{lec['id']}/delete-preview").json()
    assert preview["mode"] == "archive"
    assert preview["history"]["sessions"] >= 1
    assert "archived" in preview["message"].lower()

    deleted = client.delete(f"/api/lectures/{lec['id']}").json()
    assert deleted["mode"] == "archived"

    # --- the lecture is gone from the active surfaces ---
    active = client.get(f"/api/lectures?subject_id={_subject()['id']}").json()
    assert lec["id"] not in [x["id"] for x in active]
    topics = client.get(f"/api/topics?subject_id={_subject()['id']}").json()
    assert topic["id"] not in [t["id"] for t in topics]
    overview = client.get(f"/api/teacher/overview?subject_id={_subject()['id']}").json()
    assert topic["id"] not in [t["id"] for t in overview["topic_stats"]]
    assert all(o["topic_name"] != topic["name"] for o in overview["recent_interactions"])

    # --- but every student record survives, intact and readable ---
    assert client.get(f"/api/lectures/{lec['id']}").json()["archived"] is True
    assert client.get(f"/api/topics/{topic['id']}").status_code == 200
    db = SessionLocal()
    try:
        session = db.get(TeachSession, session_id)
        assert session is not None and session.completed
        assert session.summary_text
        assert db.query(Observation).filter(Observation.session_id == session_id).count() == 1
        assert db.query(ActivityCompletion).filter(
            ActivityCompletion.topic_id == topic["id"]).count() >= 1
        for attempt in db.query(QuizAttempt).all():
            assert db.get(Quiz, attempt.quiz_id) is not None, "orphaned quiz attempt"
        for obs in db.query(Observation).all():
            assert obs.topic_id is None or db.get(Topic, obs.topic_id) is not None
        for ts in db.query(TeachSession).all():
            assert db.get(Topic, ts.topic_id) is not None, "orphaned session"
    finally:
        db.close()

    progress = client.get(f"/api/students/{student['id']}/progress").json()
    assert any(o["topic_name"] == topic["name"] for o in progress["timeline"])

    # --- an archived lecture is read-only and cannot host new sessions ---
    assert client.post("/api/sessions/start", json={
        "student_id": student["id"], "topic_id": topic["id"]}).status_code == 400
    assert client.put(f"/api/lectures/{lec['id']}", json={"title": "x"}).status_code == 400
    assert client.post(f"/api/lectures/{lec['id']}/publish").status_code == 400

    # --- restore brings it back without having lost anything ---
    restored = client.post(f"/api/lectures/{lec['id']}/restore").json()
    assert restored["archived"] is False
    assert topic["id"] in [t["id"] for t in
                           client.get(f"/api/topics?subject_id={_subject()['id']}").json()]
    assert client.post("/api/sessions/start", json={
        "student_id": student["id"], "topic_id": topic["id"]}).status_code == 200
    client.delete(f"/api/lectures/{lec['id']}")  # leave the demo data tidy


def test_archived_lectures_are_listable_on_request():
    lec, _ = _make_lecture("Queues archived listing", publish=True)
    db = SessionLocal()
    try:
        from datetime import datetime
        row = db.get(Lecture, lec["id"])
        row.archived_at = datetime.utcnow()
        row.status = "archived"
        db.get(Topic, row.topic_id).archived_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()
    sid = _subject()["id"]
    assert lec["id"] not in [x["id"] for x in client.get(f"/api/lectures?subject_id={sid}").json()]
    archived = client.get(f"/api/lectures?subject_id={sid}&include_archived=true").json()
    entry = next(x for x in archived if x["id"] == lec["id"])
    assert entry["archived"] is True and entry["status"] == "archived"
    client.post(f"/api/lectures/{lec['id']}/restore")
    client.delete(f"/api/lectures/{lec['id']}")


def test_restore_rejects_a_lecture_that_is_not_archived():
    lec, _ = _make_lecture("Queues not archived")
    r = client.post(f"/api/lectures/{lec['id']}/restore")
    assert r.status_code == 400
    client.delete(f"/api/lectures/{lec['id']}")


def test_subject_isolation_survives_deletion():
    """Deleting a Python lecture must not touch the Neural Networks subject."""
    nn_id = _subject("Neural Networks")["id"]
    before = {t["id"] for t in client.get(f"/api/topics?subject_id={nn_id}").json()}
    lec, _ = _make_lecture("Queues isolation check", publish=True)
    client.delete(f"/api/lectures/{lec['id']}")
    after = {t["id"] for t in client.get(f"/api/topics?subject_id={nn_id}").json()}
    assert before == after
