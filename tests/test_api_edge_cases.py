"""The API must fail safely and usefully, never with a stack trace.

A demo breaks when a stale tab posts to a deleted lecture, a double-click
submits twice, or someone opens a bookmarked URL for a record that is gone.
These check that every such case returns a sensible status and a message a
person could act on.
"""
import base64

import pytest
from fastapi.testclient import TestClient

from app.hmm.model import hmm_available
from app.main import app

client = TestClient(app)

MISSING = 999999
MATERIAL = """# Stacks

## Push

Push adds a new element on top of the stack.
The most recently added element sits at the top.

## Pop

Pop removes the element currently on top of the stack.
The last element added is the first one removed.
"""


def _subject():
    teachers = client.get("/api/teachers").json()
    return next(s for t in teachers for s in t["subjects"]
                if s["name"] == "Python Programming")


def _student():
    return client.get("/api/students").json()[0]


# ------------------------------------------------------------- unknown ids

@pytest.mark.parametrize("method,path", [
    ("get", f"/api/lectures/{MISSING}"),
    ("get", f"/api/lectures/{MISSING}/delete-preview"),
    ("get", f"/api/topics/{MISSING}"),
    ("get", f"/api/topics/{MISSING}/quiz"),
    ("get", f"/api/students/{MISSING}"),
    ("get", f"/api/students/{MISSING}/progress"),
    ("get", f"/api/activities/{MISSING}"),
    ("delete", f"/api/lectures/{MISSING}"),
    ("post", f"/api/lectures/{MISSING}/publish"),
    ("post", f"/api/lectures/{MISSING}/restore"),
])
def test_unknown_ids_return_404_not_a_stack_trace(method, path):
    r = getattr(client, method)(path)
    assert r.status_code == 404, f"{path} -> {r.status_code}"
    assert r.json().get("detail"), "a 404 must carry a message"


def test_unknown_ids_on_posted_bodies_return_404():
    assert client.post("/api/sessions/start",
                       json={"student_id": MISSING, "topic_id": 1}).status_code == 404
    assert client.post(f"/api/sessions/{MISSING}/respond",
                       json={"text": "hello"}).status_code == 404
    assert client.post(f"/api/quiz/{MISSING}/submit", json={
        "student_id": _student()["id"],
        "answers": [{"question_id": 1, "selected_index": 0}]}).status_code == 404
    assert client.post("/api/activities/complete", json={
        "student_id": MISSING, "title": "x"}).status_code == 404
    assert client.post("/api/activities/complete", json={
        "student_id": _student()["id"], "activity_id": MISSING}).status_code == 404


# --------------------------------------------------------- malformed input

@pytest.mark.parametrize("payload,expected", [
    ({"subject_id": MISSING, "title": "X", "material_text": MATERIAL}, 404),
    ({"subject_id": None, "title": "X", "material_text": MATERIAL}, 422),
    ({"title": "X", "material_text": MATERIAL}, 422),
])
def test_lecture_creation_rejects_bad_references(payload, expected):
    assert client.post("/api/lectures", json=payload).status_code == expected


def test_empty_and_trivial_material_is_refused_with_a_reason():
    sid = _subject()["id"]
    for material in ("", "   ", "Stacks."):
        r = client.post("/api/lectures", json={
            "subject_id": sid, "title": "Stacks", "material_text": material})
        assert r.status_code == 400
        assert r.json()["detail"]
    r = client.post("/api/lectures", json={
        "subject_id": sid, "title": "   ", "material_text": MATERIAL})
    assert r.status_code == 400


def test_upload_of_a_broken_or_unsupported_file_is_refused_cleanly():
    payload = base64.b64encode(b"not a pdf at all").decode()
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pdf", "content_base64": payload})
    assert r.status_code == 400 and r.json()["detail"]
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pptx", "content_base64": payload})
    assert r.status_code == 400
    r = client.post("/api/lectures/extract",
                    json={"filename": "deck.pdf", "content_base64": "!!!not base64!!!"})
    assert r.status_code == 400


def test_material_that_parses_to_nothing_still_creates_a_reviewable_draft():
    """Garbage in must not crash: the teacher gets an empty draft and is told
    the material could not be interpreted, rather than a 500."""
    sid = _subject()["id"]
    r = client.post("/api/lectures", json={
        "subject_id": sid, "title": "Nonsense",
        "material_text": ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
                          "eiusmod tempor incididunt ut labore et dolore magna aliqua ut "
                          "enim ad minim veniam quis nostrud exercitation ullamco")})
    assert r.status_code == 200
    lec = r.json()
    assert isinstance(lec["draft"].get("concepts"), list)
    assert client.post(f"/api/lectures/{lec['id']}/publish").status_code in (200, 400)
    client.delete(f"/api/lectures/{lec['id']}")


# ------------------------------------------------------------- lifecycle

def test_publishing_a_draft_with_no_usable_concepts_is_refused():
    sid = _subject()["id"]
    lec = client.post("/api/lectures", json={
        "subject_id": sid, "title": "Stacks empty", "material_text": MATERIAL}).json()
    client.put(f"/api/lectures/{lec['id']}", json={"concepts": []})
    r = client.post(f"/api/lectures/{lec['id']}/publish")
    assert r.status_code == 400 and "concept" in r.json()["detail"].lower()
    client.delete(f"/api/lectures/{lec['id']}")


def test_publishing_twice_is_idempotent_not_duplicated():
    sid = _subject()["id"]
    lec = client.post("/api/lectures", json={
        "subject_id": sid, "title": "Stacks twice", "material_text": MATERIAL}).json()
    first = client.post(f"/api/lectures/{lec['id']}/publish").json()
    second = client.post(f"/api/lectures/{lec['id']}/publish").json()
    assert first["topic"]["id"] == second["topic"]["id"]
    topics = client.get(f"/api/topics?subject_id={sid}").json()
    assert sum(1 for t in topics if t["name"] == "Stacks twice") == 1
    client.delete(f"/api/lectures/{lec['id']}")


def test_deleting_twice_is_safe():
    sid = _subject()["id"]
    lec = client.post("/api/lectures", json={
        "subject_id": sid, "title": "Stacks delete twice",
        "material_text": MATERIAL}).json()
    assert client.delete(f"/api/lectures/{lec['id']}").status_code == 200
    assert client.delete(f"/api/lectures/{lec['id']}").status_code == 404


# --------------------------------------------------------------- sessions

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_a_finished_session_refuses_further_input():
    topic = client.get("/api/topics").json()[0]
    student = _student()
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "text kept inside quotes"})
    first = client.post(f"/api/sessions/{sid}/finish",
                        json={"attention": 6, "confidence": 6, "difficulty": 5})
    assert first.status_code == 200
    # a double-click on Finish, and a stale tab still answering
    again = client.post(f"/api/sessions/{sid}/finish",
                        json={"attention": 6, "confidence": 6, "difficulty": 5})
    assert again.status_code == 400 and again.json()["detail"]
    late = client.post(f"/api/sessions/{sid}/respond", json={"text": "one more thought"})
    assert late.status_code == 400


def test_finishing_a_session_with_no_answers_is_refused():
    topic = client.get("/api/topics").json()[0]
    start = client.post("/api/sessions/start",
                        json={"student_id": _student()["id"],
                              "topic_id": topic["id"]}).json()
    r = client.post(f"/api/sessions/{start['session_id']}/finish",
                    json={"attention": 5, "confidence": 5, "difficulty": 5})
    assert r.status_code == 400 and "response" in r.json()["detail"].lower()


@pytest.mark.parametrize("body", [
    {"attention": 11, "confidence": 5, "difficulty": 5},
    {"attention": -1, "confidence": 5, "difficulty": 5},
    {"confidence": 5, "difficulty": 5},
])
def test_out_of_range_self_reports_are_rejected(body):
    topic = client.get("/api/topics").json()[0]
    start = client.post("/api/sessions/start",
                        json={"student_id": _student()["id"],
                              "topic_id": topic["id"]}).json()
    r = client.post(f"/api/sessions/{start['session_id']}/finish", json=body)
    assert r.status_code == 422


def test_empty_answer_text_is_rejected():
    topic = client.get("/api/topics").json()[0]
    start = client.post("/api/sessions/start",
                        json={"student_id": _student()["id"],
                              "topic_id": topic["id"]}).json()
    r = client.post(f"/api/sessions/{start['session_id']}/respond", json={"text": ""})
    assert r.status_code == 422


# ------------------------------------------------------------------- quiz

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_quiz_rejects_a_mismatched_session_and_topic():
    topics = client.get("/api/topics").json()
    with_quiz = [t for t in topics
                 if client.get(f"/api/topics/{t['id']}/quiz").json().get("available")]
    assert len(with_quiz) >= 2, "need two quizzed topics for this check"
    a, b = with_quiz[0], with_quiz[1]
    quiz_a = client.get(f"/api/topics/{a['id']}/quiz").json()
    student = _student()
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": b["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "some explanation here"})
    client.post(f"/api/sessions/{sid}/finish",
                json={"attention": 6, "confidence": 6, "difficulty": 5})
    answers = [{"question_id": q["id"], "selected_index": 0} for q in quiz_a["questions"]]
    r = client.post(f"/api/quiz/{quiz_a['quiz_id']}/submit",
                    json={"student_id": student["id"], "session_id": sid, "answers": answers})
    assert r.status_code == 400
    assert "lecture" in r.json()["detail"].lower()


def test_quiz_answer_index_is_validated():
    topics = client.get("/api/topics").json()
    quiz = next((client.get(f"/api/topics/{t['id']}/quiz").json() for t in topics
                 if client.get(f"/api/topics/{t['id']}/quiz").json().get("available")), None)
    assert quiz is not None
    r = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": _student()["id"],
        "answers": [{"question_id": quiz["questions"][0]["id"], "selected_index": 9}]})
    assert r.status_code == 422


def test_student_quiz_view_never_leaks_the_answers():
    topics = client.get("/api/topics").json()
    for t in topics:
        quiz = client.get(f"/api/topics/{t['id']}/quiz").json()
        if not quiz.get("available"):
            continue
        for q in quiz["questions"]:
            assert "correct_index" not in q
            assert "explanation" not in q
