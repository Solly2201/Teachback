"""Topic lifecycle: create -> use -> delete/archive/restore.

Topic Management can now remove a topic, and the rule it must obey is the one
the lecture delete already obeys: a teacher tidying their topic list must
never destroy student learning records. A topic nobody has used is deleted
outright along with everything it owns; a topic with history is archived,
disappearing from the active list while every session, observation, quiz
attempt and activity completion stays intact and readable.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.hmm.model import hmm_available
from app.main import app
from app.models import (Activity, ActivityCompletion, Concept, ConceptRelationship,
                        Lecture, Misconception, Observation, Quiz, QuizAttempt,
                        QuizQuestion, TeachSession, Topic)

client = TestClient(app)

MATERIAL = """# Stacks

## Push

Push adds a new element to the top of the stack.
The stack keeps the order in which elements were added.

## Pop

Pop removes the element currently at the top of the stack.
The element added most recently is the first one removed.
"""


def _subject(name="Neural Networks"):
    teachers = client.get("/api/teachers").json()
    return next(s for t in teachers for s in t["subjects"] if s["name"] == name)


def _student():
    return client.get("/api/students").json()[0]


def _topic_payload(name, subject_id, **over):
    payload = {
        "name": name,
        "subject_id": subject_id,
        "description": "d",
        "reference_explanation": "ref",
        "concepts": [{"name": "c1", "description": "concept one",
                      "main_question": "What is c1?", "probe_question": "More on c1?"}],
        "relationships": [{"source": "c1", "label": "leads to", "target": "c2",
                           "description": "c1 leads to c2."}],
        "misconceptions": [{"name": "m1", "description": "wrong claim",
                            "clarification": "right claim"}],
        "activities": [{"title": "a1", "description": "act", "kind": "practice",
                        "target_state": "unclear", "content": "read this",
                        "question": "what did you notice?"}],
    }
    payload.update(over)
    return payload


def _create_topic(name, subject_id=None):
    subject_id = subject_id or _subject()["id"]
    r = client.post("/api/topics", json=_topic_payload(name, subject_id))
    assert r.status_code == 200, r.text
    return r.json()


def _active_names(subject_id):
    return {t["name"] for t in client.get(f"/api/topics?subject_id={subject_id}").json()}


def _run_session(topic_id, student_id):
    """A real TeachBack session, so the topic genuinely has student history."""
    start = client.post("/api/sessions/start",
                        json={"student_id": student_id, "topic_id": topic_id}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond",
                json={"text": "Push puts a new item on the top and pop takes the newest one off."})
    client.post(f"/api/sessions/{sid}/finish",
                json={"attention": 7, "confidence": 6, "difficulty": 4})
    return sid


# ---------------------------------------------------------------------------
# missing topics
# ---------------------------------------------------------------------------

def test_deleting_a_nonexistent_topic_is_404():
    assert client.delete("/api/topics/999999").status_code == 404
    assert client.get("/api/topics/999999/delete-preview").status_code == 404
    assert client.post("/api/topics/999999/restore").status_code == 404


# ---------------------------------------------------------------------------
# unused topic -> deleted outright
# ---------------------------------------------------------------------------

def test_preview_of_an_unused_topic_promises_deletion():
    topic = _create_topic("Doomed topic")
    preview = client.get(f"/api/topics/{topic['id']}/delete-preview").json()
    assert preview["mode"] == "delete"
    assert preview["history"]["total"] == 0
    assert "permanently deleted" in preview["message"]


def test_deleting_an_unused_topic_removes_it():
    subject_id = _subject()["id"]
    topic = _create_topic("Doomed topic", subject_id)
    assert "Doomed topic" in _active_names(subject_id)

    r = client.delete(f"/api/topics/{topic['id']}")
    assert r.status_code == 200
    assert r.json()["mode"] == "deleted"

    assert "Doomed topic" not in _active_names(subject_id)
    assert client.get(f"/api/topics/{topic['id']}").status_code == 404
    # not merely hidden: it is absent from the archived listing too
    everything = client.get(f"/api/topics?subject_id={subject_id}&include_archived=true").json()
    assert all(t["id"] != topic["id"] for t in everything)


def test_deleting_an_unused_topic_cascades_its_owned_rows():
    """Concepts, relationships, misconceptions, activities, the quiz and the
    quiz's questions are owned by the topic and must go with it."""
    topic = _create_topic("Doomed topic")
    tid = topic["id"]
    generated = client.post(f"/api/topics/{tid}/quiz/generate")
    assert generated.status_code in (200, 400)  # a tiny topic may yield no questions

    db = SessionLocal()
    try:
        quiz = db.query(Quiz).filter(Quiz.topic_id == tid).first()
        quiz_id = quiz.id if quiz else None
        assert db.query(Concept).filter(Concept.topic_id == tid).count() == 1
    finally:
        db.close()

    assert client.delete(f"/api/topics/{tid}").json()["mode"] == "deleted"

    db = SessionLocal()
    try:
        assert db.get(Topic, tid) is None
        assert db.query(Concept).filter(Concept.topic_id == tid).count() == 0
        assert db.query(ConceptRelationship).filter(ConceptRelationship.topic_id == tid).count() == 0
        assert db.query(Misconception).filter(Misconception.topic_id == tid).count() == 0
        assert db.query(Activity).filter(Activity.topic_id == tid).count() == 0
        assert db.query(Quiz).filter(Quiz.topic_id == tid).count() == 0
        if quiz_id is not None:
            assert db.query(QuizQuestion).filter(QuizQuestion.quiz_id == quiz_id).count() == 0
    finally:
        db.close()


def test_deleting_a_published_topic_unlinks_its_lecture_without_dangling():
    """A published lecture owns its topic. Deleting the topic must not leave
    lecture.topic_id pointing at a row that no longer exists — and must not
    throw away the teacher's material either."""
    subject = _subject("Python Programming")
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    published = client.post(f"/api/lectures/{lec['id']}/publish").json()
    tid = published["topic"]["id"]

    assert client.delete(f"/api/topics/{tid}").json()["mode"] == "deleted"

    after = client.get(f"/api/lectures/{lec['id']}").json()
    assert after["topic_id"] is None
    assert after["status"] == "draft"
    assert after["material_text"].startswith("# Stacks")

    db = SessionLocal()
    try:
        dangling = (db.query(Lecture)
                    .filter(Lecture.topic_id.isnot(None))
                    .outerjoin(Topic, Topic.id == Lecture.topic_id)
                    .filter(Topic.id.is_(None)).count())
        assert dangling == 0
    finally:
        db.close()


def test_deleting_a_topic_leaves_every_other_topic_alone():
    subject_id = _subject()["id"]
    before = _active_names(subject_id)
    keep = _create_topic("Innocent bystander", subject_id)
    doomed = _create_topic("Doomed topic", subject_id)

    client.delete(f"/api/topics/{doomed['id']}")

    after = _active_names(subject_id)
    assert after == before | {"Innocent bystander"}
    kept = client.get(f"/api/topics/{keep['id']}").json()
    assert len(kept["concepts"]) == 1 and len(kept["activities"]) == 1


def test_deleting_a_topic_does_not_touch_another_subject():
    nn_id = _subject("Neural Networks")["id"]
    py_id = _subject("Python Programming")["id"]
    py_before = _active_names(py_id)
    doomed = _create_topic("Doomed topic", nn_id)

    client.delete(f"/api/topics/{doomed['id']}")

    assert _active_names(py_id) == py_before
    assert "Doomed topic" not in _active_names(nn_id)
    assert "Doomed topic" not in _active_names(py_id)


# ---------------------------------------------------------------------------
# topic with history -> archived, never destroyed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_preview_of_a_used_topic_promises_archiving():
    subject = _subject("Python Programming")
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, _student()["id"])

    preview = client.get(f"/api/topics/{tid}/delete-preview").json()
    assert preview["mode"] == "archive"
    assert preview["history"]["sessions"] >= 1
    assert "archived so existing student history is preserved" in preview["message"]
    assert "TeachBack session" in preview["history_summary"]


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_deleting_a_topic_with_history_archives_and_keeps_everything():
    subject = _subject("Python Programming")
    student = _student()
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    session_id = _run_session(tid, student["id"])

    activity = client.get(f"/api/topics/{tid}").json()["activities"][0]
    client.post("/api/activities/complete",
                json={"student_id": student["id"], "activity_id": activity["id"],
                      "answer": "my answer"})

    db = SessionLocal()
    try:
        before = (db.query(TeachSession).filter(TeachSession.topic_id == tid).count(),
                  db.query(Observation).filter(Observation.topic_id == tid).count(),
                  db.query(ActivityCompletion).filter(ActivityCompletion.topic_id == tid).count())
    finally:
        db.close()
    assert all(count > 0 for count in before)

    r = client.delete(f"/api/topics/{tid}").json()
    assert r["mode"] == "archived"

    db = SessionLocal()
    try:
        topic = db.get(Topic, tid)
        assert topic is not None and topic.archived_at is not None
        after = (db.query(TeachSession).filter(TeachSession.topic_id == tid).count(),
                 db.query(Observation).filter(Observation.topic_id == tid).count(),
                 db.query(ActivityCompletion).filter(ActivityCompletion.topic_id == tid).count())
        assert after == before
        # the session itself, not just the count
        assert db.get(TeachSession, session_id).topic_id == tid
        # its teaching material is preserved too, so restore is meaningful
        assert db.query(Concept).filter(Concept.topic_id == tid).count() > 0
    finally:
        db.close()


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_a_quiz_attempt_alone_is_enough_history_to_archive():
    """Knowledge-check attempts point at the topic through their quiz, so they
    must count as history even when there is no session or observation."""
    subject = _subject("Python Programming")
    student = _student()
    topics = client.get(f"/api/topics?subject_id={subject['id']}").json()
    seeded = next(t for t in topics if t["name"] == "Strings in Python")

    quiz = client.get(f"/api/topics/{seeded['id']}/quiz").json()
    assert quiz["available"]
    answers = [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]
    submitted = client.post(f"/api/quiz/{quiz['quiz_id']}/submit",
                            json={"student_id": student["id"], "answers": answers})
    assert submitted.status_code == 200, submitted.text

    preview = client.get(f"/api/topics/{seeded['id']}/delete-preview").json()
    assert preview["mode"] == "archive"
    assert preview["history"]["quiz_attempts"] >= 1

    db = SessionLocal()
    try:
        attempts_before = (db.query(QuizAttempt)
                           .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
                           .filter(Quiz.topic_id == seeded["id"]).count())
    finally:
        db.close()

    assert client.delete(f"/api/topics/{seeded['id']}").json()["mode"] == "archived"

    db = SessionLocal()
    try:
        assert (db.query(QuizAttempt).join(Quiz, Quiz.id == QuizAttempt.quiz_id)
                .filter(Quiz.topic_id == seeded["id"]).count()) == attempts_before
    finally:
        db.close()


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_an_archived_topic_leaves_the_active_list_and_archives_its_lecture():
    subject = _subject("Python Programming")
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, _student()["id"])

    client.delete(f"/api/topics/{tid}")

    assert "Stacks" not in _active_names(subject["id"])
    everything = client.get(
        f"/api/topics?subject_id={subject['id']}&include_archived=true").json()
    archived = next(t for t in everything if t["id"] == tid)
    assert archived["archived"] is True and archived["archived_at"]
    # the lecture that published it goes with it, so the lecture list cannot
    # offer a "live" lecture whose topic no longer accepts sessions
    active_lectures = client.get(f"/api/lectures?subject_id={subject['id']}").json()
    assert all(l["id"] != lec["id"] for l in active_lectures)
    assert client.get(f"/api/lectures/{lec['id']}").json()["archived"] is True


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_an_archived_topic_cannot_start_a_new_session():
    subject = _subject("Python Programming")
    student = _student()
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, student["id"])

    client.delete(f"/api/topics/{tid}")

    r = client.post("/api/sessions/start", json={"student_id": student["id"], "topic_id": tid})
    assert r.status_code == 400


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_archived_history_still_shows_in_student_progress():
    subject = _subject("Python Programming")
    student = _student()
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, student["id"])

    before = client.get(f"/api/students/{student['id']}/progress").json()
    client.delete(f"/api/topics/{tid}")
    after = client.get(f"/api/students/{student['id']}/progress").json()

    assert len(after["timeline"]) == len(before["timeline"])
    assert any(o["topic_id"] == tid for o in after["timeline"])


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_an_archived_topic_cannot_take_a_new_knowledge_check():
    """Blocking new sessions is not enough on its own: the knowledge check is
    the other way a student can add history to a topic."""
    subject = _subject("Python Programming")
    student = _student()
    topics = client.get(f"/api/topics?subject_id={subject['id']}").json()
    seeded = next(t for t in topics if t["name"] == "Strings in Python")
    quiz = client.get(f"/api/topics/{seeded['id']}/quiz").json()
    answers = [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]

    _run_session(seeded["id"], student["id"])          # give it history...
    assert client.delete(f"/api/topics/{seeded['id']}").json()["mode"] == "archived"

    r = client.post(f"/api/quiz/{quiz['quiz_id']}/submit",
                    json={"student_id": student["id"], "answers": answers})
    assert r.status_code == 400
    assert "no longer available" in r.json()["detail"].lower()


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_a_check_that_follows_an_existing_session_still_completes():
    """A student who finished a session before it was archived may still hand
    in the knowledge check for it — that is finishing, not starting."""
    subject = _subject("Python Programming")
    student = _student()
    topics = client.get(f"/api/topics?subject_id={subject['id']}").json()
    seeded = next(t for t in topics if t["name"] == "Strings in Python")
    quiz = client.get(f"/api/topics/{seeded['id']}/quiz").json()
    session_id = _run_session(seeded["id"], student["id"])

    client.delete(f"/api/topics/{seeded['id']}")

    r = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": student["id"], "session_id": session_id,
        "answers": [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]})
    assert r.status_code == 200, r.text
    assert r.json()["n_questions"] == len(quiz["questions"])


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_restoring_an_archived_topic_brings_it_and_its_lecture_back():
    subject = _subject("Python Programming")
    student = _student()
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, student["id"])
    client.delete(f"/api/topics/{tid}")

    r = client.post(f"/api/topics/{tid}/restore")
    assert r.status_code == 200
    assert r.json()["id"] == tid

    assert "Stacks" in _active_names(subject["id"])
    assert client.get(f"/api/lectures/{lec['id']}").json()["status"] == "published"
    # and it accepts new sessions again
    assert client.post("/api/sessions/start",
                       json={"student_id": student["id"], "topic_id": tid}).status_code == 200


def test_restoring_a_topic_that_is_not_archived_is_rejected():
    topic = _create_topic("Doomed topic")
    assert client.post(f"/api/topics/{topic['id']}/restore").status_code == 400


# ---------------------------------------------------------------------------
# the delete action must not become a way around subject isolation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_an_archived_topic_stays_out_of_both_subject_lists():
    nn_id = _subject("Neural Networks")["id"]
    py_id = _subject("Python Programming")["id"]
    subject = _subject("Python Programming")
    lec = client.post("/api/lectures", json={
        "subject_id": subject["id"], "title": "Stacks", "material_text": MATERIAL}).json()
    tid = client.post(f"/api/lectures/{lec['id']}/publish").json()["topic"]["id"]
    _run_session(tid, _student()["id"])

    client.delete(f"/api/topics/{tid}")

    assert "Stacks" not in _active_names(py_id)
    assert "Stacks" not in _active_names(nn_id)


# ---------------------------------------------------------------------------
# the frontend action itself
# ---------------------------------------------------------------------------
# An endpoint nobody can reach is not a feature. These read the source rather
# than the DOM (scripts/verify_ui.py drives a real browser for that), and
# exist so the wiring cannot be removed silently.

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "src"


def test_the_api_client_exposes_the_topic_lifecycle():
    source = (FRONTEND / "services" / "api.js").read_text(encoding="utf-8")
    assert "topicDeletePreview:" in source
    assert "deleteTopic:" in source and "method: 'DELETE'" in source
    assert "restoreTopic:" in source


def test_topic_management_renders_a_real_delete_action():
    source = (FRONTEND / "pages" / "Topics.jsx").read_text(encoding="utf-8")
    # a labelled button, not a hover-only icon
    assert 'aria-label={`Delete topic ${t.name}`}' in source
    assert "askDelete(t.id)" in source
    # ...that opens the shared confirmation before anything is destroyed
    assert "ConfirmDeleteDialog" in source
    assert "api.topicDeletePreview" in source
    assert "api.deleteTopic" in source
    assert "api.restoreTopic" in source
    # ...and reloads from the backend afterwards rather than patching local state
    assert source.count("load()") >= 3


def test_the_confirmation_dialog_is_shared_with_the_lecture_delete():
    """One dialog for one rule: the two workflows cannot drift apart."""
    dialog = (FRONTEND / "components" / "ConfirmDeleteDialog.jsx").read_text(encoding="utf-8")
    assert "preview.mode === 'archive'" in dialog
    assert "onCancel" in dialog and "onConfirm" in dialog
    lectures = (FRONTEND / "pages" / "Lectures.jsx").read_text(encoding="utf-8")
    assert "ConfirmDeleteDialog" in lectures
    assert "function DeleteDialog" not in lectures  # the local copy is gone


# The browser may persist the demo role and the selected teacher/subject, and
# nothing else — those are UI selections, not data.
ALLOWED_STORAGE_KEYS = {"'teachback_user'", "STORAGE_KEY"}


def test_no_topic_data_is_cached_in_the_browser():
    """SQLite is the source of truth: a deleted topic cannot come back from
    localStorage on the next page load."""
    calls = []
    for path in sorted(FRONTEND.rglob("*.jsx")) + sorted(FRONTEND.rglob("*.js")):
        source = path.read_text(encoding="utf-8")
        for key in re.findall(r"localStorage\.\w+\(\s*([^,)]+)", source):
            calls.append((path.name, key.strip()))
    assert calls, "expected the teacher/subject switcher to use localStorage"
    unexpected = [c for c in calls if c[1] not in ALLOWED_STORAGE_KEYS]
    assert not unexpected, f"unexpected browser-persisted data: {unexpected}"
