"""Teacher evidence inspection, and closing an evaluation.

Two capabilities, one rule between them.

A teacher must be able to check the system's work: read a student's exact
answer next to what TeachBack concluded from it, and disagree. That is the
whole point of the evidence endpoints — TeachBack assists the teacher, it does
not replace them.

Closing the evaluation is the other end of that: once the teacher has read the
answers, the raw free text has served its purpose and is permanently deleted,
while every structured conclusion drawn from it stays. The distinction these
tests exist to protect is:

    OPEN     the response is visible
    CLOSED   the response is gone, and no endpoint can produce it
    ALWAYS   the concept, relationship, misconception, knowledge-check,
             learning-state and progress records remain
"""
import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.hmm.model import hmm_available
from app.main import app
from app.models import Observation, QuizAttempt, Response, TeachSession, Topic

client = TestClient(app)

pytestmark = pytest.mark.skipif(not hmm_available(), reason="HMM not trained")

MATERIAL = """# Stacks

## Push

Push adds a new element to the top of the stack.
The stack keeps the order in which elements were added.

## Pop

Pop removes the element currently at the top of the stack.
The element added most recently is the first one removed.
"""


def _subjects():
    teachers = client.get("/api/teachers").json()
    subjects = {s["name"]: s["id"] for t in teachers for s in t["subjects"]}
    return subjects["Neural Networks"], subjects["Python Programming"]


def _student(name="Shreshtha Bindal"):
    return next(s for s in client.get("/api/students").json() if s["name"] == name)


def _strings_topic(py_id):
    return next(t for t in client.get(f"/api/topics?subject_id={py_id}").json()
                if "Strings" in t["name"])


def _run_session(topic_id, student_id, answers=None, takeaway="", feedback_text=""):
    """A real session, so there is genuine raw text to inspect and then delete."""
    start = client.post("/api/sessions/start",
                        json={"student_id": student_id, "topic_id": topic_id}).json()
    sid = start["session_id"]
    for text in answers or ["Strings are any words or text that can be stored using quotes.",
                            "indexing picks one character and the first one is at position zero"]:
        client.post(f"/api/sessions/{sid}/respond", json={"text": text})
    client.post(f"/api/sessions/{sid}/finish", json={
        "attention": 8, "confidence": 7, "difficulty": 4, "summary": takeaway,
        "pace": "just right", "feedback_choices": ["more examples"],
        "feedback_text": feedback_text})
    return sid


def _full_history(topic_id, student_id):
    """A session plus a knowledge-check attempt plus a completed activity."""
    sid = _run_session(topic_id, student_id,
                       takeaway="Strings hold text and you can pick characters out by position.",
                       feedback_text="the slicing part went fast")
    quiz = client.get(f"/api/topics/{topic_id}/quiz").json()
    client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": student_id, "session_id": sid,
        "answers": [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]})
    activity = client.get(f"/api/topics/{topic_id}").json()["activities"][0]
    client.post("/api/activities/complete",
                json={"student_id": student_id, "activity_id": activity["id"], "answer": "done"})
    return sid, quiz


# ---------------------------------------------------------------------------
# 1-3. the teacher can inspect the evidence, scoped to their subject
# ---------------------------------------------------------------------------

def test_a_teacher_can_read_a_students_responses_before_closure():
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    student = _student()
    sid = _run_session(topic["id"], student["id"])

    listing = client.get(f"/api/teacher/topics/{topic['id']}/evidence?subject_id={py_id}")
    assert listing.status_code == 200
    body = listing.json()
    assert body["responses_available"] is True
    assert body["evaluation_closed"] is False
    row = next(s for s in body["sessions"] if s["session_id"] == sid)
    assert row["student_name"] == student["name"]
    assert row["concepts_total"] > 0

    detail = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    assert detail["responses_available"] is True
    assert len(detail["responses"]) == 2


def test_each_response_shows_the_question_the_answer_and_the_interpretation():
    """The three must be distinguishable, or the teacher cannot check anything."""
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    sid = _run_session(topic["id"], _student()["id"])

    detail = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    first = detail["responses"][0]
    assert first["question"], "the question asked must be shown"
    # the student's exact words, unmodified
    assert first["answer"] == "Strings are any words or text that can be stored using quotes."
    # ...kept separate from what the system made of them
    assert first["concepts"], "the interpretation must name the concepts it judged"
    for c in first["concepts"]:
        assert c["status"] in ("covered", "partial", "missing")
        assert c["status_label"] in ("Evidence for", "Partial evidence for", "No evidence for")
        assert c["why"], "every judgement needs a plain-language reason"
    assert isinstance(first["contributed_to_coverage"], bool)
    # a concept it credited quotes the sentence it matched, so the teacher can
    # check the claim against what the student actually wrote
    credited = [c for c in first["concepts"] if c["status"] != "missing"]
    assert credited, "this answer should have demonstrated something"
    assert any(c["why"].startswith("The student said") for c in credited)


def test_the_evidence_endpoints_never_expose_internal_machinery():
    """Teacher verification, not debugging: no scores, thresholds or vectors."""
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    sid = _run_session(topic["id"], _student()["id"])
    body = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").text.lower()
    for jargon in ("cosine", "similarity", "threshold", "posterior",
                   "feature_vector", "embedding", "state_index"):
        assert jargon not in body, f"{jargon} leaked into the teacher-facing payload"


def test_evidence_is_scoped_to_the_subject_that_owns_the_topic():
    nn_id, py_id = _subjects()
    topic = _strings_topic(py_id)
    sid = _run_session(topic["id"], _student()["id"])

    # the right subject works...
    assert client.get(
        f"/api/teacher/topics/{topic['id']}/evidence?subject_id={py_id}").status_code == 200
    # ...and a topic or session id cannot be used to reach another teacher's students
    assert client.get(
        f"/api/teacher/topics/{topic['id']}/evidence?subject_id={nn_id}").status_code == 404
    assert client.get(
        f"/api/teacher/sessions/{sid}/evidence?subject_id={nn_id}").status_code == 404
    # the subject is required, never optional
    assert client.get(f"/api/teacher/sessions/{sid}/evidence").status_code == 422


def test_no_student_facing_endpoint_returns_response_text():
    """A student must never be able to read another student's answers."""
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    a, b = _student(), _student("Aarav Shah")
    _run_session(topic["id"], a["id"], answers=["a secret sentence only Shreshtha wrote"])

    for path in (f"/api/students/{a['id']}", f"/api/students/{a['id']}/progress",
                 f"/api/students/{b['id']}", f"/api/students/{b['id']}/progress",
                 "/api/students", f"/api/topics/{topic['id']}"):
        assert "a secret sentence only Shreshtha wrote" not in client.get(path).text, path


# ---------------------------------------------------------------------------
# 5-7. closing, and what closure blocks
# ---------------------------------------------------------------------------

def test_the_preview_says_what_will_be_destroyed_and_what_will_not():
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    _full_history(topic["id"], _student()["id"])

    preview = client.get(f"/api/topics/{topic['id']}/close-preview").json()
    assert preview["already_closed"] is False
    assert preview["raw"]["responses"] == 2
    assert preview["raw"]["takeaways"] == 1
    assert preview["raw"]["written_feedback"] == 1
    assert "stop new TeachBack sessions" in preview["message"]
    assert any("response" in line for line in preview["removed"])
    assert any("progress" in line for line in preview["kept"])


def test_a_teacher_can_close_an_open_evaluation():
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    _run_session(topic["id"], _student()["id"])

    r = client.post(f"/api/topics/{topic['id']}/close-evaluation")
    assert r.status_code == 200
    body = r.json()
    assert body["evaluation_closed"] is True and body["closed_at"]
    assert body["removed"]["responses"] == 2
    listed = next(t for t in client.get(f"/api/topics?subject_id={py_id}").json()
                  if t["id"] == topic["id"])
    assert listed["evaluation_closed"] is True


def test_closure_blocks_new_teachback_sessions():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    _run_session(topic["id"], student["id"])
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    r = client.post("/api/sessions/start",
                    json={"student_id": student["id"], "topic_id": topic["id"]})
    assert r.status_code == 400
    assert "closed" in r.json()["detail"].lower()
    # and it is no longer offered to students at all
    startable = client.get("/api/topics?startable=true").json()
    assert all(t["id"] != topic["id"] for t in startable)


def test_closure_blocks_new_standalone_knowledge_checks():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    quiz = client.get(f"/api/topics/{topic['id']}/quiz").json()
    _run_session(topic["id"], student["id"])
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    r = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": student["id"],
        "answers": [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]})
    assert r.status_code == 400
    assert "closed" in r.json()["detail"].lower()


def test_closure_blocks_further_answers_in_an_unfinished_session():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid = client.post("/api/sessions/start",
                      json={"student_id": student["id"], "topic_id": topic["id"]}
                      ).json()["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "Strings are text in quotes."})
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    assert client.post(f"/api/sessions/{sid}/respond",
                       json={"text": "more"}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/finish",
                       json={"attention": 5, "confidence": 5, "difficulty": 5}).status_code == 400


# ---------------------------------------------------------------------------
# 8-14. what survives closure
# ---------------------------------------------------------------------------

def test_everything_structured_survives_closure():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid, quiz = _full_history(topic["id"], student["id"])

    before = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    progress_before = client.get(f"/api/students/{student['id']}/progress").json()

    client.post(f"/api/topics/{topic['id']}/close-evaluation")
    after = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    progress_after = client.get(f"/api/students/{student['id']}/progress").json()

    # 11/12/13: concept, relationship and misconception evidence unchanged
    assert after["concept_summary"] == before["concept_summary"]
    assert after["relationship_summary"] == before["relationship_summary"]
    assert after["misconceptions_detected"] == before["misconceptions_detected"]
    assert after["misconceptions_resolved"] == before["misconceptions_resolved"]
    # 14: the HMM state and its evidence notes
    assert after["state"] == before["state"] and after["state"] is not None
    assert after["evidence_notes"] == before["evidence_notes"]
    # 9: the knowledge-check result
    assert after["knowledge_check"] == before["knowledge_check"]
    assert after["knowledge_check"]["n_questions"] == 10
    # 10: the completed activity
    assert after["activity_completions"] == before["activity_completions"]
    assert len(after["activity_completions"]) == 1
    # the self-report, pace and the structured feedback choices
    assert after["self_report"] == before["self_report"]
    assert after["pace"] == before["pace"] == "just right"
    assert after["feedback_choices"] == before["feedback_choices"] == ["more examples"]
    # the structured evidence DERIVED from the takeaway, though the text is gone
    assert after["summary_insights"] == before["summary_insights"]
    # a recommendation can still be produced from what remains
    assert after["recommendation"] is not None
    # 8: the session is still in the student's progress, with the same history
    assert len(progress_after["timeline"]) == len(progress_before["timeline"])
    assert any(o["topic_id"] == topic["id"] for o in progress_after["timeline"])


def test_the_session_row_itself_is_never_deleted():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid = _run_session(topic["id"], student["id"])
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    db = SessionLocal()
    try:
        ts = db.get(TeachSession, sid)
        assert ts is not None
        # the minimum useful structured record for longitudinal progress
        assert ts.student_id == student["id"] and ts.topic_id == topic["id"]
        assert ts.started_at is not None and ts.completed is True
        assert ts.plan and ts.plan.get("concepts")
        assert db.query(Observation).filter(Observation.session_id == sid).count() == 1
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 15-17. what is destroyed
# ---------------------------------------------------------------------------

def test_the_raw_response_rows_are_deleted_from_the_database():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid = _run_session(topic["id"], student["id"])

    db = SessionLocal()
    try:
        assert db.query(Response).filter(Response.session_id == sid).count() == 2
    finally:
        db.close()

    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    db = SessionLocal()
    try:
        assert db.query(Response).filter(Response.session_id == sid).count() == 0
    finally:
        db.close()


def test_no_endpoint_can_produce_a_deleted_response():
    """Deleted must mean unreachable, by any route."""
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    secret = "the stack keeps whatever I pushed on last right at the top"
    sid = _run_session(topic["id"], student["id"], answers=[secret],
                       takeaway="a takeaway nobody should read afterwards",
                       feedback_text="a written comment nobody should read afterwards")

    detail = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}")
    assert secret in detail.text, "precondition: the teacher can read it while open"

    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    paths = [
        f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}",
        f"/api/teacher/topics/{topic['id']}/evidence?subject_id={py_id}",
        f"/api/teacher/overview?subject_id={py_id}",
        f"/api/students/{student['id']}",
        f"/api/students/{student['id']}/progress",
        f"/api/topics/{topic['id']}",
    ]
    for path in paths:
        text = client.get(path).text
        assert secret not in text, f"raw response still reachable via {path}"
        assert "a takeaway nobody should read afterwards" not in text, path
        assert "a written comment nobody should read afterwards" not in text, path

    body = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    assert body["responses"] == [] and body["responses_available"] is False
    assert body["takeaway"] == "" and body["takeaway_removed"] is True


def test_the_written_takeaway_goes_but_its_structured_evidence_stays():
    """The retention policy for raw free text: the words go, the evidence stays."""
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid = _run_session(topic["id"], student["id"],
                       takeaway="Strings hold text and you can pick characters out by position.")

    before = client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}").json()
    assert before["takeaway"]
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    db = SessionLocal()
    try:
        ts = db.get(TeachSession, sid)
        assert ts.summary_text == ""
        assert ts.feedback_text == ""
        # what the analysis of the takeaway concluded is not raw text
        assert ts.summary_insights == before["summary_insights"]
    finally:
        db.close()
    # and the takeaway no longer appears among the student's own summaries
    progress = client.get(f"/api/students/{student['id']}/progress").json()
    assert all("pick characters out by position" not in (s["text"] or "")
               for s in progress["summaries"])


# ---------------------------------------------------------------------------
# 18-20, 23. edges
# ---------------------------------------------------------------------------

def test_closing_twice_is_rejected_rather_than_silently_repeated():
    _, py_id = _subjects()
    topic = _strings_topic(py_id)
    _run_session(topic["id"], _student()["id"])
    assert client.post(f"/api/topics/{topic['id']}/close-evaluation").status_code == 200
    second = client.post(f"/api/topics/{topic['id']}/close-evaluation")
    assert second.status_code == 400
    assert "already closed" in second.json()["detail"].lower()
    assert client.get(f"/api/topics/{topic['id']}/close-preview").json()["already_closed"] is True


def test_closing_a_topic_nobody_has_used_works():
    nn_id, _ = _subjects()
    created = client.post("/api/topics", json={
        "name": "Untouched topic", "subject_id": nn_id, "description": "d",
        "reference_explanation": "ref", "concepts": [], "relationships": [],
        "misconceptions": [], "activities": []}).json()

    preview = client.get(f"/api/topics/{created['id']}/close-preview").json()
    assert preview["raw"] == {"sessions": 0, "responses": 0, "takeaways": 0,
                              "written_feedback": 0}
    r = client.post(f"/api/topics/{created['id']}/close-evaluation")
    assert r.status_code == 200 and r.json()["removed"]["responses"] == 0


def test_closing_a_missing_topic_is_404():
    assert client.post("/api/topics/999999/close-evaluation").status_code == 404
    assert client.get("/api/topics/999999/close-preview").status_code == 404


def test_closing_one_subjects_evaluation_leaves_the_other_alone():
    nn_id, py_id = _subjects()
    py_topic, student = _strings_topic(py_id), _student()
    nn_topic = next(t for t in client.get(f"/api/topics?subject_id={nn_id}").json()
                    if t["name"] == "Backpropagation")
    nn_sid = _run_session(nn_topic["id"], student["id"],
                          answers=["the loss measures how wrong the prediction was"])
    _run_session(py_topic["id"], student["id"])

    client.post(f"/api/topics/{py_topic['id']}/close-evaluation")

    nn_after = client.get(f"/api/teacher/topics/{nn_topic['id']}/evidence?subject_id={nn_id}").json()
    assert nn_after["evaluation_closed"] is False
    assert nn_after["responses_available"] is True
    nn_detail = client.get(f"/api/teacher/sessions/{nn_sid}/evidence?subject_id={nn_id}").json()
    assert len(nn_detail["responses"]) == 1
    # and the other subject's topic can still start sessions
    assert client.post("/api/sessions/start",
                       json={"student_id": student["id"],
                             "topic_id": nn_topic["id"]}).status_code == 200


def test_closure_and_archiving_are_independent_lifecycles():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    _run_session(topic["id"], student["id"])

    client.post(f"/api/topics/{topic['id']}/close-evaluation")
    # closing does NOT archive: the topic stays in the teacher's active list
    assert any(t["id"] == topic["id"]
               for t in client.get(f"/api/topics?subject_id={py_id}").json())

    # archiving on top of a closed evaluation still works...
    assert client.delete(f"/api/topics/{topic['id']}").json()["mode"] == "archived"
    # ...and restoring does not reopen the evaluation or bring responses back
    restored = client.post(f"/api/topics/{topic['id']}/restore")
    assert restored.status_code == 200
    body = client.get(f"/api/teacher/topics/{topic['id']}/evidence?subject_id={py_id}").json()
    assert body["evaluation_closed"] is True
    assert body["responses_available"] is False
    assert client.post("/api/sessions/start",
                       json={"student_id": student["id"],
                             "topic_id": topic["id"]}).status_code == 400


def test_an_archived_topic_can_still_have_its_evaluation_closed():
    """Closure is data minimisation, so it must reach retired material too."""
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    _run_session(topic["id"], student["id"])
    assert client.delete(f"/api/topics/{topic['id']}").json()["mode"] == "archived"

    r = client.post(f"/api/topics/{topic['id']}/close-evaluation")
    assert r.status_code == 200 and r.json()["removed"]["responses"] == 2
    db = SessionLocal()
    try:
        topic_row = db.get(Topic, topic["id"])
        assert topic_row.archived_at is not None
        assert topic_row.evaluation_closed_at is not None
    finally:
        db.close()


def test_deleting_an_unused_closed_topic_still_removes_it_cleanly():
    nn_id, _ = _subjects()
    created = client.post("/api/topics", json={
        "name": "Untouched topic", "subject_id": nn_id, "description": "d",
        "reference_explanation": "ref", "concepts": [], "relationships": [],
        "misconceptions": [], "activities": []}).json()
    client.post(f"/api/topics/{created['id']}/close-evaluation")
    assert client.delete(f"/api/topics/{created['id']}").json()["mode"] == "deleted"
    assert client.get(f"/api/topics/{created['id']}").status_code == 404


# ---------------------------------------------------------------------------
# 21-22, 24. integrity, atomicity, and the dashboard afterwards
# ---------------------------------------------------------------------------

def test_closure_leaves_no_orphans_or_broken_foreign_keys():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    _full_history(topic["id"], student["id"])
    client.post(f"/api/topics/{topic['id']}/close-evaluation")

    db = SessionLocal()
    try:
        # no response left pointing at a session, and no session left broken
        assert (db.query(Response)
                .outerjoin(TeachSession, TeachSession.id == Response.session_id)
                .filter(TeachSession.id.is_(None)).count()) == 0
        assert (db.query(TeachSession)
                .outerjoin(Topic, Topic.id == TeachSession.topic_id)
                .filter(Topic.id.is_(None)).count()) == 0
        assert (db.query(Observation).filter(Observation.session_id.isnot(None))
                .outerjoin(TeachSession, TeachSession.id == Observation.session_id)
                .filter(TeachSession.id.is_(None)).count()) == 0
        assert (db.query(QuizAttempt).filter(QuizAttempt.session_id.isnot(None))
                .outerjoin(TeachSession, TeachSession.id == QuizAttempt.session_id)
                .filter(TeachSession.id.is_(None)).count()) == 0
        conn = db.connection()
        assert list(conn.exec_driver_sql("PRAGMA foreign_key_check")) == []
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar() == "ok"
    finally:
        db.close()


def test_a_failure_partway_through_closure_changes_nothing(monkeypatch):
    """Either the evaluation closes AND the responses go, or neither happens.

    A half-closed evaluation is the worst outcome available: a topic still
    accepting sessions while some students' answers have already been erased.
    """
    import app.api.topics as topics_api

    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    sid = _run_session(topic["id"], student["id"])

    class Boom:
        @staticmethod
        def utcnow():
            raise RuntimeError("disk fell over")

    # fails after the responses have been deleted in the session, before commit
    monkeypatch.setattr(topics_api, "datetime", Boom)
    r = client.post(f"/api/topics/{topic['id']}/close-evaluation")
    assert r.status_code == 500
    monkeypatch.undo()

    db = SessionLocal()
    try:
        assert db.get(Topic, topic["id"]).evaluation_closed_at is None
        assert db.query(Response).filter(Response.session_id == sid).count() == 2
    finally:
        db.close()
    # and the topic is genuinely still open, not merely un-flagged
    assert client.get(f"/api/teacher/sessions/{sid}/evidence?subject_id={py_id}"
                      ).json()["responses_available"] is True
    assert client.post("/api/sessions/start",
                       json={"student_id": student["id"],
                             "topic_id": topic["id"]}).status_code == 200


def test_the_teacher_dashboard_still_works_after_closure():
    _, py_id = _subjects()
    topic, student = _strings_topic(py_id), _student()
    _full_history(topic["id"], student["id"])

    before = client.get(f"/api/teacher/overview?subject_id={py_id}").json()
    client.post(f"/api/topics/{topic['id']}/close-evaluation")
    after = client.get(f"/api/teacher/overview?subject_id={py_id}")

    assert after.status_code == 200
    body = after.json()
    # closure is not archiving: the topic keeps counting in every aggregate
    assert len(body["topic_stats"]) == len(before["topic_stats"])
    assert body["live_session_count"] == before["live_session_count"]
    assert [k["attempts"] for k in body["knowledge_checks"]] == \
           [k["attempts"] for k in before["knowledge_checks"]]
    assert sum(d["count"] for d in body["distribution"]) == \
           sum(d["count"] for d in before["distribution"])
    # the written comments are gone from the feedback panel, the choices remain
    for f in body["topic_feedback"]:
        if f["id"] == topic["id"]:
            assert f["recent_comments"] == []
            assert f["pace"], "structured pace feedback is retained"


# ---------------------------------------------------------------------------
# the frontend action exists
# ---------------------------------------------------------------------------

def test_the_evidence_screen_and_closure_control_are_wired_up():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "frontend" / "src"
    page = (src / "pages" / "Evidence.jsx").read_text(encoding="utf-8")
    # the student's words and the system's reading are separately labelled
    assert "Student response" in page and "System interpretation" in page
    # inspection is a real button, not a hover
    assert "Inspect" in page and "aria-label=" in page
    # the destructive warning says what it means
    assert "individual student responses are permanently deleted" in page
    assert "raw free-text responses cannot be recovered" in page
    assert "Close Evaluation" in page and "Cancel" in page
    assert "Raw student responses removed" in page

    api = (src / "services" / "api.js").read_text(encoding="utf-8")
    for call in ("topicEvidence:", "sessionEvidence:", "closePreview:", "closeEvaluation:"):
        assert call in api

    layout = (src / "components" / "Layout.jsx").read_text(encoding="utf-8")
    assert "/evidence" in layout
    app_jsx = (src / "App.jsx").read_text(encoding="utf-8")
    assert 'path="/evidence"' in app_jsx and "Evidence" in app_jsx
    # students are only offered topics they can actually start
    teachback = (src / "pages" / "TeachBack.jsx").read_text(encoding="utf-8")
    assert "api.topics(null, false, true)" in teachback
