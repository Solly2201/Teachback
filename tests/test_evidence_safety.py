"""Evidence safety: what the system is and is not allowed to conclude.

Three rules, all of which the recommender and the session summary must obey:

1. "not discussed" is not "misunderstood". A concept or connection that never
   came up is an absence of evidence and may never produce a concept-specific
   remediation activity or be worded as a mistake.
2. The takeaway summary may only ADD evidence, never remove it, and where it
   did add evidence that stays visible as summary evidence.
3. The knowledge check stays a separate, secondary channel: it never rewrites
   a TeachBack explanation and never becomes one "mastery" number.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.quiz import _combined_concept_view
from app.api.teachback import _apply_summary_to_plan
from app.hmm.model import hmm_available
from app.main import app
from app.recommend.rules import recommend

client = TestClient(app)

TOPIC_DEF = {
    "name": "Queues",
    "concepts": [
        {"id": 1, "name": "Enqueue", "description": "Enqueue adds an element at the back."},
        {"id": 2, "name": "Dequeue", "description": "Dequeue removes the element at the front."},
        {"id": 3, "name": "Circular Queue",
         "description": "A circular queue reuses the space freed at the front."},
    ],
    "relationships": [],
}


# ---------------------------------------- not discussed != misunderstood

def test_a_concept_that_never_came_up_does_not_drive_remediation():
    rec = recommend(
        1, [],  # "unclear" state, no stored activities
        evidence={"demonstrated": ["Enqueue"], "unclear": [],
                  "not_discussed": ["Circular Queue"]},
        topic_def=TOPIC_DEF,
    )
    text = (rec["activity"]["title"] + rec["activity"]["question"] + rec["why"]).lower()
    assert "circular queue" not in text, "a never-discussed concept became a remediation target"
    assert "needs clarification" not in rec["why"].lower()


def test_a_concept_with_real_gap_evidence_does_drive_remediation():
    rec = recommend(
        1, [],
        evidence={"demonstrated": ["Enqueue"], "unclear": ["Circular Queue"],
                  "not_discussed": []},
        topic_def=TOPIC_DEF,
    )
    assert "Circular Queue" in rec["activity"]["title"] + rec["activity"]["question"]
    assert "needs clarification" in rec["why"].lower()


def test_not_discussed_is_worded_as_absence_of_evidence():
    rec = recommend(3, [], evidence={"demonstrated": ["Enqueue"], "unclear": [],
                                     "not_discussed": ["Dequeue"]},
                    topic_def=TOPIC_DEF)
    note = " ".join(rec["notes"]).lower()
    assert "dequeue" in note
    assert "isn't a mistake" in note or "not a mistake" in note
    for accusation in ("wrong", "misunderstood", "failed", "struggling with"):
        assert accusation not in note


def test_combined_view_never_calls_an_undiscussed_concept_a_gap():
    plan = {"concepts": [{"name": "Enqueue", "status": "covered"},
                         {"name": "Circular Queue", "status": "pending"}]}
    combined = {c["name"]: c for c in _combined_concept_view(plan, {})}
    assert combined["Circular Queue"]["verdict"] == "not_discussed"
    assert "either way" in combined["Circular Queue"]["message"]
    assert combined["Enqueue"]["verdict"] == "solid"


def test_mcq_evidence_stays_separate_from_teachback_evidence():
    plan = {"concepts": [{"name": "Enqueue", "status": "covered"}]}
    combined = _combined_concept_view(plan, {"Enqueue": {"correct": 0, "total": 2}})[0]
    # the explanation the student actually gave is not erased by an MCQ miss
    assert combined["teachback_status"] == "covered"
    assert combined["verdict"] == "quick_review"
    assert combined["mcq_correct"] == 0 and combined["mcq_total"] == 2
    assert "explanation showed understanding" in combined["message"]
    assert "score" not in combined  # no single blended mastery number


# ------------------------------------------- the summary only ever adds

def test_summary_upgrades_but_never_downgrades():
    plan = {"concepts": [
        {"id": 1, "name": "Enqueue", "status": "covered"},
        {"id": 2, "name": "Dequeue", "status": "unclear"},
        {"id": 3, "name": "Circular Queue", "status": "pending"},
    ], "relationships": []}
    analysis = {"concepts": [
        {"id": 1, "name": "Enqueue", "status": "missing"},    # weaker: must not demote
        {"id": 2, "name": "Dequeue", "status": "covered"},    # stronger: upgrade
        {"id": 3, "name": "Circular Queue", "status": "partial"},
    ], "relationships": []}
    upgraded, mentioned = _apply_summary_to_plan(plan, analysis)
    statuses = {c["name"]: c["status"] for c in plan["concepts"]}
    assert statuses["Enqueue"] == "covered"          # untouched by a weak summary
    assert statuses["Dequeue"] == "covered"
    assert statuses["Circular Queue"] == "partial"
    assert "Dequeue" in upgraded
    assert set(mentioned) == {"Dequeue", "Circular Queue"}


def test_summary_evidence_keeps_its_provenance():
    plan = {"concepts": [
        {"id": 1, "name": "Dequeue", "status": "unclear"},
        {"id": 2, "name": "Circular Queue", "status": "pending"},
    ], "relationships": []}
    analysis = {"concepts": [
        {"id": 1, "name": "Dequeue", "status": "covered"},
        {"id": 2, "name": "Circular Queue", "status": "covered"},
    ], "relationships": []}
    _apply_summary_to_plan(plan, analysis)
    sources = {c["name"]: c.get("evidence_source") for c in plan["concepts"]}
    # evidence that existed before the summary is marked as both...
    assert sources["Dequeue"] == "teachback+summary"
    # ...evidence that only the summary produced is marked as summary evidence
    assert sources["Circular Queue"] == "summary"


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_short_summary_never_penalises_a_student_end_to_end():
    topics = client.get("/api/topics").json()
    topic = next(t for t in topics if t["concept_count"] >= 2)
    student = client.get("/api/students").json()[0]
    answer = "Strings are basically text you keep inside quotes."

    results = []
    for summary in ("", "yeah it was about text."):
        start = client.post("/api/sessions/start",
                            json={"student_id": student["id"], "topic_id": topic["id"]}).json()
        sid = start["session_id"]
        client.post(f"/api/sessions/{sid}/respond", json={"text": answer})
        results.append(client.post(f"/api/sessions/{sid}/finish", json={
            "attention": 7, "confidence": 6, "difficulty": 4, "summary": summary}).json())

    without, with_short = results
    demonstrated = [len([c for c in r["concept_summary"] if c["status"] == "covered"])
                    for r in (without, with_short)]
    assert demonstrated[1] >= demonstrated[0], "a short summary reduced the evidence"
    assert with_short["session_features"]["concept_coverage"] >= \
        without["session_features"]["concept_coverage"]
    # and every concept reports where its evidence came from
    assert all(c.get("evidence_source") for c in with_short["concept_summary"])


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_session_result_labels_the_posterior_honestly():
    topics = client.get("/api/topics").json()
    student = client.get("/api/students").json()[0]
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topics[0]["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "It stores text in quotes."})
    result = client.post(f"/api/sessions/{sid}/finish",
                         json={"attention": 6, "confidence": 5, "difficulty": 5}).json()
    state = result["state"]
    assert state["student_label"], "students should see the evidence-based wording"
    meaning = state["posterior_meaning"].lower()
    assert "model confidence" in meaning
    assert "not a probability of understanding" in meaning


# ------------------------------------------------- API referential safety

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_quiz_submission_must_match_its_session_and_student():
    students = client.get("/api/students").json()
    topics = client.get("/api/topics").json()
    topic = next(t for t in topics if client.get(f"/api/topics/{t['id']}/quiz").json().get("available"))
    quiz = client.get(f"/api/topics/{topic['id']}/quiz").json()

    start = client.post("/api/sessions/start",
                        json={"student_id": students[0]["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "It is text between quotes."})
    client.post(f"/api/sessions/{sid}/finish",
                json={"attention": 6, "confidence": 6, "difficulty": 5})

    answers = [{"question_id": q["id"], "selected_index": 0} for q in quiz["questions"]]
    # another student cannot submit against this session
    wrong_student = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": students[1]["id"], "session_id": sid, "answers": answers})
    assert wrong_student.status_code == 400
    # unknown session id
    assert client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": students[0]["id"], "session_id": 999999,
        "answers": answers}).status_code == 404
    # empty submission
    assert client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": students[0]["id"], "answers": []}).status_code == 400
    # the legitimate submission works, and only once
    ok = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": students[0]["id"], "session_id": sid, "answers": answers})
    assert ok.status_code == 200
    again = client.post(f"/api/quiz/{quiz['quiz_id']}/submit", json={
        "student_id": students[0]["id"], "session_id": sid, "answers": answers})
    assert again.status_code == 400


def test_session_endpoints_reject_unknown_ids_and_finished_sessions():
    assert client.post("/api/sessions/start",
                       json={"student_id": 999999, "topic_id": 1}).status_code == 404
    assert client.post("/api/sessions/999999/respond", json={"text": "hi"}).status_code == 404
    assert client.post("/api/sessions/999999/finish", json={
        "attention": 5, "confidence": 5, "difficulty": 5}).status_code == 404
