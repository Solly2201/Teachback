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


# --------------------------------------- the state and the session agree in words

@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_a_strong_session_under_a_weak_state_is_explained_not_contradicted():
    """The learning state reads the student's whole history, so a strong
    session can sit under a cautious state. Shown side by side without a word
    of explanation that reads as the system contradicting itself."""
    topics = client.get("/api/topics").json()
    topic = next(t for t in topics if "Strings" in t["name"])
    tdef = client.get(f"/api/topics/{topic['id']}").json()
    student = client.get("/api/students").json()[0]

    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    answers = {
        "Strings": "a string is text you keep between quotes",
        "String assignment": "you save the text under a name so you can use it again",
        "Characters": "it is made of single letters sitting in a fixed order",
        "Indexing": "you use the position number to pull out one letter, and the first is zero",
        "Slicing": "you take a part of the text between a start and an end position",
        "split() and join()": "split breaks the text into a list of pieces and join puts them back",
    }
    question = start["question"]
    guard = 0
    while question is not None and guard < 14:
        guard += 1
        concept = (question.get("concept") or "").split(" → ")[0]
        text = answers.get(concept, "it is about working with text in python")
        step = client.post(f"/api/sessions/{sid}/respond", json={"text": text}).json()
        if step["awaiting_self_report"]:
            break
        question = step.get("followup")

    result = client.post(f"/api/sessions/{sid}/finish", json={
        "attention": 8, "confidence": 7, "difficulty": 4}).json()
    demonstrated = [c for c in result["concept_summary"] if c["status"] == "covered"]
    assert len(demonstrated) >= 4, result["concept_summary"]
    # whenever a strong session sits under a low state, the result must say why
    if result["state"]["index"] <= 1:
        note = result["state"]["note"]
        assert note, "a strong session under a low state was left unexplained"
        assert "recent sessions" in note
    assert len(tdef["concepts"]) == len(result["concept_summary"])


def test_a_takeaway_that_only_repeats_lecture_vocabulary_adds_nothing():
    """"backpropagation gradient weight loss optimization" names the lecture's
    topics without saying anything about them. It is stored and shown back to
    the student, but it must not manufacture evidence."""
    from app.nlp.analyzer import is_term_list

    assert is_term_list("backpropagation gradient weight loss optimization")
    assert is_term_list("markov hidden states observations transitions")
    # ...while ordinary terse writing is not a term list
    for real in ("text in quotes", "the first position is zero",
                 "single letters in order", "split makes a list of pieces"):
        assert not is_term_list(real), real


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_keyword_salad_takeaway_creates_no_evidence_end_to_end():
    topics = client.get("/api/topics").json()
    topic = next(t for t in topics if "Strings" in t["name"])
    student = client.get("/api/students").json()[0]
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond", json={"text": "i am not sure about this"})
    result = client.post(f"/api/sessions/{sid}/finish", json={
        "attention": 5, "confidence": 5, "difficulty": 5,
        "summary": "strings indexing slicing characters assignment"}).json()
    assert result["summary_insights"] in ({}, None) or \
        not result["summary_insights"].get("new_concepts_demonstrated")
    # the student still sees their own words back
    progress = client.get(f"/api/students/{student['id']}/progress").json()
    assert any("strings indexing slicing" in s["text"] for s in progress["summaries"])


# ------------------------------- asking is not explaining (and not confessing)

# A teach-back is evidence because the student produced the explanation. A
# question produces none — but it sits in the concept's own vocabulary and is
# semantically about it, so it used to score as a demonstration. The worst
# case credited the concept to a student who had just said, in the same
# sentence, that they could not explain it.

QUESTION_TOPIC = {
    "name": "Backpropagation",
    "reference_explanation": "The loss measures the error and gradients update the weights.",
    "concepts": [
        {"id": 1, "name": "Gradient",
         "description": "The gradient is how much the loss changes when a weight changes.",
         "facts": ["The gradient points in the direction the loss increases."]},
    ],
    "misconceptions": [
        {"id": 1, "name": "Gradient is the loss",
         "description": "The gradient and the loss are the same thing.",
         "clarification": "The loss is the error; the gradient is its rate of change."},
    ],
    "relationships": [],
}

ASKED_NOT_ANSWERED = [
    "What is a gradient?",
    "wait, can you explain what a gradient is?",
    "sorry what does gradient descent mean?",
    "can you explain the gradient again?",
    "I don't understand the gradient, could you go over it?",
    "remind me what the gradient does?",
]


@pytest.mark.parametrize("text", ASKED_NOT_ANSWERED)
def test_asking_about_a_concept_never_demonstrates_it(text):
    from app.nlp.analyzer import analyze_response

    result = analyze_response(text, QUESTION_TOPIC)
    assert [c["status"] for c in result["concepts"]] == ["missing"], text
    assert result["features"]["concept_coverage"] == 0.0
    # correctness and depth measure the explanation, so they cannot be earned
    # by a question either
    assert result["features"]["semantic_correctness"] == 0.0
    assert result["features"]["explanation_depth"] == 0.0
    # ...but asking is still engagement, and effort says so
    assert result["features"]["response_effort"] > 0.0


def test_asking_whether_a_misconception_is_true_is_not_holding_it():
    """An unanswered question must never become an accusation."""
    from app.nlp.analyzer import analyze_response

    asked = analyze_response("Is the gradient the same thing as the loss?", QUESTION_TOPIC)
    assert asked["detected_misconceptions"] == []
    assert asked["features"]["misconception_score"] == 0.0
    # the same idea ASSERTED is still detected — the rule removes false
    # accusations, it does not switch misconception detection off
    claimed = analyze_response("The gradient is the same thing as the loss.", QUESTION_TOPIC)
    assert claimed["detected_misconceptions"] == ["Gradient is the loss"]


def test_an_explanation_keeps_its_credit_when_it_ends_with_a_question():
    """Only the asking part is discounted, not the whole answer."""
    from app.nlp.analyzer import analyze_response

    result = analyze_response(
        "The gradient tells us how much the loss changes when a weight changes. "
        "Does that sound right?", QUESTION_TOPIC)
    assert result["concepts"][0]["status"] == "covered"


def test_a_rhetorical_question_does_not_hide_the_answer_that_follows():
    from app.nlp.analyzer import analyze_response

    result = analyze_response(
        "What is a gradient? It is how much the loss changes when you change a weight.",
        QUESTION_TOPIC)
    # the point is that the answer still counts; which tier it reaches depends
    # on how much reference text the concept carries
    assert result["concepts"][0]["status"] in ("covered", "partial")


def test_the_per_question_check_applies_the_same_rule():
    """The short-answer path has its own scorer; it must agree."""
    from app.nlp.analyzer import targeted_concept_check

    concept = QUESTION_TOPIC["concepts"][0]
    asked = targeted_concept_check("can you explain what a gradient is?", concept,
                                   topic_name="Backpropagation")
    assert asked["informative"] is False
    assert asked["corroborated"] is False
    assert asked["plain"] == 0.0 and asked["contextual"] == 0.0

    answered = targeted_concept_check(
        "it is how much the loss changes when a weight changes", concept,
        topic_name="Backpropagation")
    assert answered["informative"] is True
