"""Tests for the Quick knowledge check: generation quality, grounding,
submission, combined TeachBack+MCQ evidence, and the spec's A-F combinations.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.quiz import _combined_concept_view
from app.database import SessionLocal
from app.hmm.model import hmm_available
from app.main import app
from app.models import Quiz, QuizAnswer, QuizAttempt
from app.nlp.quiz_gen import (generate_quiz_candidates, generate_quiz_questions,
                              validate_question)
from app.seed_content import PYTHON_LECTURE, TOPICS

client = TestClient(app)

STRINGS_DEF = {
    "name": PYTHON_LECTURE["title"],
    "concepts": PYTHON_LECTURE["reviewed_concepts"],
    "misconceptions": PYTHON_LECTURE["reviewed_misconceptions"],
    "relationships": PYTHON_LECTURE["reviewed_relationships"],
}


def _material_text():
    lec = PYTHON_LECTURE
    parts = [lec["material"]]
    for c in lec["reviewed_concepts"]:
        parts.append(c["name"] + " " + c["description"] + " " + " ".join(c.get("facts", []))
                     + " " + " ".join(c.get("examples", [])))
    for m in lec["reviewed_misconceptions"]:
        parts.append(m["description"] + " " + m["clarification"])
    for r in lec["reviewed_relationships"]:
        parts.append(r["source"] + " " + r["target"] + " " + r["description"]
                     + " " + r.get("contradiction", ""))
    return " ".join(parts).lower()


# ---------------------------------------------------------------- generation

def test_strings_quiz_has_10_valid_mixed_questions():
    qs = generate_quiz_questions(STRINGS_DEF)
    assert len(qs) == 10
    assert all(validate_question(q) for q in qs)
    kinds = {q["kind"] for q in qs}
    assert {"basic", "application", "misconception", "relationship"} <= kinds
    # the correct answer is not always in the same position
    assert len({q["correct_index"] for q in qs}) >= 3
    # every question is tied to a concept
    assert all(q["concept_name"] for q in qs)


def test_questions_grounded_in_teacher_material():
    """Every option of every question comes from the reviewed material."""
    material = _material_text()
    for q in generate_quiz_questions(STRINGS_DEF):
        for opt in q["options"]:
            core = opt.rstrip("…").strip().lower()[:40]
            assert core[:25] in material, f"option not grounded: {opt!r}"


def test_generation_is_deterministic():
    assert generate_quiz_questions(STRINGS_DEF) == generate_quiz_questions(STRINGS_DEF)


def test_validation_rejects_malformed_questions():
    good = generate_quiz_questions(STRINGS_DEF)[0]
    assert validate_question(good)
    assert not validate_question({**good, "options": good["options"][:3]})       # 3 options
    assert not validate_question({**good, "options": good["options"][:3] + [good["options"][0]]})  # duplicate
    assert not validate_question({**good, "correct_index": 4})                    # out of range
    assert not validate_question({**good, "explanation": ""})                     # no explanation
    assert not validate_question({**good, "concept_name": ""})                    # no concept
    # accidental answer clue: correct option appears inside the question text
    clue = {**good, "question": good["question"] + " " + good["options"][good["correct_index"]]}
    assert not validate_question(clue)


def test_generation_works_for_hand_authored_topic():
    qs = generate_quiz_questions(TOPICS[0])  # Backpropagation, no facts/examples
    assert len(qs) >= 5
    assert all(validate_question(q) for q in qs)


def test_misconception_question_flags_the_false_statement():
    qs = generate_quiz_candidates(STRINGS_DEF)
    miscons = [q for q in qs if q["kind"] == "misconception"]
    assert len(miscons) >= 2
    claims = {m["description"].rstrip(".") for m in PYTHON_LECTURE["reviewed_misconceptions"]}
    for q in miscons:
        correct_opt = q["options"][q["correct_index"]].rstrip(".")
        assert correct_opt in claims  # the "false statement" is the teacher's wrong claim
        assert q["explanation"]  # the clarification is shown after answering


# ---------------------------------------------------------------- API flow

def _strings_topic():
    teachers = client.get("/api/teachers").json()
    py = next(s for t in teachers for s in t["subjects"] if s["name"] == "Python Programming")
    topics = client.get(f"/api/topics?subject_id={py['id']}").json()
    return next(t for t in topics if "Strings" in t["name"])


def _student():
    return next(s for s in client.get("/api/students").json() if s["name"] == "Shreshtha Bindal")


def _answer_key(quiz_id):
    db = SessionLocal()
    quiz = db.get(Quiz, quiz_id)
    key = {q.id: q.correct_index for q in quiz.questions}
    concept = {q.id: q.concept_name for q in quiz.questions}
    db.close()
    return key, concept


def test_student_quiz_view_hides_answers():
    topic = _strings_topic()
    q = client.get(f"/api/topics/{topic['id']}/quiz").json()
    assert q["available"] and len(q["questions"]) == 10
    assert all("correct_index" not in x and "explanation" not in x for x in q["questions"])
    assert "not a replacement for your explanation" in q["intro"]


def test_submit_records_attempt_and_per_concept_evidence():
    topic = _strings_topic()
    q = client.get(f"/api/topics/{topic['id']}/quiz").json()
    key, _ = _answer_key(q["quiz_id"])
    # answer everything correctly except the first question
    answers = []
    for i, item in enumerate(q["questions"]):
        correct = key[item["id"]]
        answers.append({"question_id": item["id"],
                        "selected_index": (correct + 1) % 4 if i == 0 else correct})
    r = client.post(f"/api/quiz/{q['quiz_id']}/submit",
                    json={"student_id": _student()["id"], "answers": answers}).json()
    assert r["n_correct"] == 9 and r["n_questions"] == 10
    assert r["headline"] == "9/10 questions correct"
    assert sum(c["total"] for c in r["per_concept"]) == 10
    # the DB stores per-question answers, not just the percentage
    db = SessionLocal()
    attempt = db.get(QuizAttempt, r["attempt_id"])
    assert attempt.n_correct == 9
    assert len(attempt.answers) == 10
    assert sum(1 for a in attempt.answers if not a.correct) == 1
    assert all(isinstance(a.selected_index, int) for a in attempt.answers)
    db.close()


# ------------------------------------------------- TeachBack x MCQ combinations

def test_combined_view_combinations_a_to_d():
    plan = {"concepts": [
        {"name": "Indexing", "status": "covered"},   # A: TB strong
        {"name": "Slicing", "status": "covered"},    # B: TB strong, MCQ weak
        {"name": "Strings", "status": "unclear"},    # C: TB weak, MCQ strong
        {"name": "Characters", "status": "unclear"}, # D: TB weak, MCQ weak
    ]}
    mcq = {
        "Indexing": {"correct": 2, "total": 2},
        "Slicing": {"correct": 1, "total": 2},
        "Strings": {"correct": 2, "total": 2},
        "Characters": {"correct": 0, "total": 2},
    }
    combined = {c["name"]: c for c in _combined_concept_view(plan, mcq)}
    # A: both strong -> strong supporting evidence
    assert combined["Indexing"]["verdict"] == "solid"
    # B: TeachBack evidence remains, the MCQ gap is named gently
    assert combined["Slicing"]["verdict"] == "quick_review"
    assert combined["Slicing"]["teachback_status"] == "covered"  # not erased
    assert "explanation showed understanding" in combined["Slicing"]["message"]
    # C: recognises but can't explain -> NOT called confused
    assert combined["Strings"]["verdict"] == "practice_explaining"
    assert "own words" in combined["Strings"]["message"]
    # D: both weak -> stronger case for review
    assert combined["Characters"]["verdict"] == "revisit"


@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_quiz_after_session_updates_recommendation_not_state():
    topic = _strings_topic()
    student = _student()
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]}).json()
    sid = start["session_id"]
    client.post(f"/api/sessions/{sid}/respond",
                json={"text": "Strings are basically text stored between quotes."})
    fin = client.post(f"/api/sessions/{sid}/finish",
                      json={"attention": 7, "confidence": 6, "difficulty": 4}).json()
    assert fin["quiz"] and fin["quiz"]["n_questions"] == 10
    state_before = fin["state"]["label"]

    q = client.get(f"/api/topics/{topic['id']}/quiz").json()
    key, _ = _answer_key(q["quiz_id"])
    answers = [{"question_id": x["id"], "selected_index": key[x["id"]]} for x in q["questions"]]
    r = client.post(f"/api/quiz/{q['quiz_id']}/submit",
                    json={"student_id": student["id"], "session_id": sid,
                          "answers": answers}).json()
    assert r["n_correct"] == 10
    # the recommendation is refreshed with combined evidence...
    assert r["updated_recommendation"] is not None
    assert r["updated_recommendation"]["state_key"]
    # ...but the HMM state and the 8-dim observation are untouched
    prog = client.get(f"/api/students/{student['id']}/progress").json()
    latest = prog["timeline"][-1]
    assert latest["state_label"] == state_before
    assert len(latest["features"]) == 8
    # the knowledge-check result appears as an evidence NOTE only
    assert any("Knowledge check: 10/10" in e for e in latest["evidence"])


def test_lecture_draft_quiz_review_and_regenerate():
    teachers = client.get("/api/teachers").json()
    py = next(s for t in teachers for s in t["subjects"] if s["name"] == "Python Programming")
    lec = client.post("/api/lectures", json={
        "subject_id": py["id"], "title": "Quiz Review Test",
        "material_text": PYTHON_LECTURE["material"]}).json()
    quiz = lec["draft"]["quiz"]
    assert len(quiz) == 10 and all(validate_question(q) for q in quiz)

    # teacher edits a question and it persists
    quiz[0]["question"] = "Edited question about strings?"
    quiz[0]["explanation"] = "Edited explanation."
    lec = client.put(f"/api/lectures/{lec['id']}", json={"quiz": quiz}).json()
    assert lec["draft"]["quiz"][0]["question"] == "Edited question about strings?"

    # regenerating one question replaces it with a different valid one
    before = lec["draft"]["quiz"][1]["question"]
    lec = client.post(f"/api/lectures/{lec['id']}/quiz/regenerate", json={"index": 1}).json()
    after = lec["draft"]["quiz"][1]
    assert after["question"] != before
    assert validate_question(after)

    # deleting a question persists
    remaining = lec["draft"]["quiz"][1:]
    lec = client.put(f"/api/lectures/{lec['id']}", json={"quiz": remaining}).json()
    assert len(lec["draft"]["quiz"]) == 9

    # publishing publishes the reviewed quiz (9 questions, incl. the edit)
    r = client.post(f"/api/lectures/{lec['id']}/publish").json()
    topic_id = r["topic"]["id"]
    published = client.get(f"/api/topics/{topic_id}/quiz").json()
    assert published["available"] and len(published["questions"]) == 9


def test_knowledge_check_stats_scoped_by_subject():
    teachers = client.get("/api/teachers").json()
    subs = {s["name"]: s["id"] for t in teachers for s in t["subjects"]}
    py = client.get(f"/api/teacher/overview?subject_id={subs['Python Programming']}").json()
    nn = client.get(f"/api/teacher/overview?subject_id={subs['Neural Networks']}").json()
    py_topic_names = {t["name"] for t in client.get(f"/api/topics?subject_id={subs['Python Programming']}").json()}
    nn_topic_names = {t["name"] for t in client.get(f"/api/topics?subject_id={subs['Neural Networks']}").json()}
    assert all(k["name"] in py_topic_names for k in py.get("knowledge_checks", []))
    assert all(k["name"] in nn_topic_names for k in nn.get("knowledge_checks", []))
    # the Strings attempts from these tests appear under Python, with
    # per-concept MCQ and TeachBack numbers kept separate
    strings_stats = next((k for k in py["knowledge_checks"] if "Strings" in k["name"]), None)
    assert strings_stats is not None and strings_stats["attempts"] >= 1
    assert all("mcq_percent" in c and "teachback_percent" in c for c in strings_stats["concepts"])


# ---------------------------------------------------------------- positions

def test_correct_answer_position_is_not_a_predictable_cycle():
    """Rotating by the question's index produced A, B, C, D, A, B ... in every
    quiz — a student who spotted it could score full marks without reading.
    The rotation now comes from the question's own text."""
    positions = [q["correct_index"] for q in generate_quiz_questions(STRINGS_DEF)]
    cycle = [i % 4 for i in range(len(positions))]
    assert positions != cycle, "the correct answer follows its position in the quiz"
    assert len(set(positions)) >= 3, positions


def test_position_rotation_is_stable_across_topics_and_runs():
    """Content-derived, so the same material always produces the same quiz —
    but different material does not share a pattern."""
    for tdef in (STRINGS_DEF, *TOPICS):
        first = [q["correct_index"] for q in generate_quiz_questions(tdef)]
        assert first == [q["correct_index"] for q in generate_quiz_questions(tdef)]
        assert first != [i % 4 for i in range(len(first))], tdef["name"]


def test_every_seeded_topic_produces_a_full_valid_quiz():
    for tdef in (STRINGS_DEF, *TOPICS):
        qs = generate_quiz_questions(tdef)
        assert len(qs) == 10, f"{tdef['name']}: {len(qs)}"
        assert all(validate_question(q) for q in qs)
        stems = [q["question"].strip().lower() for q in qs]
        assert len(stems) == len(set(stems)), f"duplicate stem in {tdef['name']}"
        for q in qs:
            assert len({o.strip().lower() for o in q["options"]}) == 4
            # the correct option must not stand out by length alone
            others = [len(o) for i, o in enumerate(q["options"]) if i != q["correct_index"]]
            assert len(q["options"][q["correct_index"]]) <= 2.5 * max(others)
