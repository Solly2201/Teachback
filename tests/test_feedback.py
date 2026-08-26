"""Student summary, lecture feedback, and signal-aware recommendation tests.

Confidence/difficulty are observations that steer WHICH activity is
recommended; they never assign the HMM state. The student's own summary is an
upgrade-only evidence source and a short summary is never a penalty.
"""
import pytest
from fastapi.testclient import TestClient

from app.hmm.model import hmm_available
from app.main import app
from app.recommend.rules import recommend
from app.states import STATE_KEYS

client = TestClient(app)

TDEF = {
    "name": "Demo Topic",
    "concepts": [
        {"name": "Alpha", "description": "Alpha is the first idea."},
        {"name": "Beta", "description": "Beta is the second idea."},
    ],
    "relationships": [{"source": "Alpha", "label": "leads to", "target": "Beta",
                       "description": "Alpha leads to Beta."}],
}

GOOD = ("The loss function measures how wrong the prediction is. The gradient tells us how the "
        "loss changes when we change a weight, and gradient descent updates the weights to reduce the loss.")


def _start_backprop(student_idx=0):
    students = client.get("/api/students").json()
    topics = client.get("/api/topics").json()
    backprop = next(t for t in topics if t["name"] == "Backpropagation")
    r = client.post("/api/sessions/start",
                    json={"student_id": students[student_idx]["id"], "topic_id": backprop["id"]})
    return r.json()["session_id"]


# 18-21. Summary, pace, confidence, difficulty, and optional feedback are stored
@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_finish_stores_summary_and_feedback():
    sid = _start_backprop()
    client.post(f"/api/sessions/{sid}/respond", json={"text": GOOD})
    fin = client.post(f"/api/sessions/{sid}/finish", json={
        "attention": 8, "confidence": 7, "difficulty": 5,
        "summary": "We measure the error and use gradients to update the weights step by step.",
        "pace": "A little fast",
        "feedback_choices": ["More examples", "More practice"],
        "feedback_text": "A worked example would help.",
    })
    assert fin.status_code == 200
    assert "summary_insights" in fin.json()

    # stored fields surface in the teacher's lecture-feedback aggregates
    overview = client.get("/api/teacher/overview").json()
    row = next(f for f in overview["topic_feedback"] if f["name"] == "Backpropagation")
    assert any(p["label"] == "A little fast" for p in row["pace"])
    assert any(c["label"] == "More examples" for c in row["common_requests"])
    assert "A worked example would help." in row["recent_comments"]
    assert row["avg_confidence"] is not None and row["avg_difficulty"] is not None


# 15-17. The summary is stored and contributes upgrade-only evidence
@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_summary_upgrades_missed_concept():
    sid = _start_backprop(1)
    # conversation only demonstrates the loss concept
    client.post(f"/api/sessions/{sid}/respond", json={
        "text": "The loss function measures how wrong the network's prediction is."})
    fin = client.post(f"/api/sessions/{sid}/finish", json={
        "attention": 7, "confidence": 6, "difficulty": 5,
        "summary": ("The gradient is the derivative of the loss with respect to a weight — it tells "
                    "us how much the loss changes when we change that weight."),
    }).json()
    statuses = {c["name"]: c["status"] for c in fin["concept_summary"]}
    assert statuses["Gradient"] in ("covered", "partial")
    insights = fin["summary_insights"]
    assert "Gradient" in insights["concepts_mentioned"]

    # the summary is stored on the student's progress page
    students = client.get("/api/students").json()
    progress = client.get(f"/api/students/{students[1]['id']}/progress").json()
    assert any("derivative of the loss" in s["text"] for s in progress["summaries"])


# 14. A short/unhelpful summary never lowers the session features
@pytest.mark.skipif(not hmm_available(), reason="HMM not trained")
def test_short_summary_not_penalized():
    results = []
    for summary in ("", "Nice lecture."):
        sid = _start_backprop(2)
        client.post(f"/api/sessions/{sid}/respond", json={"text": GOOD})
        fin = client.post(f"/api/sessions/{sid}/finish", json={
            "attention": 7, "confidence": 6, "difficulty": 5, "summary": summary}).json()
        results.append(fin["session_features"])
    without, with_short = results
    assert with_short["concept_coverage"] >= without["concept_coverage"]
    assert with_short["semantic_correctness"] == without["semantic_correctness"]


# 23. Under-challenged: strong evidence + high confidence + low difficulty
#     -> optional extension/challenge, NOT remedial material
def test_under_challenged_gets_extension():
    rec = recommend(3, None, None,
                    evidence={"demonstrated": ["Alpha", "Beta"], "unclear": []},
                    signals={"understanding": 0.9, "confidence": 0.9, "difficulty": 0.2},
                    topic_def=TDEF)
    assert rec["activity_state_key"] == "confident"
    assert rec["activity"]["kind"] == "challenge"
    assert "comfortable" in rec["why"]
    # neutral language: no ability labels
    assert "gifted" not in rec["why"].lower() and "better" not in rec["why"].lower()


# 24. Good understanding + low confidence -> confidence-building application
def test_low_confidence_good_understanding_gets_application():
    rec = recommend(4, None, None,
                    evidence={"demonstrated": ["Alpha", "Beta"], "unclear": []},
                    signals={"understanding": 0.85, "confidence": 0.2, "difficulty": 0.6},
                    topic_def=TDEF)
    assert rec["activity_state_key"] == "understanding"
    assert rec["activity"]["kind"] == "application"
    assert "confidence" in rec["why"].lower()


# 22. Low understanding + high confidence -> support activity plus a gentle check note
def test_low_understanding_high_confidence_gets_check_note():
    rec = recommend(1, None, None,
                    evidence={"demonstrated": [], "unclear": ["Alpha"]},
                    signals={"understanding": 0.2, "confidence": 0.9, "difficulty": 0.4},
                    topic_def=TDEF)
    assert rec["activity_state_key"] == "unclear"       # support material stays
    assert rec["activity"]["kind"] == "concept_review"
    assert any("confidence is high" in n for n in rec["notes"])


# 34-35. Confidence/difficulty never assign a state: the reported state_key is
#        always the HMM state passed in, whatever the signals say
def test_signals_never_change_the_state():
    for idx in range(5):
        for signals in ({"understanding": 0.9, "confidence": 1.0, "difficulty": 0.0},
                        {"understanding": 0.0, "confidence": 1.0, "difficulty": 1.0},
                        None):
            rec = recommend(idx, None, None, signals=signals, topic_def=TDEF)
            assert rec["state_key"] == STATE_KEYS[idx]


# 25 + template activities: a lecture-created topic WITHOUT stored activities
#    still yields an actionable activity built from its own concepts
def test_template_activity_for_lecture_topic():
    tdef = {
        "name": "Loops in Python",
        "concepts": [
            {"name": "For loops", "description": "A for loop repeats a block once per item."},
            {"name": "While loops", "description": "A while loop repeats while a condition holds."},
        ],
        "relationships": [],
        "activities": [],
    }
    for state_idx in range(5):
        rec = recommend(state_idx, tdef["activities"],
                        evidence={"demonstrated": [], "unclear": ["For loops"]}, topic_def=tdef)
        a = rec["activity"]
        assert a["content"] and a["question"], f"state {state_idx} activity not performable"
    # the review template focuses the concept that needs clarification
    rec = recommend(1, tdef["activities"],
                    evidence={"demonstrated": [], "unclear": ["For loops"]}, topic_def=tdef)
    assert "For loops" in rec["activity"]["title"]

    # a lecture topic WITH reviewed activities prefers the stored ones (the
    # seeded Strings lecture publishes its reviewed activities)
    topics = client.get("/api/topics").json()
    strings = next(t for t in topics if "Strings" in t["name"])
    sdef = client.get(f"/api/topics/{strings['id']}").json()
    assert sdef["activities"], "seeded Strings topic should carry reviewed activities"
    rec = recommend(1, sdef["activities"], topic_def=sdef)
    assert rec["activity"]["title"] in {a["title"] for a in sdef["activities"]}
