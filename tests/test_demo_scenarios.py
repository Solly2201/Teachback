"""End-to-end demo scenarios (spec: Demo A-K) against the seeded Strings topic.

These test the educational behaviour, not just the plumbing: simple language
is accepted, "I don't know" gets support instead of credit, misconceptions get
clarified and can be resolved, the session does not escalate into an exam, and
the student's own takeaway summary can only ever ADD evidence.
"""
import pytest
from fastapi.testclient import TestClient

from app.hmm.model import hmm_available
from app.main import app
from app.recommend.rules import recommend
from app.states import UNCLEAR, UNDERSTANDING

client = TestClient(app)

pytestmark = pytest.mark.skipif(not hmm_available(), reason="HMM not trained")

# casual, non-textbook answers a normal student would give
CASUAL_ANSWERS = {
    "Strings": "Strings are basically text that we store inside quotes.",
    "String assignment": "You can put the string into a variable using =.",
    "Characters": "They're basically individual letters or symbols inside the string.",
    "Indexing": "You use the position to get a character, and Python starts from zero.",
    "Slicing": "You can take a part of the string using start and end positions.",
    "split() and join()": "split breaks the text into pieces and join puts the pieces back together.",
}


def _strings_topic():
    teachers = client.get("/api/teachers").json()
    py = next(s for t in teachers for s in t["subjects"] if s["name"] == "Python Programming")
    topics = client.get(f"/api/topics?subject_id={py['id']}").json()
    return next(t for t in topics if "Strings" in t["name"])


def _student():
    students = client.get("/api/students").json()
    return next(s for s in students if s["name"] == "Shreshtha Bindal")


def _start():
    r = client.post("/api/sessions/start",
                    json={"student_id": _student()["id"], "topic_id": _strings_topic()["id"]})
    assert r.status_code == 200
    return r.json()


def _respond(sid, text):
    r = client.post(f"/api/sessions/{sid}/respond", json={"text": text})
    assert r.status_code == 200
    return r.json()


def _run_casual_session():
    """Answer every question casually; returns (session_id, steps)."""
    start = _start()
    sid = start["session_id"]
    prompt_concept = start["question"]["concept"]
    steps = []
    for _ in range(14):
        answer = CASUAL_ANSWERS.get(prompt_concept and prompt_concept.split(" → ")[0],
                                    "I think they are connected because one gives you the other.")
        step = _respond(sid, answer)
        steps.append(step)
        if step["awaiting_self_report"]:
            break
        prompt_concept = step["followup"]["concept"]
    return sid, steps


# Demo A + B + the central no-exam behaviour (spec 11/17): casual answers are
# accepted, concepts complete, and the session never escalates
def test_casual_answers_accepted_without_escalation():
    sid, steps = _run_casual_session()
    assert steps[-1]["awaiting_self_report"], "session did not finish"
    # simple answers earn the concepts — most of the timeline is done
    done = [c for c in steps[-1]["timeline"] if c["status"] == "done"]
    assert len(done) >= 4, f"casual answers under-credited: {steps[-1]['timeline']}"
    # no extension/application escalation just because answers were correct
    kinds = {s["followup"]["kind"] for s in steps if s.get("followup")}
    assert "extension" not in kinds
    # a correct answer moves ON (next concept), it does not trigger a probe
    first = steps[0]
    assert first["followup"]["concept"] != "Strings"
    # the whole check stays short
    assert len(steps) <= 8


# Demo E: "I don't know" gets no credit and an easier question
def test_i_dont_know_gets_easier_question_not_credit():
    start = _start()
    step = _respond(start["session_id"], "I don't know.")
    assert step["followup"]["kind"] == "easier"
    assert all(c["status"] != "done" for c in step["timeline"])
    assert step["analysis"]["detected_misconceptions"] == []


# Demo H: off-topic answers earn nothing and never become a misconception
def test_off_topic_answer_no_credit_no_misconception():
    start = _start()
    step = _respond(start["session_id"], "I really liked today's lecture, the room was nice.")
    assert step["analysis"]["detected_misconceptions"] == []
    assert all(c["status"] != "done" for c in step["timeline"])
    assert step["followup"] is not None  # supportive continuation


# Demo D: misconception detected -> clarification -> student corrects -> resolved
def test_misconception_clarified_and_resolved():
    start = _start()
    sid = start["session_id"]
    step = _respond(sid, "The first character of a string is at index 1.")
    assert step["misconception"] is not None
    assert step["misconception"]["name"] == "Indexing starts at 1"
    assert "0" in step["misconception"]["clarification"]
    assert step["followup"]["kind"] == "misconception"
    corrected = _respond(sid, "Oh right, Python starts at zero — the first character is at index 0.")
    assert corrected["resolved_misconception"] == "Indexing starts at 1"


# Demo G: a reasonable analogy is not rejected — the tutor asks to connect it back
def test_analogy_gets_connect_back_not_rejection():
    start = _start()
    step = _respond(start["session_id"],
                    "A string is like a row of numbered boxes with one letter in each box.")
    # either credited (it does express the idea) or probed — never a flat move-on
    assert step["followup"] is not None
    assert step["analysis"]["detected_misconceptions"] == []


# Demo K: the takeaway summary can add evidence and never penalises
def test_summary_adds_evidence_never_penalises():
    def run(summary):
        start = _start()
        sid = start["session_id"]
        _respond(sid, CASUAL_ANSWERS["Strings"])
        r = client.post(f"/api/sessions/{sid}/finish", json={
            "attention": 7, "confidence": 6, "difficulty": 4, "summary": summary})
        assert r.status_code == 200
        return r.json()

    base = run("")
    short = run("ok")
    good = run("I understood that strings store text, you can get characters using their "
               "positions starting at zero, and you can take parts of strings using slicing.")
    cc = lambda res: res["session_features"]["concept_coverage"]
    assert cc(short) >= cc(base), "a short summary must never lower evidence"
    assert cc(good) >= cc(base)
    assert good["summary_insights"]["concepts_mentioned"], "good summary not analysed"
    covered = {c["name"] for c in good["concept_summary"] if c["status"] in ("covered", "partial")}
    assert "Indexing" in covered or "Slicing" in covered, \
        "summary evidence did not upgrade any concept"


# Demo I/J + confidence separation (spec 27/28): recommendations react to the
# confidence signals, the HMM state does not
def test_confidence_signals_shape_recommendation_only():
    topic = client.get(f"/api/topics/{_strings_topic()['id']}").json()
    acts = topic["activities"]

    # I: under-challenged -> optional extension
    rec = recommend(UNDERSTANDING, acts, [], evidence={"demonstrated": ["Strings"], "unclear": []},
                    signals={"understanding": 0.9, "confidence": 0.9, "difficulty": 0.2},
                    topic_def=topic)
    assert rec["activity_state_key"] == "confident"
    assert "comfortable" in rec["why"]

    # strong evidence + low confidence -> confidence-building application
    rec = recommend(UNDERSTANDING, acts, [], evidence={"demonstrated": ["Strings"], "unclear": []},
                    signals={"understanding": 0.9, "confidence": 0.3, "difficulty": 0.5},
                    topic_def=topic)
    assert rec["activity_state_key"] == "understanding"
    assert "confidence" in rec["why"]

    # J: high confidence + weak evidence -> gentle double-check note
    rec = recommend(UNCLEAR, acts, [], evidence={"demonstrated": [], "unclear": ["Slicing"]},
                    signals={"understanding": 0.2, "confidence": 0.9, "difficulty": 0.2},
                    topic_def=topic)
    assert any("double-check" in n for n in rec["notes"])
    # the recommended activity stays grounded in this topic's own material
    assert rec["activity"]["title"] in {a["title"] for a in acts} or rec["activity"].get("generated")


# Activities recommended after a session come from the lecture's own reviewed set
def test_recommended_activity_grounded_in_lecture():
    sid, steps = _run_casual_session()
    r = client.post(f"/api/sessions/{sid}/finish",
                    json={"attention": 8, "confidence": 8, "difficulty": 3})
    assert r.status_code == 200
    result = r.json()
    topic = client.get(f"/api/topics/{_strings_topic()['id']}").json()
    titles = {a["title"] for a in topic["activities"]}
    act = result["recommendation"]["activity"]
    assert act["title"] in titles or act.get("generated"), \
        f"activity {act['title']!r} is neither teacher-reviewed nor topic-derived"
    assert result["recommendation"]["why"]
