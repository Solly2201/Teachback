"""End-to-end backend simulation of a real faculty + student journey.

This is deliberately NOT a unit test. It drives the live HTTP API the way the
frontend does, in order, as one faculty user and one student would, and prints
what the system actually produced at every step so the output can be read and
judged by a human rather than only asserted on.

    python scripts/simulate_user.py            # resets the demo database first
    python scripts/simulate_user.py --keep     # run against the current database

Faculty: switch subject -> create a lecture from an uploaded slide PDF ->
inspect the cleaned extraction -> analyse -> inspect concepts, provenance,
meanings, facts, questions -> edit one concept -> publish.
Student: TeachBack in casual language, one "I don't know", an easier question,
a relationship left undiscussed and one answered, a takeaway summary,
confidence/difficulty/pace/feedback, the 10-question knowledge check, the
recommendation and the activity.
Faculty again: dashboard, feedback aggregation, subject isolation, delete the
lecture, and confirm the student's history survived intact.
"""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.seed import seed_db  # noqa: E402

# Start from the pristine demo dataset so the run is reproducible and the
# output is readable; --keep runs against whatever is already there.
seed_db(force="--keep" not in sys.argv)
client = TestClient(app)

STEP = [0]
FAILURES: list[str] = []


def step(title: str) -> None:
    STEP[0] += 1
    print(f"\n{'=' * 78}\n{STEP[0]:>2}. {title}\n{'=' * 78}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"    [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def get(path):
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def post(path, payload=None, expect=200):
    r = client.post(path, json=payload if payload is not None else {})
    if r.status_code != expect:
        raise AssertionError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


# --------------------------------------------------------------------------
step("Faculty switches to the Python Programming subject")
teachers = get("/api/teachers")
subjects = {s["name"]: s for t in teachers for s in t["subjects"]}
py = subjects["Python Programming"]
nn = subjects["Neural Networks"]
print(f"    teachers: {[t['name'] for t in teachers]}")
print(f"    selected: {py['name']} (id={py['id']})")
check("two demo subjects available", len(subjects) >= 2)

# --------------------------------------------------------------------------
step("Faculty uploads today's slide PDF")
from pdf_decks import cloud_deck  # noqa: E402

pdf = cloud_deck()
extract = post("/api/lectures/extract",
               {"filename": "lecture-slides.pdf",
                "content_base64": base64.b64encode(pdf).decode()})
report = extract["report"]
print(f"    pages: {report['page_count']}  extractor: {report['extractor']}")
print(f"    removed {report['removed_total']} decoration lines:")
for r in report["removed_by_reason"]:
    print(f"      {r['count']:>3}x {r['label']:<32} e.g. {r['examples'][0][:52]!r}")
print(f"    empty (boilerplate-only) pages skipped: {report['empty_pages']}")

# --------------------------------------------------------------------------
step("Faculty inspects the cleaned extraction")
material = extract["text"]
print("    ---- cleaned material (first 900 chars) ----")
print("    " + material[:900].replace("\n", "\n    "))
lowered = material.lower()
check("no copyright text survived", "copyright" not in lowered and "©" not in material)
check("no 'all rights reserved'", "all rights reserved" not in lowered)
check("running header removed", "cloud infrastructure and services" not in lowered)
check("running footer removed", "module 1: introduction to cloud computing" not in lowered)
check("page provenance markers present", "<!-- page" in material)
check("raw extraction still available", "Copyright" in extract["raw_text"])

# --------------------------------------------------------------------------
step("Faculty analyses the material (deterministic NLP, no LLM)")
lecture = post("/api/lectures", {
    "subject_id": py["id"],
    "title": "Cloud Computing Basics",
    "description": "What the cloud is and what makes it a cloud.",
    "objectives": ["Explain what cloud computing is.",
                   "Explain what on-demand self-service means."],
    "material_text": material,
    "ingestion": report,
})
concepts = lecture["draft"]["concepts"]
print(f"    lecture id={lecture['id']} status={lecture['status']}")
notes = lecture["suggestions"]["structure"]["notes"]
print(f"    draft quality notes: {notes or 'none — every suggestion is well supported'}")

# --------------------------------------------------------------------------
step("Faculty inspects the extracted concepts, provenance, meanings and facts")
for c in concepts:
    where = f"page {c['source_page']}" if c.get("source_page") else "pasted notes"
    print(f"\n    CONCEPT   {c['name']}   [{c.get('confidence_label')}]")
    print(f"    SOURCE    {where}, under {c['source_section']!r}")
    print(f"    MEANING   {c['description'][:110]}")
    for f in c["facts"]:
        print(f"    FACT      {f[:105]}")
    for e in c["examples"]:
        print(f"    EXAMPLE   {e[:105]}")
    print(f"    WHY       {c['confidence_reason'][:150]}")

banned = ("copyright", "all rights reserved", "©")
clean = all(bad not in " ".join([c["name"], c["description"]] + c["facts"]).lower()
            for c in concepts for bad in banned)
check("no boilerplate became a concept/meaning/fact", clean)
check("every concept has real supporting evidence",
      all(c["description"] or c["facts"] for c in concepts))
check("every concept keeps page provenance",
      all(c.get("source_page") or c.get("source_pages") for c in concepts))
names = {c["name"].lower() for c in concepts}
check("document structure did not become a concept",
      not ({"thank you", "lesson: cloud computing overview"} & names),
      f"suggested: {sorted(names)}")
check("concept count follows the evidence, not a fixed number",
      0 < len(concepts) <= 8, f"{len(concepts)} concepts")

# --------------------------------------------------------------------------
step("Faculty inspects the drafted questions")
for c in concepts[:2]:
    print(f"    {c['name']}:")
    for key in ("main_question", "easier_question", "probe_question", "application_question"):
        if c.get(key):
            print(f"      {key:<22} {c[key][:100]}")
check("main questions are conversational, not exam-style",
      all(c["main_question"].startswith("What did you understand about") for c in concepts))
check("application questions are optional extras",
      all("application_question" in c for c in concepts))

# --------------------------------------------------------------------------
step("Faculty edits one concept and publishes")
edited = [dict(c) for c in concepts]
edited[0]["description"] = (
    "Cloud computing means using computing resources over the network, on demand, "
    "instead of owning the hardware yourself."
)
lecture = client.put(f"/api/lectures/{lecture['id']}", json={
    "concepts": edited,
    "relationships": lecture["draft"]["relationships"],
    "activities": lecture["draft"]["activities"],
    "quiz": lecture["draft"]["quiz"],
}).json()
check("teacher edit persisted",
      lecture["draft"]["concepts"][0]["description"].startswith("Cloud computing means using"))

published = post(f"/api/lectures/{lecture['id']}/publish")
topic = published["topic"]
print(f"    published topic id={topic['id']} with {len(topic['concepts'])} concepts, "
      f"{len(topic['relationships'])} relationships, {len(topic['activities'])} activities")
check("lecture is live", published["lecture"]["status"] == "published")
check("published concepts carry provenance",
      all(c["source"].get("section") for c in topic["concepts"]))

# --------------------------------------------------------------------------
step("Published lecture is retrievable and appears for the right subject only")
fetched = get(f"/api/topics/{topic['id']}")
check("topic fetches", fetched["id"] == topic["id"])
py_topics = {t["id"] for t in get(f"/api/topics?subject_id={py['id']}")}
nn_topics = {t["id"] for t in get(f"/api/topics?subject_id={nn['id']}")}
check("new topic is in the Python subject", topic["id"] in py_topics)
check("new topic is NOT in Neural Networks", topic["id"] not in nn_topics)
check("subjects share no topics", not (py_topics & nn_topics))

# --------------------------------------------------------------------------
step("Student starts TeachBack")
students = get("/api/students")
student = next(s for s in students if s["name"] == "Shreshtha Bindal")
session = post("/api/sessions/start", {"student_id": student["id"], "topic_id": topic["id"]})
sid = session["session_id"]
print(f"    intro:  {session['intro'][:120]}")
print(f"    Q1:     {session['prompt']}")


def answer(text):
    out = post(f"/api/sessions/{sid}/respond", {"text": text})
    print(f"\n    STUDENT  {text}")
    print(f"    TUTOR    {out['feedback']}")
    if out.get("followup"):
        print(f"    NEXT     {out['followup']['text']}")
        print(f"    WHY      {out['followup']['reason']}")
    return out


# --------------------------------------------------------------------------
step("Student answers casually, says 'I don't know' once, then tries the easier question")
# realistic, casual answers — one per concept, plus one honest blank
BY_CONCEPT = {
    "Cloud Computing": "its basically renting computers over the internet instead of buying them",
    "On-demand Self-service": "you just ask for a server and you get one, nobody has to set it up for you",
    "Broad Network Access": "you can reach it from a laptop or a phone over the normal network",
    "Service Models": "its how much the provider looks after - iaas is just machines, saas is the whole app",
    "Measured Service": "they keep track of how much you use and bill you for that much",
}
turn = answer(BY_CONCEPT["Cloud Computing"])
gave_blank = False
while turn.get("followup") and not turn["awaiting_self_report"]:
    followup = turn["followup"]
    if followup["kind"] == "relationship":
        # one relationship answered, any others left undiscussed on purpose
        turn = answer("the service model decides how much of it you manage yourself")
        continue
    if not gave_blank:
        # one honest "I don't know", then the easier question is attempted
        gave_blank = True
        turn = answer("i don't know")
        continue
    concept = followup.get("concept") or ""
    turn = answer(BY_CONCEPT.get(concept, "i think it means you use it over the network"))
print(f"\n    conversation finished after {turn['question_no']} questions")
print(f"    closing: {turn.get('closing')}")

# --------------------------------------------------------------------------
step("Student gives a short takeaway summary and their self-report")
result = post(f"/api/sessions/{sid}/finish", {
    "attention": 7, "confidence": 6, "difficulty": 4,
    "summary": "cloud is basically using someone else's computers over the network whenever you need them.",
    "pace": "just right",
    "feedback_choices": ["More examples"],
    "feedback_text": "the service model slide went a bit fast",
})
print("    CONCEPT EVIDENCE")
for c in result["concept_summary"]:
    print(f"      {c['status']:<9} {c['name']:<28} (evidence from: {c.get('evidence_source')})")
print("    RELATIONSHIP EVIDENCE")
for r in result["relationship_summary"]:
    print(f"      {r['status_label']:<20} {r['source']} -> {r['target']}")
print(f"    summary insights: {result['summary_insights']}")
check("summary evidence is labelled as such",
      all(c.get("evidence_source") for c in result["concept_summary"]))
check("'not discussed' is a separate state from 'needs clarification'",
      {r["status"] for r in result["relationship_summary"]} <=
      {"demonstrated", "not_discussed", "needs_clarification"})

# --------------------------------------------------------------------------
step("Learning state, evidence and recommendation")
print(f"    state (faculty wording): {result['state']['label']}")
print(f"    state (student wording): {result['state']['student_label']}")
print(f"    posterior meaning:       {result['state']['posterior_meaning']}")
print(f"    posterior:               {result['state']['posterior']}")
print("    WHY (evidence bullets):")
for e in result["observation"]["evidence"]:
    print(f"      - {e}")
rec = result["recommendation"]
print(f"    recommended: {rec['activity']['title']}")
print(f"    why:         {rec['why']}")
for note in rec["notes"]:
    print(f"    note:        {note}")
check("posterior is described as model confidence, not understanding",
      "model confidence" in result["state"]["posterior_meaning"].lower())
check("8-dimensional observation preserved", len(result["observation"]["features"]) == 8)
not_discussed = [r for r in result["relationship_summary"] if r["status"] == "not_discussed"]
if not_discussed:
    joined = (rec["why"] + " ".join(rec["notes"])).lower()
    check("undiscussed material is never called a mistake",
          "mistake" not in joined or "isn't a mistake" in joined or "not a mistake" in joined)

# --------------------------------------------------------------------------
step("Student completes the 10-question knowledge check")
quiz = get(f"/api/topics/{topic['id']}/quiz")
if quiz.get("available"):
    print(f"    {len(quiz['questions'])} questions")
    for q in quiz["questions"][:3]:
        print(f"      [{q['kind']}] {q['question'][:95]}")
        for i, opt in enumerate(q["options"]):
            print(f"          {'ABCD'[i]}. {opt[:80]}")
    # read the answer key straight from the database: regenerating the quiz
    # would replace the question rows the student is looking at
    from app.database import SessionLocal as _S
    from app.models import QuizQuestion as _Q
    _db = _S()
    try:
        key = {q.id: q.correct_index
               for q in _db.query(_Q).filter(_Q.quiz_id == quiz["quiz_id"]).all()}
    finally:
        _db.close()
    # answer most correctly, get one wrong on purpose
    answers = []
    for i, q in enumerate(quiz["questions"]):
        correct = key[q["id"]]
        answers.append({"question_id": q["id"],
                        "selected_index": correct if i % 4 else (correct + 1) % 4})
    outcome = post(f"/api/quiz/{quiz['quiz_id']}/submit",
                   {"student_id": student["id"], "session_id": sid, "answers": answers})
    print(f"    result: {outcome['headline']}")
    print("    COMBINED EVIDENCE (TeachBack and MCQ kept separate)")
    for c in outcome["combined"]:
        print(f"      {c['name']:<28} teachback={str(c['teachback_status']):<8} "
              f"mcq={c['mcq_correct']}/{c['mcq_total']}  -> {c['verdict']}")
    check("no single blended mastery score is produced",
          all("score" not in c and "mastery" not in c for c in outcome["combined"]))
    check("MCQ never erases a TeachBack explanation",
          all(c["teachback_status"] is not None or c["mcq_total"] for c in outcome["combined"]))
    check("recommendation refreshed with combined evidence",
          outcome["updated_recommendation"] is not None)
else:
    check("knowledge check available", False, "no quiz generated")
    outcome = None

# --------------------------------------------------------------------------
step("HMM state is untouched by the knowledge check")
progress = get(f"/api/students/{student['id']}/progress")
latest = progress["timeline"][-1]
check("state label unchanged after the quiz", latest["state_label"] == result["state"]["label"],
      f"{latest['state_label']} vs {result['state']['label']}")
check("observation vector still 8-dimensional", len(latest["features"]) == 8)
check("quiz recorded only as an evidence note",
      any("Knowledge check" in e for e in latest["evidence"]))
print(f"    evidence notes: {latest['evidence']}")

# --------------------------------------------------------------------------
step("Student completes the recommended activity")
activity = (outcome["updated_recommendation"] if outcome else result["recommendation"])["activity"]
completion = post("/api/activities/complete", {
    "student_id": student["id"],
    "activity_id": activity.get("id"),
    "topic_id": topic["id"],
    "title": activity["title"], "kind": activity["kind"],
    "answer": "a school could rent servers only during results week instead of buying them.",
})
print(f"    {completion['message']}")
check("activity completion recorded", completion["completed"])

# --------------------------------------------------------------------------
step("Student progress page")
progress = get(f"/api/students/{student['id']}/progress")
print(f"    sessions in timeline: {len(progress['timeline'])}")
print(f"    completions: {[c['title'] for c in progress['completions'][:3]]}")
print(f"    saved takeaways: {len(progress['summaries'])}")
check("the student's own summary is shown back to them",
      any("someone else" in s["text"] for s in progress["summaries"]))

# --------------------------------------------------------------------------
step("Teacher dashboard, feedback aggregation and subject isolation")
py_view = get(f"/api/teacher/overview?subject_id={py['id']}")
nn_view = get(f"/api/teacher/overview?subject_id={nn['id']}")
print(f"    Python topics on dashboard:  {[t['name'] for t in py_view['topic_stats']]}")
print(f"    Neural Nets topics:          {[t['name'] for t in nn_view['topic_stats']]}")
fb = next((f for f in py_view["topic_feedback"] if f["id"] == topic["id"]), None)
if fb:
    print(f"    feedback for this lecture: pace={fb['pace']} requests={fb['common_requests']}")
    print(f"    recent comments: {fb['recent_comments']}")
check("this lecture's feedback reached the dashboard", fb is not None)
check("student comment aggregated",
      bool(fb and any("service model" in c for c in fb["recent_comments"])))
py_ids = {t["id"] for t in py_view["topic_stats"]}
nn_ids = {t["id"] for t in nn_view["topic_stats"]}
check("the two subjects share no topics", not (py_ids & nn_ids))
check("no Neural Networks session leaked into Python",
      all(o["topic_id"] in py_ids for o in py_view["recent_interactions"]))
check("no Python session leaked into Neural Networks",
      all(o["topic_id"] in nn_ids for o in nn_view["recent_interactions"]))

# --------------------------------------------------------------------------
step("Faculty deletes the lecture — student history must survive")
preview = get(f"/api/lectures/{lecture['id']}/delete-preview")
print(f"    mode: {preview['mode']}")
print(f"    history: {preview['history']}")
print(f"    dialog: {preview['message']}")
check("deletion is recognised as an archive, not an erase", preview["mode"] == "archive")

deleted = client.delete(f"/api/lectures/{lecture['id']}").json()
print(f"    result: {deleted['message']}")
check("archived rather than deleted", deleted["mode"] == "archived")

# --------------------------------------------------------------------------
step("Verify the deleted lecture is gone from every active surface")
active = [x["id"] for x in get(f"/api/lectures?subject_id={py['id']}")]
check("gone from the active lecture list", lecture["id"] not in active)
check("gone from the active topic list",
      topic["id"] not in {t["id"] for t in get(f"/api/topics?subject_id={py['id']}")})
after_view = get(f"/api/teacher/overview?subject_id={py['id']}")
check("gone from the subject dashboard",
      topic["id"] not in {t["id"] for t in after_view["topic_stats"]})
check("gone from the dashboard's recent interactions",
      all(o["topic_id"] != topic["id"] for o in after_view["recent_interactions"]))
check("new sessions are refused on it",
      client.post("/api/sessions/start",
                  json={"student_id": student["id"], "topic_id": topic["id"]}).status_code == 400)
archived_list = get(f"/api/lectures?subject_id={py['id']}&include_archived=true")
check("visible in the archived list", any(x["id"] == lecture["id"] and x["archived"]
                                          for x in archived_list))

# --------------------------------------------------------------------------
step("Verify no student record was destroyed and no reference dangles")
from app.database import SessionLocal  # noqa: E402
from app.models import (ActivityCompletion, Observation, Quiz, QuizAttempt,  # noqa: E402
                        TeachSession, Topic)

db = SessionLocal()
try:
    session_row = db.get(TeachSession, sid)
    check("the TeachBack session still exists", session_row is not None and session_row.completed)
    check("the student's takeaway summary is intact", bool(session_row.summary_text))
    check("the observation still exists",
          db.query(Observation).filter(Observation.session_id == sid).count() == 1)
    check("the completed activity still exists",
          db.query(ActivityCompletion).filter(
              ActivityCompletion.topic_id == topic["id"]).count() >= 1)
    if outcome:
        check("the quiz attempt still exists",
              db.query(QuizAttempt).filter(QuizAttempt.session_id == sid).count() == 1)
    dangling_sessions = [t.id for t in db.query(TeachSession).all()
                         if db.get(Topic, t.topic_id) is None]
    dangling_obs = [o.id for o in db.query(Observation).all()
                    if o.topic_id is not None and db.get(Topic, o.topic_id) is None]
    dangling_attempts = [a.id for a in db.query(QuizAttempt).all()
                         if db.get(Quiz, a.quiz_id) is None]
    check("no orphaned sessions", not dangling_sessions, str(dangling_sessions))
    check("no orphaned observations", not dangling_obs, str(dangling_obs))
    check("no orphaned quiz attempts", not dangling_attempts, str(dangling_attempts))
finally:
    db.close()

progress = get(f"/api/students/{student['id']}/progress")
check("the student's progress page still reads correctly",
      any(o["topic_name"] == topic["name"] for o in progress["timeline"]))
check("the archived topic is still fetchable for history",
      client.get(f"/api/topics/{topic['id']}").status_code == 200)

# --------------------------------------------------------------------------
step("Faculty restores the lecture, then removes it again to leave demo data clean")
restored = post(f"/api/lectures/{lecture['id']}/restore")
check("restore works", restored["archived"] is False)
check("restored topic is active again",
      topic["id"] in {t["id"] for t in get(f"/api/topics?subject_id={py['id']}")})
client.delete(f"/api/lectures/{lecture['id']}")

# --------------------------------------------------------------------------
step("Health and HMM integrity")
health = get("/api/health")
print(f"    health: {health}")
meta = get("/api/meta/states")
print(f"    HMM validation: ok={meta['hmm_validation']['ok']} "
      f"problems={meta['hmm_validation']['problems']}")
print(f"    faculty state names: {meta['state_names']}")
print(f"    student state names: {meta['state_student_names']}")
check("backend healthy", health["status"] == "ok" and health["hmm_trained"])
check("HMM validates", meta["hmm_validation"]["ok"])

print(f"\n{'=' * 78}")
if FAILURES:
    print(f"SIMULATION FINISHED WITH {len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"SIMULATION COMPLETE — all {STEP[0]} steps passed.")
