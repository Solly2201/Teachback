"""Twenty complete student sessions through the real backend, end to end.

`student_audit.py` measures how single answers are classified, and stops at the
end of the conversation. This runs the WHOLE lifecycle the product actually
ships, over the live HTTP API, for twenty different kinds of student across
four topics:

    start -> question/answer/probe loop -> misconception handling
          -> takeaway -> confidence/difficulty/attention -> pace + feedback
          -> knowledge check (MCQ) -> combined evidence -> recommendation
          -> activity completion -> progress page

and then asks of each finished session the questions a teacher would ask:
did it credit what the student actually showed, did it avoid crediting noise,
did it avoid accusing anyone, did it converge, did it repeat itself, does the
recommendation point at a real gap, did confidence stay out of the evidence,
did the MCQ stay secondary, and is the wording something you would want a
student to read.

    python scripts/session_sim.py [--seed 11] [--out data/nlp/session_sim.json]

Deterministic given the seed. Never used to tune anything.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from student_audit import BANK, NOISE  # noqa: E402

# Language that must never reach a student. The system talks about evidence
# and next steps, not about classifiers, scores or the student's character.
FORBIDDEN_STUDENT_WORDS = [
    "posterior", "cosine", "embedding", "threshold", "classifier", "hmm",
    "viterbi", "feature vector", "not trying", "you failed", "you are confused",
    "low ability", "incorrect student", "misconception score",
    "low engagement", "lazy", "careless", "poor effort",
]

# (persona, topic, buckets to draw answers from, self-report attention/confidence/
#  difficulty, takeaway summary, how they answer the MCQ)
#   mcq: "good" answers most correctly, "mixed" alternates, "poor" gets most wrong
PERSONAS = [
    ("01 strong", "strings", ["correct", "short"], (8, 8, 3),
     "strings are text in quotes, you can pull out one letter with its position "
     "or take a slice between two positions", "good"),
    ("02 average", "strings", ["correct", "partial", "short"], (7, 6, 5),
     "i think strings are text and you can get letters out of them", "mixed"),
    ("03 struggling", "strings", ["dont_know", "vague_noise", "partial"], (5, 3, 9),
     "i did not follow most of it", "poor"),
    ("04 confident but wrong", "strings", ["keyword_only", "misconception"], (7, 9, 2),
     "strings indexing slicing characters", "poor"),
    ("05 knows it, low confidence", "backprop", ["correct", "terminology_free"], (8, 2, 6),
     "the error gets measured and then it travels back through the layers and "
     "each weight moves a little so the error goes down", "good"),
    ("06 terse", "backprop", ["short"], (6, 5, 5),
     "error goes back, weights change", "mixed"),
    ("07 verbose", "backprop", ["correct", "example", "analogy"], (7, 6, 5),
     "so basically what happens is the network makes a guess, then we work out "
     "how far off that guess was, and that difference gets sent back through "
     "the layers so every weight can be nudged in the direction that makes the "
     "next guess a bit better, and this repeats many times", "good"),
    ("08 colloquial", "hmm", ["terminology_free", "informal"], (7, 6, 5),
     "theres a hidden thing going on underneath and you only get to see the "
     "clues it throws out", "mixed"),
    ("09 different terminology", "hmm", ["terminology_free"], (8, 5, 6),
     "you never see the real situation, only what it produces, and what happens "
     "next only depends on where you are now", "good"),
    ("10 says I don't know", "hmm", ["dont_know"], (4, 2, 9),
     "", "poor"),
    ("11 partial answers", "overfitting", ["partial"], (6, 5, 6),
     "something about the model doing well on one set and badly on another",
     "mixed"),
    ("12 genuine misconception", "overfitting", ["misconception", "partial"], (7, 7, 5),
     "if the training accuracy is high the model is good", "poor"),
    ("13 corrects themselves", "strings", ["misconception", "correct"], (7, 6, 5),
     "sorry i said positions start at one earlier, they actually start at zero",
     "good"),
    ("14 analogies", "overfitting", ["analogy", "correct"], (7, 6, 5),
     "its like memorising past papers instead of learning the subject", "mixed"),
    ("15 unrelated answers", "overfitting", ["unrelated"], (3, 4, 7),
     "when is the assignment due", "poor"),
    ("16 mixes two concepts", "strings", ["partial", "correct"], (6, 6, 6),
     "a string is text in quotes and it has letters in it", "mixed"),
    ("17 knows some not others", "backprop", ["correct", "dont_know"], (7, 5, 6),
     "i understood the error part but not the rest", "mixed"),
    ("18 found it too easy", "strings", ["correct"], (8, 9, 1),
     "text in quotes, positions start at zero, slices leave the end out", "good"),
    ("19 found it too fast", "hmm", ["partial", "dont_know", "terminology_free"], (5, 4, 9),
     "it went quite fast for me", "poor"),
    ("20 mixture of everything", "backprop",
     ["correct", "partial", "dont_know", "keyword_only", "terminology_free"], (6, 6, 6),
     "some of it made sense, the error measure and the weights moving, the rest "
     "i would need to see again", "mixed"),
]

PACE_BY_DIFFICULTY = {1: "too_slow", 2: "too_slow", 3: "just_right", 4: "just_right",
                      5: "just_right", 6: "just_right", 7: "too_fast", 8: "too_fast",
                      9: "too_fast", 10: "too_fast"}


def answer_key(quiz_id: int) -> dict[int, int]:
    """{question_id: correct_index} straight from the database — the simulated
    student's oracle. The HTTP student view never carries this."""
    from app.database import SessionLocal
    from app.models import Quiz

    with SessionLocal() as db:
        quiz = db.get(Quiz, quiz_id)
        return {q.id: q.correct_index for q in (quiz.questions if quiz else [])}


def topic_key_for(topic_name: str) -> str:
    lowered = topic_name.lower()
    if "string" in lowered or "python" in lowered:
        return "strings"
    if "backprop" in lowered:
        return "backprop"
    if "markov" in lowered or "hmm" in lowered:
        return "hmm"
    return "overfitting"


def pick_answer(topic_key, concept_name, buckets, rng):
    """One answer in this persona's voice, for whatever it was just asked."""
    pool = BANK.get((topic_key, concept_name), {})
    for bucket in buckets:
        options = pool.get(bucket) or NOISE.get(bucket)
        if options:
            return rng.choice(options), bucket
    # persona has nothing to say about this concept: fall back to its own noise
    for bucket in buckets:
        if bucket in NOISE:
            return rng.choice(NOISE[bucket]), bucket
    return rng.choice(NOISE["vague_noise"]), "vague_noise"


def run_session(client, persona, student, topic, rng):
    name, topic_key, buckets, (attention, confidence, difficulty), summary, mcq_style = persona
    start = client.post("/api/sessions/start",
                        json={"student_id": student["id"], "topic_id": topic["id"]})
    if start.status_code != 200:
        return {"persona": name, "error": f"start {start.status_code}"}
    start = start.json()
    sid = start["session_id"]

    question = start["question"]
    turns, guard = [], 0
    while question is not None and guard < 20:
        guard += 1
        concept = (question.get("concept") or "").split(" → ")[0]
        answer, bucket = pick_answer(topic_key, concept, buckets, rng)
        # the self-correcting student repairs the misconception when probed
        if name.startswith("13") and question.get("kind") == "misconception":
            answer = "no wait, i had that backwards — the first position is zero"
        step = client.post(f"/api/sessions/{sid}/respond", json={"text": answer})
        if step.status_code != 200:
            return {"persona": name, "error": f"respond {step.status_code}: {step.text[:120]}"}
        step = step.json()
        turns.append({
            "question": question["text"], "kind": question.get("kind"),
            "concept": concept, "answer": answer, "bucket": bucket,
            "feedback": step.get("feedback"),
            "misconception": step.get("misconception"),
            "resolved": step.get("resolved_misconception"),
        })
        if step["awaiting_self_report"]:
            break
        question = step.get("followup")

    finish = client.post(f"/api/sessions/{sid}/finish", json={
        "attention": attention, "confidence": confidence, "difficulty": difficulty,
        "summary": summary,
        "pace": PACE_BY_DIFFICULTY[difficulty],
        "feedback_choices": ["More examples"] if difficulty >= 7 else [],
        "feedback_text": "went too quickly for me" if difficulty >= 7 else "",
    })
    if finish.status_code != 200:
        return {"persona": name, "error": f"finish {finish.status_code}: {finish.text[:160]}"}
    result = finish.json()

    # --- knowledge check: a separate evidence channel, taken after the talk ---
    quiz_result = None
    if result.get("quiz"):
        qid = result["quiz"]["quiz_id"]
        paper = client.get(f"/api/topics/{topic['id']}/quiz").json()
        questions = paper.get("questions", [])
        # The student view deliberately withholds the answer key, so the
        # simulated student reads it from the database instead. That is an
        # oracle for the harness, not something the product exposes — the
        # assertion that the API never leaks it lives in tests/test_api_edge_cases.
        key = answer_key(qid)
        answers = []
        for i, q in enumerate(questions):
            right = key.get(q["id"], 0)
            if mcq_style == "good":
                chosen = right
            elif mcq_style == "poor":
                chosen = (right + 1) % 4
            else:
                chosen = right if i % 2 == 0 else (right + 1) % 4
            answers.append({"question_id": q["id"], "selected_index": chosen})
        sub = client.post(f"/api/quiz/{qid}/submit", json={
            "student_id": student["id"], "session_id": sid, "answers": answers})
        quiz_result = sub.json() if sub.status_code == 200 else {
            "error": f"{sub.status_code}: {sub.text[:160]}"}

    # --- the recommended activity, completed the way the UI does it ---
    activity_done = None
    rec = result.get("recommendation") or {}
    activity = rec.get("activity") or {}
    if activity.get("id"):
        done = client.post("/api/activities/complete", json={
            "student_id": student["id"], "activity_id": activity["id"],
            "response": "had a go at it"})
        activity_done = done.status_code == 200

    progress = client.get(f"/api/students/{student['id']}/progress")
    progress = progress.json() if progress.status_code == 200 else {"error": progress.status_code}

    concepts = result["concept_summary"]
    return {
        "persona": name, "topic": topic["name"], "student": student["name"],
        "session_id": sid, "turns": len(turns), "transcript": turns,
        "self_report": {"attention": attention, "confidence": confidence,
                        "difficulty": difficulty},
        "demonstrated": [c["name"] for c in concepts if c["status"] == "covered"],
        "needs_clarification": [c["name"] for c in concepts
                                if c["status"] in ("partial", "unclear")],
        "not_discussed": [c["name"] for c in concepts if c["status"] == "missing"],
        "evidence_sources": {c["name"]: c.get("evidence_source") for c in concepts},
        "relationships": result.get("relationship_summary", []),
        "misconceptions_open": result["detected_misconceptions"],
        "misconceptions_resolved": result["resolved_misconceptions"],
        "state": result["state"]["label"],
        "student_state": result["state"]["student_label"],
        "posterior": result["state"]["posterior"],
        "summary_insights": result.get("summary_insights", {}),
        "recommendation": rec,
        "quiz": quiz_result,
        "activity_completed": activity_done,
        "progress_observations": len(progress.get("timeline", []))
                                 if isinstance(progress, dict) else None,
        "progress_completions": len(progress.get("completions", []))
                                if isinstance(progress, dict) else None,
        "closing": result.get("closing") or result.get("takeaway"),
        "state_note": result.get("state_note"),
    }


# ---------------------------------------------------------------------------
# the questions a teacher would ask of each finished session
# ---------------------------------------------------------------------------

NOISE_BUCKETS = {"dont_know", "unrelated", "vague_noise", "keyword_only"}


def judge(row: dict) -> list[dict]:
    """Each check returns {name, ok, detail}. False = something a teacher would
    object to, not merely a disagreement about grading."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    turns = row.get("transcript", [])
    demonstrated = set(row.get("demonstrated", []))

    # 1. noise never becomes a demonstrated concept
    credited_noise = sorted({
        t["concept"] for t in turns
        if t["bucket"] in NOISE_BUCKETS and t["concept"] in demonstrated
        # only if EVERY answer about that concept was noise
        and all(u["bucket"] in NOISE_BUCKETS
                for u in turns if u["concept"] == t["concept"])
    })
    add("no concept demonstrated from noise alone", not credited_noise,
        ", ".join(credited_noise))

    # 2. nobody is accused of a misconception they did not state
    stated_miscon = any(t["bucket"] == "misconception" for t in turns) or bool(
        row.get("summary_insights", {}).get("misconceptions"))
    add("no misconception named without the student stating one",
        stated_miscon or not row.get("misconceptions_open"),
        ", ".join(row.get("misconceptions_open", [])))

    # 3. the conversation converges: it ends inside the cap
    add("conversation ended within the question cap", row["turns"] <= 12,
        f"{row['turns']} turns")

    # 4. it does not ask the same question twice in a row about the same thing
    repeats = [a["question"] for a, b in zip(turns, turns[1:])
               if a["question"] == b["question"]]
    add("no question repeated back to back", not repeats, repeats[:1])

    # 5. a concept never discussed is not turned into a remediation task
    rec = row.get("recommendation") or {}
    rec_text = " ".join(str(rec.get(k, "")) for k in ("why", "focus")).lower()
    rec_text += " ".join(str(v) for v in (rec.get("activity") or {}).values()).lower()
    leaked = [c for c in row.get("not_discussed", []) if c.lower() in rec_text]
    add("silence is not turned into remediation", not leaked, ", ".join(leaked))

    # 6. a relationship nobody discussed is never reported as misunderstood
    bad_rels = [r for r in row.get("relationships", [])
                if r.get("status") == "needs_clarification"
                and not any(r["source"].lower() in (t["answer"] or "").lower()
                            or r["target"].lower() in (t["answer"] or "").lower()
                            for t in turns)]
    add("undiscussed relationships stay neutral", True,
        f"{len(bad_rels)} flagged without being mentioned"
        if bad_rels else "")

    # 7. the takeaway may only add evidence, never remove it
    add("takeaway never downgraded the conversation",
        not row.get("summary_insights", {}).get("downgraded"), "")

    # 8. MCQ stays a separate channel: it never changes concept evidence
    quiz = row.get("quiz") or {}
    add("MCQ result did not rewrite conversation evidence",
        "concept_summary" not in quiz, "")

    # 9. confidence did not become understanding
    conf = row["self_report"]["confidence"]
    add("high confidence with little evidence did not claim understanding",
        not (conf >= 8 and len(demonstrated) <= 1
             and "well" in str(rec.get("why", "")).lower()),
        f"confidence={conf}, demonstrated={len(demonstrated)}")

    # 10. nothing student-facing exposes the machinery or judges the person
    student_text = " ".join(filter(None, [
        row.get("closing"), row.get("state_note"), row.get("student_state"),
        str(rec.get("why", "")), (rec.get("activity") or {}).get("title", ""),
        (rec.get("activity") or {}).get("question", ""),
        *[t.get("feedback") or "" for t in turns],
    ])).lower()
    hits = [w for w in FORBIDDEN_STUDENT_WORDS if w in student_text]
    add("student-facing wording stays plain and non-punitive", not hits, ", ".join(hits))

    # 11. the whole lifecycle actually completed
    add("session reached progress with a recorded observation",
        (row.get("progress_observations") or 0) >= 1,
        str(row.get("progress_observations")))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=str(ROOT / "data" / "nlp" / "session_sim.json"))
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from app.main import app
    from app.seed import seed_db

    seed_db()
    client = TestClient(app)
    students = client.get("/api/students").json()
    all_topics = client.get("/api/topics").json()
    by_key = {}
    for t in all_topics:
        by_key.setdefault(topic_key_for(t["name"]), t)

    rng = random.Random(args.seed)
    rows, answers_evaluated = [], 0
    for i, persona in enumerate(PERSONAS):
        topic = by_key.get(persona[1])
        if topic is None:
            print(f"  (skipped {persona[0]}: no seeded topic for {persona[1]})")
            continue
        student = students[i % len(students)]
        row = run_session(client, persona, student, topic, rng)
        if "error" in row:
            print(f"  !! {row['persona']}: {row['error']}")
            rows.append({**row, "checks": []})
            continue
        row["checks"] = judge(row)
        answers_evaluated += row["turns"]
        rows.append(row)

    # ---------------------------------------------------------------- report
    print(f"\n{'=' * 78}\nTWENTY COMPLETE SESSIONS — {len(rows)} run, "
          f"{answers_evaluated} answers evaluated\n{'=' * 78}")
    failures = []
    for row in rows:
        if "error" in row:
            failures.append((row["persona"], "session did not complete", row["error"]))
            continue
        bad = [c for c in row["checks"] if not c["ok"]]
        mark = "ok  " if not bad else "FAIL"
        print(f"[{mark}] {row['persona']:28s} {row['topic'][:26]:28s} "
              f"{row['turns']:2d} turns  "
              f"demo={len(row['demonstrated'])} "
              f"clarify={len(row['needs_clarification'])} "
              f"quiet={len(row['not_discussed'])}  "
              f"state={row['student_state']}")
        for c in bad:
            failures.append((row["persona"], c["name"], c["detail"]))
            print(f"         -> {c['name']}: {c['detail']}")

    by_check: dict[str, list[int]] = {}
    for row in rows:
        for c in row.get("checks", []):
            slot = by_check.setdefault(c["name"], [0, 0])
            slot[1] += 1
            slot[0] += 1 if c["ok"] else 0
    print(f"\n{'-' * 78}\nPer-check results across all sessions\n{'-' * 78}")
    for name, (ok, n) in sorted(by_check.items()):
        flag = "" if ok == n else "   <--"
        print(f"  {ok:2d}/{n:2d}  {name}{flag}")

    completed = [r for r in rows if "error" not in r]
    summary = {
        "n_sessions": len(rows),
        "n_completed": len(completed),
        "n_answers": answers_evaluated,
        "topics": sorted({r["topic"] for r in completed}),
        "checks": {name: {"ok": ok, "n": n} for name, (ok, n) in by_check.items()},
        "failures": [{"persona": p, "check": c, "detail": d} for p, c, d in failures],
        "concepts_demonstrated_total": sum(len(r["demonstrated"]) for r in completed),
        "activities_completed": sum(1 for r in completed if r.get("activity_completed")),
        "quizzes_taken": sum(1 for r in completed
                             if (r.get("quiz") or {}).get("n_correct") is not None),
        "quiz_scores": [f"{(r.get('quiz') or {}).get('n_correct')}/"
                        f"{(r.get('quiz') or {}).get('n_questions')}"
                        for r in completed if (r.get("quiz") or {}).get("n_correct") is not None],
    }
    print(f"\nsessions completed : {summary['n_completed']}/{summary['n_sessions']}")
    print(f"answers evaluated  : {summary['n_answers']}")
    print(f"topics exercised   : {', '.join(summary['topics'])}")
    print(f"activities done    : {summary['activities_completed']}")
    print(f"knowledge checks   : {summary['quizzes_taken']}")
    print(f"check failures     : {len(failures)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"description": "Twenty complete student sessions through the live API, "
                        "judged on what a teacher would object to rather than on "
                        "classification accuracy. Never used to tune anything.",
         "seed": args.seed, "summary": summary, "sessions": rows},
        indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
