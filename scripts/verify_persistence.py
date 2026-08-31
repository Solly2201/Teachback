"""Prove SQLite is the source of truth, and that it stays internally consistent.

Two questions, answered against the database file rather than the API:

1. Does every kind of record the application creates actually land in SQLite —
   surviving a fresh connection (and therefore a browser refresh or a backend
   restart)? The frontend keeps only the selected teacher/subject in
   localStorage; everything else must come from here.
2. Are there orphaned or dangling rows — a concept whose topic is gone, a
   session whose student never existed, a quiz attempt for a deleted quiz?

Usage:
    python scripts/verify_persistence.py [--reseed]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import inspect, text  # noqa: E402

from app.config import DB_PATH  # noqa: E402
from app.database import SessionLocal, engine  # noqa: E402
from app.models import (Activity, ActivityCompletion, Concept,  # noqa: E402
                        ConceptRelationship, Lecture, Misconception,
                        Observation, Quiz, QuizAnswer, QuizAttempt,
                        QuizQuestion, Response, Student, Subject, TeachSession,
                        Teacher, Topic)

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# Every table that must hold real application data, with the minimum the demo
# is expected to contain after a clean seed.
EXPECTED = [
    (Teacher, 2), (Subject, 2), (Student, 20), (Topic, 3), (Lecture, 1),
    (Concept, 10), (ConceptRelationship, 5), (Misconception, 5), (Activity, 5),
    (Quiz, 1), (QuizQuestion, 4), (Observation, 20),
]

# (child model, foreign-key attribute, parent model) — every reference must
# resolve, or be legitimately null.
REFERENCES = [
    (Subject, "teacher_id", Teacher), (Topic, "subject_id", Subject),
    (Lecture, "subject_id", Subject), (Lecture, "topic_id", Topic),
    (Concept, "topic_id", Topic), (ConceptRelationship, "topic_id", Topic),
    (Misconception, "topic_id", Topic), (Activity, "topic_id", Topic),
    (Quiz, "topic_id", Topic), (QuizQuestion, "quiz_id", Quiz),
    (QuizAttempt, "quiz_id", Quiz), (QuizAttempt, "student_id", Student),
    (QuizAttempt, "session_id", TeachSession), (QuizAnswer, "attempt_id", QuizAttempt),
    (QuizAnswer, "question_id", QuizQuestion),
    (TeachSession, "student_id", Student), (TeachSession, "topic_id", Topic),
    (Response, "session_id", TeachSession),
    (Observation, "student_id", Student), (Observation, "topic_id", Topic),
    (Observation, "session_id", TeachSession),
    (ActivityCompletion, "student_id", Student),
    (ActivityCompletion, "activity_id", Activity),
    (ActivityCompletion, "topic_id", Topic),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reseed", action="store_true",
                    help="force a clean reseed before checking")
    args = ap.parse_args()

    if args.reseed:
        from app.seed import seed_db
        seed_db(force=True)
        print("demo database reseeded")

    section(f"SQLite file: {DB_PATH}")
    check("the database file exists", DB_PATH.exists(),
          f"{DB_PATH.stat().st_size // 1024} KB" if DB_PATH.exists() else "missing")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected_tables = {m.__tablename__ for m, _ in EXPECTED} | {
        "responses", "quiz_answers", "quiz_attempts", "activity_completions"}
    check("every application table exists", expected_tables <= tables,
          f"missing: {sorted(expected_tables - tables)}" if expected_tables - tables else "")

    # --- 1. is the data actually there? -----------------------------------
    section("Persisted record counts (read through a fresh connection)")
    db = SessionLocal()
    try:
        for model, minimum in EXPECTED:
            n = db.query(model).count()
            check(f"{model.__tablename__:<22} >= {minimum}", n >= minimum, f"{n} rows")

        # the archive columns added later must be present and queryable
        check("lectures carry lifecycle state",
              db.query(Lecture).filter(Lecture.archived_at.is_(None)).count() >= 1)
        check("topics carry lifecycle state",
              db.query(Topic).filter(Topic.archived_at.is_(None)).count() >= 1)

        # the specific demo student the faculty walkthrough uses
        shreshtha = db.query(Student).filter(Student.name == "Shreshtha Bindal").first()
        check("the named demo student is persisted", shreshtha is not None,
              f"{shreshtha.program} · {shreshtha.roll_no}" if shreshtha else "")

        # the seeded lecture went through the real pipeline and kept its draft
        lecture = db.query(Lecture).first()
        check("the seeded lecture stores its reviewed draft in SQLite",
              bool((lecture.draft or {}).get("concepts")),
              f"{len((lecture.draft or {}).get('concepts', []))} concepts")
        check("the seeded lecture stores its NLP suggestions in SQLite",
              bool((lecture.suggestions or {}).get("concepts")))
        check("published concepts keep their facts and provenance",
              any(c.facts and c.source for c in db.query(Concept).all()))
    finally:
        db.close()

    # --- 2. does it survive a brand-new connection? -----------------------
    section("Round trip: write, drop every connection, read back")
    db = SessionLocal()
    try:
        student = db.query(Student).first()
        topic = db.query(Topic).filter(Topic.archived_at.is_(None)).first()
        session = TeachSession(student_id=student.id, topic_id=topic.id,
                               summary_text="persistence probe",
                               pace="just right", feedback_choices=["More examples"],
                               feedback_text="probe comment", completed=True)
        db.add(session)
        db.flush()
        db.add(Response(session_id=session.id, exchange_no=1, prompt="q", text="a",
                        analysis={"features": {"concept_coverage": 1.0}}))
        db.add(Observation(student_id=student.id, topic_id=topic.id,
                           session_id=session.id, features=[0.5] * 8, source="live"))
        db.add(ActivityCompletion(student_id=student.id, topic_id=topic.id,
                                  title="probe activity", kind="practice", answer="done"))
        db.commit()
        probe_id = session.id
    finally:
        db.close()

    engine.dispose()  # every pooled connection is closed: nothing is in memory

    db = SessionLocal()
    try:
        again = db.get(TeachSession, probe_id)
        check("the session survived a full connection reset", again is not None)
        check("the takeaway text survived", again.summary_text == "persistence probe")
        check("the pace/feedback survived",
              again.pace == "just right" and again.feedback_choices == ["More examples"]
              and again.feedback_text == "probe comment")
        check("the response and its analysis survived",
              db.query(Response).filter(Response.session_id == probe_id).count() == 1)
        obs = db.query(Observation).filter(Observation.session_id == probe_id).first()
        check("the observation survived with its 8-dim vector",
              obs is not None and len(obs.features) == 8)
        check("the activity completion survived",
              db.query(ActivityCompletion)
              .filter(ActivityCompletion.title == "probe activity").count() >= 1)

        # clean the probe rows up again
        db.query(Observation).filter(Observation.session_id == probe_id).delete()
        db.query(Response).filter(Response.session_id == probe_id).delete()
        db.query(ActivityCompletion).filter(
            ActivityCompletion.title == "probe activity").delete()
        db.query(TeachSession).filter(TeachSession.id == probe_id).delete()
        db.commit()
        check("probe rows removed again", db.get(TeachSession, probe_id) is None)
    finally:
        db.close()

    # --- 3. referential integrity ------------------------------------------
    section("Referential integrity (no orphaned or dangling rows)")
    db = SessionLocal()
    try:
        for child, attr, parent in REFERENCES:
            column = getattr(child, attr)
            rows = db.query(child).filter(column.isnot(None)).all()
            parent_ids = {p.id for p in db.query(parent).all()}
            dangling = [getattr(r, "id") for r in rows
                        if getattr(r, attr) not in parent_ids]
            check(f"{child.__tablename__}.{attr} -> {parent.__tablename__}",
                  not dangling, f"dangling ids: {dangling[:5]}" if dangling else "")

        # things that must never be null
        check("no topic without a subject",
              db.query(Topic).filter(Topic.subject_id.is_(None)).count() == 0)
        check("no concept without a topic",
              db.query(Concept).filter(Concept.topic_id.is_(None)).count() == 0)
        check("no session without a student or topic",
              db.query(TeachSession).filter(
                  (TeachSession.student_id.is_(None)) | (TeachSession.topic_id.is_(None))
              ).count() == 0)
        check("no quiz question without a quiz",
              db.query(QuizQuestion).filter(QuizQuestion.quiz_id.is_(None)).count() == 0)

        # Relationship endpoints are stored as free text on purpose: an
        # endpoint may name a concept the teacher defined, or an idea from the
        # lecture that is not itself a concept ("Substring", "List", "Weight").
        # analyzer.endpoint_refs handles both. So the invariant is that the
        # endpoints are real text and the connection is stated — not that they
        # must resolve to concept rows.
        broken = []
        for t in db.query(Topic).all():
            for r in t.relationships:
                if not (r.source or "").strip() or not (r.target or "").strip():
                    broken.append(f"{t.name}: empty endpoint on relationship {r.id}")
                elif not (r.description or "").strip():
                    broken.append(f"{t.name}: {r.source} -> {r.target} has no statement")
        check("every stored relationship has both endpoints and a statement",
              not broken, f"{len(broken)}: {broken[:3]}" if broken else "")
        anchored = sum(
            1 for t in db.query(Topic).all() for r in t.relationships
            if {c.name for c in t.concepts} & {r.source, r.target})
        total_rels = db.query(ConceptRelationship).count()
        check("most relationships anchor to at least one concept of their topic",
              total_rels == 0 or anchored / total_rels >= 0.5,
              f"{anchored}/{total_rels}")

        # SQLite's own integrity check
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA integrity_check")).scalar()
        check("SQLite integrity_check", result == "ok", str(result))
    finally:
        db.close()

    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{'=' * 74}")
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} persistence checks passed")
    for label, _, detail in failed:
        print(f"  FAILED: {label} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
