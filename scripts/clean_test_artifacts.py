"""Remove script- and test-created rows from the demo SQLite database.

Before the suite was isolated (tests/conftest.py), every test ran against
``backend/teachback.db`` — the same file the running app uses. Topics like
"Test Topic v2", "Homeless topic" and "Decision Trees Live" therefore appeared
in the real Topic Management screen, along with the lectures, sessions,
observations, quiz attempts and activity completions those runs produced.

The leak itself is fixed at the source; this script cleans up what earlier
runs already left behind, and stays useful for any database that predates the
fix or was written to by scripts/simulate_user.py.

What counts as intentional is taken from app.seed_content, not from a list of
names typed here, so it cannot drift away from what seeding actually creates:

    topics    - the TOPICS entries plus the sample PYTHON_LECTURE topic
    lectures  - the sample PYTHON_LECTURE
    students  - the DEMO_STUDENTS plus the "Student NN" background cohort
    history   - observations with source="seed"

Everything else was produced by a test or a simulation run and is removed,
together with the records that hang off it. Teachers, subjects and the seeded
observation histories are never touched.

Usage:
    python scripts/clean_test_artifacts.py              # report only
    python scripts/clean_test_artifacts.py --apply      # actually delete
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from sqlalchemy import text  # noqa: E402

from app.config import DB_PATH  # noqa: E402
from app.database import SessionLocal, engine  # noqa: E402
from app.models import (Activity, ActivityCompletion, Concept,  # noqa: E402
                        ConceptRelationship, Lecture, Misconception,
                        Observation, Quiz, QuizAnswer, QuizAttempt,
                        QuizQuestion, Response, Student, TeachSession, Topic)
from app.seed_content import DEMO_STUDENTS, PYTHON_LECTURE, TOPICS  # noqa: E402

BACKGROUND_STUDENT = re.compile(r"^Student \d{2}$")


def seeded_topic_names() -> set[str]:
    return {t["name"] for t in TOPICS} | {PYTHON_LECTURE["title"]}


def seeded_lecture_titles() -> set[str]:
    return {PYTHON_LECTURE["title"]}


def seeded_student_names() -> set[str]:
    return {d["name"] for d in DEMO_STUDENTS}


def is_seeded_student(name: str) -> bool:
    return name in seeded_student_names() or bool(BACKGROUND_STUDENT.match(name or ""))


def survey(db) -> dict:
    """Classify everything currently in the database."""
    topic_names = seeded_topic_names()
    lecture_titles = seeded_lecture_titles()

    topics = db.query(Topic).order_by(Topic.id).all()
    keep_topics = [t for t in topics if t.name in topic_names]
    drop_topics = [t for t in topics if t.name not in topic_names]

    lectures = db.query(Lecture).order_by(Lecture.id).all()
    drop_lectures = [l for l in lectures if l.title not in lecture_titles]

    students = db.query(Student).order_by(Student.id).all()
    drop_students = [s for s in students if not is_seeded_student(s.name)]

    # Every TeachBack session in a demo database came from a test or a
    # simulation run — seeding creates observation histories, never sessions.
    sessions = db.query(TeachSession).order_by(TeachSession.id).all()
    live_observations = db.query(Observation).filter(Observation.source != "seed").all()
    attempts = db.query(QuizAttempt).order_by(QuizAttempt.id).all()
    completions = db.query(ActivityCompletion).order_by(ActivityCompletion.id).all()

    return {
        "keep_topics": keep_topics,
        "drop_topics": drop_topics,
        "drop_lectures": drop_lectures,
        "drop_students": drop_students,
        "sessions": sessions,
        "live_observations": live_observations,
        "attempts": attempts,
        "completions": completions,
    }


def report(plan: dict) -> None:
    print(f"Database: {DB_PATH}\n")
    print("Keeping (seeded/demo):")
    for t in plan["keep_topics"]:
        flag = " [ARCHIVED]" if t.archived_at else ""
        print(f"  topic {t.id:>3}  {t.name}{flag}")
    print()
    print("Removing (created by tests or simulation runs):")
    for t in plan["drop_topics"]:
        print(f"  topic    {t.id:>3}  {t.name}")
    for l in plan["drop_lectures"]:
        print(f"  lecture  {l.id:>3}  {l.title}")
    for s in plan["drop_students"]:
        print(f"  student  {s.id:>3}  {s.name}")
    print(f"  {len(plan['sessions'])} TeachBack sessions")
    print(f"  {len(plan['live_observations'])} live observations "
          "(seeded histories are kept)")
    print(f"  {len(plan['attempts'])} knowledge-check attempts")
    print(f"  {len(plan['completions'])} activity completions")


def apply(db, plan: dict) -> None:
    """Delete in dependency order so nothing is ever left pointing at a gap."""
    # 1. student activity: answers -> attempts, responses -> sessions
    for attempt in plan["attempts"]:
        db.delete(attempt)          # cascades to its QuizAnswer rows
    for completion in plan["completions"]:
        db.delete(completion)
    for observation in plan["live_observations"]:
        db.delete(observation)
    for session in plan["sessions"]:
        db.delete(session)          # cascades to its Response rows
    db.flush()

    # 2. lectures let go of their topics before the topics disappear
    drop_topic_ids = {t.id for t in plan["drop_topics"]}
    for lecture in db.query(Lecture).all():
        if lecture.topic_id in drop_topic_ids:
            lecture.topic_id = None
            if lecture.status == "published":
                lecture.status = "draft"
    db.flush()

    for lecture in plan["drop_lectures"]:
        db.delete(lecture)

    # 3. the topics themselves, with the quiz that is not cascade-linked
    for topic in plan["drop_topics"]:
        for quiz in db.query(Quiz).filter(Quiz.topic_id == topic.id).all():
            db.delete(quiz)         # cascades to its QuizQuestion rows
        db.delete(topic)            # cascades to concepts/relationships/
        #                             misconceptions/activities
    db.flush()

    for student in plan["drop_students"]:
        db.delete(student)

    db.commit()


ORPHAN_CHECKS = [
    ("concepts without a topic", Concept, Concept.topic_id, Topic),
    ("relationships without a topic", ConceptRelationship, ConceptRelationship.topic_id, Topic),
    ("misconceptions without a topic", Misconception, Misconception.topic_id, Topic),
    ("activities without a topic", Activity, Activity.topic_id, Topic),
    ("quizzes without a topic", Quiz, Quiz.topic_id, Topic),
    ("quiz questions without a quiz", QuizQuestion, QuizQuestion.quiz_id, Quiz),
    ("quiz attempts without a quiz", QuizAttempt, QuizAttempt.quiz_id, Quiz),
    ("quiz answers without an attempt", QuizAnswer, QuizAnswer.attempt_id, QuizAttempt),
    ("responses without a session", Response, Response.session_id, TeachSession),
    ("sessions without a topic", TeachSession, TeachSession.topic_id, Topic),
    ("sessions without a student", TeachSession, TeachSession.student_id, Student),
    ("observations without a student", Observation, Observation.student_id, Student),
    ("lectures pointing at a deleted topic", Lecture, Lecture.topic_id, Topic),
]


def integrity(db) -> bool:
    """Every foreign key resolves, and SQLite's own checks pass."""
    print("\nIntegrity:")
    ok = True
    for label, model, column, target in ORPHAN_CHECKS:
        bad = (db.query(model)
               .filter(column.isnot(None))
               .outerjoin(target, target.id == column)
               .filter(target.id.is_(None))
               .count())
        ok &= bad == 0
        print(f"  [{'PASS' if bad == 0 else 'FAIL'}] {label}: {bad}")
    with engine.connect() as conn:
        fk = list(conn.exec_driver_sql("PRAGMA foreign_key_check"))
        integrity_check = conn.exec_driver_sql("PRAGMA integrity_check").scalar()
    ok &= not fk and integrity_check == "ok"
    print(f"  [{'PASS' if not fk else 'FAIL'}] PRAGMA foreign_key_check: {len(fk)} violations")
    print(f"  [{'PASS' if integrity_check == 'ok' else 'FAIL'}] PRAGMA integrity_check: {integrity_check}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="perform the deletions (default is a report only)")
    args = parser.parse_args()

    if not Path(DB_PATH).exists():
        print(f"No database at {DB_PATH} — nothing to clean.")
        return 0

    db = SessionLocal()
    try:
        plan = survey(db)
        report(plan)
        nothing_to_do = not any(plan[k] for k in
                                ("drop_topics", "drop_lectures", "drop_students",
                                 "sessions", "live_observations", "attempts", "completions"))
        if nothing_to_do:
            print("\nNothing to remove — the database holds seeded demo data only.")
            integrity(db)
            return 0
        if not args.apply:
            print("\nReport only. Re-run with --apply to remove these rows.")
            return 0
        apply(db, plan)
        print("\nRemoved.")
        db.expire_all()
        ok = integrity(db)
        print("\nRemaining topics:")
        for t in db.query(Topic).order_by(Topic.id).all():
            print(f"  {t.id:>3}  {t.name}  (subject {t.subject_id})"
                  + (" [ARCHIVED]" if t.archived_at else ""))
        return 0 if ok else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
