"""Database initialisation and seeding.

Seeds:
* the teacher-authored topics (seed_content.py),
* 8 named demo students plus ~24 background students,
* per-student observation histories drawn from the synthetic generator
  (so dashboards show realistic trajectories), with states assigned by the
  trained HMM so that everything on screen comes from real inference.
"""
import random
from datetime import datetime, timedelta

from .database import Base, SessionLocal, engine
from .hmm.model import hmm_available, infer_sequence
from .hmm.synthetic import generate_dataset
from .models import (Activity, ActivityCompletion, Concept, ConceptRelationship,
                     Lecture, Misconception, Observation, Quiz, QuizAnswer,
                     QuizAttempt, QuizQuestion, Response, Student, Subject,
                     TeachSession, Teacher, Topic)
from .seed_content import DEMO_STUDENTS, PYTHON_LECTURE, TEACHERS, TOPIC_SUBJECT, TOPICS
from .states import STATE_NAMES


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate()


def _migrate():
    """Add columns introduced after the first release to an existing SQLite DB."""
    additions = [
        ("concepts", "main_question", "TEXT DEFAULT ''"),
        ("concepts", "easier_question", "TEXT DEFAULT ''"),
        ("concepts", "application_question", "TEXT DEFAULT ''"),
        ("teach_sessions", "plan", "JSON"),
        ("observations", "evidence_notes", "JSON"),
        ("students", "roll_no", "VARCHAR(20) DEFAULT ''"),
        ("activities", "content", "TEXT DEFAULT ''"),
        ("activities", "question", "TEXT DEFAULT ''"),
        ("topics", "subject_id", "INTEGER"),
        ("teach_sessions", "summary_text", "TEXT DEFAULT ''"),
        ("teach_sessions", "summary_insights", "JSON"),
        ("teach_sessions", "pace", "VARCHAR(20) DEFAULT ''"),
        ("teach_sessions", "feedback_choices", "JSON"),
        ("teach_sessions", "feedback_text", "TEXT DEFAULT ''"),
        ("concepts", "facts", "JSON"),
        ("concepts", "examples", "JSON"),
        ("concepts", "source", "JSON"),
    ]
    with engine.connect() as conn:
        for table, col, ddl in additions:
            existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            if existing and col not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        conn.commit()


def seed_db(force: bool = False) -> bool:
    init_db()
    db = SessionLocal()
    try:
        if db.query(Topic).count() > 0 and not force:
            return False
        if force:
            for model in (QuizAnswer, QuizAttempt, QuizQuestion, Quiz,
                          ActivityCompletion, Observation, Response, TeachSession, Activity,
                          Misconception, ConceptRelationship, Concept, Lecture, Student,
                          Topic, Subject, Teacher):
                db.query(model).delete()
            db.commit()

        topics = []
        for t in TOPICS:
            topic = Topic(
                name=t["name"],
                description=t["description"],
                reference_explanation=t["reference_explanation"],
                opening_prompt=t["opening_prompt"],
                extension_question=t["extension_question"],
            )
            for i, c in enumerate(t["concepts"]):
                topic.concepts.append(
                    Concept(name=c["name"], description=c["description"],
                            main_question=c.get("main_question", ""),
                            easier_question=c.get("easier_question", ""),
                            probe_question=c["probe_question"],
                            application_question=c.get("application_question", ""),
                            position=i)
                )
            for i, r in enumerate(t.get("relationships", [])):
                topic.relationships.append(
                    ConceptRelationship(source=r["source"], label=r.get("label", "relates to"),
                                        target=r["target"], description=r["description"],
                                        contradiction=r.get("contradiction", ""),
                                        probe_question=r.get("probe_question", ""), position=i)
                )
            for m in t["misconceptions"]:
                topic.misconceptions.append(
                    Misconception(name=m["name"], description=m["description"],
                                  clarification=m["clarification"], probe_question=m["probe_question"])
                )
            for a in t["activities"]:
                topic.activities.append(
                    Activity(title=a["title"], description=a["description"],
                             kind=a["kind"], target_state=a["target_state"],
                             content=a.get("content", ""), question=a.get("question", ""))
                )
            db.add(topic)
            topics.append(topic)
        db.commit()

        # --- teachers, subjects, and the sample lecture workflow ---
        # (before observation seeding, so subject scoping can be exercised
        # with realistic per-subject histories)
        for tdata in TEACHERS:
            teacher = Teacher(name=tdata["name"])
            db.add(teacher)
            db.flush()
            for sname in tdata["subjects"]:
                db.add(Subject(name=sname, teacher_id=teacher.id))
        db.commit()
        subjects = {s.name: s for s in db.query(Subject).all()}

        # existing topics belong to the first teacher's subject
        nn_subject = subjects.get(TOPIC_SUBJECT)
        if nn_subject:
            for topic in topics:
                topic.subject_id = nn_subject.id
            db.commit()

        python_topic = _seed_python_lecture(db, subjects.get(PYTHON_LECTURE["subject"]))
        db.commit()

        # every seeded topic gets its generated Quick knowledge check (the
        # same generation a published lecture gets; teachers can regenerate)
        from .api.quiz import build_quiz_for_topic
        for topic in topics:
            build_quiz_for_topic(db, topic)
        db.commit()

        # students: named demo students first, then background students
        rng = random.Random(11)
        students = []
        for d in DEMO_STUDENTS:
            s = Student(name=d["name"], program=d["program"],
                        roll_no=d.get("roll_no", ""), is_demo=True)
            db.add(s)
            students.append(s)
        for i in range(24):
            s = Student(name=f"Student {i + 9:02d}", program=rng.choice(["B.Tech CSE", "B.Tech AI & DS", "MBA Tech"]))
            db.add(s)
            students.append(s)
        db.commit()

        # observation histories from the synthetic generator (small run, distinct seed)
        mini = generate_dataset(n_students=len(students), seed=123)
        use_hmm = hmm_available()
        now = datetime.utcnow()
        for s_idx, (student, synth) in enumerate(zip(students, mini["students"])):
            seq = [s["features"] for s in synth["sessions"]]
            if use_hmm:
                inf = infer_sequence(seq)
                states = inf["states"]
            else:
                states = [s["true_state"] for s in synth["sessions"]]
            # every third student also takes the Python subject, so both
            # subject dashboards have their own (non-overlapping) histories
            pool = list(topics)
            if python_topic is not None and s_idx % 3 == 0:
                pool.append(python_topic)
            topic_cycle = rng.sample(pool, k=len(pool))
            for t_idx, (sess, st) in enumerate(zip(synth["sessions"], states)):
                days_ago = (len(seq) - t_idx) * rng.randint(2, 4)
                obs_topic = topic_cycle[t_idx % len(topic_cycle)]
                # high misconception_score in the synthetic features -> tag a
                # plausible misconception from this topic so the teacher
                # dashboard has meaningful aggregate data
                miscon_names = []
                if sess["features"][2] > 0.42 and obs_topic.misconceptions:
                    miscon_names = [rng.choice(obs_topic.misconceptions).name]
                db.add(
                    Observation(
                        student_id=student.id,
                        topic_id=obs_topic.id,
                        features=sess["features"],
                        state_index=int(st),
                        state_label=STATE_NAMES[int(st)],
                        misconception_names=miscon_names,
                        source="seed",
                        created_at=now - timedelta(days=days_ago, hours=rng.randint(0, 8)),
                    )
                )
        db.commit()
        return True
    finally:
        db.close()


def _seed_python_lecture(db, subject) -> Topic | None:
    """Run the sample Python lecture through the REAL lecture pipeline.

    Material -> NLP preparation (suggestions stored verbatim) -> the curated
    review from seed_content playing the teacher's quick-edit step -> publish
    into a Topic via the same code path the API uses. Nothing is bypassed and
    nothing is topic-specific.
    """
    if subject is None:
        return None
    from .api.lectures import apply_draft_to_topic
    from .nlp.lecture_prep import prepare_lecture

    lec_def = PYTHON_LECTURE
    prep = prepare_lecture(
        lec_def["material"], title=lec_def["title"], description=lec_def["description"],
        objectives=lec_def["objectives"],
        known_misconceptions=[
            {"name": m.name, "description": m.description, "clarification": m.clarification,
             "probe_question": m.probe_question}
            for m in db.query(Misconception).all()
        ],
    )
    lecture = Lecture(
        subject_id=subject.id,
        title=lec_def["title"],
        description=lec_def["description"],
        material_text=lec_def["material"],
        objectives=lec_def["objectives"],
        suggestions=prep,
        # the teacher's review: curated concepts/relationships/misconceptions/
        # activities (light edits of the automatic suggestions)
        draft={
            "concepts": lec_def["reviewed_concepts"],
            "relationships": lec_def["reviewed_relationships"],
            "misconceptions": lec_def["reviewed_misconceptions"],
            "activities": lec_def["reviewed_activities"],
        },
        status="draft",
    )
    db.add(lecture)
    db.flush()

    topic = Topic()
    db.add(topic)
    apply_draft_to_topic(topic, lecture)
    db.flush()
    # publish the knowledge check the same way the API publish endpoint does
    from .api.quiz import build_quiz_for_topic
    build_quiz_for_topic(db, topic)
    lecture.topic_id = topic.id
    lecture.status = "published"
    return topic
