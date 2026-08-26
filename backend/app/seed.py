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
from .models import (Activity, Concept, ConceptRelationship, Misconception,
                     Observation, Response, Student, TeachSession, Topic)
from .seed_content import DEMO_STUDENTS, TOPICS
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
            for model in (Observation, Response, TeachSession, Activity,
                          Misconception, ConceptRelationship, Concept, Student, Topic):
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
                             kind=a["kind"], target_state=a["target_state"])
                )
            db.add(topic)
            topics.append(topic)
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
        for student, synth in zip(students, mini["students"]):
            seq = [s["features"] for s in synth["sessions"]]
            if use_hmm:
                inf = infer_sequence(seq)
                states = inf["states"]
            else:
                states = [s["true_state"] for s in synth["sessions"]]
            topic_cycle = rng.sample(topics, k=len(topics))
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
