from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Teacher(Base):
    """A demo faculty account. No authentication — the UI has a simple
    teacher/subject context switcher instead."""

    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

    subjects: Mapped[list["Subject"]] = relationship(back_populates="teacher")


class Subject(Base):
    """A subject taught by one teacher; topics and lectures hang off it."""

    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teachers.id"))

    teacher: Mapped["Teacher"] = relationship(back_populates="subjects")


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    program: Mapped[str] = mapped_column(String(100), default="B.Tech CSE")
    roll_no: Mapped[str] = mapped_column(String(20), default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    sessions: Mapped[list["TeachSession"]] = relationship(back_populates="student")
    observations: Mapped[list["Observation"]] = relationship(back_populates="student")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(150))
    description: Mapped[str] = mapped_column(Text, default="")
    # Short "model answer" used for the semantic correctness feature.
    reference_explanation: Mapped[str] = mapped_column(Text, default="")
    opening_prompt: Mapped[str] = mapped_column(Text, default="")
    extension_question: Mapped[str] = mapped_column(Text, default="")

    concepts: Mapped[list["Concept"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan", order_by="Concept.position"
    )
    misconceptions: Mapped[list["Misconception"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    relationships: Mapped[list["ConceptRelationship"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan", order_by="ConceptRelationship.position"
    )
    activities: Mapped[list["Activity"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )
    subject: Mapped["Subject"] = relationship()


class Lecture(Base):
    """One lecture prepared for TeachBack (the quick faculty workflow).

    The teacher provides material text (pasted or extracted from a file);
    the NLP preparation step stores its candidate concepts/relationships/
    objectives in `draft` for the teacher to review and edit. Publishing
    ("Start TeachBack") builds/updates the linked Topic — the same knowledge
    structure the detailed Topic Management workflow edits.
    """

    __tablename__ = "lectures"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"))
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    material_text: Mapped[str] = mapped_column(Text, default="")
    objectives: Mapped[list] = mapped_column(JSON, default=list)
    # reviewed knowledge draft: {"concepts": [...], "relationships": [...],
    # "misconceptions": [...]} — edited in place by the teacher before publish
    draft: Mapped[dict] = mapped_column(JSON, default=dict)
    # untouched NLP suggestions, kept so the review step can show provenance
    suggestions: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft | published
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    subject: Mapped["Subject"] = relationship()
    topic: Mapped["Topic"] = relationship()


class Concept(Base):
    __tablename__ = "concepts"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    name: Mapped[str] = mapped_column(String(150))
    description: Mapped[str] = mapped_column(Text, default="")
    # Question bank for the guided conversation: a short main question, an
    # easier fallback, a deeper probe, and an optional application question.
    main_question: Mapped[str] = mapped_column(Text, default="")
    easier_question: Mapped[str] = mapped_column(Text, default="")
    probe_question: Mapped[str] = mapped_column(Text, default="")
    application_question: Mapped[str] = mapped_column(Text, default="")
    # Important facts a student may mention as evidence ("Indexes start at 0."),
    # lecture examples, and provenance ({"section", "sentences"}) — all taken
    # from the reviewed lecture draft so evaluation and explanations can point
    # back at the actual source material.
    facts: Mapped[list] = mapped_column(JSON, default=list)
    examples: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[dict] = mapped_column(JSON, default=dict)
    position: Mapped[int] = mapped_column(Integer, default=0)

    topic: Mapped["Topic"] = relationship(back_populates="concepts")


class ConceptRelationship(Base):
    """A teacher-authored link between two concepts (a tiny relationship list,
    not a knowledge graph). `description` is the correct sentence expressing
    the relationship; `contradiction` is an optional wrong version of it —
    content words that appear only in the contradiction act as cue words for
    detecting a conceptually wrong (but semantically similar) statement."""

    __tablename__ = "concept_relationships"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    source: Mapped[str] = mapped_column(String(150))
    label: Mapped[str] = mapped_column(String(100), default="relates to")
    target: Mapped[str] = mapped_column(String(150))
    description: Mapped[str] = mapped_column(Text, default="")
    contradiction: Mapped[str] = mapped_column(Text, default="")
    probe_question: Mapped[str] = mapped_column(Text, default="")
    position: Mapped[int] = mapped_column(Integer, default=0)

    topic: Mapped["Topic"] = relationship(back_populates="relationships")


class Misconception(Base):
    __tablename__ = "misconceptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    name: Mapped[str] = mapped_column(String(200))
    # The wrong claim, phrased the way a student would say it.
    description: Mapped[str] = mapped_column(Text, default="")
    # The correct contrast statement; a sentence is only flagged if it is
    # closer to the wrong claim than to this correction.
    clarification: Mapped[str] = mapped_column(Text, default="")
    probe_question: Mapped[str] = mapped_column(Text, default="")

    topic: Mapped["Topic"] = relationship(back_populates="misconceptions")


class Activity(Base):
    __tablename__ = "activities"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(50), default="practice")
    # Which learning state this activity is meant for (state key, e.g. "unclear").
    target_state: Mapped[str] = mapped_column(String(30), default="understanding")
    # The material the student actually reads/uses when doing the activity,
    # and the short task/question they answer to complete it.
    content: Mapped[str] = mapped_column(Text, default="")
    question: Mapped[str] = mapped_column(Text, default="")

    topic: Mapped["Topic"] = relationship(back_populates="activities")


class ActivityCompletion(Base):
    """A student's record of completing a recommended activity.

    activity_id is null when the completed activity was a generic fallback
    (no stored Activity row); the title/kind snapshot keeps the record
    readable either way.
    """

    __tablename__ = "activity_completions"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    activity_id: Mapped[int] = mapped_column(ForeignKey("activities.id"), nullable=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200), default="")
    kind: Mapped[str] = mapped_column(String(50), default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    student: Mapped["Student"] = relationship()
    topic: Mapped["Topic"] = relationship()


class Quiz(Base):
    """The optional "Quick knowledge check" for a topic: exactly one quiz per
    topic, holding the teacher-reviewed MCQ questions. It is SECONDARY
    evidence — TeachBack (explaining in your own words) stays primary."""

    __tablename__ = "quizzes"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    title: Mapped[str] = mapped_column(String(200), default="Quick knowledge check")

    topic: Mapped["Topic"] = relationship()
    questions: Mapped[list["QuizQuestion"]] = relationship(
        back_populates="quiz", cascade="all, delete-orphan", order_by="QuizQuestion.position"
    )


class QuizQuestion(Base):
    """One single-answer MCQ, grounded in the teacher-reviewed material.

    `concept_name` ties the question to a concept for concept-level evidence
    (stored by name so it survives topic re-publishing, which replaces
    Concept rows). `kind` records the intended difficulty mix: basic /
    application / misconception / relationship.
    """

    __tablename__ = "quiz_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"))
    concept_name: Mapped[str] = mapped_column(String(150), default="")
    kind: Mapped[str] = mapped_column(String(30), default="basic")
    question: Mapped[str] = mapped_column(Text, default="")
    options: Mapped[list] = mapped_column(JSON, default=list)  # exactly 4 strings
    correct_index: Mapped[int] = mapped_column(Integer, default=0)
    explanation: Mapped[str] = mapped_column(Text, default="")
    position: Mapped[int] = mapped_column(Integer, default=0)

    quiz: Mapped["Quiz"] = relationship(back_populates="questions")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_id: Mapped[int] = mapped_column(ForeignKey("quizzes.id"))
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    # the TeachBack session this check followed, for combined evidence
    session_id: Mapped[int] = mapped_column(ForeignKey("teach_sessions.id"), nullable=True)
    n_correct: Mapped[int] = mapped_column(Integer, default=0)
    n_questions: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    quiz: Mapped["Quiz"] = relationship()
    student: Mapped["Student"] = relationship()
    answers: Mapped[list["QuizAnswer"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class QuizAnswer(Base):
    __tablename__ = "quiz_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("quiz_attempts.id"))
    question_id: Mapped[int] = mapped_column(ForeignKey("quiz_questions.id"))
    selected_index: Mapped[int] = mapped_column(Integer, default=-1)
    correct: Mapped[bool] = mapped_column(Boolean, default=False)

    attempt: Mapped["QuizAttempt"] = relationship(back_populates="answers")
    question: Mapped["QuizQuestion"] = relationship()


class TeachSession(Base):
    __tablename__ = "teach_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    exchange_count: Mapped[int] = mapped_column(Integer, default=0)
    # State of the guided concept-by-concept conversation (see nlp/conversation.py).
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    # The student's own end-of-session lecture takeaway, plus what the NLP
    # analysis of it added (upgrade-only evidence, see api/teachback.py).
    summary_text: Mapped[str] = mapped_column(Text, default="")
    summary_insights: Mapped[dict] = mapped_column(JSON, default=dict)
    # Fast lecture feedback: pace choice, selected request chips, free text.
    pace: Mapped[str] = mapped_column(String(20), default="")
    feedback_choices: Mapped[list] = mapped_column(JSON, default=list)
    feedback_text: Mapped[str] = mapped_column(Text, default="")

    student: Mapped["Student"] = relationship(back_populates="sessions")
    topic: Mapped["Topic"] = relationship()
    responses: Mapped[list["Response"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Response.exchange_no"
    )


class Response(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("teach_sessions.id"))
    exchange_no: Mapped[int] = mapped_column(Integer, default=1)
    prompt: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text, default="")
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped["TeachSession"] = relationship(back_populates="responses")


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), nullable=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("teach_sessions.id"), nullable=True)
    features: Mapped[list] = mapped_column(JSON, default=list)  # 8 floats, FEATURE_NAMES order
    state_index: Mapped[int] = mapped_column(Integer, nullable=True)
    state_label: Mapped[str] = mapped_column(String(40), nullable=True)
    misconception_names: Mapped[list] = mapped_column(JSON, default=list)
    # Human-readable evidence bullets recorded at session end (live sessions);
    # seeded observations derive their bullets from the features instead.
    evidence_notes: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(20), default="live")  # live | seed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    student: Mapped["Student"] = relationship(back_populates="observations")
    topic: Mapped["Topic"] = relationship()
