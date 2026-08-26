from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


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

    topic: Mapped["Topic"] = relationship(back_populates="activities")


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
