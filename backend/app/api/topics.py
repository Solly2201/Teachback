"""Topic Management: the detailed editor for a topic's knowledge structure.

GET    /api/topics                      active topics (optionally by subject)
GET    /api/topics/{id}                 full definition
POST   /api/topics                      create
PUT    /api/topics/{id}                 replace the knowledge structure
GET    /api/topics/{id}/delete-preview  what deleting would do, for the dialog
DELETE /api/topics/{id}                 delete a topic no student has used, or
                                        ARCHIVE it when history exists
POST   /api/topics/{id}/restore         bring an archived topic back
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Activity, Concept, ConceptRelationship, Lecture, Misconception,
                      Quiz, Subject, Topic)
from .helpers import history_summary, topic_def, topic_history

router = APIRouter(prefix="/api/topics", tags=["topics"])


class ConceptIn(BaseModel):
    name: str
    description: str = ""
    main_question: str = ""
    easier_question: str = ""
    probe_question: str = ""
    application_question: str = ""


class RelationshipIn(BaseModel):
    source: str
    label: str = "relates to"
    target: str
    description: str = ""
    contradiction: str = ""
    probe_question: str = ""


class MisconceptionIn(BaseModel):
    name: str
    description: str = ""
    clarification: str = ""
    probe_question: str = ""


class ActivityIn(BaseModel):
    title: str
    description: str = ""
    kind: str = "practice"
    target_state: str = "understanding"
    content: str = ""
    question: str = ""


class TopicIn(BaseModel):
    name: str
    subject_id: int | None = None
    description: str = ""
    reference_explanation: str = ""
    opening_prompt: str = ""
    extension_question: str = ""
    concepts: list[ConceptIn] = Field(default_factory=list)
    relationships: list[RelationshipIn] = Field(default_factory=list)
    misconceptions: list[MisconceptionIn] = Field(default_factory=list)
    activities: list[ActivityIn] = Field(default_factory=list)


@router.get("")
def list_topics(subject_id: int | None = None, include_archived: bool = False,
                db: Session = Depends(get_db)):
    """Active topics. A topic whose lecture was archived is hidden here (so no
    new TeachBack can start on it) but is still fetchable by id, because old
    sessions and progress records reference it."""
    q = db.query(Topic).order_by(Topic.id)
    if subject_id is not None:
        q = q.filter(Topic.subject_id == subject_id)
    if not include_archived:
        q = q.filter(Topic.archived_at.is_(None))
    topics = q.all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "subject_id": t.subject_id,
            "subject_name": t.subject.name if t.subject else None,
            "description": t.description,
            "concept_count": len(t.concepts),
            "relationship_count": len(t.relationships),
            "misconception_count": len(t.misconceptions),
            "activity_count": len(t.activities),
            "archived": t.archived_at is not None,
            "archived_at": t.archived_at.isoformat() if t.archived_at else None,
        }
        for t in topics
    ]


@router.get("/{topic_id}")
def get_topic(topic_id: int, db: Session = Depends(get_db)):
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    return topic_def(t)


def _apply(t: Topic, data: TopicIn):
    t.name = data.name
    if data.subject_id is not None:
        t.subject_id = data.subject_id
    t.description = data.description
    t.reference_explanation = data.reference_explanation
    t.opening_prompt = data.opening_prompt or f"Teach me what you understand about {data.name}, as if I have never learned it."
    t.extension_question = data.extension_question
    t.concepts = [
        Concept(name=c.name, description=c.description, position=i,
                main_question=c.main_question, easier_question=c.easier_question,
                probe_question=c.probe_question, application_question=c.application_question)
        for i, c in enumerate(data.concepts)
    ]
    t.relationships = [
        ConceptRelationship(source=r.source, label=r.label, target=r.target,
                            description=r.description, contradiction=r.contradiction,
                            probe_question=r.probe_question, position=i)
        for i, r in enumerate(data.relationships)
    ]
    t.misconceptions = [
        Misconception(name=m.name, description=m.description, clarification=m.clarification,
                      probe_question=m.probe_question)
        for m in data.misconceptions
    ]
    t.activities = [
        Activity(title=a.title, description=a.description, kind=a.kind, target_state=a.target_state,
                 content=a.content, question=a.question)
        for a in data.activities
    ]


def _require_subject(subject_id: int | None, db: Session) -> None:
    """Every topic belongs to a subject, which belongs to a teacher.

    A topic created without one is invisible to every subject-scoped list (the
    only way the UI reaches topics) yet still reachable by id, so it escapes
    subject isolation while remaining startable. The frontend has always sent
    a subject; nothing but an unvalidated request could produce one.
    """
    if subject_id is None:
        raise HTTPException(400, "A topic must belong to a subject.")
    if db.get(Subject, subject_id) is None:
        raise HTTPException(404, "Subject not found")


@router.post("")
def create_topic(data: TopicIn, db: Session = Depends(get_db)):
    _require_subject(data.subject_id, db)
    t = Topic()
    _apply(t, data)
    db.add(t)
    db.commit()
    db.refresh(t)
    return topic_def(t)


@router.put("/{topic_id}")
def update_topic(topic_id: int, data: TopicIn, db: Session = Depends(get_db)):
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    # an edit may move a topic between subjects, but never out of all of them
    _require_subject(data.subject_id if data.subject_id is not None else t.subject_id, db)
    _apply(t, data)
    db.commit()
    db.refresh(t)
    return topic_def(t)


# ---------------------------------------------------------------------------
# delete / archive / restore
# ---------------------------------------------------------------------------
# A Topic is what every learning record points at: TeachSession, Observation,
# QuizAttempt (through its Quiz) and ActivityCompletion all carry its id.
# Deleting one outright would therefore erase real student history because a
# teacher tidied their topic list, so the action has two modes chosen from the
# data rather than from a flag — the same rule the lecture delete already uses,
# reading the same counts (helpers.topic_history):
#
#   deleted   - no student has ever used this topic: the topic and everything
#               it owns (concepts, relationships, misconceptions, activities,
#               its quiz and that quiz's questions) are removed outright, and
#               any lecture that published it is unlinked back to a draft so
#               the teacher keeps their material and no foreign key dangles.
#   archived  - student history exists: the topic is stamped archived_at (the
#               existing lifecycle the lecture archive already uses) and so is
#               the lecture that owns it. Both leave the active lists, no new
#               TeachBack can start, and every historical record stays intact
#               and readable. Restore reverses exactly this.


def _topic_lectures(db: Session, topic_id: int) -> list[Lecture]:
    return db.query(Lecture).filter(Lecture.topic_id == topic_id).all()


def _delete_message(topic: Topic) -> str:
    return (f'Delete "{topic.name}"? This topic and its teaching material — concepts, '
            "relationships, misconceptions, activities and its knowledge check — will be "
            "permanently deleted. No student has worked on it, so no learning record is "
            "affected.")


def _archive_message(topic: Topic, history: dict) -> str:
    return (f'"{topic.name}" already has student learning records ({history_summary(history)}). '
            "This topic will be archived so existing student history is preserved: it "
            "disappears from your active topics and no new TeachBack can start on it, while "
            "every one of those records stays intact and readable.")


@router.get("/{topic_id}/delete-preview")
def topic_delete_preview(topic_id: int, db: Session = Depends(get_db)):
    """What deleting this topic would do — for the confirmation dialog."""
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    history = topic_history(db, t.id)
    mode = "archive" if history["total"] else "delete"
    lectures = _topic_lectures(db, t.id)
    return {
        "topic_id": t.id,
        "name": t.name,
        "subject_id": t.subject_id,
        "archived": t.archived_at is not None,
        "history": history,
        "history_summary": history_summary(history),
        "lecture_titles": [lec.title for lec in lectures],
        "mode": mode,
        "message": _archive_message(t, history) if mode == "archive" else _delete_message(t),
    }


@router.delete("/{topic_id}")
def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    history = topic_history(db, t.id)
    name = t.name
    lectures = _topic_lectures(db, t.id)

    if history["total"]:
        now = datetime.utcnow()
        t.archived_at = now
        for lec in lectures:
            if lec.archived_at is None:
                lec.archived_at = now
                lec.status = "archived"
        db.commit()
        return {
            "mode": "archived", "topic_id": topic_id, "name": name,
            "history": history, "history_summary": history_summary(history),
            "message": (f'"{name}" was archived, not erased. Its {history_summary(history)} '
                        "remain intact and still show in student history; the topic no longer "
                        "appears in your active list and cannot start new sessions."),
        }

    # Nothing to preserve: remove the topic and everything it owns. A lecture
    # that published it keeps its material and goes back to being a draft —
    # deleting the teacher's source text was never what was asked for, and
    # leaving topic_id pointing at a deleted row would dangle.
    for lec in lectures:
        lec.topic_id = None
        if lec.status == "published":
            lec.status = "draft"
    quiz = db.query(Quiz).filter(Quiz.topic_id == t.id).first()
    if quiz is not None:
        db.delete(quiz)  # cascades to its questions
    db.delete(t)         # cascades to concepts / relationships / misconceptions / activities
    db.commit()
    return {"mode": "deleted", "topic_id": topic_id, "name": name, "history": history,
            "history_summary": history_summary(history),
            "unlinked_lectures": [lec.id for lec in lectures],
            "message": f'"{name}" was deleted. No student records were affected.'}


@router.post("/{topic_id}/restore")
def restore_topic(topic_id: int, db: Session = Depends(get_db)):
    """Bring an archived topic (and the lecture that owns it) back."""
    t = db.get(Topic, topic_id)
    if not t:
        raise HTTPException(404, "Topic not found")
    if t.archived_at is None:
        raise HTTPException(400, "This topic is not archived.")
    t.archived_at = None
    for lec in _topic_lectures(db, t.id):
        if lec.archived_at is not None:
            lec.archived_at = None
            lec.status = "published" if lec.topic_id else "draft"
    db.commit()
    db.refresh(t)
    return topic_def(t)
