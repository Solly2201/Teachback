from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Activity, Concept, ConceptRelationship, Misconception, Topic
from .helpers import topic_def

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


@router.post("")
def create_topic(data: TopicIn, db: Session = Depends(get_db)):
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
    _apply(t, data)
    db.commit()
    db.refresh(t)
    return topic_def(t)
