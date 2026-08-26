"""Activity detail and completion endpoints.

A recommendation points at a stored Activity (by id) whenever the topic has
one for the state; the student opens it, reads the content, answers the short
task and completes it. Completion is a simple record — no second scoring
pipeline. Generic fallback recommendations (no stored row) can still be
completed; they are recorded by title with a null activity_id.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Activity, ActivityCompletion, Student

router = APIRouter(prefix="/api/activities", tags=["activities"])


def activity_out(a: Activity) -> dict:
    return {
        "id": a.id,
        "topic_id": a.topic_id,
        "topic_name": a.topic.name if a.topic else None,
        "title": a.title,
        "description": a.description,
        "kind": a.kind,
        "target_state": a.target_state,
        "content": a.content,
        "question": a.question,
    }


def completion_out(c: ActivityCompletion) -> dict:
    return {
        "id": c.id,
        "activity_id": c.activity_id,
        "topic_id": c.topic_id,
        "topic_name": c.topic.name if c.topic else None,
        "title": c.title,
        "kind": c.kind,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


class CompleteIn(BaseModel):
    student_id: int
    activity_id: int | None = None
    topic_id: int | None = None
    title: str = ""
    kind: str = ""
    answer: str = Field(default="", max_length=5000)


@router.get("/{activity_id}")
def get_activity(activity_id: int, db: Session = Depends(get_db)):
    a = db.get(Activity, activity_id)
    if not a:
        raise HTTPException(404, "Activity not found")
    return activity_out(a)


@router.post("/complete")
def complete_activity(data: CompleteIn, db: Session = Depends(get_db)):
    student = db.get(Student, data.student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    title, kind, topic_id = data.title, data.kind, data.topic_id
    if data.activity_id is not None:
        a = db.get(Activity, data.activity_id)
        if not a:
            raise HTTPException(404, "Activity not found")
        title, kind, topic_id = a.title, a.kind, a.topic_id
    if not title:
        raise HTTPException(400, "Activity title required")

    completion = ActivityCompletion(
        student_id=student.id,
        activity_id=data.activity_id,
        topic_id=topic_id,
        title=title,
        kind=kind,
        answer=data.answer,
    )
    db.add(completion)
    db.commit()
    db.refresh(completion)
    return {"completed": True, **completion_out(completion),
            "message": f"You completed {title}."}
