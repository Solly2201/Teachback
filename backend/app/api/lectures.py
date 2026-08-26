"""Quick Lecture workflow: material in -> NLP draft -> faculty review -> Topic.

POST /api/lectures            create a lecture; runs the deterministic NLP
                              preparation and stores the suggestions as an
                              editable draft
PUT  /api/lectures/{id}       save the teacher's review edits (concepts /
                              relationships / misconceptions / objectives)
POST /api/lectures/{id}/publish
                              "Start TeachBack": build or update the linked
                              Topic from the reviewed draft — the same Topic
                              structure the detailed Topic Management editor
                              uses, so both workflows share one knowledge
                              system
POST /api/lectures/extract    extract text from an uploaded .txt/.md/.pdf
                              (sent base64-encoded to avoid a multipart dep)
GET  /api/teachers            demo teachers with their subjects, for the
                              lightweight teacher/subject switcher
"""
import base64

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Activity, Concept, ConceptRelationship, Lecture,
                      Misconception, Subject, Teacher, Topic)
from ..nlp.lecture_prep import extract_text, prepare_lecture
from ..nlp.note_template import AI_PREP_PROMPT, NOTE_TEMPLATE
from ..nlp.quiz_gen import generate_quiz_candidates, generate_quiz_questions
from .helpers import topic_def
from .quiz import build_quiz_for_topic

router = APIRouter(prefix="/api", tags=["lectures"])

MAIN_QUESTION_TEMPLATE = 'What did you understand about "{name}"?'


class LectureIn(BaseModel):
    subject_id: int
    title: str
    description: str = ""
    material_text: str = Field(default="", max_length=100_000)
    objectives: list[str] = Field(default_factory=list)


class DraftConcept(BaseModel):
    name: str
    description: str = ""
    facts: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    source_section: str = ""
    source_sentences: list[str] = Field(default_factory=list)
    main_question: str = ""
    easier_question: str = ""
    probe_question: str = ""
    application_question: str = ""


class DraftRelationship(BaseModel):
    source: str
    label: str = "relates to"
    target: str
    description: str = ""
    contradiction: str = ""


class DraftMisconception(BaseModel):
    name: str
    description: str = ""
    clarification: str = ""
    probe_question: str = ""


class DraftActivity(BaseModel):
    target_state: str = "understanding"
    kind: str = "practice"
    title: str
    description: str = ""
    content: str = ""
    question: str = ""


class DraftQuizQuestion(BaseModel):
    concept_name: str = ""
    kind: str = "basic"
    question: str
    options: list[str]
    correct_index: int = 0
    explanation: str = ""


class LectureUpdateIn(BaseModel):
    title: str | None = None
    description: str | None = None
    objectives: list[str] | None = None
    concepts: list[DraftConcept] | None = None
    relationships: list[DraftRelationship] | None = None
    misconceptions: list[DraftMisconception] | None = None
    activities: list[DraftActivity] | None = None
    quiz: list[DraftQuizQuestion] | None = None


def _draft_topic_def(draft: dict, title: str) -> dict:
    """A topic_def-shaped dict built from a lecture draft, for quiz generation."""
    return {
        "name": title,
        "concepts": draft.get("concepts") or [],
        "relationships": draft.get("relationships") or [],
        "misconceptions": draft.get("misconceptions") or [],
    }


class ExtractIn(BaseModel):
    filename: str
    content_base64: str


def lecture_out(lec: Lecture, include_material: bool = True) -> dict:
    out = {
        "id": lec.id,
        "subject_id": lec.subject_id,
        "subject_name": lec.subject.name if lec.subject else None,
        "topic_id": lec.topic_id,
        "title": lec.title,
        "description": lec.description,
        "objectives": lec.objectives or [],
        "draft": lec.draft or {},
        "suggestions": lec.suggestions or {},
        "status": lec.status,
        "created_at": lec.created_at.isoformat() if lec.created_at else None,
    }
    if include_material:
        out["material_text"] = lec.material_text
    return out


@router.get("/teachers")
def list_teachers(db: Session = Depends(get_db)):
    teachers = db.query(Teacher).order_by(Teacher.id).all()
    return [
        {"id": t.id, "name": t.name,
         "subjects": [{"id": s.id, "name": s.name} for s in sorted(t.subjects, key=lambda s: s.id)]}
        for t in teachers
    ]


@router.get("/lectures/prep-prompt")
def prep_prompt():
    """Recommended note format + the copyable external-AI preparation prompt.

    TeachBack never calls an LLM: this endpoint only returns text the teacher
    may paste into an external assistant of their own choosing.
    """
    return {"template": NOTE_TEMPLATE, "prompt": AI_PREP_PROMPT}


@router.post("/lectures/extract")
def extract_material(data: ExtractIn):
    try:
        raw = base64.b64decode(data.content_base64)
        text = extract_text(data.filename, raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file.")
    if not text.strip():
        raise HTTPException(400, "No text could be extracted from the file.")
    return {"text": text}


@router.get("/lectures")
def list_lectures(subject_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(Lecture).order_by(Lecture.id.desc())
    if subject_id is not None:
        q = q.filter(Lecture.subject_id == subject_id)
    return [lecture_out(l, include_material=False) for l in q.all()]


@router.get("/lectures/{lecture_id}")
def get_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    return lecture_out(lec)


def _known_misconceptions(db: Session) -> list[dict]:
    return [
        {"name": m.name, "description": m.description,
         "clarification": m.clarification, "probe_question": m.probe_question}
        for m in db.query(Misconception).all()
    ]


@router.post("/lectures")
def create_lecture(data: LectureIn, db: Session = Depends(get_db)):
    subject = db.get(Subject, data.subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    if not data.material_text.strip():
        raise HTTPException(400, "Provide the lecture material (paste notes or upload a file).")

    prep = prepare_lecture(
        data.material_text, title=data.title, description=data.description,
        objectives=data.objectives, known_misconceptions=_known_misconceptions(db),
    )
    draft = {
        # the full evidence-carrying concept suggestions (facts, examples,
        # provenance, drafted questions) become the editable draft
        "concepts": [
            {k: c.get(k) for k in ("name", "description", "facts", "examples",
                                   "source_section", "source_sentences",
                                   "main_question", "easier_question",
                                   "probe_question", "application_question")}
            for c in prep["concepts"]
        ],
        "relationships": [{"source": r["source"], "label": r["label"], "target": r["target"],
                           "description": r["description"]} for r in prep["relationships"]],
        "misconceptions": [],  # suggestions must be explicitly accepted by the teacher
        "activities": list(prep.get("activities", [])),
    }
    # suggested knowledge-check questions, generated from the draft structure
    # and reviewable/editable like everything else
    draft["quiz"] = generate_quiz_questions(_draft_topic_def(draft, data.title))
    lec = Lecture(
        subject_id=subject.id, title=data.title, description=data.description,
        material_text=data.material_text, objectives=prep["objectives"],
        draft=draft, suggestions=prep, status="draft",
    )
    db.add(lec)
    db.commit()
    db.refresh(lec)
    return lecture_out(lec)


@router.put("/lectures/{lecture_id}")
def update_lecture(lecture_id: int, data: LectureUpdateIn, db: Session = Depends(get_db)):
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    if data.title is not None:
        lec.title = data.title
    if data.description is not None:
        lec.description = data.description
    if data.objectives is not None:
        lec.objectives = [o.strip() for o in data.objectives if o.strip()]
    draft = dict(lec.draft or {})
    if data.concepts is not None:
        draft["concepts"] = [c.model_dump() for c in data.concepts if c.name.strip()]
    if data.relationships is not None:
        draft["relationships"] = [r.model_dump() for r in data.relationships
                                  if r.source.strip() and r.target.strip()]
    if data.misconceptions is not None:
        draft["misconceptions"] = [m.model_dump() for m in data.misconceptions if m.name.strip()]
    if data.activities is not None:
        draft["activities"] = [a.model_dump() for a in data.activities if a.title.strip()]
    if data.quiz is not None:
        draft["quiz"] = [q.model_dump() for q in data.quiz if q.question.strip()]
    lec.draft = draft
    db.commit()
    db.refresh(lec)
    return lecture_out(lec)


def apply_draft_to_topic(topic: Topic, lec: Lecture) -> None:
    """Build the Topic's knowledge structure from a lecture's reviewed draft.

    Shared by the publish endpoint and the seeder so the sample lecture goes
    through exactly the same pipeline a teacher would use.
    """
    draft = lec.draft or {}
    concepts = draft.get("concepts") or []
    topic.subject_id = lec.subject_id
    topic.name = lec.title
    topic.description = lec.description
    # bounded reference for the semantic-correctness feature: the reviewed
    # concept explanations, which come from the lecture material itself
    topic.reference_explanation = " ".join(
        c.get("description", "") for c in concepts if c.get("description")
    ) or lec.material_text[:1000]
    topic.opening_prompt = (
        f"Tell me what you took away from the lecture on {lec.title} — in your own words."
    )
    topic.extension_question = ""  # no advanced questions unless the teacher adds one

    topic.concepts = [
        Concept(
            name=c["name"], description=c.get("description", ""),
            main_question=c.get("main_question") or MAIN_QUESTION_TEMPLATE.format(name=c["name"]),
            easier_question=c.get("easier_question", "") or "",
            probe_question=c.get("probe_question", "") or "",
            # the application question is optional extension material only —
            # it is never required for demonstrating lecture understanding
            application_question=c.get("application_question", "") or "",
            facts=c.get("facts") or [],
            examples=c.get("examples") or [],
            source={"section": c.get("source_section", ""),
                    "sentences": c.get("source_sentences") or []},
            position=i,
        )
        for i, c in enumerate(concepts)
    ]
    topic.relationships = [
        ConceptRelationship(source=r["source"], label=r.get("label", "relates to"),
                            target=r["target"], description=r.get("description", ""),
                            contradiction=r.get("contradiction", "") or "", position=i)
        for i, r in enumerate(draft.get("relationships") or [])
    ]
    topic.misconceptions = [
        Misconception(name=m["name"], description=m.get("description", ""),
                      clarification=m.get("clarification", ""),
                      probe_question=m.get("probe_question", ""))
        for m in (draft.get("misconceptions") or [])
    ]
    # reviewed lecture activities become the topic's stored activities, so
    # recommendations stay grounded in this lecture's own material
    topic.activities = [
        Activity(title=a["title"], description=a.get("description", ""),
                 kind=a.get("kind", "practice"),
                 target_state=a.get("target_state", "understanding"),
                 content=a.get("content", ""), question=a.get("question", ""))
        for a in (draft.get("activities") or [])
    ]


@router.post("/lectures/{lecture_id}/publish")
def publish_lecture(lecture_id: int, db: Session = Depends(get_db)):
    """Build/update the Topic from the reviewed draft ("Start TeachBack")."""
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    if not (lec.draft or {}).get("concepts"):
        raise HTTPException(400, "Keep at least one concept before starting TeachBack.")

    topic = db.get(Topic, lec.topic_id) if lec.topic_id else None
    if topic is None:
        topic = Topic()
        db.add(topic)
    apply_draft_to_topic(topic, lec)
    db.flush()
    # publish the reviewed knowledge-check questions with the topic (fall back
    # to fresh generation from the published structure if the draft has none)
    reviewed_quiz = (lec.draft or {}).get("quiz")
    build_quiz_for_topic(db, topic, questions=reviewed_quiz if reviewed_quiz else None)
    lec.topic_id = topic.id
    lec.status = "published"
    db.commit()
    db.refresh(topic)
    return {"lecture": lecture_out(lec, include_material=False), "topic": topic_def(topic)}


class RegenerateIn(BaseModel):
    index: int | None = None  # regenerate one question, or all when omitted


@router.post("/lectures/{lecture_id}/quiz/regenerate")
def regenerate_quiz(lecture_id: int, data: RegenerateIn, db: Session = Depends(get_db)):
    """Regenerate the drafted quiz (all questions, or one by index).

    Single-question regeneration picks the next unused valid candidate,
    preferring the same kind, so the teacher can cycle through alternatives.
    """
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    draft = dict(lec.draft or {})
    tdef = _draft_topic_def(draft, lec.title)
    current = list(draft.get("quiz") or [])

    if data.index is None:
        draft["quiz"] = generate_quiz_questions(tdef)
    else:
        if not 0 <= data.index < len(current):
            raise HTTPException(400, "No question at that index")
        used = {q["question"].strip().lower() for q in current}
        old = current[data.index]
        candidates = generate_quiz_candidates(tdef)
        replacement = next(
            (c for c in sorted(candidates, key=lambda c: c["kind"] != old.get("kind"))
             if c["question"].strip().lower() not in used),
            None,
        )
        if replacement is None:
            raise HTTPException(400, "No alternative question available for this material")
        current[data.index] = replacement
        draft["quiz"] = current
    lec.draft = draft
    db.commit()
    db.refresh(lec)
    return lecture_out(lec)
