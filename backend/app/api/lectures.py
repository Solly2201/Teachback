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
                              (sent base64-encoded to avoid a multipart dep);
                              PDFs go through the layout-aware extractor and
                              the deterministic boilerplate cleanup, and come
                              back as notes plus an ingestion report
DELETE /api/lectures/{id}     delete a lecture with no student history, or
                              ARCHIVE it when history exists — a UI action
                              must never destroy learning records
POST /api/lectures/{id}/restore
                              bring an archived lecture back
GET  /api/teachers            demo teachers with their subjects, for the
                              lightweight teacher/subject switcher
"""
import base64
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (Activity, Concept, ConceptRelationship, Lecture,
                      Misconception, Quiz, Subject, Teacher, Topic)
from ..nlp.lecture_prep import ScannedPdfError, extract_material, prepare_lecture
from ..nlp.note_template import AI_PREP_PROMPT, NOTE_TEMPLATE
from ..nlp.quiz_gen import (QUIZ_SIZE, generate_quiz_candidates,
                            generate_quiz_questions)
from .helpers import history_summary, topic_def, topic_history
from .quiz import build_quiz_for_topic

router = APIRouter(prefix="/api", tags=["lectures"])

MAIN_QUESTION_TEMPLATE = 'What did you understand about "{name}"?'
# below this the "material" is a title, not a lecture — analysing it would
# only produce noise for the teacher to delete
MIN_MATERIAL_WORDS = 12


class LectureIn(BaseModel):
    subject_id: int
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    material_text: str = Field(default="", max_length=400_000)
    objectives: list[str] = Field(default_factory=list)
    # the ingestion report from /lectures/extract, so the review screen can
    # show what the PDF cleanup removed (never used for scoring)
    ingestion: dict = Field(default_factory=dict)


class DraftConcept(BaseModel):
    name: str
    description: str = ""
    facts: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    # provenance + honesty about the extraction, kept through the teacher's
    # edits so the review screen can always explain where a suggestion came
    # from and how strong its evidence was
    source_section: str = ""
    source_page: int | None = None
    source_pages: list[int] = Field(default_factory=list)
    source_sentences: list[str] = Field(default_factory=list)
    confidence: str = ""
    confidence_label: str = ""
    confidence_reason: str = ""
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


def _quiz_note(questions: list[dict]) -> str:
    """Tell the teacher when the material could not support a full check."""
    n = len(questions or [])
    if n >= QUIZ_SIZE:
        return ""
    if n == 0:
        return ("No knowledge-check questions could be built from this material. The check is "
                "optional — add misconceptions or examples to the draft, or write questions yourself.")
    return (f"Only {n} of {QUIZ_SIZE} knowledge-check questions could be built from this "
            "material. Adding a common mistake or a worked example to the draft gives the "
            "generator more to work with; padding with weak questions would not help students.")


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
        "archived": lec.archived_at is not None,
        "archived_at": lec.archived_at.isoformat() if lec.archived_at else None,
        "ingestion": lec.ingestion or {},
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
def extract_upload(data: ExtractIn):
    """Turn an uploaded file into lecture notes plus an ingestion report.

    The returned ``text`` is the CLEANED material (what the teacher reviews
    and what the parser sees); ``raw_text`` is the untouched extraction, kept
    for the optional "view raw extraction" transparency view.
    """
    try:
        raw = base64.b64decode(data.content_base64)
    except Exception:
        raise HTTPException(400, "The uploaded file could not be decoded.")
    try:
        text, report = extract_material(data.filename, raw)
    except ScannedPdfError as exc:
        # an image-only PDF: say so plainly instead of pretending it worked
        raise HTTPException(400, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file.")
    if not text.strip():
        raise HTTPException(400, "No usable text could be extracted from the file.")
    raw_text = report.pop("raw_text", "")
    return {"text": text, "raw_text": raw_text, "report": report}


@router.get("/lectures")
def list_lectures(subject_id: int | None = None, include_archived: bool = False,
                  db: Session = Depends(get_db)):
    """Active lectures for a subject. Archived ones are hidden by default."""
    q = db.query(Lecture).order_by(Lecture.id.desc())
    if subject_id is not None:
        q = q.filter(Lecture.subject_id == subject_id)
    if not include_archived:
        q = q.filter(Lecture.archived_at.is_(None))
    return [lecture_out(l, include_material=False) for l in q.all()]


@router.get("/lectures/{lecture_id}")
def get_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    return lecture_out(lec)


def _known_misconceptions(db: Session, subject_id: int) -> list[dict]:
    """Previously authored misconceptions THIS subject may reuse.

    The preparation step matches stored misconceptions against the new
    material and offers the close ones as suggestions. Reading the whole table
    made that a cross-subject leak: a Python lecture that mentions error and
    weights was offered the Neural Networks teacher's authored misconceptions,
    ready to publish into a Python topic. A misconception belongs to a topic
    and a topic to a subject, so the catalog is scoped the same way every
    faculty-facing aggregate is. Archived topics are left out, like everywhere
    else — retired material should not seed new lectures.
    """
    rows = (db.query(Misconception)
            .join(Topic, Topic.id == Misconception.topic_id)
            .filter(Topic.subject_id == subject_id, Topic.archived_at.is_(None))
            .all())
    return [
        {"name": m.name, "description": m.description,
         "clarification": m.clarification, "probe_question": m.probe_question}
        for m in rows
    ]


@router.post("/lectures")
def create_lecture(data: LectureIn, db: Session = Depends(get_db)):
    subject = db.get(Subject, data.subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    if not data.title.strip():
        raise HTTPException(400, "Give the lecture a title.")
    if not data.material_text.strip():
        raise HTTPException(400, "Provide the lecture material (paste notes or upload a file).")
    if len(data.material_text.split()) < MIN_MATERIAL_WORDS:
        raise HTTPException(
            400,
            f"The lecture material is too short to analyse (at least {MIN_MATERIAL_WORDS} words). "
            "Paste the notes, or upload the slides.")

    prep = prepare_lecture(
        data.material_text, title=data.title, description=data.description,
        objectives=data.objectives, known_misconceptions=_known_misconceptions(db, subject.id),
    )
    draft = {
        # the full evidence-carrying concept suggestions (facts, examples,
        # provenance, drafted questions) become the editable draft
        "concepts": [
            {k: c.get(k) for k in ("name", "description", "facts", "examples",
                                   "source_section", "source_page", "source_pages",
                                   "source_sentences", "confidence", "confidence_label",
                                   "confidence_reason",
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
    # and reviewable/editable like everything else. The generator only ever
    # uses reviewed lecture material, so material that supports fewer than a
    # full check yields fewer questions — said out loud rather than padded.
    draft["quiz"] = generate_quiz_questions(_draft_topic_def(draft, data.title))
    draft["quiz_note"] = _quiz_note(draft["quiz"])
    lec = Lecture(
        subject_id=subject.id, title=data.title, description=data.description,
        material_text=data.material_text, objectives=prep["objectives"],
        draft=draft, suggestions=prep, status="draft",
        ingestion=data.ingestion or {},
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
    if lec.archived_at is not None:
        raise HTTPException(400, "This lecture is archived. Restore it before editing.")
    if data.title is not None:
        if not data.title.strip():
            raise HTTPException(400, "The lecture title cannot be empty.")
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
        draft["quiz_note"] = _quiz_note(draft["quiz"])
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

    topic.archived_at = None  # republishing an archived lecture reactivates it
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
                    "page": c.get("source_page"),
                    "pages": c.get("source_pages") or [],
                    "confidence": c.get("confidence", ""),
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
    """Build/update the Topic from the reviewed draft ("Start TeachBack").

    One lecture owns one published topic. Publishing REPLACES that topic's
    knowledge structure, so before reusing a topic we check that no other
    lecture points at it — otherwise republishing lecture A would silently
    overwrite lecture B's published material and invalidate its concepts.
    """
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    if lec.archived_at is not None:
        raise HTTPException(400, "This lecture is archived. Restore it before publishing.")
    concepts = [c for c in ((lec.draft or {}).get("concepts") or [])
                if str(c.get("name", "")).strip()]
    if not concepts:
        raise HTTPException(400, "Keep at least one named concept before starting TeachBack.")
    if not any(str(c.get("description", "")).strip() for c in concepts):
        raise HTTPException(
            400,
            "At least one concept needs a meaning — that is what a student's explanation is "
            "compared against.")

    topic = db.get(Topic, lec.topic_id) if lec.topic_id else None
    if topic is not None:
        shared_with = (db.query(Lecture)
                       .filter(Lecture.topic_id == topic.id, Lecture.id != lec.id)
                       .first())
        if shared_with is not None:
            # another lecture owns this topic: give this one its own topic
            # instead of destroying the other lecture's published structure
            topic = None
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
        draft["quiz_note"] = _quiz_note(draft["quiz"])
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


# ---------------------------------------------------------------------------
# delete / archive
# ---------------------------------------------------------------------------
# Deleting a lecture must never delete what students did. A published lecture
# owns a Topic, and TeachSession / Observation / QuizAttempt / ActivityCompletion
# rows point at that Topic. Cascading through them would erase real learning
# records because a teacher tidied up their lecture list.
#
# So the delete action has two modes, chosen from the data, not from a flag:
#
#   deleted   - no student ever touched this lecture's topic: the lecture,
#               its topic and the topic's owned rows (concepts, relationships,
#               misconceptions, activities, quiz) are removed outright.
#   archived  - student history exists: the lecture and its topic are marked
#               archived. They vanish from the active lecture/topic lists and
#               from the subject dashboard aggregates, no new session can be
#               started on them, and every historical record stays readable
#               with valid foreign keys.


def lecture_history(db: Session, lec: Lecture) -> dict:
    """How much student history hangs off this lecture's published topic.

    The counting itself lives in helpers.topic_history so that deleting a
    topic directly (Topic Management) and deleting the lecture that owns it
    reach the identical delete-or-archive decision.
    """
    return topic_history(db, lec.topic_id)


@router.get("/lectures/{lecture_id}/delete-preview")
def delete_preview(lecture_id: int, db: Session = Depends(get_db)):
    """What deleting this lecture would do — for the confirmation dialog."""
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    history = lecture_history(db, lec)
    mode = "archive" if history["total"] else "delete"
    return {
        "lecture_id": lec.id,
        "title": lec.title,
        "status": lec.status,
        "history": history,
        "history_summary": history_summary(history),
        "mode": mode,
        "message": (
            f'"{lec.title}" already has student learning records '
            f"({history_summary(history)}). It will be ARCHIVED rather than erased: it "
            "disappears from your active lectures and no new TeachBack can start on it, "
            "while every one of those records stays intact and readable."
            if mode == "archive" else
            f'Delete "{lec.title}"? This removes the lecture and its TeachBack configuration. '
            "No student has worked on it, so no learning record is affected."
        ),
    }


@router.delete("/lectures/{lecture_id}")
def delete_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    history = lecture_history(db, lec)
    title = lec.title
    topic = db.get(Topic, lec.topic_id) if lec.topic_id else None

    if history["total"]:
        now = datetime.utcnow()
        lec.archived_at = now
        lec.status = "archived"
        if topic is not None:
            topic.archived_at = now
        db.commit()
        return {
            "mode": "archived", "lecture_id": lecture_id, "title": title,
            "history": history, "history_summary": history_summary(history),
            "message": (f'"{title}" was archived, not erased. Its {history_summary(history)} '
                        "remain intact and still show in student history; the lecture no "
                        "longer appears in your active list and cannot start new sessions."),
        }

    # nothing to preserve: remove the lecture and the topic it owns
    if topic is not None:
        quiz = db.query(Quiz).filter(Quiz.topic_id == topic.id).first()
        if quiz is not None:
            db.delete(quiz)  # cascades to its questions
        db.delete(topic)     # cascades to concepts / relationships / misconceptions / activities
    db.delete(lec)
    db.commit()
    return {"mode": "deleted", "lecture_id": lecture_id, "title": title, "history": history,
            "history_summary": history_summary(history),
            "message": f'"{title}" was deleted. No student records were affected.'}


@router.post("/lectures/{lecture_id}/restore")
def restore_lecture(lecture_id: int, db: Session = Depends(get_db)):
    """Bring an archived lecture (and its topic) back into the active lists."""
    lec = db.get(Lecture, lecture_id)
    if not lec:
        raise HTTPException(404, "Lecture not found")
    if lec.archived_at is None:
        raise HTTPException(400, "This lecture is not archived.")
    lec.archived_at = None
    lec.status = "published" if lec.topic_id else "draft"
    topic = db.get(Topic, lec.topic_id) if lec.topic_id else None
    if topic is not None:
        topic.archived_at = None
    db.commit()
    db.refresh(lec)
    return lecture_out(lec, include_material=False)
