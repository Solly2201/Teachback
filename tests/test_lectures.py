"""Lecture workflow tests: material -> NLP draft -> faculty review -> publish
-> TeachBack, plus the seeded Python sample and teacher/subject scoping."""
import base64

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MATERIAL = (
    "A decision tree splits the data using simple questions about the features. "
    "Each split tries to separate the classes as cleanly as possible. "
    "The leaves of the tree hold the final prediction. "
    "Deep trees can overfit the training data, so pruning removes branches that do not help."
)


def _subjects():
    teachers = client.get("/api/teachers").json()
    return {s["name"]: s for t in teachers for s in t["subjects"]}, teachers


# 1. Demo teachers and subjects exist and are properly related
def test_teachers_and_subjects_listed():
    subjects, teachers = _subjects()
    assert len(teachers) >= 2
    assert "Neural Networks" in subjects and "Python Programming" in subjects
    by_name = {t["name"]: t for t in teachers}
    assert any(s["name"] == "Python Programming" for s in by_name["Prof. Arjun Rao"]["subjects"])
    assert any(s["name"] == "Neural Networks" for s in by_name["Prof. Meera Krishnan"]["subjects"])


# 2. The Python sample lecture (Strings) was seeded through the real pipeline
def test_python_sample_lecture_seeded():
    subjects, _ = _subjects()
    lectures = client.get(f"/api/lectures?subject_id={subjects['Python Programming']['id']}").json()
    assert len(lectures) >= 1
    seeded = next(l for l in lectures if l["title"] == "Strings in Python")
    lec = client.get(f"/api/lectures/{seeded['id']}").json()
    assert lec["status"] == "published"
    assert lec["topic_id"] is not None
    # the NLP suggestions were actually computed from the material (not bypassed)
    assert lec["suggestions"]["concepts"], "no NLP suggestions stored"
    suggested = {c["name"] for c in lec["suggestions"]["concepts"]}
    assert {"Strings", "Indexing", "Slicing"} <= suggested
    # the reviewed draft is the teacher-curated version
    reviewed = {c["name"] for c in lec["draft"]["concepts"]}
    assert {"Strings", "String assignment", "Characters", "Indexing", "Slicing",
            "split() and join()"} <= reviewed
    assert lec["objectives"], "learning objectives missing"
    # published topic carries the reviewed concepts, with facts and provenance
    topic = client.get(f"/api/topics/{lec['topic_id']}").json()
    assert {c["name"] for c in topic["concepts"]} == reviewed
    assert topic["subject_name"] == "Python Programming"
    indexing = next(c for c in topic["concepts"] if c["name"] == "Indexing")
    assert "Indexes start at 0 in Python." in indexing["facts"]
    assert indexing["source"]["section"] == "Indexing"
    # the reviewed activities were published with the topic
    assert topic["activities"], "reviewed lecture activities missing from topic"
    assert any("slicing" in a["title"].lower() for a in topic["activities"])


# 3. Subject scoping: one teacher's topics don't leak into another subject
def test_topics_scoped_by_subject():
    subjects, _ = _subjects()
    nn = client.get(f"/api/topics?subject_id={subjects['Neural Networks']['id']}").json()
    py = client.get(f"/api/topics?subject_id={subjects['Python Programming']['id']}").json()
    nn_names = {t["name"] for t in nn}
    py_names = {t["name"] for t in py}
    assert "Backpropagation" in nn_names and "Backpropagation" not in py_names
    assert any("Strings" in n for n in py_names)
    assert not any("Strings" in n for n in nn_names)


# 4. Creating a lecture extracts candidate concepts + objectives from material
def test_create_lecture_extracts_concepts():
    subjects, _ = _subjects()
    lec = client.post("/api/lectures", json={
        "subject_id": subjects["Neural Networks"]["id"],
        "title": "Decision Trees",
        "description": "How decision trees classify data.",
        "material_text": MATERIAL,
    }).json()
    concepts = lec["draft"]["concepts"]
    assert concepts, "no concepts extracted"
    names = " ".join(c["name"].lower() for c in concepts)
    assert "tree" in names or "split" in names or "pruning" in names
    # descriptions are suggested from the material itself
    assert any(c["description"] for c in concepts)
    # objectives auto-drafted when the teacher gives none
    assert lec["objectives"]


# 5. Teacher can edit/remove/add concepts and the edits persist
def test_teacher_can_edit_draft():
    subjects, _ = _subjects()
    lec = client.post("/api/lectures", json={
        "subject_id": subjects["Neural Networks"]["id"],
        "title": "Decision Trees v2", "material_text": MATERIAL,
    }).json()
    edited = [{"name": "Splits", "description": "A split separates the data using a feature question."},
              {"name": "Pruning", "description": "Pruning removes branches that do not help."}]
    lec = client.put(f"/api/lectures/{lec['id']}", json={"concepts": edited}).json()
    assert [c["name"] for c in lec["draft"]["concepts"]] == ["Splits", "Pruning"]


# 6. Teacher-provided objectives take priority and are preserved verbatim
def test_teacher_objectives_preserved():
    subjects, _ = _subjects()
    objectives = ["Explain what a split does.", "Explain why pruning helps."]
    lec = client.post("/api/lectures", json={
        "subject_id": subjects["Neural Networks"]["id"],
        "title": "Decision Trees v3", "material_text": MATERIAL,
        "objectives": objectives,
    }).json()
    assert lec["objectives"] == objectives


# 7. Publish creates a working topic; a student can TeachBack it with simple
#    words, and no advanced/extension questioning is forced
def test_publish_creates_working_teachback():
    subjects, _ = _subjects()
    lec = client.post("/api/lectures", json={
        "subject_id": subjects["Neural Networks"]["id"],
        "title": "Decision Trees Live", "material_text": MATERIAL,
    }).json()
    r = client.post(f"/api/lectures/{lec['id']}/publish")
    assert r.status_code == 200
    topic = r.json()["topic"]
    assert topic["concepts"]
    # conversational main questions, and no forced extension question
    assert topic["concepts"][0]["main_question"].startswith("What did you understand about")
    assert topic["extension_question"] == ""

    students = client.get("/api/students").json()
    start = client.post("/api/sessions/start",
                        json={"student_id": students[0]["id"], "topic_id": topic["id"]}).json()
    assert start["prompt"].startswith("What did you understand about")
    step = client.post(f"/api/sessions/{start['session_id']}/respond", json={
        "text": "It keeps asking simple questions about the data to split it into cleaner groups."}).json()
    assert step["feedback"]

    # republishing after an edit updates the SAME topic, not a duplicate
    client.put(f"/api/lectures/{lec['id']}", json={
        "concepts": [{"name": "Splitting", "description": "Each split separates the classes."}]})
    r2 = client.post(f"/api/lectures/{lec['id']}/publish").json()
    assert r2["topic"]["id"] == topic["id"]
    assert [c["name"] for c in r2["topic"]["concepts"]] == ["Splitting"]


# 8. Text can be extracted from an uploaded file
def test_extract_text_file():
    payload = base64.b64encode("Lecture notes about trees.".encode()).decode()
    r = client.post("/api/lectures/extract", json={"filename": "notes.txt", "content_base64": payload})
    assert r.status_code == 200
    assert "trees" in r.json()["text"]
    # unsupported types are rejected with a clear message
    r = client.post("/api/lectures/extract", json={"filename": "deck.pptx", "content_base64": payload})
    assert r.status_code == 400
