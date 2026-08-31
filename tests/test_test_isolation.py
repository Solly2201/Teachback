"""The test suite must not write to the developer/demo database.

Topics like "Test Topic v2", "Homeless topic", "Kept topic" and
"Decision Trees Live" once showed up in the real Topic Management screen:
tests created them through the API, and the API was pointed at
``backend/teachback.db``. conftest.py fixes that structurally by redirecting
``TEACHBACK_DB_PATH`` at a throwaway copy before the app is imported. These
tests assert the redirection actually holds, from the outside, so the
pollution cannot come back unnoticed.
"""
import sqlite3

import pytest

from conftest import (DEMO_DB_FINGERPRINT_AT_START, DEMO_DB_PATH, TEMPLATE_DB_PATH,
                      TEST_DB_PATH, _fingerprint)

# Names produced by tests that used to end up in the demo database.
KNOWN_TEST_ARTEFACTS = [
    "Test Topic", "Test Topic v2", "Homeless topic", "Kept topic",
    "Decision Trees Live", "Doomed topic", "Innocent bystander",
    "Queues republish", "Queues owner", "Queues intruder", "Queues with history",
    "Quiz Review Test", "Cloud Computing Basics",
]


def _demo_db_names(table, column="name"):
    """Read the demo database directly, read-only, without going near the app."""
    uri = f"file:{DEMO_DB_PATH.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return {row[0] for row in con.execute(f"SELECT {column} FROM {table}")}
    finally:
        con.close()


def test_the_app_is_pointed_at_the_throwaway_database():
    from app.config import DB_PATH, DEMO_DB_PATH as CONFIG_DEMO_DB

    assert DB_PATH == TEST_DB_PATH
    assert DB_PATH != CONFIG_DEMO_DB
    from app.database import engine
    assert str(TEST_DB_PATH) in str(engine.url)


def test_the_demo_database_file_is_never_written_to():
    """Size and mtime are unchanged since before the first test ran."""
    if DEMO_DB_FINGERPRINT_AT_START is None:
        pytest.skip("no demo database on this machine")
    assert _fingerprint(DEMO_DB_PATH) == DEMO_DB_FINGERPRINT_AT_START


def test_objects_created_by_a_test_do_not_reach_the_demo_database():
    """Create the exact kinds of rows that used to leak, then look for them."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    subject_id = client.get("/api/teachers").json()[0]["subjects"][0]["id"]
    topic = client.post("/api/topics", json={
        "name": "Pollution probe topic", "subject_id": subject_id, "description": "d",
        "reference_explanation": "ref", "concepts": [], "relationships": [],
        "misconceptions": [], "activities": []}).json()
    lecture = client.post("/api/lectures", json={
        "subject_id": subject_id, "title": "Pollution probe lecture",
        "material_text": "Probes exist only to be looked for. " * 6}).json()

    # present in the database the tests are actually using
    assert any(t["id"] == topic["id"] for t in client.get("/api/topics").json())
    assert lecture["id"]

    if not DEMO_DB_PATH.exists():
        pytest.skip("no demo database on this machine")
    assert "Pollution probe topic" not in _demo_db_names("topics")
    assert "Pollution probe lecture" not in _demo_db_names("lectures", "title")


def test_the_demo_database_holds_no_known_test_artefacts():
    """The demo database itself is clean of the names tests are known to make."""
    if not DEMO_DB_PATH.exists():
        pytest.skip("no demo database on this machine")
    leftovers = _demo_db_names("topics") & set(KNOWN_TEST_ARTEFACTS)
    assert not leftovers, f"test-created topics found in the demo database: {sorted(leftovers)}"
    lectures = _demo_db_names("lectures", "title") & set(KNOWN_TEST_ARTEFACTS)
    assert not lectures, f"test-created lectures found in the demo database: {sorted(lectures)}"


# --- the per-test snapshot: tests cannot see each other's writes either ----
# These two run in file order; the first leaves a row behind on purpose.

LEAK_PROBE = "Cross-test leak probe"


def test_a_topic_created_here_is_deliberately_not_cleaned_up():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    subject_id = client.get("/api/teachers").json()[0]["subjects"][0]["id"]
    r = client.post("/api/topics", json={
        "name": LEAK_PROBE, "subject_id": subject_id, "description": "d",
        "reference_explanation": "ref", "concepts": [], "relationships": [],
        "misconceptions": [], "activities": []})
    assert r.status_code == 200
    assert LEAK_PROBE in {t["name"] for t in client.get("/api/topics").json()}


def test_the_next_test_starts_from_the_pristine_snapshot():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    assert LEAK_PROBE not in {t["name"] for t in client.get("/api/topics").json()}


def test_the_snapshot_contains_the_seeded_demo_data():
    """Isolation must not cost the tests the demo data they rely on."""
    con = sqlite3.connect(f"file:{TEMPLATE_DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        topics = {row[0] for row in con.execute("SELECT name FROM topics")}
        students = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    finally:
        con.close()
    assert {"Backpropagation", "Overfitting and Regularization",
            "Hidden Markov Models"} <= topics
    assert students >= 8
    assert not topics & set(KNOWN_TEST_ARTEFACTS)
