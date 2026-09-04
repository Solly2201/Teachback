"""Test-database isolation.

The suite exercises the real API against a real SQLite file: it creates
topics, publishes lectures, starts sessions, submits knowledge checks and
completes activities. Those are ordinary writes, and for a while they landed
in ``backend/teachback.db`` — the developer's demo database — leaving rows
like "Test Topic v2" and "Homeless topic" visible in Topic Management.

Cleaning up after each test would only ever cover the objects someone
remembered to list, so the isolation is structural instead:

1. ``TEACHBACK_DB_PATH`` is pointed at a throwaway file in a temp directory
   BEFORE anything imports ``app.config``, so the engine, every session and
   ``seed_db()`` all address that file and the demo database is never opened.
2. The throwaway file is seeded once per session and snapshotted.
3. Every test starts from a fresh copy of that snapshot, so tests cannot see
   each other's writes either — no ordering dependencies, no accumulated junk.

test_test_isolation.py asserts all of this from the outside.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# The database the running app uses. The suite must never touch it.
DEMO_DB_PATH = BACKEND_DIR / "teachback.db"

_TMP_DIR = Path(tempfile.mkdtemp(prefix="teachback-tests-"))
TEST_DB_PATH = _TMP_DIR / "teachback.db"
TEMPLATE_DB_PATH = _TMP_DIR / "template.db"

# Must happen at import time: pytest imports conftest before any test module,
# and app.config reads this once, when it is first imported.
os.environ["TEACHBACK_DB_PATH"] = str(TEST_DB_PATH)

# The experimental LLM layer must be OFF for the whole suite regardless of
# what the developer's .env says (app.config loads .env with setdefault
# semantics, so real environment variables — these — always win). Tests that
# exercise the feature enable it explicitly with monkeypatch and injected
# fake providers; no automated test may ever call a real LLM API.
os.environ["TEACHBACK_LLM_ENABLED"] = "false"
os.environ["TEACHBACK_GENERATIVE_PROBES"] = "false"
os.environ["GEMINI_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""


def _fingerprint(path: Path) -> tuple | None:
    """Enough of a file's identity to prove it was not written to."""
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


# Captured before the first test runs, compared afterwards by the isolation test.
DEMO_DB_FINGERPRINT_AT_START = _fingerprint(DEMO_DB_PATH)


def _sqlite_files(path: Path):
    """The main file plus the journal/WAL siblings SQLite may leave beside it."""
    return [path, path.with_name(path.name + "-journal"),
            path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")]


def _swap_in(source: Path, target: Path) -> None:
    from app.database import engine

    # Close pooled connections first: replacing the file under an open handle
    # is exactly how a "database disk image is malformed" happens.
    engine.dispose()
    for stale in _sqlite_files(target)[1:]:
        stale.unlink(missing_ok=True)
    shutil.copyfile(source, target)


@pytest.fixture(scope="session", autouse=True)
def _seeded_test_database():
    """Create, migrate and seed the throwaway database, then snapshot it.

    TestClient only fires the app's startup hook inside a ``with`` block, so
    without this the suite would run against whatever schema the file happens
    to have. Seeding here also means the demo data every test relies on
    (teachers, subjects, the seeded topics, demo students and their observation
    histories) is present and identical for each test.
    """
    from app.config import DB_PATH

    assert Path(DB_PATH) == TEST_DB_PATH, (
        f"tests must run against the throwaway database, not {DB_PATH}"
    )

    from app.database import engine
    from app.seed import seed_db

    seed_db()
    engine.dispose()
    shutil.copyfile(TEST_DB_PATH, TEMPLATE_DB_PATH)
    yield
    engine.dispose()
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def pristine_database(_seeded_test_database):
    """Restore the seeded snapshot before each test.

    Restoring rather than cleaning up means a test cannot leave anything
    behind even if it fails halfway through creating it.
    """
    _swap_in(TEMPLATE_DB_PATH, TEST_DB_PATH)
    yield
