import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


@pytest.fixture(scope="session", autouse=True)
def _initialise_database():
    """Create/migrate/seed the demo database before any test runs.

    TestClient only fires the app's startup hook inside a ``with`` block, so
    without this the suite would run against whatever schema the local SQLite
    file happens to have — and a column added since it was created would fail
    every query. seed_db() is idempotent: it migrates, then seeds only when
    the database is empty.
    """
    from app.seed import seed_db

    seed_db()
