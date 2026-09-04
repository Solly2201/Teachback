import os
from pathlib import Path

# Force transformers to use the PyTorch backend only (a TensorFlow install with
# Keras 3 on the same machine would otherwise break the import).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (deliberately no python-dotenv dependency).

    KEY=VALUE lines become environment DEFAULTS: a variable already set in
    the real environment always wins, which is what lets tests/conftest.py
    pin the LLM feature flags off and the DB path at a throwaway file no
    matter what the developer's .env says. Values (API keys live here) are
    never printed or logged.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # an unquoted value may carry an inline comment ("KEY=x  # note")
        if not (value.startswith('"') or value.startswith("'")):
            value = value.split(" #", 1)[0].split("\t#", 1)[0].rstrip()
        value = value.strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(PROJECT_DIR / ".env")

DATA_DIR = PROJECT_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# The developer/demo database. It is the source of truth for the running app
# and holds real seeded and student data, so nothing automated may write to it.
DEMO_DB_PATH = BACKEND_DIR / "teachback.db"

# TEACHBACK_DB_PATH redirects every engine, session and seeding call at a
# different SQLite file. tests/conftest.py sets it to a throwaway copy before
# the app is imported, which is what keeps the suite from leaving topics,
# lectures and sessions behind in the demo database.
DB_PATH = Path(os.environ.get("TEACHBACK_DB_PATH") or DEMO_DB_PATH)
DATABASE_URL = f"sqlite:///{DB_PATH}"

HMM_MODEL_PATH = ARTIFACTS_DIR / "hmm_model.joblib"
HMM_MAPPING_PATH = ARTIFACTS_DIR / "hmm_state_mapping.json"
EVAL_RESULTS_PATH = ARTIFACTS_DIR / "evaluation_results.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

for _d in (DATA_DIR, SYNTHETIC_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
