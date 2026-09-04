"""MANUAL smoke test for the real Gemini/Groq providers. Not part of pytest.

Sends one tiny controller decision + teacher-grounded context through the
real LLMService using the .env configuration, and prints what came back:
which provider/model answered, whether failover happened, the latency, and
the generated question. Never prints keys, headers or raw payloads.

    python scripts/llm_smoke_test.py            # normal: primary, fallback on error
    python scripts/llm_smoke_test.py --provider groq   # force one provider only

The automated test suite never runs this file (its name does not match
test_*.py) and never calls real APIs.
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))

from app.llm.errors import LLMUnavailable  # noqa: E402
from app.llm.providers import PROVIDER_CLASSES  # noqa: E402
from app.llm.service import LLMService  # noqa: E402
from app.llm.settings import llm_settings  # noqa: E402

DECISION = {
    "action": "ASK_PROBE",
    "target_type": "relationship",
    "target_id": 17,
    "target_name": "String -> Sequence of characters",
    "difficulty": "easy",
}
CONTEXT = [
    {"id": "relationship:17", "kind": "relationship_explanation",
     "text": "A string is an ordered sequence of characters."},
    {"id": "relationship_contradiction:17", "kind": "known_wrong_claim",
     "text": "A wrong version students state: a string is a single indivisible value."},
    {"id": "concept:4", "kind": "concept_explanation",
     "text": "String: text data, written in quotes, made up of individual characters."},
]
PREVIOUS_ANSWER = "A string is basically a variable containing text."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "groq"],
                        help="force a single provider instead of the configured chain")
    args = parser.parse_args()

    settings = llm_settings()
    if not settings.enabled:
        raise SystemExit("Set TEACHBACK_LLM_ENABLED=true in .env first.")
    if args.provider:
        p = settings.providers[args.provider]
        if not p.configured:
            raise SystemExit(f"No API key configured for {args.provider}.")
        service = LLMService(settings=settings, providers=[
            PROVIDER_CLASSES[p.name](p, settings.timeout_seconds)])
    else:
        service = LLMService(settings=settings)
        if not service.providers:
            raise SystemExit("No provider has an API key in .env.")

    print("provider chain:", [p.name for p in service.providers])
    try:
        probe, meta = service.generate_probe(DECISION, CONTEXT, PREVIOUS_ANSWER)
    except LLMUnavailable as e:
        print("FAILED — all providers unavailable:")
        for attempt in e.attempts:
            print("  ", attempt)
        raise SystemExit(1)

    if meta.get("failed_attempts"):
        print("skipped       :", meta["failed_attempts"])
    print("provider_used :", meta["provider_used"])
    print("model_used    :", meta["model_used"])
    print("fallback_used :", meta["fallback_used"])
    print("prompt_version:", meta["prompt_version"])
    print("latency_ms    :", meta["latency_ms"])
    print("grounding_ids :", probe.grounding_ids)
    print("rationale     :", probe.rationale)
    print("question      :", probe.question)


if __name__ == "__main__":
    main()
