"""Deterministic fake LLM providers/services for the generative-probe tests.

No test in this suite may call a real provider: conftest.py pins the feature
flags off and blanks both API keys, and every scenario here is scripted.
"""
import json

from app.llm.prompts import PROMPT_VERSION
from app.llm.providers import LLMProvider
from app.llm.schema import GeneratedProbe
from app.llm.settings import LLMSettings, ProviderSettings

FAKE_KEY = "fake-key-not-a-secret"


def fake_settings(**overrides) -> LLMSettings:
    defaults = dict(enabled=True, generative_probes=True, primary="gemini",
                    fallback="groq", max_retries=1, timeout_seconds=15, providers={})
    defaults.update(overrides)
    return LLMSettings(**defaults)


def probe_json(decision: dict, grounding_ids: list[str],
               question: str = "In your own words, what does that idea mean?",
               **overrides) -> str:
    """A provider response that correctly echoes the controller decision."""
    data = {
        "action": decision["action"],
        "target_type": decision["target_type"],
        "target_id": decision["target_id"],
        "difficulty": decision["difficulty"],
        "question": question,
        "grounding_ids": grounding_ids,
        "rationale": "Asks for the idea in the student's own words.",
    }
    data.update(overrides)
    return json.dumps(data)


class FakeProvider(LLMProvider):
    """Scripted provider: each call pops the next behavior.

    A behavior is either a string (returned as the raw response text), an
    Exception instance (raised), or a callable taking the user prompt.
    """

    def __init__(self, name: str, script: list):
        super().__init__(ProviderSettings(name=name, api_key=FAKE_KEY,
                                          model=f"fake-{name}-model"), 15)
        self.name = name
        self.script = list(script)
        self.calls = 0
        self.user_prompts: list[str] = []

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.user_prompts.append(user_prompt)
        assert self.script, f"FakeProvider {self.name} called more times than scripted"
        behavior = self.script.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            return behavior(user_prompt)
        return behavior


class FakeService:
    """Drop-in for LLMService in endpoint tests: echoes the decision, or fails."""

    def __init__(self, question: str = "Here is a fake generated question — can you explain the idea?",
                 fail: Exception | None = None, meta: dict | None = None):
        self.question = question
        self.fail = fail
        self.meta = meta or {"provider_used": "gemini", "model_used": "fake-gemini-model",
                             "fallback_used": False, "prompt_version": PROMPT_VERSION,
                             "latency_ms": 1}
        self.calls: list[dict] = []

    def generate_probe(self, decision, context_items, previous_answer):
        self.calls.append({"decision": decision, "items": context_items,
                           "previous_answer": previous_answer})
        if self.fail is not None:
            raise self.fail
        probe = GeneratedProbe(
            action=decision["action"], target_type=decision["target_type"],
            target_id=decision["target_id"], difficulty=decision["difficulty"],
            question=self.question,
            grounding_ids=[context_items[0]["id"]] if context_items else [],
            rationale="Fake rationale for the teacher.")
        return probe, dict(self.meta)
