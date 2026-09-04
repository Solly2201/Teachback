"""Runtime configuration for the experimental LLM layer.

Read from the environment on every call (not at import time) so the feature
can be toggled per process and tests can flip flags with monkeypatch without
re-importing the app. All defaults are OFF / empty: a checkout with no .env
behaves exactly like v1.

API keys live only in these settings objects on the server. They are never
persisted, never returned by any endpoint and never included in error text
(see errors.sanitize).
"""
import os
from dataclasses import dataclass, field

# Importing app.config loads .env into the environment (setdefault semantics).
# Needed here so standalone scripts that import only the LLM layer still see
# the .env configuration; in the running app config is imported long before.
from .. import config  # noqa: F401


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ProviderSettings:
    name: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


@dataclass
class LLMSettings:
    enabled: bool
    generative_probes: bool
    primary: str
    fallback: str
    max_retries: int
    timeout_seconds: float
    providers: dict = field(default_factory=dict)

    def provider_order(self) -> list[ProviderSettings]:
        """Primary then fallback, keeping only providers with an API key.

        A missing key silently removes that provider from the chain rather
        than producing auth errors: with only GROQ_API_KEY set, Groq simply
        becomes the first (and only) provider tried.
        """
        order = []
        for name in (self.primary, self.fallback):
            p = self.providers.get(name)
            if p is not None and p.configured and p not in order:
                order.append(p)
        return order


# Documented defaults; both are free-tier models at the time of writing
# (September 2026 — gemini-2.5-flash is already retired for new API users).
# They are deliberately NOT hard-coded anywhere else: when a provider retires
# a model, set GEMINI_MODEL / GROQ_MODEL in .env instead of editing code.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def llm_settings() -> LLMSettings:
    return LLMSettings(
        enabled=_flag("TEACHBACK_LLM_ENABLED"),
        generative_probes=_flag("TEACHBACK_GENERATIVE_PROBES"),
        primary=os.environ.get("TEACHBACK_LLM_PRIMARY", "gemini").strip().lower(),
        fallback=os.environ.get("TEACHBACK_LLM_FALLBACK", "groq").strip().lower(),
        max_retries=int(os.environ.get("TEACHBACK_LLM_MAX_RETRIES", "1") or 1),
        timeout_seconds=float(os.environ.get("TEACHBACK_LLM_TIMEOUT_SECONDS", "15") or 15),
        providers={
            "gemini": ProviderSettings(
                name="gemini",
                api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
                model=os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
            ),
            "groq": ProviderSettings(
                name="groq",
                api_key=os.environ.get("GROQ_API_KEY", "").strip(),
                model=os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL,
            ),
        },
    )


def generative_probes_enabled() -> bool:
    """The one switch the conversation flow checks. Both flags must be on."""
    s = llm_settings()
    return s.enabled and s.generative_probes
