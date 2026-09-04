"""LLMService — the single application-level entry point to any LLM.

The rest of the codebase calls generate_probe() and gets back a validated
GeneratedProbe plus observability metadata; which provider actually answered
is this module's business. Failover policy (see errors.py for the taxonomy):

  * transient (quota / rate limit / 5xx / timeout / network): move to the
    next provider immediately — no same-provider retry, so a quota storm
    cannot become a retry storm
  * auth/model config error: also move on — the other provider has its own
    credentials and model name
  * malformed request (our fault): raise immediately; both providers would
    reject the same payload
  * invalid output (bad JSON / schema / decision mismatch): retry the SAME
    provider up to max_retries times (default 1), then move on

If every configured provider fails, LLMUnavailable is raised and the caller
falls back to deterministic v1 behavior — a student session never blocks on
an LLM.
"""
import time

from .errors import (LLMError, LLMOutputError, LLMRequestError, LLMUnavailable)
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .providers import PROVIDER_CLASSES
from .schema import GeneratedProbe, parse_probe, validate_against_decision
from .settings import LLMSettings, llm_settings


class LLMService:
    def __init__(self, settings: LLMSettings | None = None, providers: list | None = None):
        """`providers` overrides the constructed chain (used by tests and the
        evaluation harness to inject fakes)."""
        self.settings = settings or llm_settings()
        if providers is not None:
            self.providers = providers
        else:
            self.providers = [
                PROVIDER_CLASSES[p.name](p, self.settings.timeout_seconds)
                for p in self.settings.provider_order()
                if p.name in PROVIDER_CLASSES
            ]

    def generate_probe(self, decision: dict, context_items: list[dict],
                       previous_answer: str) -> tuple[GeneratedProbe, dict]:
        """Generate one probe for an already-made controller decision.

        Returns (probe, meta) where meta records which provider and model
        produced the wording — enough for reproducibility, with no prompt
        text, no student text and no secrets.
        """
        if not self.settings.enabled:
            raise LLMUnavailable([{"provider": None, "error_kind": "disabled"}])
        if not self.providers:
            raise LLMUnavailable([{"provider": None, "error_kind": "no_provider_configured"}])

        user_prompt = build_user_prompt(decision, context_items, previous_answer)
        allowed = {item["id"] for item in context_items}
        primary_name = self.providers[0].name
        attempts: list[dict] = []

        for provider in self.providers:
            output_retries_left = max(0, self.settings.max_retries)
            while True:
                started = time.perf_counter()
                try:
                    raw = provider.generate_structured(SYSTEM_PROMPT, user_prompt)
                    probe = parse_probe(provider.name, raw)
                    probe = validate_against_decision(provider.name, probe, decision, allowed)
                except LLMRequestError:
                    # our payload is malformed — the other provider would
                    # reject it too, and retrying would only add noise
                    raise
                except LLMOutputError as e:
                    attempts.append(_attempt(e))
                    if output_retries_left > 0:
                        output_retries_left -= 1
                        continue
                    break  # next provider
                except LLMError as e:  # transient or auth/config
                    attempts.append(_attempt(e))
                    break  # next provider, immediately — never a retry storm
                meta = {
                    "provider_used": provider.name,
                    "model_used": provider.settings.model,
                    "fallback_used": provider.name != primary_name,
                    "prompt_version": PROMPT_VERSION,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
                if attempts:
                    # why earlier providers were skipped — error kind and HTTP
                    # status only, already sanitized, never payloads or keys
                    meta["failed_attempts"] = attempts
                return probe, meta

        raise LLMUnavailable(attempts)


def _attempt(e: LLMError) -> dict:
    # e.message is sanitized at construction (errors.LLMError) — short,
    # key-free, payload-free
    return {"provider": e.provider, "error_kind": type(e).__name__,
            "status_code": e.status_code, "message": e.message}
