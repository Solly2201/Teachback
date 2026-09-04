"""Error taxonomy for LLM providers, built around one question: is trying
the OTHER provider likely to help?

    transient  (429 / quota / 5xx / timeout / network)  -> yes, fail over
    auth/model config (401 / 403 / 404)                 -> yes: the other
                                                           provider has its own
                                                           key and model name
    malformed request from our side (400)               -> no: our payload is
                                                           wrong for everyone;
                                                           failing over would
                                                           just double the noise
    invalid output (bad JSON / schema mismatch)         -> retry once on the
                                                           same provider, then
                                                           fail over

Every message that could contain provider payloads goes through sanitize()
so an API key can never leak into logs, stored metadata or a response.
"""
import os
import re


def sanitize(message: str) -> str:
    """Strip anything that could be a credential from an error message.

    Belt and braces: redact the literal configured keys if present, plus
    Authorization/key-like header patterns and long opaque tokens.
    """
    text = str(message)
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        key = os.environ.get(env, "").strip()
        if key:
            text = text.replace(key, "[redacted]")
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)\S+", r"\1[redacted]", text)
    text = re.sub(r"(?i)((?:api[-_]?key|x-goog-api-key|token)\s*[:=]\s*)\S+", r"\1[redacted]", text)
    # long opaque strings (typical key shapes) are not worth keeping in any
    # log. Requiring a digit keeps long snake_case identifiers (our own
    # rejection reason codes) readable while still catching key-like tokens.
    text = re.sub(r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{30,}\b", "[redacted]", text)
    return text


class LLMError(Exception):
    """Base class. `message` is already sanitized."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        self.message = sanitize(message)
        super().__init__(f"{provider}: {self.message}")


class LLMTransientError(LLMError):
    """Quota, rate limit, 5xx, timeout, network — fail over immediately."""


class LLMAuthConfigError(LLMError):
    """Bad key / unknown model on THIS provider — the other one may still work."""


class LLMRequestError(LLMError):
    """Our request itself is malformed (4xx we caused). Do not fail over."""


class LLMOutputError(LLMError):
    """The provider answered, but not with valid schema-conforming JSON."""


class LLMUnavailable(Exception):
    """Every configured provider failed. The caller must fall back to the
    deterministic v1 behavior — this is expected, not exceptional."""

    def __init__(self, attempts: list[dict]):
        # attempts: [{provider, error_kind, status_code}] — sanitized, minimal
        self.attempts = attempts
        super().__init__("all LLM providers unavailable")


# Body substrings that mark a quota/rate condition even when the HTTP status
# alone would not (providers vary in how they surface exhaustion).
QUOTA_MARKERS = ("resource_exhausted", "rate limit", "rate_limit", "quota",
                 "too many requests", "overloaded")


def classify_http(provider: str, status: int, body: str) -> LLMError:
    """Map an HTTP error response to the taxonomy above."""
    snippet = sanitize(body[:300])
    lowered = body.lower()
    if status == 429 or any(m in lowered for m in QUOTA_MARKERS):
        return LLMTransientError(provider, f"quota/rate limit (HTTP {status})", status)
    if status >= 500:
        return LLMTransientError(provider, f"provider error (HTTP {status})", status)
    if status in (401, 403, 404):
        return LLMAuthConfigError(provider, f"auth/model configuration rejected (HTTP {status})", status)
    return LLMRequestError(provider, f"request rejected (HTTP {status}): {snippet}", status)
