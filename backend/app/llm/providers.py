"""The two API providers behind one tiny interface.

Both are plain HTTPS calls via httpx (already a backend dependency) rather
than vendor SDKs: the whole contract is "system prompt + user prompt in,
raw JSON text out", and a ~40-line client keeps the failover behavior
auditable. Model names come from settings (env), never from code.

API keys are sent only in request headers — never in URLs (which end up in
logs) and never in anything raised or returned; every error path goes
through errors.classify_http / errors.sanitize.
"""
import os

import httpx

from .errors import (LLMTransientError, classify_http, sanitize)
from .settings import ProviderSettings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMProvider:
    """One provider = one way to turn (system, user) prompts into raw text."""

    name = "base"

    def __init__(self, settings: ProviderSettings, timeout_seconds: float):
        self.settings = settings
        self.timeout = timeout_seconds
        # token counts from the most recent successful call, when the
        # provider reports them (observability only; never billed logic)
        self.last_usage: dict | None = None

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        """Return the provider's raw text response (expected to be JSON).

        Raises an LLMError subclass on any failure; never returns None.
        """
        raise NotImplementedError

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except httpx.TimeoutException:
            raise LLMTransientError(self.name, f"timed out after {self.timeout}s")
        except httpx.HTTPError as e:
            raise LLMTransientError(self.name, f"network error: {sanitize(str(e))}")
        if response.status_code != 200:
            raise classify_http(self.name, response.status_code, response.text or "")
        try:
            return response.json()
        except ValueError:
            raise LLMTransientError(self.name, "non-JSON response body", response.status_code)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{GEMINI_BASE_URL}/models/{self.settings.model}:generateContent"
        headers = {"x-goog-api-key": self.settings.api_key,
                   "Content-Type": "application/json"}
        generation_config = {"response_mime_type": "application/json",
                             "temperature": 0.4}
        # Current Gemini flash models are thinking models; at their default
        # thinking level a one-question generation measured ~20s+, past any
        # sensible conversation timeout. Probe wording needs no deep
        # reasoning, so thinking is turned down (measured ~4s). Configurable
        # because the accepted values are model-dependent; set it empty for
        # a model that rejects thinkingConfig.
        thinking = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip()
        if thinking:
            generation_config["thinkingConfig"] = {"thinkingLevel": thinking}
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        data = self._post(url, headers, payload)
        usage = data.get("usageMetadata") or {}
        self.last_usage = {"input_tokens": usage.get("promptTokenCount"),
                           "output_tokens": usage.get("candidatesTokenCount")}
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # a 200 with no usable candidate (safety block, empty output) is
            # transient from our point of view: the other provider may answer
            raise LLMTransientError(self.name, "no text candidate in response")


class GroqProvider(LLMProvider):
    name = "groq"

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{GROQ_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.4,
        }
        data = self._post(url, headers, payload)
        usage = data.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("prompt_tokens"),
                           "output_tokens": usage.get("completion_tokens")}
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMTransientError(self.name, "no message content in response")


PROVIDER_CLASSES = {"gemini": GeminiProvider, "groq": GroqProvider}
