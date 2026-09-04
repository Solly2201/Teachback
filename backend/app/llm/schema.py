"""Structured output contract for the probe generator, and the hard
server-side authorization check behind it.

Two layers, deliberately separate:

  * GeneratedProbe (Pydantic) validates SYNTAX: required fields, types,
    bounds. Unknown fields are FORBIDDEN — an output smuggling extra fields
    like "state": "understanding" is rejected outright, and nothing anywhere
    reads fields the schema does not define, so no LLM output can ever touch
    learner state, evidence, recommendations or configuration.

  * validate_against_decision() validates AUTHORIZATION: an explicit backend
    comparison of the untrusted output against the deterministic controller's
    decision. The direction of authority is strictly controller -> LLM; a
    response that tries to re-target, change the action or difficulty, cite
    unapproved material, ask more than one question, or leak the system
    prompt is rejected with a stable sanitized reason code. Rejection means
    LLMOutputError -> one retry -> fallback provider -> deterministic v1
    question; a rejected output is never shown, never stored as a generated
    probe, and never influences NLP evidence or the HMM.

The rejection is enforcement, not prompting: it holds even against a model
that ignores every instruction it was given.
"""
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import LLMOutputError
from .prompts import SYSTEM_PROMPT_MARKERS

# The server-defined pedagogical vocabulary (mirrors probe/controller.py's
# ACTION_FOR_KIND; test_prompt_injection pins the two in sync). The LLM can
# never introduce an action of its own.
ALLOWED_ACTIONS = frozenset({
    "ASK_MAIN_QUESTION", "ASK_EASIER_PROBE", "ASK_PROBE", "ASK_DEEPEN",
    "PROBE_RELATIONSHIP", "PROBE_MISCONCEPTION",
})


class GeneratedProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=40)
    target_type: str = Field(pattern="^(concept|relationship|misconception)$")
    target_id: int
    difficulty: str = Field(pattern="^(easy|standard)$")
    # the single diagnostic question shown to the student
    question: str = Field(min_length=10, max_length=600)
    # which supplied teacher-material items the wording drew on
    grounding_ids: list[str] = Field(default_factory=list, max_length=20)
    # one plain-language sentence for the teacher; never chain-of-thought
    rationale: str = Field(default="", max_length=400)


def parse_probe(provider: str, raw_text: str) -> GeneratedProbe:
    """Parse and schema-validate a provider's raw text response."""
    text = raw_text.strip()
    # tolerate a fenced ```json block — some models wrap even when asked not to
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMOutputError(provider, f"llm_output_invalid_json: {e.msg}")
    if not isinstance(data, dict):
        raise LLMOutputError(provider, "llm_output_not_an_object")
    try:
        return GeneratedProbe(**data)
    except ValidationError as e:
        fields = ", ".join(sorted({str(err["loc"][0]) for err in e.errors() if err.get("loc")}))
        raise LLMOutputError(provider, f"llm_output_schema_violation ({fields})")


def validate_against_decision(provider: str, probe: GeneratedProbe,
                              decision: dict, allowed_grounding: set[str]) -> GeneratedProbe:
    """Hard backend authorization of an untrusted LLM output against the
    immutable controller decision. Any mismatch rejects the whole output."""

    def reject(code: str):
        raise LLMOutputError(provider, code)

    if probe.action not in ALLOWED_ACTIONS:
        reject("llm_output_unsupported_action")
    if probe.action != decision["action"]:
        reject("llm_output_action_mismatch")
    if probe.target_type != decision["target_type"]:
        reject("llm_output_target_type_mismatch")
    if probe.target_id != decision["target_id"]:
        reject("llm_output_target_mismatch")
    # exact match: with the easy < standard scale this also guarantees the
    # difficulty never exceeds what the controller approved
    if probe.difficulty != decision["difficulty"]:
        reject("llm_output_difficulty_mismatch")
    if not probe.grounding_ids:
        reject("llm_output_missing_grounding")
    if any(g not in allowed_grounding for g in probe.grounding_ids):
        reject("llm_output_grounding_violation")
    # exactly one diagnostic question — no bundles, no statements
    if probe.question.count("?") != 1:
        reject("llm_output_not_one_question")
    # a tricked model quoting its own instructions must never reach a student
    lowered = (probe.question + " " + probe.rationale).lower()
    if any(marker in lowered for marker in SYSTEM_PROMPT_MARKERS):
        reject("llm_output_prompt_leak")
    return probe
