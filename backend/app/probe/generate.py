"""Orchestrator: controller -> retrieval -> constrained LLM -> generated probe.

maybe_generate_probe() is the only function the conversation flow calls. Its
contract is deliberately weak: it returns a replacement question with audit
metadata, or None — and None means "keep the deterministic v1 question",
which is always safe. Every failure mode (feature off, no decision, no
grounding material, provider quota, timeout, malformed output, both
providers down, an outright bug in this pipeline) collapses to None; a
student session can never block on, or crash because of, an LLM.

The returned metadata is what teachers and the evaluation harness see:
which target was probed and why, which teacher material grounded the
wording, and which provider/model/prompt version produced it. It contains
no student text, no prompts, no embeddings and no secrets.
"""
import logging

from ..llm.errors import LLMUnavailable, sanitize
from ..llm.service import LLMService
from ..llm.settings import generative_probes_enabled
from .controller import decide
from .retrieval import retrieve

logger = logging.getLogger(__name__)


def maybe_generate_probe(plan: dict, topic_def: dict, followup: dict,
                         student_answer: str, posterior: list[float] | None = None,
                         service: LLMService | None = None) -> dict | None:
    """Try to generate a teacher-grounded wording for the plan's follow-up."""
    if not generative_probes_enabled():
        return None
    try:
        decision = decide(plan, topic_def, followup, posterior)
        if decision is None:
            return None
        items = retrieve(topic_def, decision, plan, student_answer)
        if not items:
            return None
        service = service or LLMService()
        probe, meta = service.generate_probe(decision, items, student_answer)
    except LLMUnavailable as e:
        logger.info("generated probe unavailable, keeping v1 question: %s",
                    [a.get("error_kind") for a in e.attempts])
        return None
    except Exception as e:  # any bug here must degrade to v1, never crash
        logger.warning("generated-probe pipeline failed, keeping v1 question: %s",
                       sanitize(str(e)))
        return None

    return {
        "question": probe.question,
        "meta": {
            **meta,  # provider_used, model_used, fallback_used, prompt_version, latency_ms
            "action": decision["action"],
            "target_type": decision["target_type"],
            "target_id": decision["target_id"],
            "target_name": decision["target_name"],
            "difficulty": decision["difficulty"],
            "reason": decision["reason"],
            "plan_agrees_with_ranking": decision["plan_agrees_with_ranking"],
            "grounding_ids": probe.grounding_ids,
            "rationale": probe.rationale,
        },
    }
