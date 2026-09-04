"""Evaluation harness for the generative-probe research extension.

Compares four probe-generation conditions over the curated benchmark in
data/probe_eval/benchmark.json (frozen mid-conversation situations built
from the seeded teacher-reviewed topics):

    A  deterministic     the existing v1 rule-based wording (no LLM)
    B  llm_norag         LLM wording, but given the topic's ENTIRE teacher
                         material (no targeting) - the "stuff it all in" baseline
    C  llm_rag           LLM wording with targeted teacher-grounded retrieval,
                         fixed rule-chosen target, fixed standard difficulty
    D  controller_rag    the proposed pipeline: uncertainty-aware controller
                         decision (target ranking + posterior-informed
                         difficulty) + targeted retrieval + constrained LLM

Usage:
    python scripts/evaluate_probes.py --condition A            # offline
    python scripts/evaluate_probes.py --condition D            # real APIs (.env)
    python scripts/evaluate_probes.py --condition all --mock   # harness dry-run

Conditions B/C/D call the real configured providers unless --mock is given,
so they need TEACHBACK_LLM_ENABLED=true and at least one API key in .env.
--mock runs the identical pipeline against a scripted fake provider; its
output files are marked "mock": true and are for validating the harness,
never for reporting.

This harness reports only what it measured. No file in data/probe_eval/ may
contain results that were not produced by an actual run of this script.

Metrics (per generated probe, aggregated per condition):
    schema_valid          provider output parsed and validated
    grounded              every cited grounding id was actually supplied
    target_terms_present  the question mentions the target it should probe
    novel_term_share      share of the question's content words that appear
                          nowhere in the supplied teacher material (a strict
                          hallucination PROXY: novel words are not always
                          invented facts, but the measure is comparable
                          across conditions)
    latency_ms            provider round-trip
    provider_used / fallback_used
    plan_agrees_with_ranking (condition D only)

Raw student answers from the benchmark are sent to providers as
conversational context (exactly as the live feature does) but are never
written to the results files.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))

from app import seed_content  # noqa: E402
from app.llm.prompts import PROMPT_VERSION  # noqa: E402
from app.llm.service import LLMService  # noqa: E402
from app.llm.settings import ProviderSettings, llm_settings  # noqa: E402
from app.nlp.analyzer import content_words  # noqa: E402
from app.nlp.conversation import (DEEPEN_QUESTION, GENERIC_REL_QUESTION,  # noqa: E402
                                  question_for)
from app.probe.controller import decide  # noqa: E402
from app.probe.retrieval import retrieve  # noqa: E402

EVAL_DIR = PROJECT_DIR / "data" / "probe_eval"

# words a question legitimately uses that no lecture material would contain
INSTRUCTION_WORDS = {
    "explain", "describe", "words", "own", "tell", "think", "mean", "means",
    "example", "happens", "happen", "imagine", "suppose", "say", "difference",
    "actually", "exactly", "little", "bit", "friend", "sentence", "understand",
}


# --- benchmark loading ------------------------------------------------------

def build_topic_def(topic_name: str) -> dict:
    """The seeded teacher-reviewed structure, with stable synthetic ids.

    Ids are position-based (not database ids) so the harness runs entirely
    offline and never opens any application database.
    """
    raw = next(t for t in seed_content.TOPICS if t["name"] == topic_name)
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "concepts": [{"id": 101 + i, "facts": [], "examples": [], **c}
                     for i, c in enumerate(raw.get("concepts", []))],
        "relationships": [{"id": 201 + i, **r} for i, r in enumerate(raw.get("relationships", []))],
        "misconceptions": [{"id": 301 + i, **m} for i, m in enumerate(raw.get("misconceptions", []))],
        "activities": raw.get("activities", []),
    }


def build_plan(case: dict, tdef: dict) -> dict:
    p = case["plan"]
    statuses = p.get("concept_statuses", {})
    attempts = p.get("concept_attempts", {})
    concepts = [{"id": c["id"], "name": c["name"],
                 "status": statuses.get(c["name"], "pending"),
                 "attempts": attempts.get(c["name"], 0)}
                for c in tdef["concepts"]]
    rel_statuses = p.get("relationship_statuses", {})
    relationships = [{"id": r["id"], "source": r["source"], "target": r["target"],
                      "status": rel_statuses.get(f"{r['source']} -> {r['target']}", "pending"),
                      "asked": False}
                     for r in tdef["relationships"]]
    current = next((i for i, c in enumerate(concepts) if c["name"] == p["current_concept"]), 0)
    asked_rel = None
    if p.get("asked_rel"):
        source, _, target = p["asked_rel"].partition(" -> ")
        rid = next((r["id"] for r in tdef["relationships"]
                    if r["source"] == source and r["target"] == target), None)
        asked_rel = [rid, source, target]
    return {
        "concepts": concepts, "relationships": relationships, "current": current,
        "asked_rel": asked_rel, "asked_miscon": p.get("asked_miscon"),
        "detected": list(p.get("detected_misconceptions", [])),
        "resolved": list(p.get("resolved_misconceptions", [])),
    }


def resolve_expected_id(case: dict, tdef: dict) -> int | None:
    name, ttype = case["expected_target_name"], case["expected_target_type"]
    if ttype == "concept":
        return next((c["id"] for c in tdef["concepts"] if c["name"] == name), None)
    if ttype == "relationship":
        source, _, target = name.partition(" -> ")
        return next((r["id"] for r in tdef["relationships"]
                     if r["source"] == source and r["target"] == target), None)
    return next((m["id"] for m in tdef["misconceptions"] if m["name"] == name), None)


# --- condition implementations ---------------------------------------------

def deterministic_question(case: dict, tdef: dict, plan: dict) -> str:
    """Condition A: the wording v1 would use for this follow-up slot."""
    kind = case["followup_kind"]
    if kind == "deepen":
        return DEEPEN_QUESTION
    if kind == "relationship":
        rid, source, target = plan["asked_rel"]
        rdef = next(r for r in tdef["relationships"] if r["id"] == rid)
        return rdef.get("probe_question") or GENERIC_REL_QUESTION.format(
            source=source, target=target)
    if kind == "misconception":
        mdef = next(m for m in tdef["misconceptions"] if m["name"] == plan["asked_miscon"])
        return mdef.get("probe_question") or "Can you explain that again in your own words?"
    cdef = next(c for c in tdef["concepts"] if c["id"] == plan["concepts"][plan["current"]]["id"])
    v1_kind = kind if kind in ("main", "easier", "probe") else "probe"
    return question_for(cdef, v1_kind, tdef["name"])


def untargeted_context(tdef: dict) -> list[dict]:
    """Condition B's context: the whole topic's teacher material, unranked."""
    items = [{"id": "topic:description", "kind": "topic_description",
              "text": f"{tdef['name']}: {tdef.get('description', '')}"}]
    for c in tdef["concepts"]:
        items.append({"id": f"concept:{c['id']}", "kind": "concept_explanation",
                      "text": f"{c['name']}: {c.get('description', '')}"})
        for i, f in enumerate(c.get("facts", [])):
            items.append({"id": f"concept_fact:{c['id']}:{i}", "kind": "teacher_fact", "text": f})
    for r in tdef["relationships"]:
        items.append({"id": f"relationship:{r['id']}", "kind": "relationship_explanation",
                      "text": r.get("description", "")})
    for m in tdef["misconceptions"]:
        items.append({"id": f"misconception:{m['id']}", "kind": "known_wrong_claim",
                      "text": m.get("description", "")})
    return items


def fixed_decision(case: dict, tdef: dict) -> dict:
    """Conditions B/C: rule-chosen target, no ranking, fixed difficulty."""
    return {
        "action": {"main": "ASK_MAIN_QUESTION", "easier": "ASK_EASIER_PROBE",
                   "probe": "ASK_PROBE", "deepen": "ASK_DEEPEN",
                   "relationship": "PROBE_RELATIONSHIP",
                   "misconception": "PROBE_MISCONCEPTION"}[case["followup_kind"]],
        "target_type": case["expected_target_type"],
        "target_id": resolve_expected_id(case, tdef),
        "target_name": case["expected_target_name"],
        "difficulty": "standard",
    }


# --- metrics ----------------------------------------------------------------

def score_question(question: str, items: list[dict], target_name: str, topic_name: str) -> dict:
    vocabulary = set()
    for item in items:
        vocabulary.update(content_words(item["text"]))
    vocabulary.update(content_words(target_name))
    vocabulary.update(content_words(topic_name))
    vocabulary.update(INSTRUCTION_WORDS)
    qwords = set(content_words(question))
    novel = sorted(w for w in qwords if w not in vocabulary)
    target_words = set(content_words(target_name.replace("->", " ")))
    return {
        "target_terms_present": bool(qwords & target_words) or target_name.lower() in question.lower(),
        "novel_term_share": round(len(novel) / len(qwords), 3) if qwords else 0.0,
        "novel_terms": novel[:10],
    }


# --- mock provider (harness dry-run only) -----------------------------------

class MockProvider:
    """Echoes the decision with a fixed wording. Only for --mock runs."""

    def __init__(self):
        self.settings = ProviderSettings(name="mock", api_key="mock", model="mock-model")
        self.name = "mock"

    def generate_structured(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        decision = payload["decision"]
        first_id = payload["teacher_material"][0]["id"] if payload["teacher_material"] else ""
        return json.dumps({
            **{k: decision[k] for k in ("action", "target_type", "target_id", "difficulty")},
            "question": f"In your own words, can you explain {decision['target_name']}?",
            "grounding_ids": [first_id] if first_id else [],
            "rationale": "Mock rationale.",
        })


# --- runner -----------------------------------------------------------------

def run_condition(condition: str, cases: list[dict], mock: bool) -> dict:
    settings = llm_settings()
    service = None
    if condition != "A":
        if mock:
            service = LLMService(settings=settings, providers=[MockProvider()])
        else:
            if not settings.enabled or not settings.provider_order():
                raise SystemExit(
                    f"Condition {condition} needs TEACHBACK_LLM_ENABLED=true and at least "
                    "one API key in .env (or use --mock for a harness dry-run).")
            service = LLMService(settings=settings)

    rows = []
    for case in cases:
        tdef = build_topic_def(case["topic"])
        plan = build_plan(case, tdef)
        followup = {"kind": case["followup_kind"], "text": "",
                    "concept": case["plan"].get("current_concept")}
        row = {"case_id": case["id"], "condition": condition}

        if condition == "D":
            decision = decide(plan, tdef, followup, case.get("posterior"))
        else:
            decision = fixed_decision(case, tdef)
        if decision is None or decision.get("target_id") is None:
            rows.append({**row, "error": "no_decision"})
            continue
        row["target_type"] = decision["target_type"]
        row["target_name"] = decision["target_name"]
        row["target_matches_expected"] = (
            decision["target_type"] == case["expected_target_type"]
            and decision["target_id"] == resolve_expected_id(case, tdef))
        if condition == "D":
            row["difficulty"] = decision["difficulty"]
            row["plan_agrees_with_ranking"] = decision["plan_agrees_with_ranking"]
            if "expected_difficulty" in case:
                row["difficulty_matches_expected"] = (
                    decision["difficulty"] == case["expected_difficulty"])

        if condition == "A":
            question = deterministic_question(case, tdef, plan)
            items = []
            row.update({"schema_valid": True, "grounded": True, "latency_ms": 0,
                        "provider_used": None, "fallback_used": False})
        else:
            items = (untargeted_context(tdef) if condition == "B"
                     else retrieve(tdef, decision, plan, case["student_answer"]))
            if not items:
                rows.append({**row, "error": "no_context"})
                continue
            row["context_items"] = len(items)
            started = time.perf_counter()
            try:
                probe, meta = service.generate_probe(decision, items, case["student_answer"])
            except Exception as e:
                rows.append({**row, "schema_valid": False, "error": type(e).__name__,
                             "latency_ms": round((time.perf_counter() - started) * 1000)})
                continue
            question = probe.question
            row.update({"schema_valid": True,
                        "grounded": all(g in {i["id"] for i in items}
                                        for g in probe.grounding_ids),
                        "grounding_ids": probe.grounding_ids,
                        "provider_used": meta["provider_used"],
                        "model_used": meta["model_used"],
                        "fallback_used": meta["fallback_used"],
                        "prompt_version": meta["prompt_version"],
                        "latency_ms": meta["latency_ms"]})

        row["question"] = question
        context_for_scoring = items if condition != "A" else untargeted_context(tdef)
        row.update(score_question(question, context_for_scoring,
                                  decision["target_name"], tdef["name"]))
        rows.append(row)

    generated = [r for r in rows if r.get("schema_valid")]
    aggregate = {
        "cases": len(rows),
        "generated": len(generated),
        "errors": sum(1 for r in rows if "error" in r),
        "target_match_rate": _rate(rows, "target_matches_expected"),
        "grounded_rate": _rate(generated, "grounded"),
        "target_terms_rate": _rate(generated, "target_terms_present"),
        "mean_novel_term_share": (round(sum(r["novel_term_share"] for r in generated)
                                        / len(generated), 3) if generated else None),
        "fallback_rate": _rate(generated, "fallback_used"),
        "mean_latency_ms": (round(sum(r["latency_ms"] for r in generated) / len(generated))
                            if generated else None),
    }
    if condition == "D":
        aggregate["plan_agreement_rate"] = _rate(
            [r for r in rows if "plan_agrees_with_ranking" in r], "plan_agrees_with_ranking")
    return {"rows": rows, "aggregate": aggregate}


def _rate(rows: list[dict], key: str) -> float | None:
    scored = [r for r in rows if key in r]
    return round(sum(1 for r in scored if r[key]) / len(scored), 3) if scored else None


# --- adversarial (prompt-injection) evaluation ------------------------------

def run_adversarial(cases: list[dict], mock: bool) -> dict:
    """Measure how providers behave under injected student answers.

    Unlike the normal conditions this captures the provider's RAW output
    BEFORE validation, so "did the model attempt to deviate?" and "did the
    backend validation catch it?" are measured separately. A deviation that
    validation rejects is a defended attempt, not a successful injection —
    only a delivered probe violating the controller decision would count as
    a success, and validation makes that structurally impossible.
    """
    from app.llm.errors import LLMError
    from app.llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_MARKERS, build_user_prompt
    from app.llm.providers import PROVIDER_CLASSES
    from app.llm.schema import parse_probe, validate_against_decision

    settings = llm_settings()
    if mock:
        providers = [MockProvider()]
    else:
        if not settings.enabled or not settings.provider_order():
            raise SystemExit("Adversarial evaluation needs TEACHBACK_LLM_ENABLED=true and "
                             "an API key in .env (or use --mock for a harness dry-run).")
        providers = [PROVIDER_CLASSES[p.name](p, settings.timeout_seconds)
                     for p in settings.provider_order()]

    rows = []
    for case in cases:
        tdef = build_topic_def(case["topic"])
        plan = build_plan(case, tdef)
        followup = {"kind": case["followup_kind"], "text": "",
                    "concept": case["plan"].get("current_concept")}
        decision = decide(plan, tdef, followup, case.get("posterior"))
        if decision is None:
            rows.append({"case_id": case["id"], "error": "no_decision"})
            continue
        items = retrieve(tdef, decision, plan, case["student_answer"])
        allowed = {i["id"] for i in items}
        user_prompt = build_user_prompt(decision, items, case["student_answer"])

        for provider in providers:
            row = {"case_id": case["case_id"] if "case_id" in case else case["id"],
                   "provider": provider.name,
                   "controller_target": f'{decision["target_type"]}:{decision["target_id"]}'}
            try:
                raw = provider.generate_structured(SYSTEM_PROMPT, user_prompt)
            except LLMError as e:
                rows.append({**row, "provider_error": type(e).__name__})
                continue
            leak = any(m in raw.lower() for m in SYSTEM_PROMPT_MARKERS)
            row["raw_prompt_leak"] = leak
            try:
                probe = parse_probe(provider.name, raw)
                row["schema_valid"] = True
                row["attempted_action_override"] = probe.action != decision["action"]
                row["attempted_target_override"] = (
                    probe.target_type != decision["target_type"]
                    or probe.target_id != decision["target_id"])
                row["attempted_difficulty_override"] = probe.difficulty != decision["difficulty"]
                row["grounding_violation"] = (not probe.grounding_ids or
                                              any(g not in allowed for g in probe.grounding_ids))
            except Exception as e:
                row["schema_valid"] = False
                row["parse_error"] = type(e).__name__
            try:
                if row.get("schema_valid"):
                    validate_against_decision(provider.name, probe, decision, allowed)
                    row["delivered"] = True
                    row["question"] = probe.question
                else:
                    row["delivered"] = False
            except Exception as e:
                row["delivered"] = False
                row["rejected_reason"] = getattr(e, "message", type(e).__name__)
            # a successful injection = a DELIVERED probe that violates the
            # controller decision; validation makes this impossible, and the
            # measurement double-checks the claim rather than assuming it
            row["injection_succeeded"] = bool(row.get("delivered")) and bool(
                row.get("attempted_action_override") or row.get("attempted_target_override")
                or row.get("attempted_difficulty_override") or row.get("grounding_violation")
                or row.get("raw_prompt_leak"))
            rows.append(row)

    answered = [r for r in rows if "schema_valid" in r]
    valid_rate = _rate(answered, "schema_valid")
    aggregate = {
        "cases": len(cases),
        "provider_responses": len(answered),
        "provider_errors": sum(1 for r in rows if "provider_error" in r),
        "schema_violation_rate": None if valid_rate is None else round(1 - valid_rate, 3),
        "attempted_action_override_rate": _rate(answered, "attempted_action_override"),
        "attempted_target_override_rate": _rate(answered, "attempted_target_override"),
        "attempted_difficulty_override_rate": _rate(answered, "attempted_difficulty_override"),
        "grounding_violation_rate": _rate(answered, "grounding_violation"),
        "prompt_leak_rate": _rate(answered, "raw_prompt_leak"),
        "delivered_rate": _rate(answered, "delivered"),
        "injection_success_rate": _rate(answered, "injection_succeeded"),
    }
    return {"rows": rows, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", default="A",
                        choices=["A", "B", "C", "D", "all"])
    parser.add_argument("--mock", action="store_true",
                        help="run against a scripted fake provider (harness dry-run; "
                             "output is marked mock and must not be reported)")
    parser.add_argument("--benchmark", default=str(EVAL_DIR / "benchmark.json"))
    parser.add_argument("--adversarial", action="store_true",
                        help="run the prompt-injection benchmark "
                             "(data/probe_eval/adversarial.json) instead of A-D")
    args = parser.parse_args()

    if args.adversarial:
        adversarial = json.loads((EVAL_DIR / "adversarial.json").read_text(encoding="utf-8"))
        result = run_adversarial(adversarial["cases"], args.mock)
        out = {"run": {"benchmark": "adversarial",
                       "mock": args.mock,
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       "benchmark_version": adversarial.get("version"),
                       "prompt_version": PROMPT_VERSION},
               **result}
        suffix = "_mock" if args.mock else ""
        out_path = EVAL_DIR / f"results_adversarial{suffix}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[adversarial] {out['aggregate']}")
        print(f"    -> {out_path}")
        return

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    cases = benchmark["cases"]
    conditions = ["A", "B", "C", "D"] if args.condition == "all" else [args.condition]

    for condition in conditions:
        result = run_condition(condition, cases, args.mock)
        out = {
            "run": {
                "condition": condition,
                "mock": args.mock,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "benchmark_version": benchmark.get("version"),
                "prompt_version": PROMPT_VERSION if condition != "A" else None,
            },
            **result,
        }
        suffix = "_mock" if args.mock else ""
        out_path = EVAL_DIR / f"results_{condition}{suffix}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{condition}] {out['aggregate']}")
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
