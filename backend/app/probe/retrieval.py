"""Targeted retrieval over teacher-approved material — the "RAG" in this
extension, deliberately small.

The retrieval source is the reviewed topic structure the teacher already
approved (concepts, facts, examples, authored questions, relationships,
misconceptions) — never the raw lecture text, and never "everything about
the topic". The controller's decision names one target; this module returns
only the material items relevant to that target, each with a stable
provenance id, so the stored metadata can state exactly which teacher
material grounded a generated question.

When a concept carries more facts/examples than the context budget, the
existing MiniLM embedder ranks them against the student's last answer and
the closest ones are kept — reusing the embedding machinery the analyzer
already loads, with no new vector store. If the embedder fails for any
reason the first items are kept instead; retrieval must never take a
student session down.
"""
import numpy as np

MAX_FACT_ITEMS = 6


def _rank_by_answer(texts: list[str], student_answer: str, keep: int) -> list[int]:
    """Indices of the `keep` texts most similar to the student's answer."""
    if len(texts) <= keep:
        return list(range(len(texts)))
    try:
        from ..nlp.embedder import embed
        vectors = embed(texts + [student_answer])
        sims = vectors[:-1] @ vectors[-1]
        order = np.argsort(-sims)[:keep]
        return sorted(int(i) for i in order)
    except Exception:
        return list(range(keep))


def _concept_items(cdef: dict, student_answer: str) -> list[dict]:
    cid = cdef.get("id")
    items = []
    if cdef.get("description"):
        items.append({"id": f"concept:{cid}", "kind": "concept_explanation",
                      "text": f"{cdef['name']}: {cdef['description']}"})
    facts = [f for f in (cdef.get("facts") or []) if f]
    examples = [e for e in (cdef.get("examples") or []) if e]
    keep_facts = _rank_by_answer(facts, student_answer, MAX_FACT_ITEMS)
    items += [{"id": f"concept_fact:{cid}:{i}", "kind": "teacher_fact", "text": facts[i]}
              for i in keep_facts]
    items += [{"id": f"concept_example:{cid}:{i}", "kind": "teacher_example", "text": examples[i]}
              for i in range(min(len(examples), 3))]
    for key, label in (("main_question", "main"), ("easier_question", "easier"),
                       ("probe_question", "probe"), ("application_question", "application")):
        if cdef.get(key):
            items.append({"id": f"concept_question:{cid}:{label}",
                          "kind": "teacher_authored_question", "text": cdef[key]})
    return items


def _relationship_items(rdef: dict, topic_def: dict) -> list[dict]:
    rid = rdef.get("id")
    items = []
    text = rdef.get("description") or (
        f"{rdef['source']} {rdef.get('label', 'relates to')} {rdef['target']}")
    items.append({"id": f"relationship:{rid}", "kind": "relationship_explanation", "text": text})
    if rdef.get("contradiction"):
        items.append({"id": f"relationship_contradiction:{rid}", "kind": "known_wrong_claim",
                      "text": "A wrong version students state: " + rdef["contradiction"]})
    if rdef.get("probe_question"):
        items.append({"id": f"relationship_question:{rid}",
                      "kind": "teacher_authored_question", "text": rdef["probe_question"]})
    # the two endpoint concepts, so the question can use the teacher's wording
    for c in topic_def.get("concepts", []):
        if c["name"] in (rdef.get("source"), rdef.get("target")) and c.get("description"):
            items.append({"id": f"concept:{c.get('id')}", "kind": "concept_explanation",
                          "text": f"{c['name']}: {c['description']}"})
    return items


def _misconception_items(mdef: dict, plan: dict, topic_def: dict) -> list[dict]:
    mid = mdef.get("id")
    items = []
    if mdef.get("description"):
        items.append({"id": f"misconception:{mid}", "kind": "known_wrong_claim",
                      "text": "The wrong claim to check for: " + mdef["description"]})
    if mdef.get("clarification"):
        items.append({"id": f"misconception_clarification:{mid}", "kind": "teacher_clarification",
                      "text": mdef["clarification"]})
    if mdef.get("probe_question"):
        items.append({"id": f"misconception_question:{mid}",
                      "kind": "teacher_authored_question", "text": mdef["probe_question"]})
    # the concept currently under discussion anchors the probe in context
    concepts = plan.get("concepts") or []
    current = plan.get("current", 0)
    if 0 <= current < len(concepts):
        cdef = next((c for c in topic_def.get("concepts", [])
                     if c.get("id") == concepts[current].get("id")
                     or c["name"] == concepts[current]["name"]), None)
        if cdef and cdef.get("description"):
            items.append({"id": f"concept:{cdef.get('id')}", "kind": "concept_explanation",
                          "text": f"{cdef['name']}: {cdef['description']}"})
    return items


def retrieve(topic_def: dict, decision: dict, plan: dict,
             student_answer: str = "") -> list[dict]:
    """Teacher-approved context for one controller decision, with provenance.

    Returns [] when no usable teacher material exists for the target, which
    the orchestrator treats as "do not generate" — a probe with nothing to
    ground it would have to invent content, and inventing is forbidden.
    """
    ttype, tid = decision["target_type"], decision["target_id"]
    if ttype == "concept":
        cdef = next((c for c in topic_def.get("concepts", []) if c.get("id") == tid), None)
        items = _concept_items(cdef, student_answer) if cdef else []
    elif ttype == "relationship":
        rdef = next((r for r in topic_def.get("relationships", []) if r.get("id") == tid), None)
        items = _relationship_items(rdef, topic_def) if rdef else []
    elif ttype == "misconception":
        mdef = next((m for m in topic_def.get("misconceptions", []) if m.get("id") == tid), None)
        items = _misconception_items(mdef, plan, topic_def) if mdef else []
    else:
        items = []
    # an id list alone (no substantive text) cannot ground a question
    return items if any(i["kind"] != "teacher_authored_question" for i in items) else []
