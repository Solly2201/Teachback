"""Experimental generative-probe pipeline (research extension, off by default).

    controller.py  deterministic uncertainty-aware probe selection
    retrieval.py   targeted retrieval over teacher-approved material
    generate.py    orchestrator: controller -> retrieval -> constrained LLM

Everything pedagogically meaningful is decided here, deterministically and
inspectably, BEFORE the LLM sees anything. The LLM (app.llm) only verbalizes
the decision. On any failure the caller keeps the v1 deterministic question —
this pipeline can only ever swap the wording of a question, never the flow,
the evidence, or the learner-state estimation.
"""
