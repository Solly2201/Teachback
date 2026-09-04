"""Experimental LLM layer for the generative-probe research extension.

The LLM here is a constrained language generator and nothing more. It never
evaluates student understanding, never assigns mastery and never touches the
HMM: the deterministic NLP analyzer stays the sole source of evidence, and
the pedagogical decision (what to probe, at what difficulty) is made by the
deterministic controller in app.probe before the LLM is ever called. The LLM
only turns that already-made decision into one natural-language question,
grounded exclusively in teacher-approved material.

Everything in this package is inert unless TEACHBACK_LLM_ENABLED is set; the
default v1 application makes zero LLM calls.
"""
