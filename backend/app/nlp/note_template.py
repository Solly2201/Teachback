"""Recommended lecture-note format and the optional external-AI prompt.

TeachBack itself never calls an LLM. The prompt below is only TEXT the
teacher can copy into an external assistant (ChatGPT, Claude, ...) to convert
their own rough notes into the recommended format; the result is pasted back
into TeachBack and parsed by the deterministic pipeline like any other notes.
"""

NOTE_TEMPLATE = """# Lecture Title

## Learning Objectives
- Understand ...
- Explain ...
- Apply ...

## 1. Concept Name

Explanation of the concept, in the words students should be able to echo back.

Example:
code or a worked example

## 2. Another Concept

Explanation.

Example:
...

## Important Connections

- Concept A → relationship → Concept B
- Concept B → relationship → Concept C

## Common Mistakes

- Students may think X, but actually Y.
- A common confusion is ...

## Summary

Two or three sentences of what the lecture boils down to.
"""

AI_PREP_PROMPT = """You are helping a teacher convert their lecture notes into a \
structured format for TeachBack, a tool that checks what students took away \
from a lecture. Convert the notes below into EXACTLY this Markdown structure:

# <Lecture Title>

## Learning Objectives
- <objective 1>
- <objective 2>

## 1. <Concept Name>

<1-3 short sentences explaining the concept, phrased the way a student should \
be able to explain it back — simple words, not textbook jargon.>

Example:
<a short example or code snippet from the notes, if one exists>

## 2. <Next Concept>
...

## Important Connections

- <Concept A> → <relationship verb> → <Concept B>

## Common Mistakes

- Students may think <wrong claim>, but actually <correct clarification>.

## Summary

<2-3 sentences>

STRICT RULES:
- Do NOT invent information. Use only what is in the notes.
- Do NOT add concepts that are not present in the source notes.
- Preserve the teacher's own terminology and all examples and code exactly.
- Only list a connection or a common mistake if the notes support it.
- Keep every explanation simple enough to be said back in one or two sentences.
- If something in the notes is ambiguous, mark it with [UNCLEAR] instead of guessing.

TEACHER'S NOTES:
<paste your notes here>
"""
