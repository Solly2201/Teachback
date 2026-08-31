# TeachBack

**Teach it back. We'll listen.**

TeachBack is a browser-based learning prototype built around one question:

> **"Did the student actually understand what was taught in today's lecture — and what did they
> understand, misunderstand, or struggle with?"**

Instead of a quiz or a self-rating, students **explain the lecture back in their own words**.
Simple language, analogies, examples and short answers all count — a student who can correctly
explain what was taught has demonstrated conceptual understanding, even if they cannot answer
advanced questions that were never taught. TeachBack is *not* an exam system.

Deterministic NLP (sentence embeddings — **no generative LLM anywhere**) evaluates the evidence
in each explanation, a **Hidden Markov Model** tracks how the student's learning condition
evolves across sessions, and transparent rules recommend the next activity — which the student
can actually open and complete.

---

## 1. Problem statement

Self-assessment ("How well did you understand this? 8/10") is unreliable: students routinely
over- or under-estimate themselves, and a single rating says nothing about *what* was
misunderstood. Static labels ("weak student", "strong student") are both inaccurate and unfair —
understanding fluctuates from lecture to lecture. And asking teachers to hand-build a full
question bank after every single lecture doesn't scale.

## 2. Core idea

1. After a lecture, the teacher uploads/pastes the lecture material; NLP drafts the important
   concepts and the teacher quickly reviews them (automatic first draft + quick review — not a
   black box, and not manual data entry).
2. Students explain what they understood, in their own words, in a short guided conversation.
3. NLP evaluates the **evidence of conceptual understanding** in what they said.
4. Students add their own takeaway summary, confidence, perceived difficulty, and quick lecture
   feedback (pace, "more examples", …).
5. An HMM turns the sequence of session observations into an estimated **current learning
   condition** — a snapshot, never a permanent label.
6. A rule-based recommender picks the next activity, and the student can open and complete it.

## 3. The complete system

```text
FACULTY
  │  lecture material (+ optional learning objectives)
  ▼
NLP preparation ── candidate concepts / relationships / objectives
  ▼
Faculty review  (rename / remove / add — nothing is final until "Start TeachBack")
  ▼
TeachBack session ── student explains what they understood, in their own words
  ▼
NLP evidence ── concepts / relationships / misconceptions demonstrated
  +  student summary, confidence, difficulty, effort, lecture feedback
  ▼
HMM ── current learning condition (Viterbi over the whole session history)
  ▼
Adaptive recommendation ── review / practice / application / extension
  ▼
Actual activity ── student completes it, completion recorded
  ▼
Progress
```

### The two model roles — kept deliberately separate

**NLP answers:** *"What evidence of conceptual understanding is present in what the student
said?"* — semantic similarity, concept evidence in plain language, relationships,
misconceptions, and summary evidence.

**HMM answers:** *"How is this student's learning condition evolving over time?"* — it takes the
per-session observation vectors and infers the current state from the whole trajectory.

NLP does **not** classify the student, and the HMM is **not** a synonym for the NLP score:
NLP extracts and evaluates evidence, while the HMM models the student's state over time.

## 4. Faculty workflow (Quick Lecture mode — the normal path)

1. **Create Lecture TeachBack** (Lecture TeachBacks page).
2. Upload lecture material (`.txt`/`.md`/`.pdf`) or paste the notes.
3. Optionally enter learning objectives — these take priority over automatic extraction.
4. NLP extracts candidate concepts (each with a "possible explanation" quoted from the
   lecture), potential relationships, and draft objectives; already-authored misconceptions
   that match the lecture may be *suggested* for the teacher to accept or reject.
5. The teacher quickly reviews: rename, remove, add, edit.
6. **Start TeachBack** — the lecture is published as a topic students can explain back.

The teacher is *not* expected to hand-author every concept, question, relationship,
misconception and activity after every lecture. **Topic Management** remains available as the
optional advanced path for topics that deserve hand-tuned questions, misconception probes,
contradiction examples and custom activities (the seeded Backpropagation topic is a full
example). Both paths feed the exact same underlying topic structure — there is one knowledge
system, not two.

### Lecture lifecycle: publish, update, delete/archive

One lecture owns one published Topic. Publishing rebuilds that Topic from the reviewed draft;
if another lecture were ever pointing at the same Topic, publishing gives this lecture a fresh
Topic instead of silently overwriting the other lecture's material.

**Deleting a lecture must never delete what students did.** `DELETE /api/lectures/{id}` picks
its behaviour from the data, not from a flag:

| student history | behaviour | what happens |
|---|---|---|
| none | **deleted** | the lecture, its Topic and the Topic's owned rows (concepts, relationships, misconceptions, activities, quiz) are removed |
| any sessions / observations / quiz attempts / activity completions | **archived** | `archived_at` is set on the lecture and its Topic |

An archived lecture disappears from the active lecture list, the active topic list and every
subject-dashboard aggregate, and no new TeachBack can be started on it — while every session,
takeaway summary, observation, quiz attempt and completed activity stays in place with valid
foreign keys, still readable from the student's Progress page. `GET /api/lectures/{id}/delete-preview`
tells the UI which of the two will happen so the confirmation dialog can say it plainly, an
**Archived** section lists archived lectures, and `POST /api/lectures/{id}/restore` brings one
back. Tests: `tests/test_lecture_lifecycle.py`.

## 5. Student workflow

1. Attend the lecture.
2. Open TeachBack and pick the lecture/topic.
3. Explain what they understood — own words, short answers welcome.
4. Answer short conversational follow-ups (only where the evidence needs them).
5. Write an optional **"Your takeaway"** summary of the lecture.
6. Report confidence and perceived difficulty, plus how focused they were.
7. Give quick lecture feedback: pace (too slow … too fast) and request chips
   ("More examples", "More practice", "More visuals", …) with an optional free-text note.
8. See their current learning condition, what they demonstrated, what needs clarification,
   and *why*.
9. Get an evidence-based recommended activity — and complete it with one click.
10. Review everything on the Progress page.

The experience is designed to feel like *"tell us what you took away"*, not *"take another
exam"*.

## 6. Learning states

| # | State | Typical evidence |
|---|-------|------------------|
| 0 | Not Trying | minimal responses, low attention |
| 1 | Unclear | short, inaccurate explanations, frequent misconceptions |
| 2 | Struggling but Trying | high effort and attention, but gaps and errors |
| 3 | Understanding | mostly accurate, decent coverage |
| 4 | Confident | accurate, complete, high self-confidence |

These are **snapshots of the current learning condition, not mastery levels** — and they are
per-topic and per-time. A student can be Understanding in Python, Unclear in Statistics, and
move `Understanding → Unclear → Struggling but Trying` across weeks; modelling those temporal
transitions is exactly why an HMM is used instead of a per-session classifier.

## 7. NLP component (`backend/app/nlp/`)

The system is deliberately **bounded**: each topic defines required concepts, known
misconceptions (wrong claim + correct clarification), relationships, and a reference
explanation — authored directly by the teacher or drafted from the lecture and reviewed.
Student responses are evaluated **against that definition**.

- Sentences are embedded with the pretrained **sentence-transformers** model
  `all-MiniLM-L6-v2` (a non-generative embedding model).
- **Concept coverage** — each concept is represented by *several* reference texts: its meaning
  **plus each teacher-reviewed "important fact" from the lecture** (e.g. *"Indexes start at
  0."*). The best cosine similarity between any student sentence and any reference counts
  (full credit ≥ 0.62, partial ≥ 0.56), so a student who says *"the first position is zero"*
  matches the taught fact even with no shared textbook wording. Which specific facts the
  student expressed is tracked and shown to faculty as evidence.
- **Misconception detection** — a sentence is flagged only when it is genuinely closer to the
  wrong claim than to the *correct* account of the material: similar enough to the wrong claim
  (≥ 0.65) **and** either (a) closer to it than to both the clarification and the concepts'
  own reference texts (margin 0.08) — which stops the system inventing a misconception out of
  a correct answer — or (b) semantically about the claim while using a word unique to the
  wrong version and none unique to the clarification (embeddings are nearly blind to
  "index 1" vs "index 0", so the cue path catches polarity/number flips).
- **Semantic correctness** — cosine similarity of the whole response to the reference
  explanation, rescaled to 0–1.
- **Explanation depth / response effort** — structural richness and length measures.
- **Concept relationships** — each topic carries a small teacher-authored list of
  relationships (e.g. *Backpropagation → computes → Gradient*), each stored as one correct
  sentence and, where a natural error exists, one wrong version. Every relationship ends the
  session in exactly one of **three** states, and only the last of them is a learning gap:

  | State | Meaning |
  |---|---|
  | ✓ **Demonstrated** | the answer contains evidence for the teacher's connection |
  | ○ **Not discussed** | no evidence either way — never counted, worded or acted on as a mistake |
  | ⚠ **Needs clarification** | the connection was expressed incompletely, questionably or incorrectly |

  A relationship is demonstrated either by a **strong direct match** against the teacher's
  sentence, or by a weaker match that is **corroborated at both ends of the link** — the same
  student sentence also carries semantic and lexical evidence for the source *and* the target
  concept. The second path exists because one reference sentence under-rates ordinary
  phrasing ("they're individual letters or symbols inside the string" expresses *strings
  contain characters* at cosine 0.674, just under the direct bar) and the endpoint evidence is
  what makes accepting it safe without lowering the bar for everything.

  Because embeddings are nearly blind to polarity flips ("reduces the loss" vs "increases the
  loss" differ by ~0.002 cosine), contradictions come from explicit cues only: **cue words
  derived automatically from the teacher-authored pair** (flagged only when the sentence uses
  no word unique to the *correct* version, so "join puts the pieces back together" is not
  accused of contradicting split()), or an **explicit negation of one of the endpoints**
  ("…not characters"). A sentence that says nothing about a connection leaves it *Not
  discussed*; even a directly probed connection stays *Not discussed* when the student gave a
  blank or off-target answer — being asked is not the same as getting it wrong.

  Relationship evidence accumulates across the conversation (it is *not* added to the HMM
  feature vector, so the trained artifact stays valid).

### What counts as evidence (`analyzer.informative_terms` / `concept_evidence`)

Every reference text is stored as `"Name: explanation"`, which gives short
answers the context of the question — and, left unguarded, makes *saying the
name* score higher than *explaining the idea*. A 170-answer audit
(`scripts/student_audit.py`) found both failure directions coming from that one
spot:

| student answer | before | after |
|---|---|---|
| "python uses indexing" | credited as understanding Indexing (cosine 0.82) | no evidence |
| "you use the number to get the letter" | no evidence (no shared words) | demonstrated |

The fix is a rule, not a threshold — **no similarity bar was changed**. Before
either score is trusted, the answer must say something *about* the concept:

1. take the answer's content words;
2. remove the concept's own name, the lecture title, the lecture's other
   concept names, and a small closed set of evaluative/meta filler
   ("useful", "was covered in class", "really");
3. what remains is the evidence. Credit needs **two such words, or one that is
   also one of the teacher's own key words** for that idea.

Neither condition can be met by naming the concept, and a genuine paraphrase
that avoids the jargon satisfies them easily. The same rule governs
whole-response concept coverage, so naming a concept in passing cannot mark it
demonstrated either. Two further guards came out of the same audit: an answer
sitting closer to a taught wrong claim than to the concept itself has credit
*withheld* (never an accusation — see `MISCONCEPTION_SHADOW`), and connective
adverbs like "back" no longer count as evidence that a relationship was stated
backwards.

Measured effect (no thresholds touched), on the labelled set's held-out
portion: evidence accuracy 0.782 → 0.862, misconception precision unchanged at
1.000, relationship checks unchanged at 18/20.

#### A score has to be earned, not started with (`NAME_ONLY_LIFT`)

The lexical rule above stops an answer being credited for *naming* a concept,
but it cannot see the second half of the same problem: because the concept's
name is repeated inside every reference text, both similarity scores start
well above zero **before the student has said anything at all**.

| concept | score for an empty answer | credit bar |
|---|---|---|
| Hidden states | 0.82 | 0.62 |
| Slicing | 0.76 | 0.62 |
| Markov property | 0.76 | 0.62 |
| Weight update | 0.54 | 0.62 |

For the self-describing concepts that floor is already *past* the bar, so an
absolute threshold means a different thing for every concept, and for some of
them silence scores as well as an explanation. The adversarial audit below
found this being paid out as real credit: *"umm well you know how it is"* and
*"is this going to be in the test"* were both reported to the student as a
demonstrated concept.

Each score is now measured against **its own floor** — the identical
measurement with the answer removed — and full credit goes to the score the
*answer* lifted above it. This is a correction to the measurement, not a
change of threshold: every bar in `conversation.py` is untouched, and the
required lift was swept over 0.00–0.10 on the calibration split and left at a
bare "must beat its floor", because a wider margin bought nothing the floor
had not already bought and cost real answers.

Two smaller fixes came from the same audit:

- **Polarity, where similarity cannot help.** *"The next state depends on
  everything that happened before it"* is the Markov property inverted, and it
  shares nearly every word with the teacher's own sentence — so the embedding
  puts it **closer to the concept (0.91) than to the taught misconception it
  restates (0.80)**. No similarity rule can separate those. A concept defined
  by an exclusion ("depends *only* on…, *not* on…") therefore reads an answer
  asserting the totality it rules out as a ceiling on credit. It is
  deliberately narrow — quantifying words like "each" are excluded, so *"each
  state emits an observation"* does not trip it — and it only ever *withholds*
  credit; it never names a misconception.
- **The per-question check now gets the lecture's other concept names**, which
  the whole-response analysis always had. Without them, a bare list of the
  topic's own headings ("gradient descent backpropagation loss weights")
  counted as an explanation of whichever concept was under discussion.

### Guided conversation (`nlp/conversation.py`)

A deterministic rule engine — not a language model — walks through the concepts one **short
question** at a time and adapts:

- clearly correct → acknowledge, move on (concepts demonstrated in passing are skipped);
- **short confirmation of a yes/no question** ("yes") → positive evidence, not full credit:
  *"Right — can you explain what that means in your own words?"* — but "yes" to an open
  question earns nothing, and "I don't know" is never inflated (a content-word-overlap guard
  blocks empty answers from the contextual scorer, and topic-title words like "Python" don't
  count as overlap);
- partly right → one targeted probe;
- **a plausible analogy** ("it's like a row of numbered boxes") → neither accepted nor
  rejected: *"Interesting analogy — can you connect that back to …?"*;
- unclear → one easier question;
- misconception → explain the distinction, probe once, and mark it **resolved** if the next
  answer no longer contains it.

Question thresholds are **calibrated against the labelled answer dataset**
(`data/nlp/labeled_answers.json`, ~200 hand-written answers) with
`python scripts/evaluate_nlp.py --tune`; the chosen values are committed explicitly in
`nlp/conversation.py` — nothing retrains or self-modifies at runtime.

The session ends when every concept has been visited — the 12-question limit is a safety cap,
not a target. Depth is never confused with understanding: no advanced questions are asked
unless the teacher configured them (lecture-published topics have no extension question at
all), and lecture-mode questions are conversational (*"What did you understand about X?"*).

### PDF ingestion (`nlp/pdf_extract.py` + `nlp/pdf_clean.py`)

Real lecture PDFs — especially slide decks — are not linear documents. Flattening one with
`page.extract_text()` produced a stream in which a running footer, a slide number and a slide
title were indistinguishable, which is how `"5© Copyright 2014 EMC Corporation."` could end up
as a concept's *meaning*. PDFs therefore go through a layout-aware pipeline before they reach
the parser:

```
PDF -> layout-aware extraction -> deterministic cleanup -> Markdown notes -> same parser
TXT/MD ------------------------------------------------------------------>
```

1. **Extraction** (`pdfplumber`, already installed via pdfminer.six; `pypdf` as a fallback)
   keeps a `LectureDocument`: pages, and per page an ordered list of blocks with text,
   bounding box, font size and weight. Letter-spaced titles (`C l o u d  C o m p u t i n g`)
   are rejoined using the word gaps, and hyphenated line breaks are repaired.
2. **Cleanup** decides what is page *decoration* rather than content — and never on a single
   signal, because every single signal has a legitimate counter-example. Repetition alone is
   wrong ("Cloud Computing" may genuinely be taught on twelve slides); position alone is wrong
   (the first line of a slide is usually its title); shortness alone is wrong ("Immutability"
   is a real concept). So repetition-based removal requires **repetition across pages AND a
   consistent header/footer position AND a font size no larger than the body text AND that the
   line never behaves like a heading** (never followed by real prose on any page). Copyright /
   legal notices and page-number patterns are removed on their own, and only when short and
   standalone. Every removal records its reason, and the counts are shown to the teacher.
3. **Rebuild** emits ordinary Markdown notes — headings from relative font size/weight,
   bullets, prose paragraphs with wrapped lines rejoined — plus `<!-- page N -->` provenance
   markers. The parser reads and strips those markers, so a suggestion can say *"Found on page
   7 under 'Indexing'"* even after the teacher hand-edits the extracted text.
4. **Extractability is judged page by page, never by a document total.** A total hides the
   case that matters: the real 22-page *Regular expression* lecture has 15 pages that are
   photographs of the board and 7 typed pages, and those 7 carry enough characters that a
   document average calls the file a normal text PDF — dropping two thirds of the lecture
   without telling anyone. Two bars are used, because *sparse* and *absent* are different
   problems: below `EMPTY_PAGE_CHARS` (10) a page yielded nothing and its content **has been
   lost**; below `MIN_CHARS_PER_PAGE` (40) a page is merely sparse, like a title slide or a
   section divider, and is reported but tolerated. The verdict is one of:

   | `text_quality` | when | result |
   |---|---|---|
   | `text` | no image-only pages, or fewer than max(2, 10% of pages) | ingested; any image-only pages are still reported |
   | `mixed` | at least max(2, 10% of pages) are image-only | **refused**, naming the pages |
   | `scanned` | 80% or more of pages are image-only | **refused** |

   A refused document produces **no notes at all** — handing back the fraction that happened
   to carry text would present part of a lecture as the whole of it. The message names the
   affected pages (*"pages 1-8, 10-16 … 15 of 22 pages could not be read"*) and points at OCR,
   pasting, or the prepared-note workflow. `scanned` is still in the report for compatibility,
   but now means "this file cannot be fully extracted", which is the question every caller was
   actually asking. No OCR dependency was added: the goal here is to detect incomplete
   extraction, not to fix it.

   Measured on the two real lectures: the 108-page EMC Cloud Computing deck has **0** image-only
   pages (minimum 145 characters per page) and still ingests; the *Regular expression* lecture
   is identified as `mixed` with image pages `1-8, 10-16` and is refused.
5. **Image-heavy pages** are a separate, weaker signal, and a warning rather than a refusal. A
   page can extract perfectly and still be missing the lesson: the EMC deck has slides whose
   labels are baked into a diagram while the surrounding prose reads fine. Using only the image
   geometry pdfplumber already reports, a page is flagged when it carries an image covering
   ≥20% of the page that is neither a full-bleed background (≥90%) nor template chrome (the
   same picture, same size, same position, on ≥50% of pages — the repetition argument the
   header/footer cleanup already uses). Nothing inspects pixels, so the wording never claims
   the picture contains text: *"page 5 contains a large diagram; labels inside it may not have
   been extracted."* On the real deck this flags **18 of 108** pages, correctly ignoring the
   logo that appears on all 108 and the 26 full-bleed backgrounds. Image-only pages are
   excluded from this list — they are missing content, which is a different and worse problem.

The review screen shows *"Extracted from 10 pages. Removed 30 repeated header/footer,
page-number and copyright lines"* with examples, and a **View raw extraction** toggle for
debugging. TXT/MD ingestion is unchanged — both paths feed the same downstream pipeline.

### Lecture preparation (`nlp/lecture_parser.py` + `nlp/lecture_prep.py`)

Deterministic extraction, no LLM — and **structure first, semantics second**. The parser turns
raw notes into a document tree: headings (Markdown or plain-text), bullets, numbered items,
fenced code, `Example:` lines, and special sections (*Learning Objectives*, *Important
Connections*, *Common Mistakes*, *Summary*). Then:

- **Structured notes** (real headings): each content section becomes a **candidate**
  concept, not a concept. A heading is evidence that a concept might be there; it is not
  proof. Every candidate must earn its place:
  - it must not be document structure — `Agenda`, `Thank You`, `Lesson: X Overview`,
    `Module 3:`, a bare number or a copyright line are all recognised as navigation;
  - the lecture must actually *explain* it — a definition-like sentence, or supporting facts
    plus prose. A heading followed only by `"© 2014 EMC Corporation."` produces nothing;
  - its **meaning is chosen, not taken from the first line**: candidate sentences are scored
    on whether they mention the concept, carry an explanatory predicate (*is / means / refers
    to / enables / consists of / …*), have a reasonable length, and are not metadata or code;
  - its **facts are ranked, not the first N leftovers**: a fact must be a complete declarative
    claim about the concept, must not restate the meaning, and must not be a fragment
    (`"organizations"`, `"the coming decade"` never become facts).

  Candidates are then scored (definition + facts + examples + objective match + semantic
  centrality), the best are kept — **there is no fixed count of six**; a lecture may yield 3
  or 8 — and each keeps its provenance (`source_section`, `source_page`, `source_sentences`)
  and an honest confidence label: *Strongly supported* / *Moderately supported* /
  *⚠ Weak evidence — review carefully*. If little was well supported, the review screen says
  so rather than padding the draft.
- **Plain prose** (no headings): the n-gram fallback — content-word phrases scored by
  frequency × embedding centrality, but a candidate is only kept when the lecture actually
  *explains* it (a sentence where it appears early and gets elaborated). Generic nouns
  without explanatory support are rejected, so notes about strings no longer produce
  "Letters", "Values" or "Operator" as concepts.
- **Relationships**: explicit `A → label → B` lines from an *Important Connections* section
  are authoritative. Prose-inferred relationships are deliberately conservative — two concepts
  appearing in the same sentence is **not** a pedagogical relationship, so an explicit
  relational cue (*uses, enables, consists of, produces, converts, requires, contrasts with…*)
  must sit between them, within a short span. A false relationship is worse than a missing one
  because it later shapes how a student's answer is judged.
- **Misconception suggestions**: *Common Mistakes* lines — "Students may think X, but
  actually Y" splits into the wrong claim and its clarification — plus previously authored
  misconceptions the lecture is semantically about.
- **Teacher objectives** stay authoritative and are used to *rank* candidates (semantic
  similarity between each objective and each candidate). They never invent a concept: material
  that does not support an objective still produces nothing for it.
- **Question drafts** per concept (main / easier / probe / application), templated from the
  concept's own facts and examples — e.g. the probe quotes an actual lecture fact. The
  application question is optional extension material, never required.
- **Suggested activities** built from the lecture's own concepts and examples, one per
  learning state.

Everything is a *suggestion*; nothing becomes authoritative until the teacher reviews and
publishes it.

### Recommended note format & optional external AI preparation

The lecture-creation page shows a **recommended note template** (headings + Examples +
Important Connections + Common Mistakes) — recommended, not required; ordinary notes still
work. It also offers a **"Copy AI preparation prompt"** button: text the teacher may paste
into an external assistant (ChatGPT, Claude, …) to convert rough notes into the template. The
prompt forbids inventing information or adding outside knowledge, requires preserving the
teacher's terminology, examples, code and important qualifications, forbids omitting content
that merely looks unimportant, asks for source/page references to be kept, and marks
uncertainty as `[UNCLEAR]`. The teacher reviews the result before TeachBack analyses it.
**TeachBack itself never calls an LLM** — this is copyable text only
(`GET /api/lectures/prep-prompt`), and the recommended format remains optional: ordinary
unformatted notes are fully supported.

### Student summary (`Your takeaway`)

After the conversation the student writes what they personally took away. It is stored with
the session, shown on Progress, and analyzed as an **upgrade-only** evidence source: it can
add or strengthen concept/relationship evidence but can never lower anything — a short summary
is never a penalty.

### Quick knowledge check (MCQs) — secondary evidence, not the measure

After the conversation and takeaway, the student can *optionally* take a short
teacher-reviewed 10-question check. The two signals are deliberately different and
deliberately kept separate:

- **TeachBack asks:** *what can the student explain in their own words?* (primary)
- **The knowledge check asks:** *what can the student correctly recognise/apply?* (supporting)

Neither alone is a perfect measurement: a student may understand something and make a careless
MCQ mistake, memorise an answer they cannot explain, explain an idea in different terminology,
or misunderstand one detail inside a broadly understood concept. The system preserves this
nuance instead of collapsing it:

| TeachBack | MCQ | Interpretation |
|---|---|---|
| strong | strong | strong supporting evidence |
| strong | weak | explanation evidence **stays**; the specific MCQ gap is named as "worth a quick review" |
| weak | strong | *not* called confused — "practice putting the idea into your own words" |
| weak | weak | stronger case for review/support |

Questions are **generated deterministically** (`nlp/quiz_gen.py`) from the teacher-reviewed
structure only — concept meanings, reviewed facts, lecture examples, relationships and the
teacher's own misconceptions supply both answers and distractors, so nothing outside the
lecture is ever asked. Difficulty mix: ~4 recognition, ~3 application (e.g. *"what does
`s[0:3]` give?"* built from the lecture's own example), ~2 spot-the-false-statement (the
teacher's wrong claims among taught facts), ~1 relationship. Every question passes structural
validation (4 unique options, one correct answer, no accidental answer clues) and the teacher
can **edit / delete / regenerate / add** every question on the lecture page before publishing.

Results are stored per answer and per concept (`Quiz`/`QuizQuestion`/`QuizAttempt`/
`QuizAnswer`), shown to the student as *"8/10 questions correct"* with solid/revisit concept
lists (never an alarming grade), and to the teacher as **two side-by-side per-concept
numbers** — MCQ % and TeachBack-demonstrated % — never one "mastery" score. The score becomes
an evidence *note* on the session's observation and can refresh the recommendation (e.g.
"Quick review: Slicing"); it **never touches the 8-dimensional observation vector, the HMM
artifact, or the estimated state**.

## 8. Confidence, difficulty and lecture feedback

**Confidence is not treated as understanding.** It is a separate observation, compared with
the NLP evidence:

| NLP evidence | Confidence | Difficulty | Recommendation |
|--------------|-----------|------------|----------------|
| high | low | — | easy application task to build confidence |
| high | high | low | *optional* extension/challenge ("Today's material appears comfortable for you") |
| low | high | — | support material plus a gentle double-check note |
| low | low | — | simpler explanation / practice (state-based) |

These signals steer **which activity** is recommended — they never assign an HMM state
(`if confidence > 8: state = Confident` does not exist, and tests assert it). Students who
find a lecture comfortable get an *optional* extension in neutral language — never labels
like "gifted" or "weak", and never automatic escalation to harder material.

**Lecture feedback** (pace choice, request chips, free text) serves the *teacher*, not student
classification: the Class Overview aggregates average confidence, average perceived
difficulty, a pace distribution, common requests and recent comments per lecture — so faculty
learn how the lecture itself was experienced.

## 9. HMM component (`backend/app/hmm/`)

A `GaussianHMM` (hmmlearn, 5 states, diagonal covariance) trained **unsupervised** on
synthetic observation sequences of the fixed 8-feature vector: concept coverage, semantic
correctness, misconception score, explanation depth, response effort, attention, confidence,
difficulty.

**State interpretation.** Unsupervised state IDs are arbitrary, so each learned state's
emission mean is matched one-to-one to the closest canonical profile (`app/states.py`) with
the Hungarian algorithm; the mapping is saved to `data/artifacts/hmm_state_mapping.json` so
the assignment is auditable rather than asserted.

**Inference.** When a student finishes a session, Viterbi decoding runs over their *entire*
observation history — the current state depends on the trajectory, not just the latest
session, and earlier states may be re-interpreted in light of new evidence.

## 10. Dataset construction (`backend/app/hmm/synthetic.py`)

No real educational dataset is required. A synthetic dataset of **200 students × 5–10
sessions** is generated from five learner archetypes, each with an initial state distribution
and a Markov transition matrix; observations are drawn from per-state emission profiles with
Gaussian noise. True hidden states are kept for evaluation. The demo database is seeded the
same way (`source = "seed"`), with displayed states still coming from real HMM inference;
sessions completed in the UI are stored as `source = "live"`.

## 11. Recommendations and activities (`backend/app/recommend/`)

A transparent rule table maps the state (adjusted by the confidence/difficulty signals above)
to an activity style: re-engagement → concept review → guided practice → application →
challenge. The activity itself is resolved in order:

1. **teacher-authored activity** stored on the topic for that state (with its own content and
   question — e.g. the seeded "Hiker-on-a-hill analogy", which is demo content, not a special
   case in code);
2. **deterministic template activity** generated from the topic's own concepts (explain X in
   one sentence / give an everyday example of X / apply X / connect X and Y) — this is what
   keeps freshly published lectures fully actionable with zero extra teacher work;
3. a generic fallback.

Every recommendation is actionable: **Recommendation → [Start Activity →] → actual content and
short task → submit → completion recorded → back to Progress/Dashboard.** No dead-end
recommendation cards. Each recommendation also explains *why* it was chosen, from the actual
session evidence.

## 12. Multi-teacher / multi-subject

Data model: **Teacher → Subject → Lecture/Topic → TeachBack sessions → Students.** The faculty
interface has a lightweight teacher/subject switcher (demo-level context switching, *not*
authentication — a documented limitation). The selected subject scopes **every faculty-facing
query in the backend** — dashboard statistics, state distribution, misconception aggregates,
declining students, topic stats, lecture feedback, recent interactions and session counts all
filter by the subject's topics in `/api/teacher/overview?subject_id=…` (not by frontend
filtering), so Prof. Arjun Rao's *Python Programming* dashboard can never show Prof. Meera
Krishnan's *Neural Networks* data, and vice versa. Cross-subject isolation is asserted by
regression tests (`tests/test_subject_isolation.py`). The student topic chooser is grouped by
subject.

## 13. Demo accounts & data

- **Teachers:** Prof. Meera Krishnan (*Neural Networks*: Backpropagation, Overfitting and
  Regularization, Hidden Markov Models — fully hand-authored topics) and Prof. Arjun Rao
  (*Python Programming*: Strings in Python).
- **Students:** 9 named demo students (including **Shreshtha Bindal · B.Tech CE · B023**, the
  primary demo student) plus background students with seeded histories — some in both
  subjects, so both dashboards have their own (non-overlapping) data.
- **Strings in Python** is seeded through the *real* lecture pipeline: the stored lecture
  record contains the raw structured notes, the untouched NLP suggestions computed from them,
  and the teacher-reviewed draft (Strings, String assignment, Characters, Indexing, Slicing,
  split() and join() — with facts, examples, custom questions, relationships, misconceptions
  and activities) that was published. It exists precisely to show the system is generic and
  not hardcoded around Backpropagation — there is no Python-specific code anywhere.

## 14. How to run (Windows-friendly)

Prerequisites: Python 3.11+, Node 18+.

```bash
# 1) backend dependencies
cd teachback/backend
pip install -r requirements.txt

# 2) build everything: synthetic dataset -> HMM training/evaluation -> seeded SQLite DB
#    (first run downloads the ~90 MB embedding model)
python scripts/build_all.py --force-seed

# 3) start the API (run from teachback/backend)
python -m uvicorn app.main:app --port 8000

# 4) frontend (new terminal)
cd teachback/frontend
npm install
npm run dev        # http://localhost:5173  (dev server proxies /api to :8000)
```

Reset/regenerate everything: `python scripts/build_all.py --force-seed` (fixed random seeds,
reproducible). Production frontend build: `cd teachback/frontend && npm run build`.

### Running tests

```bash
cd teachback
python -m pytest tests -q      # NLP, conversation, PDF ingestion, concept quality,
                               # lecture lifecycle (delete/archive), evidence safety,
                               # HMM validation + artifact integrity, subject isolation,
                               # quiz, feedback, recommendations, activities, API end-to-end

python scripts/evaluate_nlp.py         # answer-evaluator metrics on the labelled set
python scripts/evaluate_nlp.py --tune  # threshold calibration sweep
python scripts/simulate_user.py        # full faculty+student journey against the live API
                                       # (resets the demo DB; --keep to run in place)
python scripts/student_audit.py        # 170-answer + 10-session student-understanding audit
python scripts/verify_persistence.py   # SQLite is the source of truth; no orphaned rows
python scripts/verify_ui.py            # headless-browser check of the real interface
                                       # (needs the backend on :8000 and vite on :5173)
```

## 15. Faculty demo (~5 minutes)

1. **Continue as Student** → *Shreshtha Bindal (B.Tech CE · B023)* → TeachBack →
   **Backpropagation**.
2. Answer in short natural sentences — e.g. *"By checking the error between the prediction and
   the actual result."* Simple wording is accepted as evidence.
3. When asked a yes/no question (e.g. *"Does the gradient tell us how the loss changes when we
   nudge a weight?"*), answer just **"yes"** — TeachBack treats it as positive evidence and
   asks you to explain it in your own words.
4. Say *"Backpropagation and gradient descent are the same thing."* — watch the misconception
   clarification and probe, then correct it and see it marked **resolved**.
5. Write a one-line **Your takeaway**, set confidence/difficulty, pick a pace and a feedback
   chip.
6. The summary shows the learning journey (previous state → new state), demonstrated concepts
   and connections, why the state is what it is, and the **recommended next activity**.
7. Click **Start Activity →**, complete the short task, and see the completion on
   **Progress** along with your takeaway.
8. Switch to **Teacher** → pick *Prof. Arjun Rao / Python Programming* in the switcher → open
   the **Strings in Python** lecture on the Lecture TeachBacks page: the NLP suggestions, the
   reviewed concepts (with their source evidence) and the suggested activities are all
   visible. Run a TeachBack on Strings as *Shreshtha Bindal* with deliberately simple answers
   — *"Strings are basically text stored in quotes"*, *"You use the position to get a
   character, and the first position is zero"* — and watch them accepted naturally, with no
   escalation once a concept is demonstrated. The Class Overview shows only the selected
   subject's data plus the lecture-feedback aggregates (pace, requests, confidence,
   difficulty).

## 16. Architecture & technology stack

```text
teachback/
├── backend/                 FastAPI + SQLAlchemy (SQLite)
│   ├── app/
│   │   ├── api/             REST endpoints: students, topics, sessions (TeachBack),
│   │   │                    lectures + teachers, activities, teacher dashboard, meta
│   │   ├── nlp/             embedder, analyzer, guided conversation engine,
│   │   │                    lecture preparation (concept extraction)
│   │   ├── hmm/             synthetic generator, HMM training / mapping / inference
│   │   ├── recommend/       rule-based recommendations + template activities
│   │   ├── evaluation/      NLP + HMM evaluation
│   │   ├── models.py        ORM: teachers, subjects, lectures, topics, concepts,
│   │   │                    relationships, misconceptions, activities, students,
│   │   │                    sessions, responses, observations, completions
│   │   ├── seed.py          database seeding (incl. the Python sample lecture,
│   │   │                    which runs through the real lecture pipeline)
│   │   └── seed_content.py  teacher-authored demo content
│   ├── scripts/build_all.py
│   └── requirements.txt
├── frontend/                React + Vite + Tailwind CSS
│   └── src/
│       ├── pages/           Student Dashboard, TeachBack, Activity, Progress,
│       │                    Teacher Dashboard, Lecture TeachBacks, Topic Management
│       ├── components/      layout, state badges, timelines, teacher/subject switcher
│       └── services/        API client
├── data/
│   ├── synthetic/           generated dataset (JSON + CSV)
│   ├── nlp/                 curated labelled answers for evaluator calibration
│   ├── nlp_eval/            hand-labelled responses for the legacy NLP evaluation
│   └── artifacts/           trained HMM, state mapping, evaluation results
├── scripts/evaluate_nlp.py  answer-evaluator metrics + threshold tuning
├── tests/                   pytest suite (100+ tests)
└── README.md
```

Python: FastAPI, SQLAlchemy, sentence-transformers, hmmlearn, SciPy, NumPy, pypdf.
Frontend: React 18, Vite, Tailwind. Storage: SQLite. **No LLM APIs anywhere.**

## 17. Evaluation methodology & results

**What is trained vs. what is curated — no fake training claims:**

- The **sentence-transformer** (`all-MiniLM-L6-v2`) is a *pretrained* embedding model used
  as-is; TeachBack trains no neural network of its own.
- `data/nlp/labeled_answers.json` (~200 hand-written answers + relationship checks) is a
  *curated evaluation/calibration set* used to tune and regression-test the deterministic
  evaluator. It is **not** real student data and the evaluator was **not** "trained on
  student answers".
- The **HMM** is the pre-existing artifact trained once on synthetic trajectories
  (`data/artifacts/hmm_model.joblib`); its SHA256 is pinned by regression tests and it was
  not retrained in the quality pass.

**Answer evaluator** — `python scripts/evaluate_nlp.py` runs every labelled answer through the
real per-turn pipeline (analysis + targeted check + verdict) and reports per-label
precision/recall/F1, a 4×4 confusion matrix, and per-category accuracy (simple language,
paraphrases, short answers, analogies, noise, …). The dataset is split deterministically into a
**calibration portion (174 items — thresholds tuned only on these)** and a **held-out portion
(87 items — never used for tuning)**:

| | Calibration (174) | Held-out (87) |
|---|---|---|
| Strict label accuracy | 0.64 | 0.62 |
| Evidence-level accuracy | 0.86 | 0.84 |
| Misconception precision | **1.00** | **1.00** |
| Misconception recall | 0.55 | 0.57 |
| Strong-evidence precision / recall | 0.78 / 0.67 | 0.83 / 0.64 |

("Evidence-level" counts strong↔partial confusion as acceptable — both mean "the student
showed understanding".) The held-out portion is small (87 items), so its numbers are a
sanity check rather than precise statistics — but they are consistent with calibration,
suggesting the thresholds are not badly overfit. Misconception *precision* is deliberately
prioritised over recall: the system must never accuse a correct answer of being a
misconception; missed paraphrased misconceptions fall through to the normal probe flow
instead. 18 of 20 relationship checks behave as expected.

**These numbers moved in both directions in the final pass, and the trade was deliberate.**
Fixing the similarity floor (`NAME_ONLY_LIFT`, above) raised strong-evidence *precision*
(0.72 → 0.78 calibration, 0.77 → 0.83 held-out) and cut the adversarial audit's false-credit
rate from 0.317 to 0.067; it cost strong-evidence *recall* (0.80 → 0.67, 0.73 → 0.64) and
about a point of overall accuracy on this set. The labelled set is dominated by cooperative
answers, so it under-weights exactly the failure the fix targets. Telling a student they have
demonstrated something they have not is the more damaging error, so it was taken.

**Legacy NLP feature evaluation** — `scripts/build_all.py` additionally scores concept /
misconception detection on `data/nlp_eval/` and stores results in
`data/artifacts/evaluation_results.json` (recomputed on every build, not hard-coded).

**HMM** — students split 80/20; trained on 160 students, Viterbi-decoded on the 40 held-out
students against the generator's true states:

| Metric | Value |
|--------|-------|
| State accuracy (297 test sessions) | 0.976 |
| Adjacent-state accuracy (off by ≤ 1) | 1.000 |

**The HMM was evaluated using synthetic student trajectories and should not be interpreted as
validated real-world student prediction.**

### Student-answer audit (`scripts/student_audit.py`)

The labelled set is small and was partly used for calibration, so it cannot
answer "would this work on real students?". The audit is a second, independent
check: 170 deterministic answers built from hand-written per-concept phrasings
plus meaning-preserving surface transformations (shorten, informalise, add
filler, add a typo), deliberately avoiding the concept name in most correct
answers, plus 10 complete conversation sessions with different student
personalities and 20 free-form takeaways. It runs through the real pipeline —
`analyze_response` + `targeted_concept_check` + the conversation engine's own
verdict — and is **never used to tune anything**.

Current result (seed 20260831, `data/nlp/student_simulation.json`):

| measure | before the final pass | after |
|---|---|---|
| strict / evidence accuracy | 0.641 / 0.753 | 0.529 / **0.759** |
| demonstrated precision / recall | 0.787 / 0.686 | **0.792** / 0.442 |
| false credit on answers carrying no evidence | 0.133 | **0.044** |
| ...on "I don't know" | 0.000 | 0.000 |
| ...on unrelated answers | 0.000 | 0.000 |
| correct answers in simple/terminology-free wording missed | 0.246 | 0.246 |
| relationship probes as expected | 6/12 | 6/12 |
| takeaways: never downgraded / no fabrication from a term list | yes / yes | yes / yes |

**Read the two accuracy figures together.** *Evidence-level* accuracy (which
treats "demonstrated" and "partly shown" as the same answer to the question
*did this student show understanding?*) went slightly **up**. *Strict*
accuracy went down because the split between those two moved: after the
similarity-floor fix, a correct answer far from the teacher's wording is more
often credited on the **follow-up** question rather than on the first one.
The student is not told they are wrong — they get one more question — but the
concept is no longer settled in a single turn, which is what the recall figure
is measuring.

The 10 sessions still discriminate as they should, one step more
conservatively than before: the concise, informal, example-driven and
misconception-correcting students demonstrate **five** of six concepts (six
before the fix); the uncertain, vague and different-terminology students
demonstrate three; the overconfident-but-wrong and genuinely-struggling
students demonstrate one. All stay inside the 12-question cap.

Disabling the floor fix and re-running the adversarial audit isolates what it
buys: false credit **0.108 → 0.067**, and 11 further answers stop being
reported as demonstrated — among them *"is this going to be in the test"* and
*"the architecture leverages an end-to-end differentiable paradigm"*. That is
the trade, measured in both directions and taken deliberately.

### Adversarial audit (`scripts/adversarial_audit.py`)

`student_audit.py` asks whether ordinary student wording is *recognised*. This
asks the opposite question — where does the system produce a **dangerous**
output? Two failures matter far more than accuracy:

- **False credit** — an answer with no understanding in it is reported as
  demonstrated. The student is told they know something they do not.
- **False accusation** — a correct or merely absent answer is named as a
  misconception, or turned into a remediation task. The student is corrected
  for something they did not do.

The full report, including the category breakdown and every bug it found, is in
[EVALUATION.md](EVALUATION.md).

151 fixed cases (139 answers + 12 relationship probes) across **four topics**,
run through the real pipeline, covering the eight families the brief calls for:
clearly correct (textbook, paraphrase, informal, colloquial, very short, long,
own terminology, analogy, example, correct-plus-irrelevant), partially correct,
incorrect-but-not-taught, taught misconceptions, no-evidence, uncertainty,
natural language variation (spelling, no punctuation, slang, Indian-English,
terse, rambling), and adversarial (concept name only, right words with wrong
meaning, negation, misconception followed by its own correction, mixed
right-and-wrong claims). **Nothing in it is ever used to tune a threshold.**

| measure | before this pass | after |
|---|---|---|
| false credit (no-evidence or wrong answer reported as demonstrated) | 0.317 | **0.067** |
| false accusation (misconception named where none was made) | 0.000 | **0.000** |
| correct answers recognised | 0.949 | 0.949 |
| misconception precision / recall | 1.000 / 0.636 | 1.000 / 0.636 |
| taught misconceptions handed full credit | 1 | **0** |
| "I don't know" / blank / question-echo given credit | 0 | 0 |
| silence reported as a misunderstood relationship | 0 | 0 |
| safety invariants passed | 7/7 | 7/7 |

The four residual false credits are all the same shape: an answer that uses
the right vocabulary about the right topic while asserting something false
("characters are the variables that hold a string", "regularization increases
the size of the weights"). Sentence embeddings score those as near-identical
to the correct account, and no deterministic rule in this codebase separates
them — see *Known limitations*. Overall agreement with the expected outcome
rose from 0.77 to 0.856 across the same 139 answers.

### Complete-session simulation (`scripts/session_sim.py`)

The two audits above judge answers. This one judges **finished sessions**: 20
kinds of student across all four topics, each driven through the whole
lifecycle over the live HTTP API — conversation, misconception handling,
takeaway, self-report, pace and feedback, the knowledge check, combined
evidence, recommendation, activity completion and the progress page.

20/20 sessions completed, **202 answers evaluated**, 20 knowledge checks, 20
activities completed, 8–12 turns each (mean 10.1, cap 12). Every session is
then checked against eleven things a teacher would object to — credit from
noise, an accusation nobody earned, a repeated question, silence turned into
remediation, an undiscussed relationship called a problem, a takeaway that
downgrades, MCQ rewriting conversation evidence, confidence becoming
understanding, machinery or judgement in student-facing wording — and all
eleven pass 20/20. Full results in [EVALUATION.md](EVALUATION.md) and
`data/nlp/session_sim.json`.

It found four defects, all in what the student is told rather than in the
scoring: four probe questions that merely restated the question the student
had just failed to answer; an acknowledgement that said *"you have the main
idea"* while recording the concept as partial; *"your recent sessions showed
very low engagement"* — a claim about effort, which the system never observes —
surviving in two places after `states.py` had been corrected for exactly that;
and a topic that could be created with no subject at all, escaping subject
isolation. All four are fixed and covered by regression tests.

## 18. Known limitations

The system estimates **evidence of conceptual understanding within a bounded, teacher-reviewed
lecture context** — it does not directly measure understanding, and it does not know what is
happening in a student's mind.

- NLP features are similarity heuristics against teacher-authored/reviewed text; paraphrases
  far from the stored descriptions can be missed, and thresholds were tuned on the same small
  labelled set they are evaluated on.
- Semantic correctness saturates for fluent on-topic text even when partially wrong.
- **A confidently wrong answer built from the right vocabulary can still be credited.**
  This is the largest remaining hole, and the adversarial audit measures it rather than
  hiding it (0.067 false credit, all four cases of this shape). *"Characters are the
  variables that hold a string"* scores as well against the Characters reference texts as
  many correct paraphrases do, because it is about characters, about strings and about
  holding text; only the claim is false. Sentence embeddings measure aboutness, not truth,
  and the two guards that do catch a false claim both need something to compare against:
  a teacher-authored misconception, or an explicit polarity marker in the concept's own
  description. An arbitrary false statement in fluent topic vocabulary has neither.
  Closing this properly needs entailment checking, which means a model TeachBack
  deliberately does not have. The practical mitigations are the ones already in place:
  teachers author the misconceptions they actually see, and the conversation asks a
  follow-up rather than settling the concept on one answer.
- The HMM is trained **and evaluated on synthetic trajectories**; its high accuracy reflects
  recovery of the generating process, not validated performance on real students.
- Relationship contradiction detection relies on cue words from the teacher-authored wrong
  version; contradictions phrased with entirely different vocabulary (or negation, e.g.
  "does not decrease") can be missed. Analogies are never auto-credited: a plausible one gets
  a "connect it back" follow-up, and one with no semantic footing gets an easier question.
- Misconception detection prioritises precision over recall: a wrong belief phrased with
  entirely different vocabulary than the stored claim (e.g. *"you start counting from one"*
  instead of *"the first character is at index 1"*) can be missed, because embeddings are
  nearly blind to polarity/number flips and the cue-word path needs at least one shared
  marked word. Missed misconceptions fall through to the normal probe/easier-question flow.
- A paraphrase that shares *no* content words with the concept's meaning or facts (e.g.
  *"you can find a particular letter by telling Python where it is"*) may receive a follow-up
  probe instead of immediate credit — the overlap guard that blocks false credit for vague
  answers also gates fully terminology-free paraphrases. The follow-up conversation usually
  recovers the credit.
- Automatic lecture concept extraction is structure + embedding heuristics, not semantic
  understanding of the lecture; unstructured prose notes still extract worse than notes with
  headings, and the faculty review step remains the authority. That review step is a design
  feature, not an afterthought.
- Evaluator thresholds were tuned on the calibration portion of the labelled set and checked
  on a small held-out portion (~87 of 261 items); with a dataset this size the held-out
  numbers are coarse, and none of it represents real students.
- **PDF ingestion depends on the deck having real text and a real visual hierarchy.** A deck
  whose titles are the same size and weight as its body text loses heading detection and falls
  back to prose mining; a deck with no selectable text is refused rather than guessed at (no
  OCR). Multi-column layouts, tables and text inside images are not reconstructed — a
  two-column page is read in line order across both columns.
- There is **no OCR**. A lecture that lives in images cannot be ingested at all — the system
  detects that and says so, but the teacher's only routes are an OCR-enabled export, pasting
  the text, or the prepared-note workflow. Adding OCR would mean a system binary or a large
  dependency, which this project deliberately avoids.
- The mixed/scanned tolerance is a heuristic on page counts, not on meaning: one image-only
  diagram slide in a twenty-page deck is accepted on the assumption that it is a decoration
  rather than the lesson. The review screen names the page in a prominent warning, but nothing
  forces the teacher to act on it.
- Image-heavy detection is geometry only. It cannot tell a photograph from a labelled diagram,
  so a slide with a large decorative picture is flagged alongside one whose labels carry the
  lesson — and conversely a diagram drawn with vector shapes rather than an embedded image is
  not flagged at all. A full-bleed image is always treated as a background, so a genuinely
  full-page screenshot with a caption is missed. All of this is why it is a warning and never
  a refusal.
- The boilerplate cleanup is multi-signal and conservative, which means it errs towards
  *keeping* text: a footer that appears on fewer than half the pages, or one typeset as large
  as the body text, will survive into the notes for the teacher to delete. Conversely a
  genuine one-line concept that only ever appears in the footer band would be removed.
- Concept extraction from slide decks inherits the deck's own quality. Slides that are pure
  bullet fragments with no explanatory sentence produce few or no concepts — by design, since
  a heading with no explanation is not something a student can be asked to explain back — so
  such a deck needs the teacher to add a sentence per idea, or the prepared-note workflow.
- Prose-inferred relationships now require an explicit relational cue between the two
  concepts. This trades recall for precision: genuine connections stated without such a cue
  ("Indexing. Characters. Both matter here.") are simply not suggested, and the teacher adds
  them in the review step or in an *Important Connections* section.
- The evidence rule is lexical at its core: it asks whether the answer contains
  words beyond the concept's own name, not whether those words mean the right
  thing. "gradients are used in neural networks" still passes it (two real
  content words) and is then judged on similarity alone. It removes the
  clear-cut cases, not every empty answer.
- Misconception RECALL stays low by design (precision is 1.000 on the labelled
  set, recall ~0.55). In the audit only 1 of 5 misconception phrasings was
  named as such; the rest had credit withheld rather than being flagged, which
  is the intended conservative behaviour but means a teacher will not always be
  told a misconception occurred.
- Relationship recall on deliberately colloquial phrasing is poor (2 of 7 in the
  audit) even though it is 11 of 12 on the labelled set. Prose that states a
  connection without any of the teacher's vocabulary is usually read as "not
  discussed" — an absence of evidence rather than a mistake, which is the safe
  direction, but it does mean connections go unrecorded.
- The audit's `upgrade_hit_rate` (0.091) is measured from a deliberately harsh
  baseline — every concept starts at "partial" and only promotion to "covered"
  counts — so it understates what takeaways contribute in a real session.
- The **real** problematic Cloud Computing PDF was not present in this environment, so the
  regression suite reproduces its structure (running header/footer, slide numbers, per-slide
  copyright, a divider slide, a boilerplate-only slide, a legitimately repeated title) as a
  generated fixture rather than testing that exact file. A different deck may still contain
  decoration patterns these fixtures do not cover.
- MCQ distractors come from the same lecture's material; a student who eliminates options by
  recognising which concept a sentence belongs to can sometimes answer without deep
  understanding — which is precisely why the knowledge check stays secondary to TeachBack.
- Student self-reported confidence and perceived difficulty are subjective observations, not
  ground truth; they influence which activity is recommended but never assign a learning state.
- Lecture pace/feedback responses are subjective and aggregated very simply.
- The student's takeaway summary is analyzed with the same bounded semantic matching as the
  conversation; it can only add evidence, and ambiguous summaries may add none.
- Authentication is a demo role selector, and the teacher/subject switcher is demo-level
  context switching; there is no real user management.
- Lecture file upload supports `.txt`/`.md`/`.pdf` text extraction; slide decks (`.pptx`) must
  be exported to PDF or pasted as text.

## 19. Future improvements

- Learn thresholds per concept from more labelled data; add a paraphrase-mined probe set.
- Replace self-reports with behavioural signals (response latency, edit patterns).
- Per-topic HMMs, or an input-output HMM conditioning transitions on the activity done.
- Pilot with real students and re-estimate the emission profiles from real data.
