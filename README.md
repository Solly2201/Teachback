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
- **Concept coverage** — best cosine similarity between any student sentence and each concept
  description (full credit ≥ 0.62, partial ≥ 0.56). Simple and informal wording counts:
  *"It tells us how much the error changes when we change a weight"* is valid evidence for
  the gradient concept.
- **Misconception detection** — a sentence is flagged only if it is similar enough to the
  wrong claim (≥ 0.65) **and** closer to the wrong claim than to the correct clarification
  (margin 0.08). This keeps correct statements about the same subject from being flagged.
- **Semantic correctness** — cosine similarity of the whole response to the reference
  explanation, rescaled to 0–1.
- **Explanation depth / response effort** — structural richness and length measures.
- **Concept relationships** — each topic carries a small teacher-authored list of
  relationships (e.g. *Backpropagation → computes → Gradient*), each stored as one correct
  sentence and, where a natural error exists, one wrong version. A student sentence
  demonstrates a relationship when similar enough to the correct sentence. Because embeddings
  are nearly blind to polarity flips ("reduces the loss" vs "increases the loss" differ by
  ~0.002 cosine), contradictions are detected with **cue words derived automatically from the
  teacher-authored pair** — and only on sentences that are semantically about that
  relationship, so a cue word alone never triggers a flag. Relationship evidence accumulates
  across the conversation (it is *not* added to the HMM feature vector, so the trained
  artifact stays valid).

### Guided conversation (`nlp/conversation.py`)

A deterministic rule engine — not a language model — walks through the concepts one **short
question** at a time and adapts:

- clearly correct → acknowledge, move on (concepts demonstrated in passing are skipped);
- **short confirmation of a yes/no question** ("yes") → positive evidence, not full credit:
  *"Right — can you explain what that means in your own words?"* — but "yes" to an open
  question earns nothing, and "I don't know" is never inflated (a content-word-overlap guard
  blocks empty answers from the contextual scorer);
- partly right → one targeted probe;
- unclear → one easier question;
- misconception → explain the distinction, probe once, and mark it **resolved** if the next
  answer no longer contains it.

The session ends when every concept has been visited — the 12-question limit is a safety cap,
not a target. Depth is never confused with understanding: no advanced questions are asked
unless the teacher configured them (lecture-published topics have no extension question at
all), and lecture-mode questions are conversational (*"What did you understand about X?"*).

### Lecture preparation (`nlp/lecture_prep.py`)

Deterministic extraction, no LLM: sentence segmentation → contiguous content-word n-grams
(verbs and note-taking noise rejected) → scored by frequency × embedding centrality to the
whole document, boosted for phrases named in the title/objectives → embedding-based
deduplication → per-concept "possible explanation" (the closest lecture sentence, verbatim) →
relationship suggestions from sentences mentioning two concepts → objective templates.
Everything is a *suggestion*; the faculty review screen exists precisely because this is
heuristic keyphrase extraction, not semantic understanding of the lecture.

### Student summary (`Your takeaway`)

After the conversation the student writes what they personally took away. It is stored with
the session, shown on Progress, and analyzed as an **upgrade-only** evidence source: it can
add or strengthen concept/relationship evidence but can never lower anything — a short summary
is never a penalty.

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
authentication — a documented limitation). The selected subject scopes the Lecture TeachBacks
and Topic Management pages, so Prof. Arjun Rao's *Python Programming* view never shows
Prof. Meera Krishnan's *Neural Networks* topics, and vice versa. The student topic chooser is
grouped by subject.

## 13. Demo accounts & data

- **Teachers:** Prof. Meera Krishnan (*Neural Networks*: Backpropagation, Overfitting and
  Regularization, Hidden Markov Models — fully hand-authored topics) and Prof. Arjun Rao
  (*Python Programming*: Python Basics).
- **Students:** 9 named demo students (including **Shreshtha Bindal · B.Tech CE · B023**, the
  primary demo student) plus background students with seeded histories.
- **Python Basics — Variables, Data Types, and Basic Operations** is seeded through the *real*
  lecture pipeline: the stored lecture record contains the raw material, the untouched NLP
  suggestions (`Variables, Python, Values, Operator, …`) and the teacher-reviewed draft
  (`Variables, Assignment, Data types, Operators, Expressions`) that was published. It exists
  precisely to show the system is generic and not hardcoded around Backpropagation — there is
  no Python-specific code anywhere.

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
python -m pytest tests -q      # 76 tests: NLP, conversation, lectures, feedback,
                               # recommendations, activities, HMM, API end-to-end
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
   the **Python Basics** lecture on the Lecture TeachBacks page: the NLP suggestions and the
   reviewed concepts are both visible. Repeat a quick TeachBack on Python Basics as a student
   to prove the whole pipeline is generic. The Class Overview shows the lecture-feedback
   aggregates (pace, requests, confidence, difficulty).

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
│   ├── nlp_eval/            hand-labelled responses for NLP evaluation
│   └── artifacts/           trained HMM, state mapping, evaluation results
├── tests/                   pytest suite (76 tests)
└── README.md
```

Python: FastAPI, SQLAlchemy, sentence-transformers, hmmlearn, SciPy, NumPy, pypdf.
Frontend: React 18, Vite, Tailwind. Storage: SQLite. **No LLM APIs anywhere.**

## 17. Evaluation methodology & results

Numbers below are produced by `scripts/build_all.py` and stored in
`data/artifacts/evaluation_results.json` — recomputed on every build, not hard-coded.

**NLP** — evaluated on 24 hand-labelled student-style responses (120 concept pairs,
84 misconception pairs):

| Task | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Concept detection | 0.767 | 0.767 | 0.767 |
| Misconception detection | 0.90 | 0.90 | 0.90 |

**HMM** — students split 80/20; trained on 160 students, Viterbi-decoded on the 40 held-out
students against the generator's true states:

| Metric | Value |
|--------|-------|
| State accuracy (297 test sessions) | 0.976 |
| Adjacent-state accuracy (off by ≤ 1) | 1.000 |

**The HMM was evaluated using synthetic student trajectories and should not be interpreted as
validated real-world student prediction.**

## 18. Known limitations

The system estimates **evidence of conceptual understanding within a bounded, teacher-reviewed
lecture context** — it does not directly measure understanding, and it does not know what is
happening in a student's mind.

- NLP features are similarity heuristics against teacher-authored/reviewed text; paraphrases
  far from the stored descriptions can be missed, and thresholds were tuned on the same small
  labelled set they are evaluated on.
- Semantic correctness saturates for fluent on-topic text even when partially wrong.
- The HMM is trained **and evaluated on synthetic trajectories**; its high accuracy reflects
  recovery of the generating process, not validated performance on real students.
- Relationship contradiction detection relies on cue words from the teacher-authored wrong
  version; contradictions phrased with entirely different vocabulary (or negation, e.g.
  "does not decrease") can be missed. Analogies with no shared content words fall back to an
  easier follow-up question rather than being credited immediately.
- Misconception detection needs the wrong belief to be phrased somewhat like the stored claim,
  and lecture-preparation misconception suggestions only draw on misconceptions a teacher has
  already authored somewhere in the system.
- Automatic lecture concept extraction is frequency + embedding-centrality heuristics, not
  semantic understanding of the lecture; it regularly needs the faculty review step to remove
  noisy phrases, rename concepts, or add missing ones. That review step is a design feature,
  not an afterthought.
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
