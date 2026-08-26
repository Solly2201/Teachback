# TeachBack

**Teach it back. We'll listen.**

TeachBack is a browser-based learning prototype in which students demonstrate understanding by
*teaching a topic back* in their own words, instead of rating themselves on a 1–10 scale.
The system analyzes each explanation with NLP, asks rule-based follow-up questions, and uses a
**Hidden Markov Model** to track how each student's *learning state* evolves across sessions —
then recommends the next activity accordingly.

---

## 1. Problem statement

Self-assessment ("How well did you understand this? 8/10") is unreliable: students routinely
over- or under-estimate themselves, and a single rating carries no information about *what*
they misunderstood. At the same time, classifying students with static labels ("weak",
"strong") is both inaccurate and unfair — understanding fluctuates from week to week.

## 2. Objective

Build a working prototype that:

1. collects genuine evidence of understanding (a free-text teach-back explanation),
2. extracts interpretable signals from it with real NLP techniques (no generative LLM),
3. models the student's learning condition **over time** with an HMM rather than labelling
   them once, and
4. adapts the next activity to the estimated state with transparent rules.

## 3. How TeachBack works

```text
                    STUDENT
                       |
                       v
              TeachBack interface  ("Teach me this concept as if I've never learned it")
                       |
                       v
                      NLP  (sentence embeddings)
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
    Concepts     Misconceptions   Semantic quality,
    detected       detected       depth, effort
        |              |              |
        +--------------+--------------+
                       |            + self-reports (attention, confidence, difficulty)
                       v
              Observation vector (8 features)
                       |
                       v
                      HMM  (5 hidden learning states, Viterbi over the whole history)
                       |
                       v
              Current learning state
                       |
                       v
          Rule-based adaptive recommendation
```

## 4. Learning states

The HMM has five hidden states, in a fixed canonical order:

| # | State | Typical evidence |
|---|-------|------------------|
| 0 | Not Trying | minimal responses, low attention |
| 1 | Unclear | short, inaccurate explanations, frequent misconceptions |
| 2 | Struggling but Trying | high effort and attention, but gaps and errors |
| 3 | Understanding | mostly accurate, decent coverage |
| 4 | Confident | accurate, complete, high self-confidence |

These are **snapshots of the current learning condition, not mastery levels**. A student can
move `Understanding → Unclear → Struggling but Trying`; modelling those temporal transitions
is exactly why an HMM is used instead of a per-session classifier.

## 5. NLP component (`backend/app/nlp/`)

The system is deliberately **bounded**: a teacher defines each topic with required concepts,
known misconceptions (each stored as the *wrong claim* plus a *correct clarification*), and a
short reference explanation. Student responses are then evaluated **against that definition**.

- Sentences are embedded with the pretrained **sentence-transformers** model
  `all-MiniLM-L6-v2` (a non-generative embedding model).
- **Concept coverage** — best cosine similarity between any student sentence and each concept
  description (full credit ≥ 0.62, partial ≥ 0.56).
- **Misconception detection** — a sentence is flagged only if it is similar enough to the
  wrong claim (≥ 0.65) **and** closer to the wrong claim than to the correct clarification
  (margin 0.08). This keeps correct statements about the same subject from being flagged.
- **Semantic correctness** — cosine similarity of the whole response to the reference
  explanation, rescaled to 0–1.
- **Explanation depth / response effort** — structural richness and length measures.

Thresholds were tuned on the labelled evaluation set (`data/nlp_eval/labeled_responses.json`).
All five outputs are heuristic 0–1 *observation features* for the HMM — they are not claimed
to be objective measurements of human understanding.

**Guided conversational teach-back** is driven by a deterministic rule engine
(`nlp/conversation.py`), *not* a language model. Instead of asking for one long essay, the
session walks through the topic's concepts one **short question** at a time (a progress
timeline shows where the student is). Each concept has a teacher-authored question bank:
a main question, an easier fallback, a deeper probe, and an application question. Fixed
rules judge each answer and pick the next move: clearly correct → encourage and move on
(concepts demonstrated in passing are skipped); partly right → one targeted probe; unclear
→ one easier question; misconception detected → explain the distinction and probe it — if
the next answer no longer contains it, the misconception is marked **resolved**. Because
short answers like "It tells us how the loss changes" rely on the question for context,
the current concept is additionally scored with the concept name prefixed to the answer,
gated by a content-word-overlap check so empty answers ("I don't know") are not inflated.
Every follow-up carries a machine-generated "why this question" explanation, and all
feedback phrases are fixed templates chosen deterministically.

## 6. HMM component (`backend/app/hmm/`)

A `GaussianHMM` (hmmlearn, 5 states, diagonal covariance) is trained **unsupervised** on the
synthetic observation sequences.

**State interpretation.** Unsupervised HMM state IDs are arbitrary — learned state "2" means
nothing by itself. Each learned state's emission mean is therefore matched one-to-one to the
closest canonical state profile (`app/states.py`) with the Hungarian algorithm; the mapping
and its distances are saved to `data/artifacts/hmm_state_mapping.json` so the assignment is
auditable rather than asserted.

**Inference.** When a student finishes a session, Viterbi decoding runs over their *entire*
observation history, so the estimated current state depends on the trajectory, not just the
latest session — and earlier states may be re-interpreted in light of new evidence.

## 7. Dataset construction (`backend/app/hmm/synthetic.py`)

No real educational dataset is required. A synthetic dataset of **200 students × 5–10
sessions** is generated from five learner archetypes (fast learner, hardworking struggler,
disengaged, inconsistent, strong), each defined by an initial state distribution and a Markov
transition matrix. Observations are drawn from per-state emission profiles with Gaussian
noise, so e.g. an "Understanding" student still occasionally produces a weak answer or
reports low confidence. The true hidden states are kept for evaluation.

The demo database is seeded the same way: the 32 visible students get short generated
observation histories (marked `source = "seed"`) so the dashboards start populated. Their
displayed states still come from real HMM inference over those observations; sessions you
complete in the UI are stored as `source = "live"`.

## 8. Adaptive recommendation (`backend/app/recommend/`)

A transparent rule table: each state maps to an activity style (re-engagement → analogy /
basic review → guided worked example → application task → edge-case challenge). If the topic
has a teacher-authored activity targeting the state it is used; otherwise a generic fallback.
Freshly detected misconceptions attach a "revisit this point first" note.

## 9. Architecture & technology stack

```text
teachback/
├── backend/            FastAPI + SQLAlchemy (SQLite)
│   ├── app/
│   │   ├── api/        REST endpoints (students, topics, sessions, teacher, meta)
│   │   ├── nlp/        embeddings, analyzer, guided conversation engine
│   │   ├── hmm/        synthetic generator, HMM training/mapping/inference
│   │   ├── recommend/  rule-based recommendations
│   │   ├── evaluation/ NLP + HMM evaluation
│   │   ├── models.py   ORM: students, topics, concepts, misconceptions,
│   │   │               activities, sessions, responses, observations
│   │   └── seed*.py    topic definitions + database seeding
│   └── scripts/build_all.py
├── frontend/           React + Vite + Tailwind CSS
│   └── src/pages/      Student Dashboard, TeachBack, Progress,
│                       Teacher Dashboard, Topic Management
├── data/
│   ├── synthetic/      generated dataset (JSON + CSV)
│   ├── nlp_eval/       hand-labelled responses for NLP evaluation
│   └── artifacts/      trained HMM, state mapping, evaluation results
└── tests/              pytest suite (NLP, HMM, recommendations, API)
```

Python: FastAPI, SQLAlchemy, sentence-transformers, hmmlearn, SciPy, NumPy.
Frontend: React 18, Vite, Tailwind. Storage: SQLite. No LLM APIs anywhere.

## 10. How to run

Prerequisites: Python 3.11+, Node 18+.

```bash
# 1) backend deps
cd teachback/backend
pip install -r requirements.txt

# 2) build everything: synthetic dataset -> HMM evaluation -> final HMM ->
#    seeded SQLite DB -> NLP evaluation  (first run downloads the ~90 MB embedding model)
python scripts/build_all.py --force-seed

# 3) start the API
python -m uvicorn app.main:app --port 8000

# 4) frontend (new terminal)
cd teachback/frontend
npm install
npm run dev        # http://localhost:5173  (dev server proxies /api to :8000)
```

Demo flow (~3 minutes): open the app → *Continue as Student* → pick *Aarav Shah*
(Struggling but Trying) → *TeachBack* → choose *Backpropagation* → answer the short
questions in one or two sentences (try answering one with "backpropagation and gradient
descent are the same thing", then correct it when probed) → watch the concept timeline
fill in → fill the three self-report sliders → the summary shows what was demonstrated,
what needs clarification, the resolved misconception, the learning journey
(previous state → new state) and the explained recommendation → *Progress* shows the
state trajectory with per-session evidence → switch to Teacher for the class overview.

### Regenerating data / retraining

`python scripts/build_all.py --force-seed` regenerates the dataset, retrains and re-evaluates
both models, and resets the database (all seeded from fixed random seeds, so results are
reproducible).

### Running tests

```bash
cd teachback
python -m pytest tests -q      # 25 tests: NLP, dialogue, generator, HMM, rules, API E2E
```

### Production frontend build

```bash
cd teachback/frontend && npm run build     # outputs to frontend/dist
npm run preview                            # serves the build at :4173 (proxies /api to :8000)
```

## 11. Evaluation methodology & results

Numbers below are produced by `scripts/build_all.py` and stored in
`data/artifacts/evaluation_results.json` — they are recomputed on every build, not hard-coded.

**NLP** — evaluated on 24 hand-labelled student-style responses (120 concept pairs,
84 misconception pairs):

| Task | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Concept detection | 0.767 | 0.767 | 0.767 |
| Misconception detection | 0.90 | 0.90 | 0.90 |

**HMM** — students split 80/20; the model is trained on 160 students and Viterbi-decodes the
40 held-out students' sequences against the generator's true states:

| Metric | Value |
|--------|-------|
| State accuracy (297 test sessions) | 0.976 |
| Adjacent-state accuracy (off by ≤ 1) | 1.000 |

A full confusion matrix and per-state precision/recall are in the evaluation artifact and are
summarized at the bottom of the Teacher Dashboard.

## 12. Known limitations

- The NLP features are similarity heuristics against teacher-authored text; paraphrases far
  from the stored descriptions can be missed, and the thresholds were tuned on the same small
  labelled set they are evaluated on.
- Semantic correctness saturates for fluent on-topic text even when partially wrong.
- The HMM is trained on synthetic trajectories; its high accuracy reflects recovery of the
  generating process, not validated performance on real students.
- Misconception detection needs the wrong belief to be phrased somewhat like the stored claim.
- Authentication is a demo role selector; there is no real user management.

## 13. Future improvements

- Learn thresholds per concept from more labelled data; add a paraphrase-mined probe set.
- Replace self-reports with behavioural signals (response latency, edit patterns).
- Per-topic HMMs, or an input-output HMM conditioning transitions on the activity done.
- Pilot with real students and re-estimate the emission profiles from real data.
