"""Canonical learning states.

The five hidden states of the learner HMM, in a fixed canonical order.
Index 0..4 is used everywhere (synthetic generator, state mapping, API).
These describe the student's *current learning condition*, not a permanent
mastery label - the whole point of the HMM is that they change over time.
"""

STATE_NAMES = [
    "Not Trying",
    "Unclear",
    "Struggling but Trying",
    "Understanding",
    "Confident",
]

STATE_KEYS = ["not_trying", "unclear", "struggling", "understanding", "confident"]

NOT_TRYING, UNCLEAR, STRUGGLING, UNDERSTANDING, CONFIDENT = range(5)

# Observation feature order used everywhere (synthetic data, HMM, live analysis).
FEATURE_NAMES = [
    "concept_coverage",      # share of required concepts demonstrated (0-1)
    "semantic_correctness",  # similarity of explanation to reference explanation (0-1)
    "misconception_score",   # strength of detected misconceptions (0-1, higher = worse)
    "explanation_depth",     # structural richness of the explanation (0-1)
    "response_effort",       # length/engagement of the response (0-1)
    "attention",             # self-reported attention (0-1)
    "confidence",            # self-reported confidence (0-1)
    "difficulty",            # self-reported perceived difficulty (0-1, higher = harder)
]

# Typical emission profile per state (means for each feature above).
# These drive the synthetic data generator AND give the learned HMM states an
# interpretable anchor: after unsupervised training, each learned state is
# matched to the closest profile (see hmm/model.py).
STATE_PROFILES = {
    NOT_TRYING:    [0.05, 0.15, 0.10, 0.05, 0.06, 0.20, 0.30, 0.50],
    UNCLEAR:       [0.25, 0.35, 0.35, 0.30, 0.35, 0.50, 0.35, 0.70],
    STRUGGLING:    [0.45, 0.50, 0.30, 0.55, 0.75, 0.80, 0.40, 0.75],
    UNDERSTANDING: [0.70, 0.70, 0.12, 0.65, 0.70, 0.75, 0.65, 0.45],
    CONFIDENT:     [0.88, 0.85, 0.05, 0.80, 0.75, 0.80, 0.85, 0.25],
}

# Per-feature noise (std dev) used by the generator so observations are noisy,
# overlapping and realistic rather than trivially separable.
STATE_NOISE = [0.10, 0.10, 0.10, 0.12, 0.12, 0.13, 0.13, 0.13]
