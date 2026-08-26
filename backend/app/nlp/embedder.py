"""Lazy singleton around the sentence-transformers embedding model.

all-MiniLM-L6-v2 is a small pretrained sentence embedding model (not a
generative LLM). It maps sentences to 384-d vectors; cosine similarity
between vectors approximates semantic similarity.
"""
import threading

import numpy as np

from ..config import EMBEDDING_MODEL_NAME

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into L2-normalised vectors."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of a and rows of b (both normalised)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    return a @ b.T
