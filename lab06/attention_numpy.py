from __future__ import annotations
from typing import Tuple
import numpy as np


ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]

class ScaledDotProductAttention:
    @staticmethod
    def _validate_inputs(
        Q: np.ndarray, K: np.ndarray, V: np.ndarray
    ) -> None:
        if Q.ndim not in (1, 2):
            raise ValueError("Q must have shape (d_k,) or (n_queries, d_k).")
        if K.ndim != 2:
            raise ValueError("K must have shape (sequence_length, d_k).")
        if V.ndim != 2:
            raise ValueError("V must have shape (sequence_length, d_v).")
        if Q.shape[-1] != K.shape[-1]:
            raise ValueError(
                "Q and K must have the same last dimension (d_k)."
            )
        if K.shape[0] != V.shape[0]:
            raise ValueError(
                "K and V must contain the same number of sequence items."
            )
        if any(dimension == 0 for array in (Q, K, V) for dimension in array.shape):
            raise ValueError("Q, K and V must have non-empty dimensions.")
        if not all(np.isfinite(array).all() for array in (Q, K, V)):
            raise ValueError("Q, K and V must contain only finite values.")

    @staticmethod
    def softmax(scores: np.ndarray) -> np.ndarray:
        """Apply numerically stable softmax along the last axis."""
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 0 or scores.size == 0:
            raise ValueError("scores must be a non-empty array.")
        if not np.isfinite(scores).all():
            raise ValueError("scores must contain only finite values.")
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    @classmethod
    def scaled_dot_product_attention(
        cls, Q: np.ndarray, K: np.ndarray, V: np.ndarray
    ) -> ArrayTriplet:
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        cls._validate_inputs(Q, K, V)

        d_k = K.shape[-1]
        scores = (Q @ K.T) / np.sqrt(d_k)
        attention_weights = cls.softmax(scores)
        context = attention_weights @ V

        return scores, attention_weights, context

