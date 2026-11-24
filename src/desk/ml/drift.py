"""Has the incoming mail changed? Three cheap signals that need no labels."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy.special import softmax

from .data import LABELS

WARN, ALARM = 0.10, 0.25  # the usual bands for the population stability index


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-3) -> float:
    e = np.maximum(expected / expected.sum(), eps)
    a = np.maximum(actual / actual.sum(), eps)
    e, a = e / e.sum(), a / a.sum()
    return float(((a - e) * np.log(a / e)).sum())


def vocabulary(texts: list[str]) -> set[str]:
    return {w for t in texts for w in re.findall(r"[a-z]+", t.lower())}


def new_word_rate(texts: list[str], known: set[str]) -> float:
    words = [w for t in texts for w in re.findall(r"[a-z]+", t.lower())]
    return sum(w not in known for w in words) / max(len(words), 1)


class Monitor:
    def __init__(self, scores: np.ndarray, texts: list[str], temperature: float = 1.0):
        self.t = temperature
        prob = softmax(scores / temperature, axis=1)
        self.mix = np.bincount(prob.argmax(1), minlength=len(LABELS)).astype(float)
        self.edges = np.quantile(prob.max(1), np.linspace(0, 1, 6)[1:-1])
        self.conf = self._bins(prob.max(1))
        self.known = vocabulary(texts)
        self.base_new = new_word_rate(texts, self._others(texts))

    @staticmethod
    def _others(texts: list[str]) -> set[str]:
        """Words used in at least two messages: the rest would be new to somebody."""
        c = Counter(w for t in texts for w in set(re.findall(r"[a-z]+", t.lower())))
        return {w for w, n in c.items() if n > 1}

    def _bins(self, conf: np.ndarray) -> np.ndarray:
        return np.bincount(np.searchsorted(self.edges, conf), minlength=5).astype(float)

    def check(self, scores: np.ndarray, texts: list[str]) -> dict:
        prob = softmax(scores / self.t, axis=1)
        mix = psi(self.mix, np.bincount(prob.argmax(1), minlength=len(LABELS)).astype(float))
        conf = psi(self.conf, self._bins(prob.max(1)))
        new = new_word_rate(texts, self.known)
        worst = max(mix, conf)
        return {
            "intent_psi": mix,
            "confidence_psi": conf,
            "new_words": new,
            "usual_new_words": self.base_new,
            "state": "alarm" if worst >= ALARM else "warn" if worst >= WARN else "ok",
        }
