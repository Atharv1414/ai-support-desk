from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from .data import LABELS, Example


def truth(data: list[Example]) -> np.ndarray:
    return np.array([LABELS.index(e.intent) for e in data])


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for k in range(len(LABELS)):
        tp = ((p == k) & (y == k)).sum()
        f.append(2 * tp / max((p == k).sum() + (y == k).sum(), 1))
    return float(np.mean(f))


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """One number that makes confidence honest, fitted on validation data only."""

    def nll(t):
        return -log_softmax(scores / t, axis=1)[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def ece(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, hit = prob.max(1), prob.argmax(1) == y
    edge = np.clip((conf * bins).astype(int), 0, bins - 1)
    return float(
        sum(
            abs(conf[edge == b].mean() - hit[edge == b].mean()) * (edge == b).mean()
            for b in range(bins)
            if (edge == b).any()
        )
    )


def bootstrap(y: np.ndarray, p: np.ndarray, reps: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(y), (reps, len(y)))
    return [float(v) for v in np.percentile((y[i] == p[i]).mean(1), [2.5, 97.5])]


def paired_gain(y, a, b, reps: int = 4000, seed: int = 0) -> dict[str, float]:
    """Accuracy of b minus a on the same messages, with a bootstrap interval."""
    d = (b == y).astype(float) - (a == y).astype(float)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"gain": float(d.mean()), "low": float(low), "high": float(high)}


def evaluate(scores: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict:
    prob = softmax(scores / temperature, axis=1)
    p = prob.argmax(1)
    conf = prob.max(1)
    curve = []
    for share in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(share * len(y))))
        top = np.argsort(-conf, kind="stable")[:k]
        curve.append({"handled": share, "accuracy": float((p[top] == y[top]).mean())})
    confusion = np.zeros((len(LABELS), len(LABELS)), int)
    np.add.at(confusion, (y, p), 1)
    bins = []
    for lo in np.arange(0, 1, 0.1):
        m = (conf >= lo) & (conf < lo + 0.1 + (lo > 0.85) * 1e-9)
        if m.any():
            bins.append(
                {
                    "confidence": float(conf[m].mean()),
                    "accuracy": float((p[m] == y[m]).mean()),
                    "n": int(m.sum()),
                }
            )
    return {
        "accuracy": float((p == y).mean()),
        "interval": bootstrap(y, p),
        "macro_f1": macro_f1(y, p),
        "ece": ece(prob, y),
        "temperature": temperature,
        "per_intent": {
            k: float((p[y == i] == i).mean()) for i, k in enumerate(LABELS) if (y == i).any()
        },
        "selective": curve,
        "reliability": bins,
        "confusion": confusion.tolist(),
    }
