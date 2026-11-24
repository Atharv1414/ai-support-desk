"""TF-IDF and logistic regression: the model to beat, and the one that serves when no GPU does."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import LABELS, Example


class Baseline:
    def __init__(self, pipe: Pipeline | None = None):
        self.pipe = pipe or Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            (
                                "chars",
                                TfidfVectorizer(
                                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
                                ),
                            ),
                        ]
                    ),
                ),
                ("model", LogisticRegression(C=10, max_iter=2000)),
            ]
        )

    def train(self, data: list[Example]) -> Baseline:
        self.pipe.fit([e.text for e in data], [LABELS.index(e.intent) for e in data])
        return self

    def scores(self, texts: list[str]) -> np.ndarray:
        return self.pipe.decision_function(texts)
