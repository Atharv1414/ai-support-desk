"""Help-centre search: BM25 for the words, a small semantic index for the gist, fused by rank.

There is no embedding service here. The semantic side is latent semantic analysis fitted on the
articles, which is weaker than a neural encoder and costs nothing to run. The evaluation says how
much it adds.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = set(
    "a an and are as at be been but by can do does for from had has have how i if in into is "
    "it its me my of on or our so some that the their them then there this to was we what "
    "when where which who why will with you your".split()
)


# chosen on the development half of the questions (rag/evaluate.py), never on the test half
FLOOR = 0.56


def stem(word: str) -> str:
    for end in ("ing", "ies", "ed", "es", "s"):
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[: -len(end)] + ("y" if end == "ies" else "")
    return word


def words(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


@dataclass(frozen=True)
class Section:
    id: str
    article: str
    heading: str
    text: str

    @property
    def body(self) -> str:
        return f"{self.article}. {self.heading}. {self.text}"


def load_sections(folder: Path) -> list[Section]:
    out = []
    for path in sorted(folder.glob("*.md")):
        title, *parts = re.split(r"^## ", path.read_text(), flags=re.M)
        for part in parts:
            heading, text = part.split("\n", 1)
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            out.append(
                Section(
                    f"{path.stem}#{slug}",
                    title.removeprefix("# ").strip(),
                    heading.strip(),
                    " ".join(text.split()),
                )
            )
    return out


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float
    support: float  # how far the section covers the question, from 0 to about 1


class Index:
    def __init__(self, sections: list[Section], k1: float = 1.4, b: float = 0.6):
        self.sections, self.k1, self.b = sections, k1, b
        self.docs = [Counter(words(s.body)) for s in sections]
        self.len = np.array([sum(d.values()) for d in self.docs], float)
        df = Counter(w for d in self.docs for w in d)
        n = len(sections)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.unknown_idf = math.log(1 + (n + 0.5) / 0.5)
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        m = self.tfidf.fit_transform([s.body for s in sections])
        self.svd = TruncatedSVD(min(24, n - 1), random_state=0)
        self.vec = self._unit(self.svd.fit_transform(m))

    @staticmethod
    def _unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def bm25(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.len.mean())
        for w in set(words(query)):
            tf = np.array([d.get(w, 0) for d in self.docs], float)
            out += self.idf.get(w, 0.0) * tf * (self.k1 + 1) / (tf + norm)
        return out

    def semantic(self, query: str) -> np.ndarray:
        return self.vec @ self._unit(self.svd.transform(self.tfidf.transform([query])))[0]

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[Hit]:
        lists = {
            "bm25": [self.bm25(query)],
            "semantic": [self.semantic(query)],
            "hybrid": [self.bm25(query), self.semantic(query)],
        }[mode]
        fused = np.zeros(len(self.docs))
        for scores in lists:
            rank = np.empty(len(scores))
            rank[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
            fused += 1 / (20 + rank)
        order = np.argsort(-fused, kind="stable")[:k]
        q = set(words(query))
        weight = {w: self.idf.get(w, self.unknown_idf) for w in q}
        total = sum(weight.values()) or 1.0
        near = self.semantic(query)
        return [
            Hit(
                self.sections[i],
                float(fused[i]),
                float(
                    0.5 * sum(v for w, v in weight.items() if w in self.docs[i]) / total
                    + 0.5 * max(near[i], 0)
                ),
            )
            for i in order
        ]

    def answerable(self, hits: list[Hit], floor: float = FLOOR) -> bool:
        """Refuse when even the best section has too little to do with the question."""
        return bool(hits) and max(h.support for h in hits) >= floor

    def quote(self, hit: Hit, query: str, sentences: int = 2) -> str:
        """The sentences of a section that answer best, word for word."""
        parts = re.split(r"(?<=[.!?])\s+", hit.section.text)
        q = set(words(query))
        score = [sum(self.idf.get(w, 0) for w in q & set(words(p))) for p in parts]
        keep = sorted(sorted(range(len(parts)), key=lambda i: (-score[i], i))[:sentences])
        return " ".join(parts[i] for i in keep)
