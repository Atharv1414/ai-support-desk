"""Retrieval quality and refusal, on questions split into a half to tune on and a half to report."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from .index import FLOOR, Index, load_sections

DATA = Path(__file__).resolve().parents[3] / "data"


def questions(half: str) -> list[dict]:
    with (DATA / "questions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    def side(r):
        # by section, so a section's wording cannot help to tune and then be reported on
        key = r["section"] or r["question"]
        return "dev" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 else "test"

    return [r for r in rows if side(r) == half]


def refusal(index: Index, rows: list[dict], floor: float) -> tuple[float, float]:
    best = [(bool(r["section"]), max(h.support for h in index.search(r["question"]))) for r in rows]
    known = [s < floor for k, s in best if k]
    unknown = [s < floor for k, s in best if not k]
    return float(np.mean(unknown)), float(np.mean(known))


def tune(index: Index) -> float:
    """The floor with the best balance of refusing the unknown and answering the known."""
    rows = questions("dev")
    grid = np.round(np.arange(0.1, 0.8, 0.01), 2)
    value = [r - w for r, w in (refusal(index, rows, f) for f in grid)]
    top = np.flatnonzero(np.isclose(value, max(value)))
    return float(grid[top[len(top) // 2]])  # the middle of the best range


def run() -> dict:
    index = Index(load_sections(DATA / "articles"))
    out: dict = {"sections": len(index.sections), "floor": FLOOR, "tuned_floor": tune(index)}
    for half in ("dev", "test"):
        rows = questions(half)
        known = [r for r in rows if r["section"]]
        part: dict = {"questions": len(known), "unanswerable": len(rows) - len(known)}
        for mode in ("bm25", "semantic", "hybrid"):
            ranks = []
            for r in known:
                ids = [h.section.id for h in index.search(r["question"], 5, mode)]
                ranks.append(ids.index(r["section"]) + 1 if r["section"] in ids else None)
            part[mode] = {
                "recall@1": sum(x == 1 for x in ranks) / len(ranks),
                "recall@3": sum(x is not None and x <= 3 for x in ranks) / len(ranks),
                "mrr": sum(1 / x for x in ranks if x) / len(ranks),
            }
        part["refuses_unanswerable"], part["refuses_answerable"] = refusal(index, rows, FLOOR)
        out[half] = part
    return out
