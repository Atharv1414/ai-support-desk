"""Model versions on disk, a champion, and a gate that decides who may become one."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics

ROOT = Path(__file__).resolve().parents[3] / "registry"


@dataclass(frozen=True)
class Gate:
    min_gain_low: float = 0.0  # lower end of the paired interval must be above this
    max_intent_drop: float = 0.17  # no intent may get worse by more than 2 of its 12 messages
    max_ece: float = 0.12

    def check(self, y, champion: dict, challenger: dict, p_old, p_new) -> dict:
        gain = metrics.paired_gain(y, p_old, p_new)
        drops = {k: champion["per_intent"][k] - v for k, v in challenger["per_intent"].items()}
        worst = max(drops, key=drops.get)
        checks = {
            "more accurate, beyond chance": gain["low"] > self.min_gain_low,
            "no intent much worse": drops[worst] <= self.max_intent_drop + 1e-9,
            "confidence is honest": challenger["ece"] <= self.max_ece,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "gain": gain,
            "worst_intent": {"intent": worst, "drop": drops[worst]},
            "ece": challenger["ece"],
        }


class Registry:
    def __init__(self, root: Path = ROOT):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    @property
    def _file(self) -> Path:
        return self.root / "stages.json"

    def stages(self) -> dict:
        if not self._file.exists():
            return {"champion": None, "history": []}
        return json.loads(self._file.read_text())

    def _write(self, data: dict) -> None:
        # written beside and moved into place, so a crash cannot leave half a file
        fd, tmp = tempfile.mkstemp(dir=self.root)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self._file)

    def register(self, version: str, manifest: dict, scores: np.ndarray) -> Path:
        folder = self.root / version
        if (folder / "manifest.json").exists():
            raise FileExistsError(f"{version} is already registered, and versions never change")
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "test_scores.npy", scores.astype("float32"))
        (folder / "manifest.json").write_text(
            json.dumps({"version": version, **manifest}, indent=1)
        )
        return folder

    def manifest(self, version: str) -> dict:
        return json.loads((self.root / version / "manifest.json").read_text())

    def versions(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/manifest.json"))

    def promote(self, version: str, y: np.ndarray, gate: Gate | None = None) -> dict:
        """Make `version` champion if it passes the gate. The first model has nothing to beat."""
        s, new = self.stages(), self.manifest(version)
        if s["champion"] is None:
            report = {"passed": True, "checks": {"first model": True}}
        else:
            old = self.manifest(s["champion"])
            if old["test_data"] != new["test_data"]:
                raise ValueError("the two models were tested on different data")

            def load(v):
                return np.load(self.root / v / "test_scores.npy").argmax(1)

            report = (gate or Gate()).check(
                y, old["metrics"], new["metrics"], load(s["champion"]), load(version)
            )
        if report["passed"]:
            s["history"].append({"from": s["champion"], "to": version})
            s["champion"] = version
            self._write(s)
        return report

    def rollback(self) -> str:
        s = self.stages()
        if not s["history"] or s["history"][-1]["from"] is None:
            raise ValueError("there is no earlier champion to go back to")
        last = s["history"].pop()
        s["champion"] = last["from"]
        self._write(s)
        return s["champion"]
