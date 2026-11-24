"""Train, test, register, and promote only through the gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import data, metrics
from .baseline import Baseline
from .registry import ROOT, Registry


def evaluate(model, val, test) -> tuple[dict, object]:
    sv, st = model.scores([e.text for e in val]), model.scores([e.text for e in test])
    t = metrics.fit_temperature(sv, metrics.truth(val))
    start = time.perf_counter()
    model.scores([e.text for e in test[:24]])
    ms = 1000 * (time.perf_counter() - start) / 24
    m = metrics.evaluate(st, metrics.truth(test), t)
    m["validation_accuracy"] = float((sv.argmax(1) == metrics.truth(val)).mean())
    return {**m, "milliseconds": ms}, st


def run(base_model: str | None, epochs: int, registry: Registry, log=print) -> dict:
    train, val, test = data.train_set(), data.validation_set(), data.test_set()
    y = metrics.truth(test)
    prints = {
        "train_data": data.fingerprint(train),
        "validation_data": data.fingerprint(val),
        "test_data": data.fingerprint(test),
        "train_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
    }
    out: dict = {}

    def submit(version: str, kind: str, model, extra: dict) -> None:
        m, scores = evaluate(model, val, test)
        folder = registry.register(version, {"kind": kind, **prints, **extra, "metrics": m}, scores)
        if hasattr(model, "save_adapter") and extra.get("training"):
            model.save_adapter(folder / "adapter.safetensors")
        out[version] = registry.promote(version, y)
        (folder / "gate.json").write_text(json.dumps(out[version], indent=1))
        log(f"{version}: accuracy {m['accuracy']:.3f}, gate {out[version]['checks']}")

    submit("v1-tfidf", "TF-IDF and logistic regression", Baseline().train(train), {})
    if base_model:
        from .slm import SmallModel

        small = SmallModel.load(base_model)
        submit("v2-smollm-zero-shot", "SmolLM2-135M, no training", small, {})
        info = small.train(train, epochs=epochs, log=log)
        submit(
            "v3-smollm-lora",
            "SmolLM2-135M with LoRA",
            small,
            {"training": {**info, "epochs": epochs, "rank": 8}},
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="desk-train")
    p.add_argument("--base-model", help="folder or hub id of SmolLM2-135M-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--registry", type=Path, default=ROOT)
    a = p.parse_args()
    run(a.base_model, a.epochs, Registry(a.registry))


if __name__ == "__main__":
    main()
