"""The fine-tuned small language model, used as a classifier that cannot answer off the list.

Nothing is generated. The model reads a prompt that ends just before the answer, and only the
scores of the ten label words are compared, so every answer is valid and comes with a probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from . import lora
from .data import LABELS, Example

# short on purpose: the list of topics is there so that the untrained model has a fair chance
PROMPT = "Customer message: {text}\nTopics: " + ", ".join(LABELS) + ".\nThe topic of the message is"


def device() -> str:
    return (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


class SmallModel:
    def __init__(self, model, tokenizer, dev: str = "cpu"):
        self.model, self.tok, self.dev = model.to(dev).eval(), tokenizer, dev
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        ids = [self.tok.encode(" " + w, add_special_tokens=False) for w in LABELS]
        self.label_ids = torch.tensor([i[0] for i in ids], device=dev)
        if len(set(self.label_ids.tolist())) != len(LABELS):
            raise ValueError("two labels start with the same token, so they cannot be told apart")

    @classmethod
    def load(cls, path: str | Path, dev: str | None = None) -> SmallModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = dev or device()
        # bfloat16 weights, but float32 for training on Apple GPUs
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        return cls(model, AutoTokenizer.from_pretrained(path), dev)

    def logits(self, texts: list[str]) -> torch.Tensor:
        """Scores of the label words at the last position of each prompt."""
        batch = self.tok(
            [PROMPT.format(text=t[:600]) for t in texts], return_tensors="pt", padding=True
        ).to(self.dev)
        last = batch["attention_mask"].sum(1) - 1
        hidden = self.model.model(**batch).last_hidden_state
        picked = hidden[torch.arange(len(texts), device=self.dev), last]
        # only the ten rows of the output matrix that matter, not all 49,152
        return picked @ self.model.lm_head.weight[self.label_ids].T

    @torch.no_grad()
    def scores(self, texts: list[str], batch: int = 24) -> np.ndarray:
        self.model.eval()
        parts = [
            self.logits(texts[i : i + batch]).float().cpu().numpy()
            for i in range(0, len(texts), batch)
        ]
        return np.concatenate(parts)

    def train(
        self,
        data: list[Example],
        epochs: int = 3,
        batch: int = 12,
        lr: float = 8e-4,
        r: int = 8,
        seed: int = 0,
        log=print,
    ) -> dict:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        trainable = lora.inject(self.model, r=r, alpha=2 * r)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total = epochs * (len(data) // batch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / total)
        )
        losses = []
        self.model.train()
        for _epoch in range(epochs):
            order = rng.permutation(len(data))
            for i in range(0, len(order) - batch + 1, batch):
                rows = [data[j] for j in order[i : i + batch]]
                y = torch.tensor([LABELS.index(e.intent) for e in rows], device=self.dev)
                loss = torch.nn.functional.cross_entropy(self.logits([e.text for e in rows]), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
                if len(losses) % 25 == 0:
                    log(f"step {len(losses)}/{total} loss {np.mean(losses[-25:]):.3f}")
        self.model.eval()
        return {
            "trainable": trainable,
            "parameters": sum(p.numel() for p in self.model.parameters()) - trainable,
            "steps": len(losses),
            "first_loss": float(np.mean(losses[:10])),
            "last_loss": float(np.mean(losses[-10:])),
        }

    def save_adapter(self, path: Path) -> None:
        save_file(lora.adapter_state(self.model), str(path))

    def load_adapter(self, path: Path, r: int = 8, merge: bool = True) -> None:
        lora.inject(self.model, r=r, alpha=2 * r, dropout=0.0)
        lora.load_adapter(self.model, load_file(str(path)))
        if merge:
            lora.merge(self.model)
