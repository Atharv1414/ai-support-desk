"""LoRA (Hu et al., 2021), written out: freeze the model, learn a low-rank change to some layers."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x. B starts at zero, so training starts from the original model."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(
            torch.empty(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> nn.Linear:
        """An ordinary linear layer with the change folded in, for serving at no extra cost."""
        out = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            self.base.bias is not None,
            device=self.A.device,
            dtype=self.A.dtype,
        )
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out


def _targets(model: nn.Module, names: tuple[str, ...], kind: type) -> list[tuple[nn.Module, str]]:
    return [
        (parent, name)
        for parent in model.modules()
        for name, child in parent.named_children()
        if name in names and isinstance(child, kind)
    ]


def inject(
    model: nn.Module,
    names: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> int:
    """Wrap every matching layer and freeze everything else. Returns the trainable count."""
    model.requires_grad_(False)
    targets = _targets(model, names, nn.Linear)
    if not targets:
        raise ValueError(f"no linear layers named {names}")
    for parent, name in targets:
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha, dropout))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().cpu().contiguous()
        for n, p in model.named_parameters()
        if n.endswith((".A", ".B")) and p.requires_grad
    }


def load_adapter(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = {n: p for n, p in model.named_parameters() if n.endswith((".A", ".B"))}
    if set(own) != set(state):
        raise ValueError("the adapter does not match this model's LoRA layers")
    with torch.no_grad():
        for n, p in own.items():
            p.copy_(state[n].to(p.device, p.dtype))


def merge(model: nn.Module, names: tuple[str, ...] = ("q_proj", "v_proj")) -> None:
    for parent, name in _targets(model, names, LoRALinear):
        setattr(parent, name, getattr(parent, name).merged())
