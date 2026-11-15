"""Healing: recover quality after pruning with a short low-rank fine-tune.

The edited model (student) is trained to reproduce the base model's (teacher's)
next-token distribution on a few dozen prompts. Only small low-rank adapters
(LoRA) on the block weights are trained, and they are merged back into the
weights afterwards, so the healed model has the same shape as the edited one.

No dataset is needed: the teacher's own predictions are the training signal.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
from torch import nn

from xray.models.adapters import LoadedModel

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """`base(x) + scale * B(A(x))`, with only A and B trainable."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight.data += (self.b @ self.a * self.scale).to(self.base.weight.dtype)
        return self.base


def _wrap(lm: LoadedModel, rank: int, alpha: float) -> list[tuple[nn.Module, str, LoRALinear]]:
    wrapped = []
    for layer in lm.refs.layers:
        for parent in (layer.attn, layer.mlp):
            for name in TARGETS:
                child = getattr(parent, name, None)
                if isinstance(child, nn.Linear):
                    lora = LoRALinear(child, rank, alpha)
                    setattr(parent, name, lora)
                    wrapped.append((parent, name, lora))
    return wrapped


def heal(
    lm: LoadedModel,
    encoded: list[torch.Tensor],
    teacher: list[torch.Tensor],
    steps: int = 150,
    rank: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
    progress=lambda *a, **k: None,
) -> dict[str, Any]:
    """Distil the teacher's predictions into the edited model. Needs a float32 model."""
    if lm.dtype != torch.float32:
        raise ValueError("Healing trains the model, which needs float32 weights.")
    for p in lm.model.parameters():
        p.requires_grad_(False)
    wrapped = _wrap(lm, rank, alpha=2.0 * rank)
    params = [p for _, _, lora in wrapped for p in (lora.a, lora.b)]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order: list[int] = []
    losses = []
    try:
        for step in range(steps):
            if step % 10 == 0:
                progress("healing", step, steps)
            if not order:
                order = list(range(len(encoded)))
                rng.shuffle(order)
            i = order.pop()
            ids = encoded[i].to(lm.device)
            target = torch.log_softmax(teacher[i].to(lm.device, torch.float32), dim=-1)
            student = torch.log_softmax(lm.model(input_ids=ids).logits[0], dim=-1)
            loss = (target.exp() * (target - student)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
    finally:
        for parent, name, lora in wrapped:
            setattr(parent, name, lora.merged())
        lm.model.zero_grad(set_to_none=True)
        # Merging put plain Linear modules back; the adapter must see them again.
        from xray.patch import _refresh

        _refresh(lm)
    head = losses[: max(1, len(losses) // 10)]
    tail = losses[-max(1, len(losses) // 10) :]
    return {
        "steps": len(losses),
        "rank": rank,
        "learning_rate": lr,
        "trained_params": sum(p.numel() for p in params),
        "train_prompts": len(encoded),
        "loss_start": sum(head) / len(head),
        "loss_end": sum(tail) / len(tail),
    }
