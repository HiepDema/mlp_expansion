"""Adapter modules: MLP expansion and the LoRA baseline.

ExpandedMLP
    A gated MLP with ``rank`` extra hidden neurons.  The original neurons are
    frozen; only the new ones train::

        y = W_down(act(W_gate x) * W_up x)  +  B(act(A_gate x) * A_up x)

    This is exactly the MLP with intermediate size k + rank whose weight matrices
    are [W_gate; A_gate], [W_up; A_up] and [W_down | B] -- see ``fold()``.  The
    non-linearity comes from the MLP itself, so the adapter costs 3*d*rank
    parameters and never touches the intermediate dimension k.

LoRALinear
    Standard LoRA:  y = base(x) + (alpha/rank)(x A) B.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandedMLP(nn.Module):
    def __init__(self, mlp, rank, init="kaiming"):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = mlp
        self.rank = rank
        w_gate, w_up = mlp.gate_proj.weight, mlp.up_proj.weight
        d, dt, dv = w_gate.shape[1], w_gate.dtype, w_gate.device
        self.a_gate = nn.Parameter(torch.empty(rank, d, dtype=dt, device=dv))
        self.a_up = nn.Parameter(torch.empty(rank, d, dtype=dt, device=dv))
        self.b = nn.Parameter(torch.zeros(d, rank, dtype=dt, device=dv))
        if init == "kaiming":
            nn.init.kaiming_uniform_(self.a_gate, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.a_up, a=math.sqrt(5))
        elif init == "match":
            # New neurons drawn from the same distribution as the existing ones.
            with torch.no_grad():
                self.a_gate.normal_(0.0, w_gate.float().std().item())
                self.a_up.normal_(0.0, w_up.float().std().item())
        else:
            raise ValueError(f"unknown init {init!r}")

    @property
    def act_fn(self):
        return self.base.act_fn

    def hidden(self, x):
        """Activations of the rank new neurons."""
        return self.act_fn(F.linear(x, self.a_gate)) * F.linear(x, self.a_up)

    def forward(self, x):
        return self.base(x) + F.linear(self.hidden(x), self.b)

    @torch.no_grad()
    def fold(self):
        """Widen the base MLP in place to k + rank hidden units and return it.

        Single wider GEMMs instead of two paths: this is the deployment form.
        """
        base = self.base
        for name, extra, dim in (("gate_proj", self.a_gate, 0),
                                 ("up_proj", self.a_up, 0),
                                 ("down_proj", self.b, 1)):
            lin = getattr(base, name)
            assert lin.bias is None, "gated MLPs with biases are not supported"
            w = torch.cat([lin.weight, extra], dim=dim)
            wide = nn.Linear(w.shape[1], w.shape[0], bias=False,
                             device=w.device, dtype=w.dtype)
            wide.weight.copy_(w)
            wide.weight.requires_grad_(False)
            setattr(base, name, wide)
        if hasattr(base, "intermediate_size"):
            base.intermediate_size = base.gate_proj.out_features
        return base

    def extra_repr(self):
        return f"rank={self.rank}, k={self.base.gate_proj.out_features}"


class LoRALinear(nn.Module):
    def __init__(self, lin, rank, alpha):
        super().__init__()
        self.base = lin
        d_in, d_out = lin.in_features, lin.out_features
        dt, dv = lin.weight.dtype, lin.weight.device
        self.down = nn.Parameter(torch.empty(d_in, rank, dtype=dt, device=dv))
        self.up = nn.Parameter(torch.zeros(rank, d_out, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)
