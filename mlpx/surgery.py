"""Model surgery: inject adapters, fold / merge them back, count parameters.

Everything operates in place.  Typical use::

    apply_expansion(model, rank=14)          # or apply_lora(model, MLP_LINEARS, 2, 2)
    # ... train ...
    fold_expansion(model)                    # or merge_lora(model, MLP_LINEARS)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mlpx.adapters import ExpandedMLP, LoRALinear

ATTN_LINEARS = ["self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj"]
MLP_LINEARS = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def get_blocks(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"cannot find transformer blocks in {type(model).__name__}")


def resolve(root, path):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_adapters(model):
    for m in model.modules():
        if isinstance(m, ExpandedMLP):
            for p in (m.a_gate, m.a_up, m.b):
                p.requires_grad_(True)
        elif isinstance(m, LoRALinear):
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)


# ------------------------------------------------------------------ inject

def apply_expansion(model, rank, init="kaiming"):
    for block in get_blocks(model):
        block.mlp = ExpandedMLP(block.mlp, rank, init)
    freeze_all(model)
    unfreeze_adapters(model)
    return model


def apply_lora(model, targets, rank, alpha):
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha))
    freeze_all(model)
    unfreeze_adapters(model)
    return model


# ------------------------------------------------------------------ fold / merge

@torch.no_grad()
def fold_expansion(model):
    """Replace every ExpandedMLP by its wider plain MLP and update the config."""
    new_k = None
    for block in get_blocks(model):
        if isinstance(block.mlp, ExpandedMLP):
            block.mlp = block.mlp.fold()
            new_k = block.mlp.gate_proj.out_features
    if new_k is not None:
        model.config.intermediate_size = new_k
    return model


@torch.no_grad()
def merge_lora(model, targets):
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, LoRALinear):
                continue
            delta = mod.scale * (mod.down.double() @ mod.up.double())
            mod.base.weight += delta.T.to(mod.base.weight.dtype)
            setattr(parent, attr, mod.base)
    return model


def merge_all(model):
    fold_expansion(model)
    return merge_lora(model, ATTN_LINEARS + MLP_LINEARS)


# ------------------------------------------------------------------ counting

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return dict(trainable=trainable, total=total)


def linear_shapes(config):
    """(out, in) of every adaptable linear in one block, keyed by target name."""
    d = config.hidden_size
    k = config.intermediate_size
    hd = getattr(config, "head_dim", None) or d // config.num_attention_heads
    kv = (getattr(config, "num_key_value_heads", None) or config.num_attention_heads) * hd
    return {"self_attn.q_proj": (config.num_attention_heads * hd, d),
            "self_attn.k_proj": (kv, d),
            "self_attn.v_proj": (kv, d),
            "self_attn.o_proj": (d, config.num_attention_heads * hd),
            "mlp.gate_proj": (k, d),
            "mlp.up_proj": (k, d),
            "mlp.down_proj": (d, k)}


def lora_params_per_layer(config, targets, rank):
    sh = linear_shapes(config)
    return rank * sum(sum(sh[t]) for t in targets)


def expansion_params_per_layer(config, rank):
    return 3 * config.hidden_size * rank


def expansion_rank_matching_lora(config, lora_rank):
    """Expansion rank whose MLP budget is closest to LoRA at lora_rank on the 3 MLP linears."""
    target = lora_params_per_layer(config, MLP_LINEARS, lora_rank)
    return max(1, round(target / expansion_params_per_layer(config, 1)))
