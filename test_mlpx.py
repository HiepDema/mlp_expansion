"""Correctness checks on a tiny random Qwen2 -- no weights downloaded, runs on CPU.

    python test_mlpx.py
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from mlpx import (ATTN_LINEARS, MLP_LINEARS, ExpandedMLP, apply_expansion,
                  apply_lora, count_params, expansion_params_per_layer,
                  expansion_rank_matching_lora, fold_expansion, get_blocks,
                  lora_params_per_layer, merge_lora)

torch.manual_seed(0)
D, K, L, R = 64, 96, 2, 5


def tiny():
    cfg = AutoConfig.for_model("qwen2", hidden_size=D, intermediate_size=K,
                               num_hidden_layers=L, num_attention_heads=4,
                               num_key_value_heads=2, vocab_size=256,
                               max_position_embeddings=64)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model, cfg


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def perturb(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1 * torch.randn_like(p))


ids = torch.randint(0, 256, (2, 7))

# --- expansion: identity at init, param count, fold equivalence, gradient routing
model, cfg = tiny()
ref = logits(model, ids)
apply_expansion(model, R)
assert torch.allclose(logits(model, ids), ref, atol=1e-6), "B=0 must leave the model unchanged"
n = count_params(model)["trainable"]
assert n == L * expansion_params_per_layer(cfg, R) == L * 3 * D * R, n
assert all(isinstance(b.mlp, ExpandedMLP) for b in get_blocks(model))

perturb(model)
before = logits(model, ids)
assert not torch.allclose(before, ref, atol=1e-3), "perturbed adapter must change the output"
model.train()
model(ids, labels=ids).loss.backward()
for name, p in model.named_parameters():
    has = p.grad is not None and p.grad.abs().sum() > 0
    assert has == p.requires_grad, f"gradient routing wrong at {name}"
model.zero_grad()
model.eval()

fold_expansion(model)
assert model.config.intermediate_size == K + R
assert all(b.mlp.gate_proj.out_features == K + R for b in get_blocks(model))
assert all(type(b.mlp).__name__ == "Qwen2MLP" for b in get_blocks(model))
assert count_params(model)["trainable"] == 0
assert torch.allclose(logits(model, ids), before, atol=1e-5), "fold must be exact"

# --- init=match draws from the base weights' scale
model, _ = tiny()
apply_expansion(model, R, init="match")
m = get_blocks(model)[0].mlp
assert abs(m.a_gate.std().item() / m.base.gate_proj.weight.std().item() - 1) < 0.5

# --- LoRA baseline: identity at init, param count, merge equivalence
model, cfg = tiny()
ref = logits(model, ids)
apply_lora(model, MLP_LINEARS, 3, 3)
assert torch.allclose(logits(model, ids), ref, atol=1e-6)
n = count_params(model)["trainable"]
assert n == L * lora_params_per_layer(cfg, MLP_LINEARS, 3) == L * 3 * 3 * (D + K), n
perturb(model)
before = logits(model, ids)
merge_lora(model, MLP_LINEARS)
assert count_params(model)["trainable"] == 0
assert torch.allclose(logits(model, ids), before, atol=1e-5), "merge must be exact"

# --- attention LoRA on top of expansion
model, cfg = tiny()
apply_expansion(model, R)
apply_lora(model, ATTN_LINEARS, 2, 2)
n = count_params(model)["trainable"]
assert n == L * (expansion_params_per_layer(cfg, R)
                 + lora_params_per_layer(cfg, ATTN_LINEARS, 2)), n

# --- budget matching on the real Qwen2.5-1.5B shape
q = AutoConfig.for_model("qwen2", hidden_size=1536, intermediate_size=8960,
                         num_hidden_layers=28, num_attention_heads=12,
                         num_key_value_heads=2)
assert lora_params_per_layer(q, MLP_LINEARS, 1) == 31488
assert expansion_params_per_layer(q, 1) == 4608
assert lora_params_per_layer(q, ATTN_LINEARS, 1) == 9728
assert expansion_rank_matching_lora(q, 2) == 14
assert expansion_rank_matching_lora(q, 4) == 27
assert expansion_rank_matching_lora(q, 8) == 55

print("all checks passed")
