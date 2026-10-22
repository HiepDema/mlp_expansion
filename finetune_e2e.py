#!/usr/bin/env python
"""MLP expansion vs LoRA on E2E NLG (and DART), decoder LLMs with a gated MLP.

  --method expand   add --rank neurons to every MLP, train only those   (3*d*r / layer)
  --method lora     LoRA on gate/up/down of every MLP                   (3*r*(d+k) / layer)
  --method none     zero-shot floor

Both arms share one data pipeline, one training loop, one decoder and one scorer,
so any difference comes from the method.  Qwen2.5-1.5B (d=1536, k=8960):

  expand r :  4 608 r per layer
  lora   r : 31 488 r per layer      -> 6.83x more at the same rank

Compare at matched budget, not matched rank:  --method expand --rank 2 --match-params
picks the expansion rank whose budget equals LoRA r=2 (14 here).

  python finetune_e2e.py --method lora   --rank 2 --model Qwen/Qwen2.5-1.5B --model-dtype bf16
  python finetune_e2e.py --method expand --rank 2 --match-params --model Qwen/Qwen2.5-1.5B --model-dtype bf16
  python summarize.py runs/summary.jsonl

Scoring is the official E2E NLG Challenge toolkit (tuetschek/e2e-metrics: BLEU, NIST,
METEOR, ROUGE-L, CIDEr), run on the *_hyps.txt / *_refs.txt each evaluation writes.
``bash setup_metrics.sh`` once; the scorer is checked before any training starts.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from mlpx import (ATTN_LINEARS, MLP_LINEARS, apply_expansion, apply_lora,
                  count_params, expansion_params_per_layer,
                  expansion_rank_matching_lora, fold_expansion,
                  lora_params_per_layer, merge_lora, official)

PARQUET = "hf://datasets/tuetschek/e2e_nlg@refs%2Fconvert%2Fparquet/default"
SEP = " ||| "
DART = ("https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/"
        "dart-v1.1.1-full-{split}.json")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# =========================================================================== data

def load_e2e(cache_dir=None):
    from datasets import load_dataset
    files = {s: f"{PARQUET}/{s}/0000.parquet" for s in ("train", "validation", "test")}
    return load_dataset("parquet", data_files=files, cache_dir=cache_dir)


class _Rows(dict):
    def select(self, idx):
        idx = list(idx)
        return _Rows({k: [v[i] for i in idx] for k, v in self.items()})

    def __len__(self):
        return len(next(iter(self.values())))


def load_dart(cache_dir="dart_cache"):
    # datasets>=3 dropped script datasets, so GEM/dart cannot be loaded; fetch the raw JSON.
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for split, key in (("train", "train"), ("dev", "validation"), ("test", "test")):
        p = os.path.join(cache_dir, f"dart-{split}.json")
        if not os.path.exists(p):
            print(f"    downloading DART {split}...", flush=True)
            urllib.request.urlretrieve(DART.format(split=split), p)
        with open(p, encoding="utf-8") as f:
            rows = json.load(f)
        mr, hr = [], []
        for it in rows:
            src = " | ".join(" : ".join(t) for t in it["tripleset"])
            for ann in it["annotations"]:
                t = ann["text"].strip()
                if src and t:
                    mr.append(src)
                    hr.append(t)
        out[key] = _Rows({"meaning_representation": mr, "human_reference": hr})
    return out["train"], out["validation"], out["test"]


def load_split(name):
    if name == "dart":
        return load_dart()
    ds = load_e2e()
    return ds["train"], ds["validation"], ds["test"]


def encode_train(ds, tok, max_len):
    """'<mr> ||| <ref><eos>', loss only on <ref>."""
    out = []
    for mr, ref in zip(ds["meaning_representation"], ds["human_reference"]):
        p = tok(mr + SEP.rstrip(), add_special_tokens=False).input_ids
        c = tok(" " + ref.strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + c)[:max_len]
        labels = ([-100] * len(p) + c)[:max_len]
        if len(labels) > len(p):
            out.append((ids, labels))
    return out


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for j, (i, l) in enumerate(batch):
        ids[j, :len(i)] = torch.tensor(i)
        lab[j, :len(l)] = torch.tensor(l)
        att[j, :len(i)] = 1
    return ids, lab, att


def make_batches(data, batch):
    """Length-grouped batches: shuffle, sort within mega-batches, shuffle batch order.

    E2E lengths vary a lot (p50=58, max=122); random batching wastes 35-55% of tokens
    on padding.
    """
    idx = list(range(len(data)))
    random.shuffle(idx)
    mega, out = batch * 50, []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: len(data[j][0]))
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    random.shuffle(out)
    return out


def group_refs(ds):
    g = collections.OrderedDict()
    for mr, ref in zip(ds["meaning_representation"], ds["human_reference"]):
        g.setdefault(mr, []).append(ref.strip())
    return list(g.items())


# =========================================================================== timing

def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn, dev, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    _sync(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def bench_model(model, a, prefix=""):
    """Forward and forward+backward latency on a fixed batch, independent of decoding."""
    ids = torch.randint(0, model.config.vocab_size,
                        (a.bench_batch, a.bench_seq), device=a.device)
    ntok = a.bench_batch * a.bench_seq
    out, was_training = {}, model.training

    model.eval()
    with torch.no_grad():
        ms = timeit(lambda: model(ids), a.device, a.bench_iters)
    out[prefix + "fwd_ms"] = round(ms, 3)
    out[prefix + "fwd_ktok_s"] = round(ntok / ms, 1)

    tp = [p for p in model.parameters() if p.requires_grad]
    if tp:
        model.train()

        def step():
            model(ids, labels=ids).loss.backward()
            for p in tp:
                p.grad = None

        ms = timeit(step, a.device, max(5, a.bench_iters // 2), warmup=3)
        out[prefix + "fwdbwd_ms"] = round(ms, 3)
        out[prefix + "fwdbwd_ktok_s"] = round(ntok / ms, 1)

    model.train(was_training)
    return out


# =========================================================================== train / eval

@torch.no_grad()
def generate(model, tok, pairs, a):
    model.eval()
    tok.padding_side = "left"
    hyps = []
    t0 = time.perf_counter()
    for i in range(0, len(pairs), a.eval_batch):
        chunk = [mr + SEP.rstrip() for mr, _ in pairs[i:i + a.eval_batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(a.device)
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                             num_beams=a.beams, length_penalty=a.length_penalty,
                             no_repeat_ngram_size=4, early_stopping=True,
                             pad_token_id=tok.eos_token_id)
        for j in range(out.shape[0]):
            gen = out[j, enc.input_ids.shape[1]:]
            hyps.append(tok.decode(gen, skip_special_tokens=True).strip().replace("\n", " "))
        if i == 0 or (i // a.eval_batch) % 20 == 0:
            done = min(i + a.eval_batch, len(pairs))
            print(f"      generated {done}/{len(pairs)}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    tok.padding_side = "right"
    return hyps, time.perf_counter() - t0


def evaluate(model, tok, pairs, a, tag):
    hyps, gen_s = generate(model, tok, pairs, a)
    refs = [r for _, r in pairs]
    os.makedirs(a.out_dir, exist_ok=True)
    hp, rp = f"{a.out_dir}/{tag}_hyps.txt", f"{a.out_dir}/{tag}_refs.txt"
    with open(hp, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    with open(rp, "w", encoding="utf-8") as f:                 # e2e-metrics multi-ref format
        f.write("\n\n".join("\n".join(r) for r in refs) + "\n")
    t0 = time.perf_counter()
    sc = official.score(rp, hp, a.e2e_metrics)
    print(f"\n    [{tag}]  BLEU {sc['bleu']:.2f}  NIST {sc['nist']:.4f}  METEOR {sc['meteor']:.2f}  "
          f"ROUGE-L {sc['rouge_l']:.2f}  CIDEr {sc['cider']:.4f}   "
          f"({len(hyps)} sentences, scored in {time.perf_counter() - t0:.0f}s)")
    print(f"    sample: {hyps[0][:110]}")
    ntok = sum(len(h.split()) for h in hyps)
    print(f"    {len(hyps)} sentences in {gen_s:.0f}s "
          f"({len(hyps)/gen_s:.2f} sent/s, {ntok/gen_s:.0f} words/s)")
    return dict(**sc, n=len(hyps), gen_s=round(gen_s, 1),
                gen_sent_s=round(len(hyps)/gen_s, 3), gen_word_s=round(ntok/gen_s, 1))


@torch.no_grad()
def eval_loss(model, data, tok, a, max_n=None):
    """Mean per-token NLL, no label smoothing, so it is comparable with perplexity."""
    was = model.training
    model.eval()
    lossf = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    idx = sorted(range(len(data)), key=lambda j: len(data[j][0]))
    if max_n:
        idx = idx[:max_n]
    tot = ntok = 0
    for i in range(0, len(idx), a.eval_batch):
        ids, lab, att = collate([data[j] for j in idx[i:i + a.eval_batch]], tok.eos_token_id)
        ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
        lg = model(input_ids=ids, attention_mask=att).logits
        tgt = lab[:, 1:].reshape(-1)
        tot += lossf(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt).item()
        ntok += int((tgt != -100).sum())
    model.train(was)
    return tot / max(ntok, 1)


def trainable_state(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def save_ckpt(path, state, meta):
    torch.save({"trainable": {n: v.detach().cpu() for n, v in state.items()},
                "meta": meta}, path)
    return os.path.getsize(path) / 2**20


def load_ckpt(path, model, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cur = trainable_state(model)
    miss = set(cur) ^ set(ck["trainable"])
    if miss:
        raise ValueError(f"checkpoint differs from current config in {len(miss)} tensors: "
                         f"{sorted(miss)[:3]}... -- check --method/--rank/--attn-lora-rank")
    bad = [n for n, v in ck["trainable"].items() if tuple(v.shape) != tuple(cur[n].shape)]
    if bad:
        raise ValueError(f"checkpoint shapes differ from current config in {len(bad)} tensors, e.g. "
                         f"{bad[0]}: {tuple(ck['trainable'][bad[0]].shape)} in file vs "
                         f"{tuple(cur[bad[0]].shape)} in model -- check --rank/--attn-lora-rank")
    with torch.no_grad():
        for n, v in ck["trainable"].items():
            cur[n].copy_(v.to(device))
    m = ck.get("meta", {})
    print(f"  loaded {path}: which={m.get('which','?')} epoch={m.get('epoch','?')} "
          f"val_loss={m.get('val_loss','?')}")
    return m


def make_sched(opt, steps, warm, kind):
    # 'constant' exists because a linear schedule drives lr to 0 at the last epoch, which
    # makes val loss always fall there and hides whether training has converged.
    if kind == "constant":
        fn = lambda s: min(1.0, s / max(warm, 1))
    elif kind == "cosine":
        fn = lambda s: (s / max(warm, 1) if s < warm else
                        0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    else:
        fn = lambda s: s / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train(model, data, val_data, tok, a):
    tp = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tp, lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(math.ceil(len(data) / a.batch) / a.accum) * a.epochs
    warm = int(a.warmup_ratio * steps) if a.warmup_ratio else min(a.warmup, steps // 10)
    sched = make_sched(opt, steps, warm, a.lr_schedule)
    lossf = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.label_smoothing)

    print(f"\n  train: {len(data)} examples, {steps} steps, lr={a.lr} ({a.lr_schedule}), "
          f"batch={a.batch}x{a.accum}")
    if str(a.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model.train()
    step, t0, run = 0, time.perf_counter(), None
    val_hist, train_hist = [], []
    best, bad, stopped, val_s = (float("inf"), -1, None), 0, None, 0.0

    for ep in range(a.epochs):
        for mi, bidx in enumerate(make_batches(data, a.batch)):
            ids, lab, att = collate([data[j] for j in bidx], tok.eos_token_id)
            ids, lab, att = ids.to(a.device), lab.to(a.device), att.to(a.device)
            logits = model(input_ids=ids, attention_mask=att).logits
            loss = lossf(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1))
            (loss / a.accum).backward()
            raw = loss.item()
            if (mi + 1) % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_(tp, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run = raw if run is None else 0.98 * run + 0.02 * raw
            step += 1
            if step % a.log_every == 0:
                el = time.perf_counter() - t0 - val_s
                print(f"    ep {ep} step {step}/{steps}  loss {run:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {step / el:.2f} it/s  "
                      f"ETA {(steps - step) / max(step / el, 1e-9) / 60:.0f}m", flush=True)

        if val_data and (ep + 1) % a.val_every == 0:
            tv = time.perf_counter()
            vl = eval_loss(model, val_data, tok, a, a.val_max)
            # Train loss measured the same way (no smoothing, eval mode) so the gap means something.
            tl = eval_loss(model, data, tok, a, a.train_eval_max) if a.train_eval_max else None
            val_s += time.perf_counter() - tv
            val_hist.append(round(vl, 4))
            if tl is not None:
                train_hist.append(round(tl, 4))
            if vl < best[0] - a.min_delta:
                best = (vl, ep, {n: v.detach().clone() for n, v in trainable_state(model).items()})
                bad, mark = 0, "  <- best"
            else:
                bad += 1
                mark = f"  (no improvement {bad}/{a.patience})" if a.patience else ""
            gap = f"  train {tl:.4f}  gap {vl - tl:+.4f}" if tl is not None else ""
            print(f"    [ep {ep}] val loss {vl:.4f}  ppl {math.exp(vl):.3f}{gap}  "
                  f"({time.perf_counter() - tv:.0f}s){mark}", flush=True)
            if a.patience and bad >= a.patience:
                stopped = ep
                print(f"    EARLY STOP: best epoch {best[1]} (val {best[0]:.4f})")
                break

    el = time.perf_counter() - t0 - val_s
    vram = (torch.cuda.max_memory_allocated() / 2**30
            if str(a.device).startswith("cuda") else 0.0)

    out = {}
    meta = dict(model=a.model, method=a.method, rank=a.rank, alpha=a.alpha, init=a.init,
                attn_lora_rank=a.attn_lora_rank, seed=a.seed, epochs_run=len(val_hist))
    if not a.no_save_ckpt:
        os.makedirs(a.out_dir, exist_ok=True)
        pl = f"{a.out_dir}/{a.tag}_last.pt"
        mb = save_ckpt(pl, trainable_state(model), meta | {"which": "last"})
        out["ckpt_last"] = pl
        print(f"    saved {pl}  ({mb:.1f} MB)")
        if best[2] is not None:
            pb = f"{a.out_dir}/{a.tag}_best.pt"
            save_ckpt(pb, best[2], meta | {"which": "best", "epoch": best[1],
                                           "val_loss": round(best[0], 4)})
            out["ckpt_best"] = pb

    if val_hist:
        out.update(val_hist=val_hist, val_loss=val_hist[-1],
                   best_val=round(best[0], 4), best_epoch=best[1],
                   val_ppl=round(math.exp(val_hist[-1]), 3),
                   train_eval_hist=train_hist,
                   train_eval_loss=train_hist[-1] if train_hist else None,
                   overfit_gap=round(val_hist[-1] - train_hist[-1], 4) if train_hist else None,
                   val_s=round(val_s, 1), epochs_run=len(val_hist), stopped_epoch=stopped)
        if a.best_epoch and best[2] is not None and best[1] != len(val_hist) - 1:
            with torch.no_grad():
                cur = trainable_state(model)
                for n, v in best[2].items():
                    cur[n].copy_(v)
            print(f"    -> restored epoch {best[1]} (val {best[0]:.4f}) "
                  f"instead of last ({val_hist[-1]:.4f})")
            out["restored_epoch"] = best[1]
    return dict(**out, final_loss=run, train_s=round(el, 1), train_it_s=round(step / el, 3),
                train_step_ms=round(el / step * 1000, 2), steps_run=step,
                peak_vram_gb=round(vram, 2))


# =========================================================================== model

def build_model(a):
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM
    # from_pretrained() renamed torch_dtype -> dtype in 4.56
    dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kw = {dtype_key: DTYPES[a.model_dtype]}
    if a.attn != "auto":
        kw["attn_implementation"] = a.attn
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    n_orig = sum(p.numel() for p in model.parameters())

    if a.method == "expand":
        apply_expansion(model, a.rank, a.init)
    elif a.method == "lora":
        apply_lora(model, MLP_LINEARS, a.rank, a.alpha)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    if a.attn_lora_rank and a.method != "none":
        apply_lora(model, ATTN_LINEARS, a.attn_lora_rank, a.attn_lora_rank)

    if a.grad_ckpt:
        # Base is frozen, so checkpointed segments see no input that needs grad;
        # without enable_input_require_grads() backward silently updates nothing.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    model = model.to(a.device)
    info = count_params(model)
    info["n_orig"] = n_orig
    return model, info


def deploy(model, a):
    """Fold expansion into wider MLPs / merge LoRA into base: the inference form."""
    if a.method == "expand":
        fold_expansion(model)
    merge_lora(model, ATTN_LINEARS + MLP_LINEARS)
    return model


# =========================================================================== main

def main():
    p = argparse.ArgumentParser(description="MLP expansion vs LoRA on E2E NLG")
    p.add_argument("--method", required=True, choices=["expand", "lora", "none"])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--dataset", choices=["e2e", "dart"], default="e2e")
    p.add_argument("--rank", type=int, default=8,
                   help="expand: neurons added per MLP; lora: LoRA rank")
    p.add_argument("--match-params", action="store_true",
                   help="method=expand: replace --rank by the expansion rank whose budget "
                        "matches LoRA at --rank on gate/up/down")
    p.add_argument("--alpha", type=float, default=None, help="LoRA alpha, default = rank")
    p.add_argument("--init", choices=["kaiming", "match"], default="kaiming",
                   help="expand: kaiming (like LoRA's A) or match the base weights' std")
    p.add_argument("--attn-lora-rank", type=int, default=0,
                   help="also LoRA q/k/v/o at this rank, in BOTH arms (0 = MLP only)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--lr-schedule", choices=["linear", "constant", "cosine"], default="linear")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--model-dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--attn", default="auto",
                   choices=["auto", "sdpa", "flash_attention_2", "eager"])
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--beams", type=int, default=10)
    p.add_argument("--length-penalty", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--eval-before", action="store_true")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-max", type=int, default=None)
    p.add_argument("--train-eval-max", type=int, default=1000)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--best-epoch", action="store_true",
                   help="restore the lowest-val-loss epoch before the final eval")
    p.add_argument("--load-ckpt", default=None, help="score a saved *_best.pt / *_last.pt, no training")
    p.add_argument("--resume-ckpt", default=None,
                   help="load the trainable weights of a saved *_last.pt / *_best.pt and keep "
                        "training for --epochs more (optimizer state restarts; pick --lr / "
                        "--lr-schedule / --warmup for the continuation)")
    p.add_argument("--e2e-metrics", default=None,
                   help="clone of tuetschek/e2e-metrics (default: $E2E_METRICS or ./e2e-metrics)")
    p.add_argument("--bench-batch", type=int, default=8)
    p.add_argument("--bench-seq", type=int, default=128)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.patience and not a.best_epoch:
        a.best_epoch = True

    # Fail here, not after an hour of training.
    a.e2e_metrics = official.check(a.e2e_metrics)
    print(f"  official scorer OK: {a.e2e_metrics}")

    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model)
    L = cfg.num_hidden_layers
    lora_rank = a.rank
    if a.method == "expand" and a.match_params:
        a.rank = expansion_rank_matching_lora(cfg, lora_rank)
        print(f"  --match-params: LoRA r={lora_rank} on gate/up/down = "
              f"{lora_params_per_layer(cfg, MLP_LINEARS, lora_rank) * L / 1e6:.3f} M"
              f"  ->  expand r={a.rank} = "
              f"{expansion_params_per_layer(cfg, a.rank) * L / 1e6:.3f} M")
    if a.method == "none":
        a.rank = 0
    if a.alpha is None:
        a.alpha = a.rank

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tag = f"{a.method}_r{a.rank}"
    if a.method == "expand" and a.init != "kaiming":
        tag += f"_{a.init}"
    if a.method == "lora" and a.alpha != a.rank:
        tag += f"_a{a.alpha:g}"
    if a.attn_lora_rank:
        tag += f"_attn{a.attn_lora_rank}"
    tag += f"_s{a.seed}"
    a.tag = tag
    print(f"\n=== {tag} ===\nmodel={a.model}  device={a.device}  dtype={a.model_dtype}")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  loading {a.dataset}...")
    tr_rows, va_rows, te_rows = load_split(a.dataset)
    if a.max_train:
        tr_rows = tr_rows.select(range(min(a.max_train, len(tr_rows))))
    data = encode_train(tr_rows, tok, a.max_len)
    val_data = None if a.no_val else encode_train(va_rows, tok, a.max_len)
    pairs = group_refs(te_rows)
    if a.limit_eval:
        pairs = pairs[:a.limit_eval]
    print(f"  train {len(data)} | val {len(val_data) if val_data else 0} "
          f"| test {len(pairs)} unique inputs")

    t0 = time.perf_counter()
    model, info = build_model(a)
    t_setup = time.perf_counter() - t0
    print(f"\n  TRAINABLE: {info['trainable'] / 1e6:.3f} M "
          f"({100 * info['trainable'] / info['n_orig']:.3f} % of {info['n_orig'] / 1e6:.2f} M)"
          f"   [setup {t_setup:.1f}s]")

    res = dict(tag=tag, config=vars(a) | {"device": str(a.device)}, params=info,
               setup_s=round(t_setup, 1))
    if a.resume_ckpt:
        m = load_ckpt(a.resume_ckpt, model, a.device)
        res["resumed_from"] = {k: m.get(k) for k in ("which", "epoch", "val_loss")} | {"path": a.resume_ckpt}
    if a.eval_before:
        res["before"] = evaluate(model, tok, pairs, a, tag + "_before")
    if a.load_ckpt:
        m = load_ckpt(a.load_ckpt, model, a.device)
        res["loaded"] = {k: m.get(k) for k in ("which", "epoch", "val_loss")} | {"path": a.load_ckpt}
    elif a.method != "none":
        res["train"] = train(model, data, val_data, tok, a)
        res["final_loss"] = res["train"]["final_loss"]
    res["after"] = evaluate(model, tok, pairs, a, tag)

    if not a.no_bench:
        print(f"\n  latency (batch {a.bench_batch} x seq {a.bench_seq}):")
        res["bench"] = bench_model(model, a)
        if a.method != "none":
            deploy(model, a)
            res["bench"].update(bench_model(model, a, prefix="deployed_"))
            res["bench"]["deployed_total"] = count_params(model)["total"]
        b = res["bench"]
        print(f"    forward            {b['fwd_ms']:8.2f} ms  ({b['fwd_ktok_s']:.1f} k tok/s)")
        if "fwdbwd_ms" in b:
            print(f"    forward+backward   {b['fwdbwd_ms']:8.2f} ms")
        if "deployed_fwd_ms" in b:
            ov = 100 * (b["deployed_fwd_ms"] / b["fwd_ms"] - 1)
            print(f"    deployed forward   {b['deployed_fwd_ms']:8.2f} ms   ({ov:+.1f} % vs "
                  f"training form; {b['deployed_total'] / 1e6:.2f} M params after "
                  f"{'fold' if a.method == 'expand' else 'merge'})")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    with open(f"{a.out_dir}/summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(tag=tag, method=a.method, rank=a.rank, alpha=a.alpha,
                                init=a.init, attn_lora_rank=a.attn_lora_rank, seed=a.seed,
                                trainable=info["trainable"], model=a.model,
                                dataset=a.dataset, dtype=a.model_dtype, lr=a.lr,
                                lr_schedule=a.lr_schedule, epochs_planned=a.epochs,
                                setup_s=res["setup_s"],
                                **res["after"], **res.get("train", {}),
                                **res.get("bench", {}),
                                **({"loaded": res["loaded"]} if "loaded" in res else {}),
                                **({"resumed_from": res["resumed_from"]} if "resumed_from" in res else {}))) + "\n")
    print(f"\n  results -> {a.out_dir}/{tag}.json  (+ one line in summary.jsonl)")


if __name__ == "__main__":
    main()
