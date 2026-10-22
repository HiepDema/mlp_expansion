#!/usr/bin/env python
"""Read summary.jsonl and print a comparison table sorted by trainable parameters."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "runs/summary.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

g = defaultdict(list)
for r in rows:
    init = r.get("init", "kaiming") if r["method"] == "expand" else "-"
    g[(r.get("model", "?"), r.get("dataset", "e2e"), r.get("dtype", "?"), r["method"],
       r["rank"], init, r.get("attn_lora_rank", 0))].append(r)


def ms(v, f, w=6, p=2):
    x = [r[f] for r in v if r.get(f) is not None]
    if not x:
        return " " * (w + 8)
    m = sum(x) / len(x)
    sd = (sum((y - m) ** 2 for y in x) / (len(x) - 1)) ** .5 if len(x) > 1 else 0.0
    return f"{m:{w}.{p}f} +/-{sd:5.{p}f}"


def one(v, f, w=8, p=2):
    x = [r[f] for r in v if r.get(f) is not None]
    return f"{sum(x)/len(x):{w}.{p}f}" if x else " " * w


for key in sorted({k[:3] for k in g}):
    ks = sorted([k for k in g if k[:3] == key], key=lambda k: g[k][0]["trainable"])
    print(f"\n{'=' * 100}\n{key[0]}  {key[1]}  {key[2]}")
    print(f"\nQUALITY (official e2e-metrics; BLEU/METEOR/ROUGE-L x100)\n"
          f"  {'method':<8s} {'r':>3s} {'attn':>4s} {'init':>7s} {'params':>8s} {'n':>2s} "
          f"{'BLEU':>14s} {'NIST':>14s} {'METEOR':>14s} {'ROUGE-L':>14s} {'CIDEr':>14s} "
          f"{'val loss':>9s} {'ep':>3s}")
    print("  " + "-" * 130)
    for k in ks:
        v = g[k]
        print(f"  {k[3]:<8s} {k[4]:>3d} {k[6]:>4d} {k[5]:>7s} {v[0]['trainable']/1e6:7.3f}M {len(v):>2d} "
              f"{ms(v,'bleu')} {ms(v,'nist',6,4)} {ms(v,'meteor')} {ms(v,'rouge_l')} {ms(v,'cider',6,4)} "
              f"{one(v,'best_val',9,4)} {one(v,'best_epoch',3,0)}")

    if any("train_s" in r for k in ks for r in g[k]):
        print(f"\nCOST\n  {'method':<8s} {'r':>3s} {'params':>8s} {'train_s':>8s} {'it/s':>6s} "
              f"{'VRAM_GB':>8s} {'fwd_ms':>7s} {'deploy_ms':>9s} {'deploy_M':>9s}")
        print("  " + "-" * 80)
        for k in ks:
            v = g[k]
            dep = one(v, 'deployed_total', 9, 2)
            if dep.strip():
                dep = f"{float(dep) / 1e6:9.2f}"
            print(f"  {k[3]:<8s} {k[4]:>3d} {v[0]['trainable']/1e6:7.3f}M "
                  f"{one(v,'train_s',8,0)} {one(v,'train_it_s',6,2)} {one(v,'peak_vram_gb',8,2)} "
                  f"{one(v,'fwd_ms',7,2)} {one(v,'deployed_fwd_ms',9,2)} {dep}")

    for k in ks:
        h = g[k][0].get("val_hist")
        if h:
            be = g[k][0].get("best_epoch")
            trend = " -> ".join(("*" if i == be else "") + f"{x:.4f}" for i, x in enumerate(h))
            print(f"\n  val by epoch [{k[3]} r={k[4]}]: {trend}   (* = best)")

print("\n'deploy_ms' is the forward after fold (expand) or merge (lora); 'deploy_M' the parameter"
      " count of that deployed model -- expansion keeps its 3*d*r extra weights, LoRA does not.")
