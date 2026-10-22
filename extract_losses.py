#!/usr/bin/env python
"""Pull the loss curves out of run logs into one long-format CSV, and print val by epoch.

    python extract_losses.py [LOGDIR[:EPOCH_OFFSET] ...] [-o loss_curves.csv]

Default: runs_Qwen2.5-1.5B_bf16/logs. Several log directories can be given; an offset
shifts their epoch numbers, so a 5-epoch run plus its 5-epoch continuation become
epochs 0-9 of one run:

    python extract_losses.py runs_Qwen2.5-1.5B_bf16/logs runs_Qwen2.5-1.5B_bf16_cont/logs:5

Rows have kind = train_step (the running EMA of the label-smoothed training loss, every
--log-every steps), val_epoch (mean per-token NLL on the validation set, no smoothing) or
train_eval_epoch (the same measure on --train-eval-max training examples, so the gap to
val_epoch is an overfitting gap). train_step and val_epoch are therefore NOT on the same
scale; compare val_epoch with train_eval_epoch.
"""
import csv
import glob
import os
import re
import sys

args = sys.argv[1:]
out = "loss_curves.csv"
if "-o" in args:
    i = args.index("-o")
    out = args[i + 1]
    del args[i:i + 2]
dirs = args or ["runs_Qwen2.5-1.5B_bf16/logs"]
if "-o" not in sys.argv:
    out = os.path.join(os.path.dirname(dirs[0].split(":")[0].rstrip("/\\")), "loss_curves.csv")

TAG = re.compile(r"^=== (\S+) ===")
STEP = re.compile(r"ep (\d+) step (\d+)/(\d+)\s+loss ([\d.]+)\s+lr ([\d.e+-]+)")
VAL = re.compile(r"\[ep (\d+)\] val loss ([\d.]+)\s+ppl ([\d.]+)"
                 r"(?:\s+train ([\d.]+)\s+gap ([+-][\d.]+))?.*?(<- best)?\s*$")
FIELDS = ["run", "host", "phase", "kind", "epoch", "step", "total_steps", "loss", "lr", "best"]

rows, vals = [], {}
for spec in dirs:
    log_dir, _, off = spec.partition(":")
    off = int(off or 0)
    phase = os.path.basename(os.path.dirname(log_dir.rstrip("/\\"))) or log_dir
    for p in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        host = os.path.basename(p).split("_")[0]
        tag = None
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = TAG.match(line.strip())
                if m:
                    tag = m.group(1)
                    continue
                if tag is None:
                    continue
                m = STEP.search(line)
                if m:
                    rows.append(dict(run=tag, host=host, phase=phase, kind="train_step", epoch=int(m[1]) + off,
                                     step=int(m[2]), total_steps=int(m[3]), loss=float(m[4]), lr=float(m[5]),
                                     best=""))
                    continue
                m = VAL.search(line)
                if m:
                    ep, vl = int(m[1]) + off, float(m[2])
                    tl = float(m[4]) if m[4] else None
                    best = "best" if m[6] else ""
                    rows.append(dict(run=tag, host=host, phase=phase, kind="val_epoch", epoch=ep, step="",
                                     total_steps="", loss=vl, lr="", best=best))
                    if tl is not None:
                        rows.append(dict(run=tag, host=host, phase=phase, kind="train_eval_epoch", epoch=ep,
                                         step="", total_steps="", loss=tl, lr="", best=""))
                    vals.setdefault(tag, []).append((ep, vl, tl, best))

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows from {len(vals)} runs -> {out}")

print(f"\n{'run':<24s} {'ep':>2s} {'val':>7s} {'train':>7s} {'gap':>8s}")
for tag, v in vals.items():
    for ep, vl, tl, best in sorted(v):
        tl_s = f"{tl:7.4f}" if tl is not None else " " * 7
        gap = f"{vl - tl:+.4f}" if tl is not None else ""
        print(f"{tag:<24s} {ep:>2d} {vl:7.4f} {tl_s} {gap:>8s} {'*' if best else ''}")
    print()
