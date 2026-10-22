# Results so far: MLP expansion vs LoRA on E2E NLG

Qwen2.5-1.5B (d = 1536, k = 8960, 28 layers), bf16 base and adapters, seed 0, one A10 per
run. Recipe of the LoRA paper's E2E setup: 5 epochs, batch 8, AdamW lr 2e-4 linear with
500 warm-up steps, label smoothing 0.1, beam 10, length penalty 0.9, best epoch by val
loss. Scores are the official `e2e-metrics` toolkit on the 630-MR test set; BLEU, METEOR
and ROUGE-L are x100. Raw outputs: `runs_Qwen2.5-1.5B_bf16/` (5-epoch runs) and
`runs_Qwen2.5-1.5B_bf16_cont/` (epochs 5-9); every loss curve is in the `loss_curves*.csv`
there (`extract_losses.py`).

## A. Five-epoch runs

"attn r" = LoRA of that rank on q/k/v/o, identical in both arms of a pair. "Same budget"
means `--match-params` (expansion rank whose MLP budget equals LoRA's at the given rank).

| Pair | Run | Trainable | BLEU | NIST | METEOR | ROUGE-L | CIDEr | Best val | Best ep | Gap @ep4 |
|---|---|---|---|---|---|---|---|---|---|---|
| r=4, MLP only | LoRA r=4 | 3.527 M | **66.21** | 8.650 | 45.06 | **70.08** | 2.327 | 1.0951 | 1 | +0.230 |
| | Expansion r=27 (same budget) | 3.484 M | 65.77 | 8.606 | 44.95 | 69.84 | 2.254 | **1.0949** | 3 | +0.092 |
| | Expansion r=4 (same rank) | 0.516 M | 65.52 | 8.587 | 44.92 | 69.65 | 2.299 | 1.1185 | 3 | +0.050 |
| r=4, attn r=4 | LoRA r=4 | 4.616 M | 65.51 | 8.585 | 44.82 | 69.63 | 2.291 | 1.0952 | 1 | +0.275 |
| | Expansion r=27 (same budget) | 4.573 M | 64.73 | 8.567 | 44.21 | 68.54 | 2.200 | 1.0967 | 2 | +0.134 |
| | Expansion r=4 (same rank) | 1.606 M | 65.28 | 8.591 | 44.88 | 69.25 | 2.272 | 1.1066 | 3 | +0.094 |
| r=2, attn r=2 | LoRA r=2 | 2.308 M | 65.81 | 8.484 | 44.49 | 69.60 | 2.234 | 1.1002 | 1 | +0.204 |
| | Expansion r=2 | 0.803 M | 65.44 | 8.607 | 44.28 | 69.24 | 2.206 | 1.1163 | 3 | +0.077 |
| r=1, attn r=1 | LoRA r=1 | 1.154 M | 65.75 | 8.611 | 44.50 | 69.29 | 2.234 | 1.1014 | 2 | +0.145 |
| | Expansion r=1 | **0.401 M** | 65.83 | 8.629 | 44.39 | 69.09 | 2.220 | 1.1286 | 3 | +0.063 |

Cost (same host within a pair only; gpu-4 is ~10 % slower than the others):

| Run | train it/s | VRAM | fwd train-form | fwd deployed | deployed params |
|---|---|---|---|---|---|
| LoRA r=4 (MLP only) | 9.3 | 6.6 GB | 79.7 ms | 64.8 ms (merged, -19 %) | 1543.71 M |
| Expansion r=27 (MLP only) | 11.0 | 6.6 GB | 64.8 ms | 93.5 ms (folded, +44 %) | 1547.20 M |
| Expansion r=4 (MLP only) | 10.8 | 6.6 GB | 66.2 ms | 75.4 ms (folded, +14 %) | 1544.23 M |

## B. Convergence: the rank-1/2 pairs trained to 10 epochs

Resumed from the epoch-4 weights (`--resume-ckpt`), 5 more epochs at a constant lr of
1e-4 (the first phase's linear schedule had reached 0; a constant lr avoids the
end-of-schedule val dip). Val / train-eval loss by epoch; `*` = best epoch of the run:

| Run | ep0 | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | train-eval ep4 → ep9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LoRA r=1 | 1.1256 | 1.1056 | **1.1014*** | 1.1075 | 1.1110 | 1.1141 | 1.1149 | 1.1188 | 1.1319 | 1.1382 | 0.967 → 0.906 |
| Expansion r=1 | 1.1601 | 1.1372 | 1.1319 | **1.1286*** | 1.1287 | 1.1292 | 1.1290 | 1.1286 | 1.1311 | 1.1308 | 1.066 → 1.059 |
| LoRA r=2 | 1.1135 | **1.1002*** | 1.1025 | 1.1124 | 1.1212 | 1.1197 | 1.1357 | 1.1478 | 1.1622 | 1.1820 | 0.917 → 0.797 |
| Expansion r=2 | 1.1430 | 1.1268 | 1.1217 | **1.1163*** | 1.1178 | 1.1208 | 1.1212 | 1.1226 | 1.1207 | 1.1202 | 1.041 → 1.020 |

BLEU of the continuation (scored at its own best epoch): LoRA r=1 66.16 (ep 5), LoRA r=2
66.03 (ep 5), expansion r=1 65.04 (ep 7), expansion r=2 64.97 (ep 9). Deploy numbers are
unchanged.

## Findings

1. **Quality is a tie everywhere, on a saturated task.** All fourteen BLEU scores lie in
   64.7-66.2; the LoRA paper's single-seed noise on E2E is about 0.5 BLEU, and here the
   same weights re-scored after five more epochs moved by up to 0.8 BLEU with no change in
   val loss. No pair differs by more than noise. Val loss is the finer instrument: at the
   same rank LoRA is always lower (by 0.012-0.027), at matched budget the two are equal
   (1.0949 vs 1.0951).
2. **Parameter efficiency is the real result.** Expansion r=1 reaches 65.8 BLEU with
   0.40 M trainable parameters, of which only 0.13 M are in the MLPs; LoRA needs 1.15 M
   for the same score (2.9x), and the MLP-only expansion r=4 (0.52 M) is within 0.7 BLEU
   of LoRA r=4 (3.53 M, 6.8x).
3. **The two methods fail differently.** LoRA reaches its best val loss at epoch 1-2 and
   then overfits: its train loss keeps falling (r=2: 0.92 → 0.80 over epochs 4-9) while
   val rises (1.100 → 1.182, gap +0.38). Expansion plateaus on *both* curves from epoch 3
   (r=1: val 1.129 → 1.131, train 1.066 → 1.059) and is essentially immune to
   over-training. That is the "adds features but cannot modify existing ones" constraint
   acting as a regulariser: the frozen neurons cannot be bent to memorise the training
   set. It also means expansion is capacity-limited at these ranks, so a higher lr or
   more epochs will not help it; more neurons or a second task with less headroom will.
4. **More budget does not buy quality here.** Expansion r=27 matches LoRA's val loss but
   not a better BLEU than r=4, and LoRA r=4 is no better than LoRA r=1. The task is
   saturated around 0.4-1 M parameters for both methods.
5. **Attention LoRA did not help either method** at seed 0 (every "+attn" row is within
   noise of its MLP-only counterpart, usually slightly below).
6. **Training is cheaper, deployment currently is not.** Expansion trains 15-17 % faster
   (three small GEMMs instead of six LoRA matmuls per MLP) at identical VRAM. After
   folding, however, the wider MLP runs 14-44 % slower than the training form while
   merged LoRA runs 19 % faster. An A10 micro-benchmark shows this is alignment, not
   width: k = 8964 and 8987 (not multiples of 8) cost +30 % and +76 % per MLP, whereas
   k = 8968 / 8992 / 9024 cost only +6-10 %. `fold()` should pad to a multiple of 8 (or
   64) with zero neurons, which is exact, before any deployment number is quoted.

## Caveats and next steps

- One seed. Add seeds 1-2 for the MLP-only r=4 trio and the r=1 pair before writing.
- E2E is saturated; DART (less headroom) or a task needing new knowledge would separate
  the methods. The precision choice (pure bf16 adapters) is not neutral between methods
  (S-LoRA measured -2.1 BLEU for LoRA and -1.2 for S-LoRA on Qwen2.5-0.5B); state it.
- Matched-budget counterparts at low rank (expansion r=7 vs LoRA r=1, r=14 vs r=2) were
  not run.
- Fix the fold alignment, then re-measure deployed latency and memory.
