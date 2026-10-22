# MLP expansion

Parameter-efficient fine-tuning of a gated MLP by **adding neurons** instead of adding a
low-rank correction.

For a SwiGLU block `y = W_down (σ(W_gate x) ⊙ W_up x)` with `d` inputs and `k` hidden
units, stack `A_gate, A_up ∈ R^{r×d}` under `W_gate, W_up` and `B ∈ R^{d×r}` beside
`W_down`. The block becomes an ordinary MLP with `k + r` hidden units whose output splits
exactly into the frozen original plus an adapter path that reuses the MLP's own
non-linearity:

```
y' = W_down(σ(W_gate x) ⊙ W_up x)  +  B(σ(A_gate x) ⊙ A_up x)
```

Three things follow:

- **Cost is `3·d·r` per layer and never touches `k`.** LoRA on the same three matrices
  costs `3·r·(d+k)`. On Qwen2.5-1.5B (`d=1536, k=8960`) that is 4 608 r against 31 488 r,
  a factor 6.83 at the same rank.
- **The adapter is non-linear and gated for free**: `A_gate` gates the `r` new neurons the
  same way `W_gate` gates the original `k`.
- **It does not merge.** The deployed model is the wider MLP (`intermediate_size + r`),
  so it keeps its `3·d·r` extra weights. `fold_expansion()` produces that model;
  `finetune_e2e.py` times it against the merged LoRA.

The FFN part of this idea appears in DimA (Zhang et al., LREC-COLING 2024) on RoBERTa and
GPT-2 with a plain GELU MLP. This repo is the gated / SwiGLU case on decoder LLMs, with a
matched-budget comparison against LoRA.

## Install

```bash
pip install torch transformers datasets
bash setup_metrics.sh                   # e2e-metrics deps + Java check + a two-sentence scoring test
```

Every number the pipeline reports is from the **official E2E NLG Challenge scorer**,
[tuetschek/e2e-metrics](https://github.com/tuetschek/e2e-metrics), cloned at
`./e2e-metrics` (git-ignored): BLEU and NIST via its mteval-v13a port (`measure_scores.py
-p`), METEOR / ROUGE-L / CIDEr via its MS-COCO scripts behind the Stanford PTB tokenizer.
It runs on the `*_hyps.txt` / `*_refs.txt` each evaluation writes, takes ~15 s per 630
sentences, and is checked at start-up, before any training. BLEU, METEOR and ROUGE-L are
reported x100, as in the LoRA paper's Table 3, so the rows are directly comparable with it.
`--e2e-metrics` / `$E2E_METRICS` point to a clone elsewhere.

The PTB tokenizer and METEOR are Java (1.8+): on Ubuntu `sudo apt-get install -y
default-jre-headless`, or without root `conda install -c conda-forge openjdk`; a JRE that
is only in `$JAVA_HOME` is picked up too. There is no in-repo reimplementation of any
metric on purpose: on two earlier runs the hand-written ROUGE-L sat 1.2 points below the
official one.

## Run

```bash
python test_mlpx.py                     # correctness on a tiny random Qwen2, CPU

# LoRA r=2 on gate/up/down:          1.763 M trainable
python finetune_e2e.py --method lora   --rank 2 --model Qwen/Qwen2.5-1.5B --model-dtype bf16 --best-epoch

# expansion at the same budget:      r=14, 1.806 M trainable
python finetune_e2e.py --method expand --rank 2 --match-params --model Qwen/Qwen2.5-1.5B --model-dtype bf16 --best-epoch

# expansion at the same rank:        r=2, 0.258 M trainable
python finetune_e2e.py --method expand --rank 2 --model Qwen/Qwen2.5-1.5B --model-dtype bf16 --best-epoch

bash run_compare.sh                     # lora r=4, expand r=4 (same rank), expand r=27 (same budget),
                                        # each with LoRA r=4 on q/k/v/o as well (ATTN=0 turns that off)
python summarize.py runs_Qwen2.5-1.5B_bf16/summary.jsonl
```

`--resume-ckpt runs/<tag>_last.pt --epochs 5` keeps training a finished run for five more
epochs (weights only; the optimizer restarts, so give it `--warmup`). For a convergence
check pair it with `--lr-schedule constant`, because the default linear schedule drives
the lr to 0 at the last epoch and makes the val loss dip there regardless.

`--attn-lora-rank R` adds LoRA on q/k/v/o at rank `R` in *both* arms, so the comparison
stays about the MLP; `run_compare.sh` sets it to the MLP rank. On Qwen2.5-1.5B that is
9 728 r per layer, 1.090 M at r=4, the same in every run, and `--match-params` still
matches the MLP part only. `--init match` draws the new neurons from the base weights'
distribution instead of Kaiming. `--dataset dart` switches to DART (raise `--max-len`).

## Files

| File | Purpose |
|------|---------|
| `mlpx/adapters.py` | `ExpandedMLP` (with `fold()`), `LoRALinear` |
| `mlpx/surgery.py` | inject / fold / merge, parameter counting, budget matching |
| `mlpx/official.py` | runs `measure_scores.py` from e2e-metrics and parses the five official scores |
| `setup_metrics.sh` | fetches e2e-metrics, installs its Python deps, proves Java and the scorer work |
| `finetune_e2e.py` | E2E / DART pipeline: length-grouped batching, per-epoch validation, best-epoch selection, beam search, latency bench |
| `summarize.py` | aggregates `summary.jsonl` across seeds |
| `extract_losses.py` | parses run logs into `loss_curves.csv` (train EMA per 100 steps, val and train-eval loss per epoch) |
| `test_mlpx.py` | identity at init, exact fold / merge, gradient routing, parameter formulas |
