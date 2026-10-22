#!/bin/bash
# MLP expansion vs LoRA on E2E NLG, Qwen2.5-1.5B -- three runs, two comparisons.
# Every run also puts LoRA r=ATTN (default: same as R) on q/k/v/o, identical in both
# arms, so the comparison stays about the MLP:
#
#                        MLP part    attn LoRA r=4    total
#   lora   r=4           3.527 M       1.090 M       4.616 M   reference
#   expand r=4           0.516 M       1.090 M       1.606 M   same MLP rank   (6.83x fewer MLP params)
#   expand r=27          3.484 M       1.090 M       4.573 M   same MLP budget (--match-params picks 27 from r=4)
#
# Per layer, Qwen2.5-1.5B (d=1536, k=8960):
#   expand r = 4 608 r      lora on gate/up/down r = 31 488 r      lora on q/k/v/o r = 9 728 r
#
#   bash run_compare.sh                # bf16 (default)
#   DT=fp32 bash run_compare.sh
#   R=2 bash run_compare.sh            # another rank: expansion follows (2 -> 14), attn LoRA too
#   ATTN=0 bash run_compare.sh         # MLP adapters only, nothing on attention
#   EXTRA="--max-train 16 --limit-eval 3 --epochs 1" MODEL=path/to/tiny bash run_compare.sh
#                                      # smoke test: EXTRA is appended to every run
set -u
PY="${PY:-python}"
DT="${DT:-bf16}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
SEED="${SEED:-0}"
R="${R:-4}"
ATTN="${ATTN:-$R}"
OUT="${OUT:-runs_$(basename "$MODEL")_${DT}}"
EXTRA="${EXTRA:-}"

COMMON="--model $MODEL --epochs 5 --best-epoch --seed $SEED --attn sdpa
        --attn-lora-rank $ATTN --eval-batch 32 --model-dtype $DT --out-dir $OUT $EXTRA"

run () {
  echo
  echo "=================================================================="
  echo "==> $DT  $*"
  echo "=================================================================="
  "$PY" -u finetune_e2e.py $COMMON "$@"
  echo "RUN_DONE $*"
}

run --method lora   --rank "$R"
run --method expand --rank "$R"
run --method expand --rank "$R" --match-params

echo "ALL_DONE"
"$PY" summarize.py "$OUT/summary.jsonl"
