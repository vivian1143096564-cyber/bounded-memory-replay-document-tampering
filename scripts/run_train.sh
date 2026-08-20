#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the parent directory containing RTM/, DocTamper_dataset/, DocTamper_official_code/, and DocTamper_pretrained_weights/}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN="$PKG_ROOT/code/train_dtd_rtm_xpu.py"
SPLIT="$PKG_ROOT/protocol/perceptual_group_split.json"
HARD_NEG="$PKG_ROOT/protocol/hard_negatives.json"
WEIGHT="$DATA_ROOT/DocTamper_pretrained_weights/dtd_doctamper.pth"
OUT_ROOT="${OUT_ROOT:-$PKG_ROOT/runs}"
METHOD="${1:?Usage: run_train.sh METHOD SEED; METHOD=direct|lwf|ewc|joint|replay}"
SEED="${2:-20260811}"
ARGS=(--output "$OUT_ROOT/${METHOD}_${SEED}" --epochs 2 --batch-size 2 --workers 2 --max-train-steps 300 --lr 1e-6 --weight-decay 5e-4 --seed "$SEED" --resume "$WEIGHT" --split-file "$SPLIT" --balanced-mix --tamper-class-weight 10 --negative-penalty-weight 0.02 --hard-negative-file "$HARD_NEG" --hard-negative-prob 0.20 --hard-background-weight 0.02 --hard-background-fraction 0.002 --far-selection-weight 0.5)
case "$METHOD" in
  direct) ARGS+=(--extra-target-batch) ;;
  lwf) ARGS+=(--lwf-weight 1.0 --lwf-temperature 2.0) ;;
  ewc) ARGS+=(--ewc-weight 1000 --ewc-samples 200) ;;
  joint) ARGS+=(--doctamper-replay-weight 1.0 --doctamper-replay-every 1 --doctamper-replay-samples 0) ;;
  replay) ARGS+=(--doctamper-replay-weight 1.0 --doctamper-replay-every 1 --doctamper-replay-samples 2000) ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
"$PYTHON_BIN" "$TRAIN" "${ARGS[@]}"
