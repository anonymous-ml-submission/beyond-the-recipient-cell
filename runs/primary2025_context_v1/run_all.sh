#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1

BIO="${BIO:?Set BIO to the original Python environment path}"
ENTRY="scripts/train_primary2025_context_v1.py"
ROOT="runs/primary2025_context_v1"

export PYTHONUNBUFFERED=1

for CTX in 24 64 128; do

    RD="$ROOT/A_raw_ctx${CTX}"
    CFG="$RD/config.yaml"
    OUT="$RD/train"
    LOG="$RD/train.log"

    if [ -e "$OUT" ]; then
        echo "STOP: output already exists: $OUT"
        exit 1
    fi

    echo
    echo "======================================================================"
    echo "START A_raw_ctx${CTX}"
    date
    echo "======================================================================"

    "$BIO" "$ENTRY" \
        --config "$CFG" \
        --out "$OUT" \
        2>&1 | tee "$LOG"

    test -f "$OUT/best.pt"
    test -f "$OUT/history.json"

    echo
    echo "======================================================================"
    echo "COMPLETE A_raw_ctx${CTX}"
    sha256sum "$OUT/best.pt"
    sha256sum "$OUT/history.json"
    date
    echo "======================================================================"

done

echo
echo "======================================================================"
echo "ALL THREE PRIMARY RAW CONTEXT MODELS COMPLETE"
echo "NEXT = HELD-OUT TEST EVALUATION"
echo "======================================================================"
