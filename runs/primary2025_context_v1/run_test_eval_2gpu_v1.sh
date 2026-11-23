#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1

BIO="${BIO:?Set BIO to the original Python environment path}"
EVALPY="scripts/evaluate_primary2025_context_v1.py"
ROOT="runs/primary2025_context_v1"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


run_eval () {

    CTX="$1"
    GPU="$2"

    RD="$ROOT/A_raw_ctx${CTX}"

    CK="$RD/${CHECKPOINT_SUBDIR:?Set CHECKPOINT_SUBDIR to the original frozen training directory}/best.pt"
    OUT="$RD/eval"
    LOG="$RD/eval.log"

    test -f "$CK"

    test ! -e "$OUT" || {
        echo "STOP existing eval: $OUT"
        return 1
    }

    TMPROOT="${TMPROOT_BASE:?Set TMPROOT_BASE to a writable scratch prefix}_${BASHPID}"

    mkdir -p "$TMPROOT"
    chmod 700 "$TMPROOT"

    echo "=============================================================="
    echo "START TEST EVAL CTX=$CTX GPU=$GPU"
    echo "$(date)"
    echo "=============================================================="

    CUDA_VISIBLE_DEVICES="$GPU" \
    TMPDIR="$TMPROOT" \
    TMP="$TMPROOT" \
    TEMP="$TMPROOT" \
    "$BIO" -u "$EVALPY" \
        --checkpoint "$CK" \
        --out "$OUT" \
        --batch-size 64 \
        --workers 6 \
        > "$LOG" \
        2>&1

    RC=$?

    rm -rf "$TMPROOT"

    if [ "$RC" -ne 0 ]; then
        return "$RC"
    fi

    echo "CTX${CTX}_TEST_EVAL=PASS"

    sha256sum \
        "$OUT/metrics_raw.csv" \
        "$OUT/metrics_raw_by_donor.csv" \
        "$OUT/metrics_clean.csv" \
        "$OUT/metrics_clean_by_donor.csv" \
        "$OUT/predictions.npz"
}


echo "======================================================================"
echo "PHASE 1 — TEST CTX24 GPU0 + CTX64 GPU1"
echo "======================================================================"

run_eval 24 0 &
P24=$!

run_eval 64 1 &
P64=$!

set +e

wait "$P24"
R24=$?

wait "$P64"
R64=$?

set -e

echo "CTX24_RC=$R24"
echo "CTX64_RC=$R64"

if [ "$R24" -ne 0 ] || [ "$R64" -ne 0 ]; then
    echo "STOP: PHASE1 evaluation failure"
    exit 1
fi


echo
echo "======================================================================"
echo "PHASE 2 — TEST CTX128 GPU0"
echo "======================================================================"

run_eval 128 0


echo
echo "======================================================================"
echo "ALL THREE HELD-OUT TEST EVALUATIONS COMPLETE"
echo "======================================================================"

# Existing frozen ranking implementation.
"$BIO" scripts/ranking_reversal.py \
    --runs "$ROOT" \
    --out "$ROOT/ranking_reversal_test.csv" \
    | tee "$ROOT/ranking_reversal_test.log"

echo
echo "======================================================================"
echo "RANKING TABLE"
echo "======================================================================"

cat "$ROOT/ranking_reversal_test.csv"

echo
echo "======================================================================"
echo "PRIMARY TEST STAGE COMPLETE"
echo "======================================================================"
