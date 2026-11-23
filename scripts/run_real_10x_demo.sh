#!/usr/bin/env bash
# REAL one-donor engineering smoke test only. Never report as patient generalization.
set -euo pipefail
ROOT=${ROOT:-data/10x_breast_idc}; PREP=${PREP:-data/prepared_10x_demo}; RUN=${RUN:-runs/tenx_demo}
mkdir -p "$RUN"
./scripts/download_10x_breast_idc.sh "$ROOT"
Rscript R/run_split_xenium.R --xenium-dir "$ROOT/xenium_outs" --out "$RUN/split" --cores 4
python scripts/prepare_tenx_from_split.py --split-dir "$RUN/split" \
  --he "$ROOT/Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif" \
  --out "$PREP" --sample-id TENX_BREAST_IDC --demo-split
python scripts/preflight.py --root "$PREP" --sample TENX_BREAST_IDC
python scripts/make_registration_montage.py --root "$PREP" --sample TENX_BREAST_IDC --out "$RUN/registration_montage.png"
python scripts/validate_target_semantics.py --root "$PREP" --samples TENX_BREAST_IDC
# The demo estimates gamma only from the training spatial blocks. This is still NOT a donor-independent estimate.
python scripts/estimate_leakage.py --root "$PREP" --samples TENX_BREAST_IDC --split train --out "$RUN/leakage.csv"
python scripts/compute_gene_stats.py --root "$PREP" --samples TENX_BREAST_IDC --split train --out "$RUN/gene_stats.csv"
python scripts/make_leakage_gene_panel.py --leakage "$RUN/leakage.csv" --stats "$RUN/gene_stats.csv" \
  --out "$RUN/gene_panel.txt" --n-high 24 --n-low 24 --min-r2 0.0
python scripts/select_markers.py --root "$PREP" --sample TENX_BREAST_IDC --cell-type-col first_type --out "$RUN/markers.csv" --top-k 5 || true
python scripts/run_ablation_matrix.py --base configs/tenx_idc_demo.yaml --out "$RUN/ablations" --execute
python scripts/evaluate_ablation_matrix.py --runs "$RUN/ablations" --root "$PREP" --samples TENX_BREAST_IDC --split test
python scripts/ranking_reversal.py --runs "$RUN/ablations" --out "$RUN/ranking.csv" || true
python scripts/context_leakage_coupling.py \
  --small "$RUN/ablations/A_raw_ctx24/eval/metrics_raw.csv" \
  --large "$RUN/ablations/A_raw_ctx128/eval/metrics_raw.csv" \
  --leakage "$RUN/leakage.csv" --stats "$RUN/gene_stats.csv" --out "$RUN/context_leakage.csv"
