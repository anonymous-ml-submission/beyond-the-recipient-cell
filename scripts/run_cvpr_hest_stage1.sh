#!/usr/bin/env bash
# Stage 1: genuine multi-donor HEST Xenium/H&E preparation with segmentation-defined geometry.
set -euo pipefail
HEST=${HEST:-data/hest_xenium}
PREP=${PREP:-data/prepared_hest}
ORGAN=${ORGAN:-Breast}
COMMON=${COMMON:-runs/hest_common_genes.txt}
python scripts/download_hest_xenium.py --out "$HEST" --organ "$ORGAN" --max-samples 12
python scripts/prepare_hest.py --hest-root "$HEST" --out "$PREP" --seed 2026
mapfile -t SAMPLES < <(find "$PREP/samples" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
python scripts/build_common_gene_universe.py --root "$PREP" --samples "${SAMPLES[@]}" --out "$COMMON"
printf '\nPrepared multi-donor HEST data with segmentation-defined centres/polygons.\n'
printf 'Common assay-safe gene universe: %s\n' "$COMMON"
printf 'Next: polygon registration QA, attach independently corrected clean.h5ad, validate targets, then estimate training-only leakage.\n'
