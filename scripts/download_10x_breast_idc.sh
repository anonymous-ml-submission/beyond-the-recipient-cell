#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-data/10x_breast_idc}
mkdir -p "$OUT"
cd "$OUT"
# Official 10x links from the Xenium Explorer image-alignment tutorial.
curl -fL --retry 5 -o Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif \
  https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FFPE_Human_Breast_IDC/Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif
curl -fL --retry 5 -o Xenium_V1_FFPE_Human_Breast_IDC_outs.zip \
  https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.0.2/Xenium_V1_FFPE_Human_Breast_IDC/Xenium_V1_FFPE_Human_Breast_IDC_outs.zip
unzip -q -o Xenium_V1_FFPE_Human_Breast_IDC_outs.zip -d xenium_outs
printf '%s\n' "Downloaded real 10x Xenium breast IDC + aligned post-Xenium H&E into $OUT"
