#!/usr/bin/env python3

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SAMPLES = [
    ("TENX191", "Patient 1"),
    ("TENX195", "Patient 2"),
    ("TENX197", "Patient 3"),
    ("TENX200", "Patient 4"),
]

ROOT = Path("data/prepared_primary2025/samples")
ANNROOT = Path("runs/independent_histo")
OUTROOT = Path("runs/independent_histo")


# Fixed BEFORE looking at source-enrichment results.
# These are biological feasibility candidates, NOT genes
# selected from this Xenium dataset by differential expression.
MARKER_SETS = {
    "Inflammatory": {
        "T_cell": [
            "CD3E",
            "TRAC",
            "CD8A",
            "CCL5",
            "IL7R",
            "CCR7",
            "CD27",
            "CTLA4",
            "TIGIT",
        ],
        "B_cell": [
            "MS4A1",
            "BANK1",
            "GPR183",
            "TNFRSF13C",
            "CD74",
        ],
        "Myeloid": [
            "CD68",
            "CD163",
            "LYZ",
            "ITGAX",
            "C15orf48",
            "APOC1",
            "CHIT1",
            "MMP12",
        ],
    },

    # This is deliberately called perivascular/contractile,
    # not generic fibroblast.
    "Connective": {
        "Perivascular_contractile": [
            "ACTA2",
            "TAGLN",
            "PDGFRB",
            "RGS5",
            "MYH11",
            "CNN1",
            "MYLK",
            "ACTG2",
        ],
        "Stromal_available": [
            "MMP2",
            "FBLN1",
            "CCDC80",
            "CXCL12",
            "SFRP4",
            "OLFML2A",
        ],
    },
}


def dense_col(X, j):
    z = X[:, j]
    if sp.issparse(z):
        return np.asarray(z.toarray()).ravel()
    return np.asarray(z).ravel()


rows = []

for sid, donor in SAMPLES:

    print("\n" + "=" * 90)
    print(sid, donor)
    print("=" * 90)

    a = ad.read_h5ad(
        ROOT / sid / "raw.h5ad"
    )

    ann = pd.read_parquet(
        ANNROOT / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = ann["cell_id"].astype(str)

    if ann["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicated annotation cell_id"
        )

    ann = ann.set_index("cell_id")

    obs = pd.Index(
        a.obs_names.astype(str),
        name="cell_id",
    )

    ann = ann.reindex(obs)

    if ann["annotation_status"].isna().any():
        raise RuntimeError(
            f"{sid}: annotations do not cover raw.h5ad cells"
        )

    exact = ann["annotation_status"].eq(
        "unique_nucleus_inside_cell"
    ).to_numpy()

    # Exclude Dead/Background from the biological comparison.
    biological = (
        exact
        & ann["hovernet_type"]
          .isin([
              "Inflammatory",
              "Connective",
              "Neoplastic",
              "Non-Neoplastic Epithelial",
          ])
          .to_numpy()
    )

    X = a.X

    library = np.asarray(
        X.sum(axis=1)
    ).ravel().astype(float)

    genes = pd.Index(
        a.var_names.astype(str)
    )

    for target_class, families in MARKER_SETS.items():

        target = (
            biological
            & ann["hovernet_type"]
              .eq(target_class)
              .to_numpy()
        )

        other = (
            biological
            & ~ann["hovernet_type"]
               .eq(target_class)
               .to_numpy()
        )

        nt = int(target.sum())
        no = int(other.sum())

        print(
            f"\n{target_class}: "
            f"target={nt:,}, other={no:,}"
        )

        if nt == 0 or no == 0:
            raise RuntimeError(
                f"{sid}/{target_class}: empty comparison"
            )

        target_lib = library[target].sum()
        other_lib = library[other].sum()

        for family, candidate_genes in families.items():

            for gene in candidate_genes:

                if gene not in genes:
                    raise RuntimeError(
                        f"Frozen candidate unexpectedly absent: {gene}"
                    )

                j = genes.get_loc(gene)
                v = dense_col(X, j).astype(float)

                tv = v[target]
                ov = v[other]

                t_sum = float(tv.sum())
                o_sum = float(ov.sum())

                t_det = float(
                    np.mean(tv > 0)
                )

                o_det = float(
                    np.mean(ov > 0)
                )

                # Gene counts per 10,000 total panel transcripts.
                t_cpt10k = (
                    1e4 * t_sum / target_lib
                    if target_lib > 0
                    else np.nan
                )

                o_cpt10k = (
                    1e4 * o_sum / other_lib
                    if other_lib > 0
                    else np.nan
                )

                eps = 1e-8

                norm_ratio = (
                    (t_cpt10k + eps)
                    /
                    (o_cpt10k + eps)
                )

                detection_ratio = (
                    (t_det + eps)
                    /
                    (o_det + eps)
                )

                total_gene = t_sum + o_sum
                target_gene_share = (
                    t_sum / total_gene
                    if total_gene > 0
                    else np.nan
                )

                target_cell_share = (
                    nt / (nt + no)
                )

                concentration_lift = (
                    target_gene_share
                    / target_cell_share
                    if total_gene > 0
                    else np.nan
                )

                rows.append({
                    "sample": sid,
                    "donor": donor,
                    "target_class": target_class,
                    "marker_family": family,
                    "gene": gene,
                    "n_target": nt,
                    "n_other": no,
                    "target_detection": t_det,
                    "other_detection": o_det,
                    "detection_ratio": detection_ratio,
                    "target_cpt10k": t_cpt10k,
                    "other_cpt10k": o_cpt10k,
                    "normalized_ratio": norm_ratio,
                    "log2_normalized_ratio": np.log2(
                        norm_ratio
                    ),
                    "target_gene_share": target_gene_share,
                    "target_cell_share": target_cell_share,
                    "concentration_lift": concentration_lift,
                    "target_raw_counts": t_sum,
                    "other_raw_counts": o_sum,
                })

    a.file.close() if getattr(a, "isbacked", False) else None


res = pd.DataFrame(rows)

detail_path = (
    OUTROOT
    / "histo_marker_feasibility_4donor_detail.csv"
)

res.to_csv(
    detail_path,
    index=False,
)


summary = (
    res.groupby(
        [
            "target_class",
            "marker_family",
            "gene",
        ],
        as_index=False,
    )
    .agg(
        donors=("sample", "nunique"),

        min_target_detection=(
            "target_detection",
            "min",
        ),

        median_target_detection=(
            "target_detection",
            "median",
        ),

        median_log2_normalized_ratio=(
            "log2_normalized_ratio",
            "median",
        ),

        min_log2_normalized_ratio=(
            "log2_normalized_ratio",
            "min",
        ),

        donors_target_gt_other=(
            "log2_normalized_ratio",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        median_concentration_lift=(
            "concentration_lift",
            "median",
        ),

        total_target_raw_counts=(
            "target_raw_counts",
            "sum",
        ),
    )
)

summary = summary.sort_values(
    [
        "target_class",
        "marker_family",
        "gene",
    ]
)

summary_path = (
    OUTROOT
    / "histo_marker_feasibility_4donor_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 120)
print("FOUR-DONOR FIXED-MARKER FEASIBILITY SUMMARY")
print("=" * 120)

for target in ["Inflammatory", "Connective"]:

    z = summary[
        summary["target_class"].eq(target)
    ]

    print("\n" + "#" * 120)
    print(target)
    print("#" * 120)

    print(
        z[
            [
                "marker_family",
                "gene",
                "min_target_detection",
                "median_target_detection",
                "median_log2_normalized_ratio",
                "min_log2_normalized_ratio",
                "donors_target_gt_other",
                "median_concentration_lift",
                "total_target_raw_counts",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


print("\nSAVED:", detail_path)
print("SAVED:", summary_path)
print(
    "\nFIXED-MARKER FEASIBILITY AUDIT: COMPLETE"
)
