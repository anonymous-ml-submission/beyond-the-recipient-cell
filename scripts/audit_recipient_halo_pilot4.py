#!/usr/bin/env python3

from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.stats import spearmanr


SAMPLES = [
    ("TENX191", "Patient 1"),
    ("TENX195", "Patient 2"),
    ("TENX197", "Patient 3"),
    ("TENX200", "Patient 4"),
]

PREP = Path("data/prepared_primary2025/samples")
ANNROOT = Path("runs/independent_histo")

LOCK = Path(
    "runs/frozen_primary2025/"
    "leakage_marker_panel_pilot4.json"
)

cfg = json.loads(LOCK.read_text())


CONTRASTS = [
    {
        "name": "Connective_to_Inflammatory",
        "source": cfg["primary_source"]["histo_class"],
        "recipient": "Inflammatory",
        "genes": cfg["primary_source"]["genes"],
        "role": "primary",
    },
    {
        "name": "Inflammatory_to_Connective",
        "source": cfg["orthogonal_source"]["histo_class"],
        "recipient": "Connective",
        "genes": cfg["orthogonal_source"]["genes"],
        "role": "orthogonal",
    },
]


# Fixed BEFORE seeing halo results.
DISTANCE_EDGES = [
    0.0,
    20.0,
    40.0,
    80.0,
    160.0,
    320.0,
    np.inf,
]

DISTANCE_LABELS = [
    "0-20",
    "20-40",
    "40-80",
    "80-160",
    "160-320",
    "320+",
]


def vector(X):
    if sp.issparse(X):
        return np.asarray(X.toarray()).ravel()
    return np.asarray(X).ravel()


def zscore(v):
    v = np.asarray(v, dtype=float)

    m = np.nanmean(v)
    s = np.nanstd(v)

    if not np.isfinite(s) or s == 0:
        return np.zeros_like(v)

    return (v - m) / s


def adjusted_beta(
    outcome,
    source_distance,
    area,
    d5,
    library,
):
    """
    Standardized descriptive partial coefficient.

    This is NOT used as a p-value test.
    It asks whether source distance remains directionally
    associated with source-marker signal after controlling
    for the known mapping-selection covariates.
    """

    y = zscore(
        np.log1p(outcome)
    )

    xd = zscore(
        np.log1p(source_distance)
    )

    xa = zscore(
        np.log1p(area)
    )

    xn = zscore(
        np.log1p(d5)
    )

    xl = zscore(
        np.log1p(library)
    )

    X = np.column_stack([
        np.ones(len(y)),
        xd,
        xa,
        xn,
        xl,
    ])

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    return float(beta[1])


bin_rows = []
summary_rows = []
gene_rows = []


for sid, donor in SAMPLES:

    print("\n" + "=" * 100)
    print(sid, donor)
    print("=" * 100)

    a = ad.read_h5ad(
        PREP / sid / "raw.h5ad"
    )

    obs = pd.Index(
        a.obs_names.astype(str),
        name="cell_id",
    )

    cells = pd.read_parquet(
        PREP / sid / "cells.parquet"
    )

    cells["cell_id"] = (
        cells["cell_id"].astype(str)
    )

    cells = (
        cells
        .set_index("cell_id")
        .reindex(obs)
    )

    required = [
        "x_um",
        "y_um",
        "polygon_area_um2",
    ]

    missing = [
        c for c in required
        if c not in cells.columns
    ]

    if missing:
        raise RuntimeError(
            f"{sid}: missing cell columns {missing}"
        )

    if cells[required].isna().any().any():
        raise RuntimeError(
            f"{sid}: missing geometry after reindex"
        )

    ann = pd.read_parquet(
        ANNROOT
        / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann["cell_id"].astype(str)
    )

    if ann["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicated annotation cell IDs"
        )

    ann = (
        ann
        .set_index("cell_id")
        .reindex(obs)
    )

    if ann["annotation_status"].isna().any():
        raise RuntimeError(
            f"{sid}: incomplete annotation join"
        )

    exact = (
        ann["annotation_status"]
        .eq("unique_nucleus_inside_cell")
        .to_numpy()
    )

    genes = pd.Index(
        a.var_names.astype(str)
    )

    library = np.asarray(
        a.X.sum(axis=1)
    ).ravel().astype(float)

    all_xy = cells[
        ["x_um", "y_um"]
    ].to_numpy(dtype=float)

    # Local density covariate:
    # distance to fifth nearest Xenium cell.
    all_tree = cKDTree(all_xy)

    dd, _ = all_tree.query(
        all_xy,
        k=6,
        workers=-1,
    )

    d5_all = dd[:, 5]

    area_all = pd.to_numeric(
        cells["polygon_area_um2"],
        errors="coerce",
    ).to_numpy(dtype=float)


    for contrast in CONTRASTS:

        cname = contrast["name"]
        source_class = contrast["source"]
        recipient_class = contrast["recipient"]
        locked_genes = contrast["genes"]

        for g in locked_genes:
            if g not in genes:
                raise RuntimeError(
                    f"{sid}/{cname}: locked gene missing: {g}"
                )

        source_mask = (
            exact
            & ann["hovernet_type"]
              .eq(source_class)
              .to_numpy()
        )

        recipient_mask = (
            exact
            & ann["hovernet_type"]
              .eq(recipient_class)
              .to_numpy()
        )

        source_idx = np.flatnonzero(
            source_mask
        )

        recipient_idx = np.flatnonzero(
            recipient_mask
        )

        if len(source_idx) == 0:
            raise RuntimeError(
                f"{sid}/{cname}: zero source cells"
            )

        if len(recipient_idx) == 0:
            raise RuntimeError(
                f"{sid}/{cname}: zero recipient cells"
            )

        source_xy = all_xy[
            source_idx
        ]

        recipient_xy = all_xy[
            recipient_idx
        ]

        tree = cKDTree(
            source_xy
        )

        distance, nearest_local = tree.query(
            recipient_xy,
            k=1,
            workers=-1,
        )

        nearest_source_idx = source_idx[
            nearest_local
        ]

        marker_idx = [
            genes.get_loc(g)
            for g in locked_genes
        ]

        module_all = np.asarray(
            a.X[:, marker_idx].sum(axis=1)
        ).ravel().astype(float)

        module = module_all[
            recipient_idx
        ]

        lib = library[
            recipient_idx
        ]

        area = area_all[
            recipient_idx
        ]

        d5 = d5_all[
            recipient_idx
        ]

        valid = (
            np.isfinite(distance)
            & np.isfinite(area)
            & np.isfinite(d5)
            & np.isfinite(lib)
            & (lib > 0)
            & (area > 0)
        )

        distance = distance[valid]
        module = module[valid]
        lib = lib[valid]
        area = area[valid]
        d5 = d5[valid]

        recipient_valid_idx = (
            recipient_idx[valid]
        )

        cpt10k = (
            1e4
            * module
            / lib
        )

        detection = (
            module > 0
        ).astype(float)

        rho_cpt = float(
            spearmanr(
                distance,
                cpt10k,
            ).statistic
        )

        rho_detection = float(
            spearmanr(
                distance,
                detection,
            ).statistic
        )

        beta = adjusted_beta(
            cpt10k,
            distance,
            area,
            d5,
            lib,
        )

        bins = pd.cut(
            distance,
            bins=DISTANCE_EDGES,
            labels=DISTANCE_LABELS,
            right=False,
            include_lowest=True,
        )

        print(
            f"\n{cname} "
            f"[{contrast['role']}]"
        )

        print(
            "source cells:",
            len(source_idx),
            "recipient cells:",
            len(recipient_idx),
            "valid recipients:",
            len(distance),
        )

        print(
            f"Spearman distance vs module CPT10k: "
            f"{rho_cpt:+.4f}"
        )

        print(
            f"Spearman distance vs module detection: "
            f"{rho_detection:+.4f}"
        )

        print(
            f"Adjusted standardized distance beta: "
            f"{beta:+.4f}"
        )

        for label in DISTANCE_LABELS:

            m = np.asarray(
                bins == label
            )

            if not m.any():
                continue

            total_module = float(
                module[m].sum()
            )

            total_lib = float(
                lib[m].sum()
            )

            module_cpt = (
                1e4
                * total_module
                / total_lib
                if total_lib > 0
                else np.nan
            )

            detect_frac = float(
                np.mean(module[m] > 0)
            )

            row = {
                "sample": sid,
                "donor": donor,
                "role": contrast["role"],
                "contrast": cname,
                "source_class": source_class,
                "recipient_class": recipient_class,
                "distance_bin_um": label,
                "n_recipient": int(m.sum()),
                "median_distance_um": float(
                    np.median(distance[m])
                ),
                "module_raw_counts": total_module,
                "library_raw_counts": total_lib,
                "module_cpt10k": module_cpt,
                "module_detection": detect_frac,
            }

            bin_rows.append(row)

            print(
                f"  {label:8s} "
                f"n={m.sum():6d} "
                f"CPT10k={module_cpt:9.4f} "
                f"detect={detect_frac:.4f}"
            )

        # Fixed near/far comparison.
        near = distance < 40.0

        far = (
            (distance >= 160.0)
            & (distance < 320.0)
        )

        def aggregate_cpt(counts, mask):
            if not mask.any():
                return np.nan

            denom = float(
                lib[mask].sum()
            )

            if denom <= 0:
                return np.nan

            return (
                1e4
                * float(counts[mask].sum())
                / denom
            )

        near_cpt = aggregate_cpt(
            module,
            near,
        )

        far_cpt = aggregate_cpt(
            module,
            far,
        )

        near_far_ratio = (
            near_cpt / far_cpt
            if (
                np.isfinite(near_cpt)
                and np.isfinite(far_cpt)
                and far_cpt > 0
            )
            else np.nan
        )

        print(
            f"Near <40um CPT10k: {near_cpt:.4f}"
        )

        print(
            f"Far 160-320um CPT10k: {far_cpt:.4f}"
        )

        print(
            f"Near/far ratio: {near_far_ratio:.4f}"
        )

        summary_rows.append({
            "sample": sid,
            "donor": donor,
            "role": contrast["role"],
            "contrast": cname,
            "source_class": source_class,
            "recipient_class": recipient_class,
            "n_source": len(source_idx),
            "n_recipient_total": len(recipient_idx),
            "n_recipient_valid": len(distance),
            "spearman_distance_cpt10k": rho_cpt,
            "spearman_distance_detection": rho_detection,
            "adjusted_distance_beta": beta,
            "near_n": int(near.sum()),
            "far_n": int(far.sum()),
            "near_cpt10k": near_cpt,
            "far_cpt10k": far_cpt,
            "near_far_ratio": near_far_ratio,
        })

        # Individual locked-gene concordance.
        for gene in locked_genes:

            j = genes.get_loc(gene)

            gv_all = vector(
                a.X[:, j]
            ).astype(float)

            gv = gv_all[
                recipient_valid_idx
            ]

            near_gene = aggregate_cpt(
                gv,
                near,
            )

            far_gene = aggregate_cpt(
                gv,
                far,
            )

            ratio = (
                near_gene / far_gene
                if (
                    np.isfinite(near_gene)
                    and np.isfinite(far_gene)
                    and far_gene > 0
                )
                else np.nan
            )

            gene_rho = float(
                spearmanr(
                    distance,
                    1e4 * gv / lib,
                ).statistic
            )

            gene_rows.append({
                "sample": sid,
                "donor": donor,
                "role": contrast["role"],
                "contrast": cname,
                "gene": gene,
                "near_cpt10k": near_gene,
                "far_cpt10k": far_gene,
                "near_far_ratio": ratio,
                "spearman_distance_cpt10k":
                    gene_rho,
            })

    del a


bins_df = pd.DataFrame(
    bin_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

genes_df = pd.DataFrame(
    gene_rows
)


bins_path = (
    ANNROOT
    / "recipient_halo_pilot4_bins.csv"
)

summary_path = (
    ANNROOT
    / "recipient_halo_pilot4_summary.csv"
)

gene_path = (
    ANNROOT
    / "recipient_halo_pilot4_gene_concordance.csv"
)

bins_df.to_csv(
    bins_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

genes_df.to_csv(
    gene_path,
    index=False,
)


print("\n" + "=" * 115)
print("PILOT-4 RECIPIENT HALO SUMMARY")
print("=" * 115)

print(
    summary_df[
        [
            "sample",
            "donor",
            "role",
            "contrast",
            "n_source",
            "n_recipient_valid",
            "spearman_distance_cpt10k",
            "spearman_distance_detection",
            "adjusted_distance_beta",
            "near_cpt10k",
            "far_cpt10k",
            "near_far_ratio",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 115)
print("CROSS-DONOR DIRECTION CHECK")
print("=" * 115)

for cname, g in summary_df.groupby(
    "contrast"
):

    print("\n", cname)

    print(
        "donors near/far > 1:",
        int(
            np.sum(
                g["near_far_ratio"] > 1
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors adjusted beta < 0:",
        int(
            np.sum(
                g["adjusted_distance_beta"] < 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors Spearman CPT10k < 0:",
        int(
            np.sum(
                g["spearman_distance_cpt10k"] < 0
            )
        ),
        "/",
        len(g),
    )

print("\nSAVED:", bins_path)
print("SAVED:", summary_path)
print("SAVED:", gene_path)

print(
    "\nRECIPIENT HALO PILOT-4: COMPLETE"
)
