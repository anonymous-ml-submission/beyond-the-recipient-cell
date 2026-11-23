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
    ("TENX192", "Patient 1"),
    ("TENX193", "Patient 1"),
    ("TENX196", "Patient 2"),
    ("TENX198", "Patient 3"),
    ("TENX199", "Patient 3"),
    ("TENX201", "Patient 4"),
    ("TENX202", "Patient 4"),
]

PREP = Path(
    "data/prepared_primary2025/samples"
)

ANN = Path(
    "runs/independent_histo"
)

CONTROL_LOCK = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

cfg = json.loads(
    CONTROL_LOCK.read_text()
)

NEAR_UM = 40.0
FAR_LO = 160.0
FAR_HI = 320.0


def vector(X):
    if sp.issparse(X):
        return np.asarray(
            X.toarray()
        ).ravel()

    return np.asarray(X).ravel()


def zscore(v):
    v = np.asarray(
        v,
        dtype=float,
    )

    mu = np.nanmean(v)
    sd = np.nanstd(v)

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return np.zeros_like(v)

    return (
        v - mu
    ) / sd


def adjusted_beta(
    outcome,
    distance,
    area,
    d5,
    library,
):
    y = zscore(
        np.log1p(outcome)
    )

    xd = zscore(
        np.log1p(distance)
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

    X = np.column_stack(
        [
            np.ones(len(y)),
            xd,
            xa,
            xn,
            xl,
        ]
    )

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    return float(beta[1])


def aggregate_cpt(
    counts,
    library,
    mask,
):
    if not mask.any():
        return np.nan

    den = float(
        library[mask].sum()
    )

    if den <= 0:
        return np.nan

    return (
        1e4
        * float(
            counts[mask].sum()
        )
        / den
    )


rows = []

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

    ann = pd.read_parquet(
        ANN
        / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann["cell_id"].astype(str)
    )

    if ann["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicated annotations"
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
        .eq(
            "unique_nucleus_inside_cell"
        )
        .to_numpy()
    )

    genes = pd.Index(
        a.var_names.astype(str)
    )

    library = np.asarray(
        a.X.sum(axis=1)
    ).ravel().astype(float)

    xy = cells[
        ["x_um", "y_um"]
    ].to_numpy(float)

    area_all = pd.to_numeric(
        cells["polygon_area_um2"],
        errors="coerce",
    ).to_numpy(float)

    all_tree = cKDTree(xy)

    dd, _ = all_tree.query(
        xy,
        k=6,
        workers=-1,
    )

    d5_all = dd[:, 5]

    for cname, c in cfg[
        "contrasts"
    ].items():

        source_mask = (
            exact
            & ann["hovernet_type"]
              .eq(c["source"])
              .to_numpy()
        )

        recipient_mask = (
            exact
            & ann["hovernet_type"]
              .eq(c["recipient"])
              .to_numpy()
        )

        source_idx = np.flatnonzero(
            source_mask
        )

        recipient_idx = np.flatnonzero(
            recipient_mask
        )

        if (
            len(source_idx) == 0
            or len(recipient_idx) == 0
        ):
            raise RuntimeError(
                f"{sid}/{cname}: empty class"
            )

        tree = cKDTree(
            xy[source_idx]
        )

        distance, _ = tree.query(
            xy[recipient_idx],
            k=1,
            workers=-1,
        )

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
            & np.isfinite(lib)
            & np.isfinite(area)
            & np.isfinite(d5)
            & (lib > 0)
            & (area > 0)
        )

        distance = distance[valid]
        lib = lib[valid]
        area = area[valid]
        d5 = d5[valid]

        ridx = recipient_idx[
            valid
        ]

        near = (
            distance < NEAR_UM
        )

        far = (
            (distance >= FAR_LO)
            & (distance < FAR_HI)
        )

        if not near.any():
            raise RuntimeError(
                f"{sid}/{cname}: zero near recipients"
            )

        if not far.any():
            print(
                f"SECONDARY_NEARFAR_UNAVAILABLE: "
                f"{sid}/{cname}: zero recipients in "
                f"[{FAR_LO}, {FAR_HI}) um; "
                f"primary beta/rho retained"
            )

        genes_needed = set(
            c["markers"]
        )

        for x in c[
            "controls"
        ].values():
            genes_needed.update(x)

        for gene in sorted(
            genes_needed
        ):

            if gene not in genes:
                raise RuntimeError(
                    f"{sid}/{cname}: "
                    f"missing frozen gene {gene}"
                )

            gv = vector(
                a.X[
                    :,
                    genes.get_loc(gene)
                ]
            ).astype(float)

            gv = gv[
                ridx
            ]

            cpt = (
                1e4
                * gv
                / lib
            )

            rho = spearmanr(
                distance,
                cpt,
            ).statistic

            rho = (
                float(rho)
                if np.isfinite(rho)
                else np.nan
            )

            beta = adjusted_beta(
                cpt,
                distance,
                area,
                d5,
                lib,
            )

            near_cpt = aggregate_cpt(
                gv,
                lib,
                near,
            )

            if far.any():
                far_cpt = aggregate_cpt(
                    gv,
                    lib,
                    far,
                )

                log2_near_far = np.log2(
                    (
                        near_cpt
                        + 1e-6
                    )
                    /
                    (
                        far_cpt
                        + 1e-6
                    )
                )
            else:
                far_cpt = np.nan
                log2_near_far = np.nan

            marker_for = None
            gene_role = None

            if gene in c["markers"]:
                marker_for = gene
                gene_role = "marker"

            else:
                for marker, controls in (
                    c["controls"].items()
                ):
                    if gene in controls:
                        marker_for = marker
                        gene_role = "control"
                        break

            if marker_for is None:
                raise RuntimeError(
                    f"Could not identify role for {gene}"
                )

            rows.append({
                "sample": sid,
                "donor": donor,
                "contrast": cname,
                "gene": gene,
                "gene_role": gene_role,
                "matched_to_marker": marker_for,

                "n_source":
                    int(len(source_idx)),

                "n_recipient":
                    int(len(distance)),

                "near_n":
                    int(near.sum()),

                "far_n":
                    int(far.sum()),

                "spearman_distance_cpt10k":
                    rho,

                "adjusted_distance_beta":
                    beta,

                "near_cpt10k":
                    near_cpt,

                "far_cpt10k":
                    far_cpt,

                "log2_near_far":
                    float(log2_near_far),
            })

    del a


res = pd.DataFrame(rows)

detail_path = (
    ANN
    / "halo_vs_matched_controls_confirmatory7_detail.csv"
)

res.to_csv(
    detail_path,
    index=False,
)


# ------------------------------------------------------------
# For each marker/donor:
# compare marker halo with MEDIAN of its 3 frozen controls.
#
# More leakage-like:
#   delta_beta < 0
#   delta_rho  < 0
#   delta_log2_near_far > 0
# ------------------------------------------------------------

cmp_rows = []

group_cols = [
    "sample",
    "donor",
    "contrast",
    "matched_to_marker",
]

for keys, g in res.groupby(
    group_cols,
    sort=False,
):

    marker = g[
        g["gene_role"].eq("marker")
    ]

    controls = g[
        g["gene_role"].eq("control")
    ]

    if len(marker) != 1:
        raise RuntimeError(
            f"{keys}: expected 1 marker, got {len(marker)}"
        )

    if len(controls) != 3:
        raise RuntimeError(
            f"{keys}: expected 3 controls, got {len(controls)}"
        )

    m = marker.iloc[0]

    ctrl_beta = controls[
        "adjusted_distance_beta"
    ].median()

    ctrl_rho = controls[
        "spearman_distance_cpt10k"
    ].median()

    ctrl_nf = controls[
        "log2_near_far"
    ].median()

    cmp_rows.append({
        "sample": keys[0],
        "donor": keys[1],
        "contrast": keys[2],
        "marker": keys[3],

        "marker_beta":
            m["adjusted_distance_beta"],

        "control_median_beta":
            ctrl_beta,

        "delta_beta":
            m["adjusted_distance_beta"]
            - ctrl_beta,

        "marker_rho":
            m["spearman_distance_cpt10k"],

        "control_median_rho":
            ctrl_rho,

        "delta_rho":
            m["spearman_distance_cpt10k"]
            - ctrl_rho,

        "marker_log2_near_far":
            m["log2_near_far"],

        "control_median_log2_near_far":
            ctrl_nf,

        "delta_log2_near_far":
            m["log2_near_far"]
            - ctrl_nf,
    })


cmp = pd.DataFrame(
    cmp_rows
)

cmp_path = (
    ANN
    / "halo_vs_matched_controls_confirmatory7_marker_comparison.csv"
)

cmp.to_csv(
    cmp_path,
    index=False,
)


# Donor is the biological replication unit.
donor = (
    cmp
    .groupby(
        [
            "sample",
            "donor",
            "contrast",
        ],
        as_index=False,
    )
    .agg(
        median_delta_beta=(
            "delta_beta",
            "median",
        ),
        median_delta_rho=(
            "delta_rho",
            "median",
        ),
        median_delta_log2_near_far=(
            "delta_log2_near_far",
            "median",
        ),
        markers_beta_stronger=(
            "delta_beta",
            lambda x: int(
                np.sum(
                    np.asarray(x) < 0
                )
            ),
        ),
        markers_rho_stronger=(
            "delta_rho",
            lambda x: int(
                np.sum(
                    np.asarray(x) < 0
                )
            ),
        ),
        markers_nearfar_stronger=(
            "delta_log2_near_far",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),
        n_markers=(
            "marker",
            "nunique",
        ),
    )
)


donor_path = (
    ANN
    / "halo_vs_matched_controls_confirmatory7_donor_summary.csv"
)

donor.to_csv(
    donor_path,
    index=False,
)


print("\n" + "=" * 120)
print("MARKER VS MATCHED-CONTROL HALO")
print("=" * 120)

print(
    cmp.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print("\n" + "=" * 120)
print("DONOR-LEVEL SUMMARY")
print("=" * 120)

print(
    donor.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print("\n" + "=" * 120)
print("CROSS-DONOR DIRECTION")
print("=" * 120)

for cname, g in donor.groupby(
    "contrast"
):

    print("\n", cname)

    print(
        "donors marker beta stronger:",
        int(
            np.sum(
                g["median_delta_beta"] < 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors marker rho stronger:",
        int(
            np.sum(
                g["median_delta_rho"] < 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors marker near/far stronger:",
        int(
            np.sum(
                g[
                    "median_delta_log2_near_far"
                ] > 0
            )
        ),
        "/",
        len(g),
    )


print("\nSAVED:", detail_path)
print("SAVED:", cmp_path)
print("SAVED:", donor_path)
print("\nMATCHED-CONTROL HALO AUDIT: COMPLETE")
