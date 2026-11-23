#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


SAMPLES = [
    ("TENX191", "Patient 1"),
    ("TENX195", "Patient 2"),
    ("TENX197", "Patient 3"),
    ("TENX200", "Patient 4"),
]

PREP = Path(
    "data/prepared_primary2025/samples"
)

ANNROOT = Path(
    "runs/independent_histo"
)

LOCK = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

NEAR_UM = 40.0
QV_MIN = 20.0
BATCH_SIZE = 500_000

cfg = json.loads(
    LOCK.read_text()
)


# ============================================================
# Frozen marker/control roles
# ============================================================

role_rows = []

for cname, c in cfg["contrasts"].items():

    for marker in c["markers"]:

        role_rows.append({
            "contrast": cname,
            "feature_name": marker,
            "gene_role": "marker",
            "matched_to_marker": marker,
        })

        controls = c["controls"][marker]

        if len(controls) != 3:
            raise RuntimeError(
                f"{cname}/{marker}: "
                f"expected exactly 3 controls"
            )

        for ctrl in controls:

            role_rows.append({
                "contrast": cname,
                "feature_name": ctrl,
                "gene_role": "control",
                "matched_to_marker": marker,
            })


roles = pd.DataFrame(role_rows)

all_cell_gene = []


# ============================================================
# Process donors
# ============================================================

for sid, donor in SAMPLES:

    print("\n" + "=" * 100)
    print(sid, donor)
    print("=" * 100)

    sdir = PREP / sid

    mpp = float(
        (sdir / "pixel_size_um.txt")
        .read_text()
        .strip()
    )

    prov = json.loads(
        (
            sdir
            / "preparation_provenance.json"
        ).read_text()
    )

    transcript_path = Path(
        prov["transcript_source"]
    )

    cells = pd.read_parquet(
        sdir / "cells.parquet"
    )

    cells["cell_id"] = (
        cells["cell_id"].astype(str)
    )

    if cells["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicated cell IDs"
        )

    cells = cells.set_index("cell_id")

    ann = pd.read_parquet(
        ANNROOT
        / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann["cell_id"].astype(str)
    )

    if ann["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicated annotation IDs"
        )

    ann = ann.set_index("cell_id")

    if not ann.index.isin(cells.index).all():
        raise RuntimeError(
            f"{sid}: annotation cells missing "
            f"from prepared cells"
        )

    exact = ann[
        ann["annotation_status"].eq(
            "unique_nucleus_inside_cell"
        )
    ]

    # --------------------------------------------------------
    # Reproduce EXACTLY the previous near-recipient geometry.
    # Source centroid -> recipient centroid.
    # --------------------------------------------------------

    meta_parts = []

    for cname, c in cfg[
        "contrasts"
    ].items():

        source_ids = exact.index[
            exact["hovernet_type"].eq(
                c["source"]
            )
        ]

        recipient_ids = exact.index[
            exact["hovernet_type"].eq(
                c["recipient"]
            )
        ]

        if (
            len(source_ids) == 0
            or len(recipient_ids) == 0
        ):
            raise RuntimeError(
                f"{sid}/{cname}: "
                f"empty source/recipient class"
            )

        source_xy = (
            cells.loc[
                source_ids,
                ["x_px", "y_px"],
            ]
            .to_numpy(float)
        )

        recipient_xy = (
            cells.loc[
                recipient_ids,
                ["x_px", "y_px"],
            ]
            .to_numpy(float)
        )

        tree = cKDTree(
            source_xy
        )

        dist_px, _ = tree.query(
            recipient_xy,
            k=1,
            workers=-1,
        )

        dist_um = (
            dist_px * mpp
        )

        valid = (
            np.isfinite(dist_um)
            & (dist_um < NEAR_UM)
        )

        near_ids = np.asarray(
            recipient_ids.astype(str)
        )[valid]

        z = pd.DataFrame({
            "cell_id": near_ids,
            "contrast": cname,
            "source_distance_um":
                dist_um[valid],
        })

        meta_parts.append(z)

        print(
            f"{cname:35s} "
            f"source={len(source_ids):7d} "
            f"recipient={len(recipient_ids):7d} "
            f"near={len(z):7d}"
        )

    meta = pd.concat(
        meta_parts,
        ignore_index=True,
    )

    if meta["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: recipient appeared "
            f"in multiple contrasts"
        )


    # ========================================================
    # Stream transcripts
    # ========================================================

    pf = pq.ParquetFile(
        transcript_path
    )

    columns = [
        "cell_id",
        "feature_name",
        "qv",
        "is_gene",
        "overlaps_nucleus",
        "nucleus_distance",
    ]

    sample_parts = []

    near_ids_set = set(
        meta["cell_id"]
    )

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns,
    ):

        x = batch.to_pandas()

        x["cell_id"] = (
            x["cell_id"].astype(str)
        )

        qv = pd.to_numeric(
            x["qv"],
            errors="coerce",
        )

        keep = (
            x["is_gene"]
            .fillna(False)
            .astype(bool)
            &
            (qv >= QV_MIN)
            &
            x["cell_id"].isin(
                near_ids_set
            )
        )

        if not keep.any():
            continue

        x = x.loc[
            keep,
            columns,
        ].copy()

        x = x.merge(
            meta[
                [
                    "cell_id",
                    "contrast",
                ]
            ],
            on="cell_id",
            how="inner",
            validate="many_to_one",
        )

        x = x.merge(
            roles,
            on=[
                "contrast",
                "feature_name",
            ],
            how="inner",
            validate="many_to_one",
        )

        if len(x) == 0:
            continue

        overlap = pd.to_numeric(
            x["overlaps_nucleus"],
            errors="coerce",
        )

        ndist = pd.to_numeric(
            x["nucleus_distance"],
            errors="coerce",
        )

        x["is_nuclear"] = (
            overlap.fillna(0) > 0
        ).astype(int)

        x["distance_valid"] = (
            np.isfinite(ndist)
        ).astype(int)

        x["nucleus_distance_sum"] = (
            np.where(
                np.isfinite(ndist),
                ndist,
                0.0,
            )
        )

        g = (
            x.groupby(
                [
                    "contrast",
                    "matched_to_marker",
                    "gene_role",
                    "feature_name",
                    "cell_id",
                ],
                as_index=False,
            )
            .agg(
                n_molecules=(
                    "feature_name",
                    "size",
                ),

                n_nuclear=(
                    "is_nuclear",
                    "sum",
                ),

                n_distance=(
                    "distance_valid",
                    "sum",
                ),

                sum_nucleus_distance=(
                    "nucleus_distance_sum",
                    "sum",
                ),
            )
        )

        sample_parts.append(g)


    if not sample_parts:
        raise RuntimeError(
            f"{sid}: no relevant transcripts"
        )

    cg = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    # Same cell/gene may occur in several parquet batches.
    cg = (
        cg.groupby(
            [
                "contrast",
                "matched_to_marker",
                "gene_role",
                "feature_name",
                "cell_id",
            ],
            as_index=False,
        )
        .agg(
            n_molecules=(
                "n_molecules",
                "sum",
            ),

            n_nuclear=(
                "n_nuclear",
                "sum",
            ),

            n_distance=(
                "n_distance",
                "sum",
            ),

            sum_nucleus_distance=(
                "sum_nucleus_distance",
                "sum",
            ),
        )
    )

    cg["nuclear_fraction"] = (
        cg["n_nuclear"]
        / cg["n_molecules"]
    )

    cg["mean_nucleus_distance"] = (
        cg["sum_nucleus_distance"]
        / cg["n_distance"].replace(
            0,
            np.nan,
        )
    )

    cg["sample"] = sid
    cg["donor"] = donor

    all_cell_gene.append(cg)


# ============================================================
# Combined cell/gene table
# ============================================================

cell_gene = pd.concat(
    all_cell_gene,
    ignore_index=True,
)

cell_gene_path = (
    ANNROOT
    / "same_recipient_nuclear_cell_gene.csv"
)

cell_gene.to_csv(
    cell_gene_path,
    index=False,
)


# ============================================================
# SAME-RECIPIENT pairing
#
# Marker molecules are compared with pooled molecules from
# the marker's three already-frozen controls IN THE SAME CELL.
# ============================================================

pair_parts = []
marker_rows = []


for (
    sid,
    donor,
    cname,
    marker
), g in cell_gene.groupby(
    [
        "sample",
        "donor",
        "contrast",
        "matched_to_marker",
    ],
    sort=False,
):

    m = g[
        g["gene_role"].eq(
            "marker"
        )
    ].copy()

    c = g[
        g["gene_role"].eq(
            "control"
        )
    ].copy()

    if m["feature_name"].nunique() != 1:
        raise RuntimeError(
            f"{sid}/{cname}/{marker}: "
            f"expected one marker gene"
        )

    # Pool the 3 frozen controls within recipient cell.
    # This prevents one sparse control transcript from
    # dominating the comparison.
    ctrl = (
        c.groupby(
            "cell_id",
            as_index=False,
        )
        .agg(
            control_n_molecules=(
                "n_molecules",
                "sum",
            ),

            control_n_nuclear=(
                "n_nuclear",
                "sum",
            ),

            control_n_distance=(
                "n_distance",
                "sum",
            ),

            control_sum_nucleus_distance=(
                "sum_nucleus_distance",
                "sum",
            ),
        )
    )

    mm = (
        m[
            [
                "cell_id",
                "n_molecules",
                "n_nuclear",
                "n_distance",
                "sum_nucleus_distance",
            ]
        ]
        .rename(
            columns={
                "n_molecules":
                    "marker_n_molecules",

                "n_nuclear":
                    "marker_n_nuclear",

                "n_distance":
                    "marker_n_distance",

                "sum_nucleus_distance":
                    "marker_sum_nucleus_distance",
            }
        )
    )

    p = mm.merge(
        ctrl,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )

    if len(p) == 0:
        print(
            f"{sid} {cname} {marker}: "
            f"NO SAME-CELL PAIRS"
        )
        continue

    p["marker_nuclear_fraction"] = (
        p["marker_n_nuclear"]
        / p["marker_n_molecules"]
    )

    p["control_nuclear_fraction"] = (
        p["control_n_nuclear"]
        / p["control_n_molecules"]
    )

    p["delta_nuclear_fraction"] = (
        p["marker_nuclear_fraction"]
        - p["control_nuclear_fraction"]
    )

    p["marker_mean_nucleus_distance"] = (
        p["marker_sum_nucleus_distance"]
        / p[
            "marker_n_distance"
        ].replace(
            0,
            np.nan,
        )
    )

    p["control_mean_nucleus_distance"] = (
        p["control_sum_nucleus_distance"]
        / p[
            "control_n_distance"
        ].replace(
            0,
            np.nan,
        )
    )

    p["delta_nucleus_distance"] = (
        p["marker_mean_nucleus_distance"]
        - p["control_mean_nucleus_distance"]
    )

    p["sample"] = sid
    p["donor"] = donor
    p["contrast"] = cname
    p["marker"] = marker

    pair_parts.append(p)

    d_nuc = p[
        "delta_nuclear_fraction"
    ].to_numpy(float)

    d_dist = p[
        "delta_nucleus_distance"
    ].to_numpy(float)

    d_dist = d_dist[
        np.isfinite(d_dist)
    ]

    marker_rows.append({
        "sample": sid,
        "donor": donor,
        "contrast": cname,
        "marker": marker,

        "paired_cells":
            int(len(p)),

        "paired_distance_cells":
            int(len(d_dist)),

        "marker_molecules":
            int(
                p[
                    "marker_n_molecules"
                ].sum()
            ),

        "control_molecules":
            int(
                p[
                    "control_n_molecules"
                ].sum()
            ),

        "mean_delta_nuclear_fraction":
            float(
                np.mean(d_nuc)
            ),

        "median_delta_nuclear_fraction":
            float(
                np.median(d_nuc)
            ),

        "fraction_cells_marker_less_nuclear":
            float(
                np.mean(
                    d_nuc < 0
                )
            ),

        "mean_delta_nucleus_distance":
            (
                float(
                    np.mean(d_dist)
                )
                if len(d_dist)
                else np.nan
            ),

        "median_delta_nucleus_distance":
            (
                float(
                    np.median(d_dist)
                )
                if len(d_dist)
                else np.nan
            ),

        "fraction_cells_marker_farther_nucleus":
            (
                float(
                    np.mean(
                        d_dist > 0
                    )
                )
                if len(d_dist)
                else np.nan
            ),
    })

    print(
        f"{sid:7s} "
        f"{cname:35s} "
        f"{marker:8s} "
        f"paired={len(p):5d} "
        f"mean_dNuc="
        f"{np.mean(d_nuc):+.4f} "
        f"P(dNuc<0)="
        f"{np.mean(d_nuc < 0):.4f} "
        f"mean_dDist="
        f"{np.mean(d_dist) if len(d_dist) else np.nan:+.4f}"
    )


pairs = pd.concat(
    pair_parts,
    ignore_index=True,
)

marker_df = pd.DataFrame(
    marker_rows
)


pairs_path = (
    ANNROOT
    / "same_recipient_nuclear_pairs.csv"
)

marker_path = (
    ANNROOT
    / "same_recipient_nuclear_marker_summary.csv"
)

pairs.to_csv(
    pairs_path,
    index=False,
)

marker_df.to_csv(
    marker_path,
    index=False,
)


# ============================================================
# DONOR-LEVEL biological replication
#
# Leakage-like:
#
# median delta nuclear fraction < 0
# median delta nucleus distance > 0
# ============================================================

donor = (
    marker_df
    .groupby(
        [
            "sample",
            "donor",
            "contrast",
        ],
        as_index=False,
    )
    .agg(
        median_marker_mean_delta_nuclear_fraction=(
            "mean_delta_nuclear_fraction",
            "median",
        ),

        median_marker_mean_delta_nucleus_distance=(
            "mean_delta_nucleus_distance",
            "median",
        ),

        markers_less_nuclear=(
            "mean_delta_nuclear_fraction",
            lambda x: int(
                np.sum(
                    np.asarray(x) < 0
                )
            ),
        ),

        markers_farther_from_nucleus=(
            "mean_delta_nucleus_distance",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        min_paired_cells=(
            "paired_cells",
            "min",
        ),

        median_paired_cells=(
            "paired_cells",
            "median",
        ),

        n_markers=(
            "marker",
            "nunique",
        ),
    )
)


donor_path = (
    ANNROOT
    / "same_recipient_nuclear_donor_summary.csv"
)

donor.to_csv(
    donor_path,
    index=False,
)


print("\n" + "=" * 125)
print("SAME-RECIPIENT NUCLEAR/PERIPHERAL MARKER SUMMARY")
print("=" * 125)

print(
    marker_df.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print("\n" + "=" * 125)
print("DONOR-LEVEL SUMMARY")
print("=" * 125)

print(
    donor.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print("\n" + "=" * 125)
print("CROSS-DONOR DIRECTION")
print("=" * 125)

for cname, g in donor.groupby(
    "contrast"
):

    print("\n", cname)

    print(
        "donors median nuclear delta < 0:",
        int(
            np.sum(
                g[
                    "median_marker_mean_delta_nuclear_fraction"
                ] < 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors >=3/5 markers less nuclear:",
        int(
            np.sum(
                g[
                    "markers_less_nuclear"
                ] >= 3
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors median nucleus-distance delta > 0:",
        int(
            np.sum(
                g[
                    "median_marker_mean_delta_nucleus_distance"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors >=3/5 markers farther from nucleus:",
        int(
            np.sum(
                g[
                    "markers_farther_from_nucleus"
                ] >= 3
            )
        ),
        "/",
        len(g),
    )


print("\nSAVED:", cell_gene_path)
print("SAVED:", pairs_path)
print("SAVED:", marker_path)
print("SAVED:", donor_path)

print(
    "\nSAME-RECIPIENT NUCLEAR LOCALIZATION AUDIT: COMPLETE"
)
