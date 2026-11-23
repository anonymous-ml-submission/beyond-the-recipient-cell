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

PREP = Path("data/prepared_primary2025/samples")
ANNROOT = Path("runs/independent_histo")

CONTROL_LOCK = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

NEAR_UM = 40.0
QV_MIN = 20.0
BATCH_SIZE = 500_000

cfg = json.loads(
    CONTROL_LOCK.read_text()
)


# ------------------------------------------------------------
# Frozen gene-role table.
# Each marker has exactly 3 controls already fixed before
# this source-facing analysis.
# ------------------------------------------------------------

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
                f"{cname}/{marker}: expected 3 controls"
            )

        for ctrl in controls:

            role_rows.append({
                "contrast": cname,
                "feature_name": ctrl,
                "gene_role": "control",
                "matched_to_marker": marker,
            })


roles = pd.DataFrame(role_rows)

all_cell_gene_parts = []


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
        (sdir / "preparation_provenance.json")
        .read_text()
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
            f"{sid}: duplicate cells"
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
            f"{sid}: duplicate annotations"
        )

    ann = ann.set_index("cell_id")

    if not ann.index.isin(cells.index).all():
        raise RuntimeError(
            f"{sid}: annotation IDs missing from cells"
        )

    exact = ann[
        ann["annotation_status"].eq(
            "unique_nucleus_inside_cell"
        )
    ]

    meta_parts = []


    # --------------------------------------------------------
    # Keep exactly the SAME source geometry as current halo:
    # nearest source centroid.
    #
    # We intentionally do not introduce polygon distance here.
    # --------------------------------------------------------

    for cname, c in cfg["contrasts"].items():

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

        source_xy = (
            cells.loc[
                source_ids,
                ["x_px", "y_px"],
            ]
            .to_numpy(dtype=float)
        )

        recipient = cells.loc[
            recipient_ids,
            [
                "x_px",
                "y_px",
                "polygon_area_um2",
            ],
        ].copy()

        recipient_xy = recipient[
            ["x_px", "y_px"]
        ].to_numpy(dtype=float)

        if (
            len(source_xy) == 0
            or len(recipient_xy) == 0
        ):
            raise RuntimeError(
                f"{sid}/{cname}: empty source/recipient"
            )

        tree = cKDTree(
            source_xy
        )

        dist_px, nearest = tree.query(
            recipient_xy,
            k=1,
            workers=-1,
        )

        source_nearest_xy = (
            source_xy[nearest]
        )

        direction = (
            source_nearest_xy
            - recipient_xy
        )

        direction_norm = np.sqrt(
            np.sum(
                direction * direction,
                axis=1,
            )
        )

        distance_um = (
            dist_px * mpp
        )

        area_um2 = pd.to_numeric(
            recipient[
                "polygon_area_um2"
            ],
            errors="coerce",
        ).to_numpy(dtype=float)

        equivalent_radius_um = np.sqrt(
            area_um2 / np.pi
        )

        valid = (
            np.isfinite(distance_um)
            & np.isfinite(direction_norm)
            & (direction_norm > 0)
            & np.isfinite(equivalent_radius_um)
            & (equivalent_radius_um > 0)
            & (distance_um < NEAR_UM)
        )

        rid = np.asarray(
            recipient_ids.astype(str)
        )[valid]

        rec_xy = recipient_xy[valid]
        direction = direction[valid]
        direction_norm = (
            direction_norm[valid]
        )

        direction_unit = (
            direction
            / direction_norm[:, None]
        )

        z = pd.DataFrame({
            "cell_id": rid,
            "contrast": cname,

            "recipient_x_px":
                rec_xy[:, 0],

            "recipient_y_px":
                rec_xy[:, 1],

            "source_dir_x":
                direction_unit[:, 0],

            "source_dir_y":
                direction_unit[:, 1],

            "source_distance_um":
                distance_um[valid],

            "equivalent_radius_um":
                equivalent_radius_um[valid],
        })

        meta_parts.append(z)

        print(
            cname,
            "source=",
            len(source_ids),
            "recipient=",
            len(recipient_ids),
            "near_recipient=",
            len(z),
        )


    meta = pd.concat(
        meta_parts,
        ignore_index=True,
    )

    if meta["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: recipient belongs to "
            f"multiple contrasts"
        )


    # Relevant genes depend on the recipient contrast.
    allowed = roles.copy()

    pf = pq.ParquetFile(
        transcript_path
    )

    sample_parts = []

    columns = [
        "cell_id",
        "feature_name",
        "qv",
        "is_gene",
        "he_x",
        "he_y",
    ]


    # --------------------------------------------------------
    # Stream transcript parquet.
    #
    # Do not load tens of millions of molecules at once.
    # --------------------------------------------------------

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
                meta["cell_id"]
            )
        )

        if not keep.any():
            continue

        x = x.loc[
            keep,
            [
                "cell_id",
                "feature_name",
                "he_x",
                "he_y",
            ],
        ].copy()

        x = x.merge(
            meta,
            on="cell_id",
            how="inner",
            validate="many_to_one",
        )

        # Remove genes not frozen for that contrast.
        x = x.merge(
            allowed,
            on=[
                "contrast",
                "feature_name",
            ],
            how="inner",
            validate="many_to_one",
        )

        if len(x) == 0:
            continue

        hx = pd.to_numeric(
            x["he_x"],
            errors="coerce",
        ).to_numpy(float)

        hy = pd.to_numeric(
            x["he_y"],
            errors="coerce",
        ).to_numpy(float)

        dx_px = (
            hx
            - x["recipient_x_px"]
              .to_numpy(float)
        )

        dy_px = (
            hy
            - x["recipient_y_px"]
              .to_numpy(float)
        )

        radial_px = np.sqrt(
            dx_px * dx_px
            + dy_px * dy_px
        )

        dir_x = x[
            "source_dir_x"
        ].to_numpy(float)

        dir_y = x[
            "source_dir_y"
        ].to_numpy(float)

        # Signed displacement toward the source.
        signed_projection_px = (
            dx_px * dir_x
            + dy_px * dir_y
        )

        ok = (
            np.isfinite(radial_px)
            & (radial_px > 0)
            & np.isfinite(
                signed_projection_px
            )
        )

        if not ok.any():
            continue

        x = x.loc[ok].copy()

        radial_px = radial_px[ok]

        signed_projection_px = (
            signed_projection_px[ok]
        )

        # User-proposed angular statistic.
        #
        # +1 = directly source-facing
        #  0 = perpendicular
        # -1 = directly away from source
        cosine_u = (
            signed_projection_px
            / radial_px
        )

        radius_um = x[
            "equivalent_radius_um"
        ].to_numpy(float)

        # Signed displacement normalized by
        # equivalent cell radius.
        #
        # Unlike cosine alone, this gives less
        # weight to molecules arbitrarily close
        # to the cell center.
        projection_norm = (
            signed_projection_px
            * mpp
            / radius_um
        )

        radial_norm = (
            radial_px
            * mpp
            / radius_um
        )

        x["cosine_u"] = cosine_u

        x["projection_norm"] = (
            projection_norm
        )

        x["radial_norm"] = (
            radial_norm
        )

        x["source_facing"] = (
            cosine_u > 0
        ).astype(int)

        # Aggregate first at recipient-cell/gene level.
        g = (
            x.groupby(
                [
                    "contrast",
                    "feature_name",
                    "gene_role",
                    "matched_to_marker",
                    "cell_id",
                ],
                as_index=False,
            )
            .agg(
                n_molecules=(
                    "cosine_u",
                    "size",
                ),

                sum_u=(
                    "cosine_u",
                    "sum",
                ),

                sum_projection_norm=(
                    "projection_norm",
                    "sum",
                ),

                sum_radial_norm=(
                    "radial_norm",
                    "sum",
                ),

                n_source_facing=(
                    "source_facing",
                    "sum",
                ),
            )
        )

        sample_parts.append(g)


    if not sample_parts:
        raise RuntimeError(
            f"{sid}: no relevant recipient transcripts"
        )

    sample_cell_gene = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    # Same cell/gene may occur in several streamed batches.
    sample_cell_gene = (
        sample_cell_gene
        .groupby(
            [
                "contrast",
                "feature_name",
                "gene_role",
                "matched_to_marker",
                "cell_id",
            ],
            as_index=False,
        )
        .agg(
            n_molecules=(
                "n_molecules",
                "sum",
            ),

            sum_u=(
                "sum_u",
                "sum",
            ),

            sum_projection_norm=(
                "sum_projection_norm",
                "sum",
            ),

            sum_radial_norm=(
                "sum_radial_norm",
                "sum",
            ),

            n_source_facing=(
                "n_source_facing",
                "sum",
            ),
        )
    )

    sample_cell_gene["mean_u"] = (
        sample_cell_gene["sum_u"]
        /
        sample_cell_gene["n_molecules"]
    )

    sample_cell_gene[
        "mean_projection_norm"
    ] = (
        sample_cell_gene[
            "sum_projection_norm"
        ]
        /
        sample_cell_gene[
            "n_molecules"
        ]
    )

    sample_cell_gene[
        "mean_radial_norm"
    ] = (
        sample_cell_gene[
            "sum_radial_norm"
        ]
        /
        sample_cell_gene[
            "n_molecules"
        ]
    )

    sample_cell_gene[
        "source_facing_fraction"
    ] = (
        sample_cell_gene[
            "n_source_facing"
        ]
        /
        sample_cell_gene[
            "n_molecules"
        ]
    )

    sample_cell_gene[
        "sample"
    ] = sid

    sample_cell_gene[
        "donor"
    ] = donor

    all_cell_gene_parts.append(
        sample_cell_gene
    )


cell_gene = pd.concat(
    all_cell_gene_parts,
    ignore_index=True,
)


cell_gene_path = (
    ANNROOT
    / "source_facing_molecules_cell_gene.csv"
)

cell_gene.to_csv(
    cell_gene_path,
    index=False,
)


# ------------------------------------------------------------
# Gene-level donor summaries.
#
# Primary quantities are CELL-BALANCED:
# each recipient cell contributes one mean value regardless
# of how many molecules it contains.
# ------------------------------------------------------------

gene_rows = []

group_cols = [
    "sample",
    "donor",
    "contrast",
    "feature_name",
    "gene_role",
    "matched_to_marker",
]

for keys, g in cell_gene.groupby(
    group_cols,
    sort=False,
):

    n_molecules = int(
        g["n_molecules"].sum()
    )

    gene_rows.append({
        "sample": keys[0],
        "donor": keys[1],
        "contrast": keys[2],
        "gene": keys[3],
        "gene_role": keys[4],
        "matched_to_marker": keys[5],

        "n_cells_with_molecule":
            int(g["cell_id"].nunique()),

        "n_molecules":
            n_molecules,

        "mean_cell_u":
            float(
                g["mean_u"].mean()
            ),

        "median_cell_u":
            float(
                g["mean_u"].median()
            ),

        "mean_cell_projection_norm":
            float(
                g[
                    "mean_projection_norm"
                ].mean()
            ),

        "median_cell_projection_norm":
            float(
                g[
                    "mean_projection_norm"
                ].median()
            ),

        "mean_cell_source_facing_fraction":
            float(
                g[
                    "source_facing_fraction"
                ].mean()
            ),

        "transcript_weighted_u":
            float(
                g["sum_u"].sum()
                / n_molecules
            ),

        "transcript_weighted_projection_norm":
            float(
                g[
                    "sum_projection_norm"
                ].sum()
                / n_molecules
            ),

        # Coordinate sanity diagnostic.
        "mean_cell_radial_norm":
            float(
                g[
                    "mean_radial_norm"
                ].mean()
            ),
    })


gene_df = pd.DataFrame(
    gene_rows
)

gene_path = (
    ANNROOT
    / "source_facing_molecules_gene_summary.csv"
)

gene_df.to_csv(
    gene_path,
    index=False,
)


# ------------------------------------------------------------
# Marker minus median of its 3 frozen controls.
#
# Leakage-like source-facing signature:
#
#   delta_u > 0
#   delta_projection > 0
#   delta_facing_fraction > 0
# ------------------------------------------------------------

cmp_rows = []

for keys, g in gene_df.groupby(
    [
        "sample",
        "donor",
        "contrast",
        "matched_to_marker",
    ],
    sort=False,
):

    marker = g[
        g["gene_role"].eq(
            "marker"
        )
    ]

    controls = g[
        g["gene_role"].eq(
            "control"
        )
    ]

    if len(marker) != 1:
        raise RuntimeError(
            f"{keys}: expected 1 marker "
            f"but got {len(marker)}"
        )

    if len(controls) != 3:
        raise RuntimeError(
            f"{keys}: expected 3 controls "
            f"but got {len(controls)}"
        )

    m = marker.iloc[0]

    control_u = controls[
        "mean_cell_u"
    ].median()

    control_projection = controls[
        "mean_cell_projection_norm"
    ].median()

    control_facing = controls[
        "mean_cell_source_facing_fraction"
    ].median()

    cmp_rows.append({
        "sample": keys[0],
        "donor": keys[1],
        "contrast": keys[2],
        "marker": keys[3],

        "marker_n_cells":
            int(m[
                "n_cells_with_molecule"
            ]),

        "marker_n_molecules":
            int(m["n_molecules"]),

        "marker_mean_cell_u":
            m["mean_cell_u"],

        "control_median_mean_cell_u":
            control_u,

        "delta_u":
            m["mean_cell_u"]
            - control_u,

        "marker_mean_projection_norm":
            m[
                "mean_cell_projection_norm"
            ],

        "control_median_projection_norm":
            control_projection,

        "delta_projection_norm":
            m[
                "mean_cell_projection_norm"
            ]
            - control_projection,

        "marker_source_facing_fraction":
            m[
                "mean_cell_source_facing_fraction"
            ],

        "control_median_facing_fraction":
            control_facing,

        "delta_source_facing_fraction":
            m[
                "mean_cell_source_facing_fraction"
            ]
            - control_facing,
    })


cmp = pd.DataFrame(
    cmp_rows
)

cmp_path = (
    ANNROOT
    / "source_facing_molecules_marker_vs_control.csv"
)

cmp.to_csv(
    cmp_path,
    index=False,
)


# ------------------------------------------------------------
# Donor is the biological replication unit.
# ------------------------------------------------------------

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
        median_delta_u=(
            "delta_u",
            "median",
        ),

        median_delta_projection_norm=(
            "delta_projection_norm",
            "median",
        ),

        median_delta_source_facing_fraction=(
            "delta_source_facing_fraction",
            "median",
        ),

        markers_u_stronger=(
            "delta_u",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        markers_projection_stronger=(
            "delta_projection_norm",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        markers_facing_stronger=(
            "delta_source_facing_fraction",
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
    ANNROOT
    / "source_facing_molecules_donor_summary.csv"
)

donor.to_csv(
    donor_path,
    index=False,
)


print("\n" + "=" * 120)
print("SOURCE-FACING MOLECULE: MARKER VS FROZEN CONTROLS")
print("=" * 120)

print(
    cmp.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print("\n" + "=" * 120)
print("DONOR-LEVEL SOURCE-FACING SUMMARY")
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
        "donors delta_u > 0:",
        int(
            np.sum(
                g["median_delta_u"] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors delta_projection > 0:",
        int(
            np.sum(
                g[
                    "median_delta_projection_norm"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors delta_facing_fraction > 0:",
        int(
            np.sum(
                g[
                    "median_delta_source_facing_fraction"
                ] > 0
            )
        ),
        "/",
        len(g),
    )


print("\nSAVED:", cell_gene_path)
print("SAVED:", gene_path)
print("SAVED:", cmp_path)
print("SAVED:", donor_path)

print(
    "\nSOURCE-FACING MOLECULE AUDIT: COMPLETE"
)
