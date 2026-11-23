#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.spatial import cKDTree
from scipy.stats import spearmanr


# ============================================================
# FROZEN CONTRACT
# ============================================================

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

ANNROOT = Path(
    "runs/independent_histo"
)

LOCK = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

PRIMARY_CONTRAST = (
    "Connective_to_Inflammatory"
)

NEAR_UM = 40.0

# Predeclared gate:
#
# donor PASS:
#   median marker-minus-control beta > 0
#   AND >=3/5 marker deltas > 0
#
# primary gate PASS:
#   >=3/4 donors PASS
#
MIN_MARKERS_PER_DONOR = 3
MIN_DONORS_PRIMARY = 6


OUT_GEOMETRY = (
    ANNROOT
    / "source_dose_confirmatory7_source_neighborhoods.csv"
)

OUT_GENE = (
    ANNROOT
    / "source_dose_confirmatory7_gene_summary.csv"
)

OUT_MARKER = (
    ANNROOT
    / "source_dose_confirmatory7_marker_vs_control.csv"
)

OUT_DONOR = (
    ANNROOT
    / "source_dose_confirmatory7_section_summary.csv"
)


cfg = json.loads(
    LOCK.read_text()
)


# ============================================================
# Utilities
# ============================================================

def dense_col(X, j):
    z = X[:, j]

    if sp.issparse(z):
        return np.asarray(
            z.toarray()
        ).ravel().astype(float)

    return np.asarray(
        z
    ).ravel().astype(float)


def standardize_required(v):
    """
    Standardize an endpoint/predictor.

    Returns None when variance is zero or values are invalid.
    """
    v = np.asarray(
        v,
        dtype=float,
    )

    if not np.isfinite(v).all():
        return None

    sd = np.std(
        v,
        ddof=0,
    )

    if (
        not np.isfinite(sd)
        or sd <= 1e-12
    ):
        return None

    return (
        v - np.mean(v)
    ) / sd


def standardize_optional(v):
    """
    Standardize a covariate.

    Constant covariates are omitted rather than causing failure.
    """
    v = np.asarray(
        v,
        dtype=float,
    )

    if not np.isfinite(v).all():
        return None

    sd = np.std(
        v,
        ddof=0,
    )

    if (
        not np.isfinite(sd)
        or sd <= 1e-12
    ):
        return None

    return (
        v - np.mean(v)
    ) / sd


def adjusted_standardized_beta(
    source_cpt10k,
    recipient_cpt10k,
    covariates,
):
    """
    Source-cell-centred model:

      log1p(recipient-neighborhood CPT10k)
          ~
      log1p(source-cell CPT10k)
          +
      frozen nuisance covariates

    Both outcome and source-dose predictor are standardized,
    making coefficients comparable between genes.

    Every source cell remains present, including source-expression
    zeros and recipient-neighborhood-expression zeros.
    """

    sx = np.asarray(
        source_cpt10k,
        dtype=float,
    )

    ry = np.asarray(
        recipient_cpt10k,
        dtype=float,
    )

    valid = (
        np.isfinite(sx)
        & np.isfinite(ry)
    )

    for v in covariates.values():
        valid &= np.isfinite(
            np.asarray(
                v,
                dtype=float,
            )
        )

    sx = sx[valid]
    ry = ry[valid]

    cov = {
        k: np.asarray(
            v,
            dtype=float,
        )[valid]
        for k, v in covariates.items()
    }

    if len(sx) < 20:
        return np.nan, int(len(sx))

    zx = standardize_required(
        np.log1p(sx)
    )

    zy = standardize_required(
        np.log1p(ry)
    )

    if zx is None or zy is None:
        return np.nan, int(len(sx))

    columns = [
        np.ones(
            len(zx),
            dtype=float,
        ),
        zx,
    ]

    # Covariates are log-transformed because all are positive/
    # nonnegative scale quantities.
    for name, v in cov.items():

        zv = standardize_optional(
            np.log1p(v)
        )

        if zv is not None:
            columns.append(zv)

    X = np.column_stack(
        columns
    )

    beta = np.linalg.lstsq(
        X,
        zy,
        rcond=None,
    )[0]

    # column 1 is the standardized source-dose predictor.
    return float(beta[1]), int(len(sx))


def safe_spearman(x, y):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if (
        np.std(x) <= 1e-12
        or np.std(y) <= 1e-12
    ):
        return np.nan

    r = spearmanr(
        x,
        y,
    ).statistic

    return (
        float(r)
        if np.isfinite(r)
        else np.nan
    )


# ============================================================
# Frozen marker/control role table
# ============================================================

role_rows = []

for cname, c in cfg[
    "contrasts"
].items():

    for marker in c["markers"]:

        role_rows.append({
            "contrast": cname,
            "gene": marker,
            "gene_role": "marker",
            "matched_to_marker": marker,
        })

        controls = list(
            c["controls"][marker]
        )

        if len(controls) != 3:
            raise RuntimeError(
                f"{cname}/{marker}: "
                "expected exactly 3 frozen controls"
            )

        for control in controls:

            role_rows.append({
                "contrast": cname,
                "gene": control,
                "gene_role": "control",
                "matched_to_marker": marker,
            })


roles = pd.DataFrame(
    role_rows
)

if roles.duplicated(
    ["contrast", "gene"]
).any():
    raise RuntimeError(
        "A gene has multiple roles within a contrast"
    )


# ============================================================
# Run donor by donor
# ============================================================

geometry_parts = []
gene_rows = []


for sid, donor in SAMPLES:

    print(
        "\n" + "=" * 110
    )
    print(
        sid,
        donor,
    )
    print(
        "=" * 110
    )

    sdir = PREP / sid

    mpp = float(
        (
            sdir
            / "pixel_size_um.txt"
        )
        .read_text()
        .strip()
    )

    a = ad.read_h5ad(
        sdir / "raw.h5ad"
    )

    obs = pd.Index(
        a.obs_names.astype(str),
        name="cell_id",
    )

    genes = pd.Index(
        a.var_names.astype(str)
    )

    cells = pd.read_parquet(
        sdir / "cells.parquet"
    )

    cells["cell_id"] = (
        cells["cell_id"]
        .astype(str)
    )

    if cells[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{sid}: duplicated cells"
        )

    cells = (
        cells
        .set_index("cell_id")
        .reindex(obs)
    )

    required_cell_cols = [
        "x_px",
        "y_px",
        "polygon_area_um2",
    ]

    if cells[
        required_cell_cols
    ].isna().any().any():

        raise RuntimeError(
            f"{sid}: incomplete cells geometry"
        )


    ann = pd.read_parquet(
        ANNROOT
        / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann["cell_id"]
        .astype(str)
    )

    if ann[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{sid}: duplicated annotations"
        )

    ann = (
        ann
        .set_index("cell_id")
        .reindex(obs)
    )

    if ann[
        "annotation_status"
    ].isna().any():

        raise RuntimeError(
            f"{sid}: incomplete annotation join"
        )


    exact = (
        ann[
            "annotation_status"
        ]
        .eq(
            "unique_nucleus_inside_cell"
        )
        .to_numpy()
    )


    # --------------------------------------------------------
    # Library depth from the exact prepared 280-gene matrix.
    # --------------------------------------------------------

    library = np.asarray(
        a.X.sum(axis=1)
    ).ravel().astype(float)


    # --------------------------------------------------------
    # Same local-density definition used in previous halo work:
    # distance to fifth nearest Xenium cell.
    # --------------------------------------------------------

    xy_all = cells[
        ["x_px", "y_px"]
    ].to_numpy(float)

    all_tree = cKDTree(
        xy_all
    )

    d_all, _ = all_tree.query(
        xy_all,
        k=6,
        workers=-1,
    )

    d5_px_all = (
        d_all[:, 5]
    )

    area_all = pd.to_numeric(
        cells[
            "polygon_area_um2"
        ],
        errors="coerce",
    ).to_numpy(float)


    # ========================================================
    # Contrast
    # ========================================================

    for cname, c in cfg[
        "contrasts"
    ].items():

        print(
            "\n",
            cname,
        )


        source_mask = (
            exact
            & ann[
                "hovernet_type"
            ]
            .eq(
                c["source"]
            )
            .to_numpy()
        )

        recipient_mask = (
            exact
            & ann[
                "hovernet_type"
            ]
            .eq(
                c["recipient"]
            )
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
                f"{sid}/{cname}: "
                "empty source or recipient population"
            )


        source_xy = (
            xy_all[source_idx]
        )

        recipient_xy = (
            xy_all[recipient_idx]
        )


        # ----------------------------------------------------
        # EXACT SAME HALO GEOMETRY:
        #
        # each recipient -> nearest exact H&E source centroid.
        # retain only recipient distance <40 um.
        # ----------------------------------------------------

        tree = cKDTree(
            source_xy
        )

        dist_px, nearest_local = (
            tree.query(
                recipient_xy,
                k=1,
                workers=-1,
            )
        )

        dist_um = (
            dist_px * mpp
        )


        recipient_lib = (
            library[
                recipient_idx
            ]
        )

        recipient_area = (
            area_all[
                recipient_idx
            ]
        )

        recipient_d5 = (
            d5_px_all[
                recipient_idx
            ]
        )


        recipient_valid = (
            np.isfinite(dist_um)
            & np.isfinite(
                recipient_lib
            )
            & (
                recipient_lib > 0
            )
            & np.isfinite(
                recipient_area
            )
            & (
                recipient_area > 0
            )
            & np.isfinite(
                recipient_d5
            )
            & (
                dist_um < NEAR_UM
            )
        )


        ridx = (
            recipient_idx[
                recipient_valid
            ]
        )

        nearest = (
            nearest_local[
                recipient_valid
            ]
            .astype(int)
        )

        rdist = (
            dist_um[
                recipient_valid
            ]
        )

        rlib = (
            recipient_lib[
                recipient_valid
            ]
        )

        rarea = (
            recipient_area[
                recipient_valid
            ]
        )

        rd5 = (
            recipient_d5[
                recipient_valid
            ]
        )


        n_source = len(
            source_idx
        )


        # ----------------------------------------------------
        # Aggregate recipient geometry BY NEAREST SOURCE CELL.
        # This makes source cell the unit of analysis.
        # ----------------------------------------------------

        n_recipients = np.bincount(
            nearest,
            minlength=n_source,
        ).astype(float)

        recipient_lib_sum = np.bincount(
            nearest,
            weights=rlib,
            minlength=n_source,
        )

        recipient_dist_sum = np.bincount(
            nearest,
            weights=rdist,
            minlength=n_source,
        )

        recipient_area_sum = np.bincount(
            nearest,
            weights=rarea,
            minlength=n_source,
        )

        recipient_d5_sum = np.bincount(
            nearest,
            weights=rd5,
            minlength=n_source,
        )


        source_lib = (
            library[
                source_idx
            ]
        )

        source_area = (
            area_all[
                source_idx
            ]
        )

        source_d5 = (
            d5_px_all[
                source_idx
            ]
        )


        # A source cell enters only if:
        # - at least one exact recipient is assigned within 40 um
        # - source geometry/library is valid
        # - recipient pooled library is valid
        source_keep = (
            (n_recipients > 0)
            & np.isfinite(
                source_lib
            )
            & (
                source_lib > 0
            )
            & np.isfinite(
                source_area
            )
            & (
                source_area > 0
            )
            & np.isfinite(
                source_d5
            )
            & np.isfinite(
                recipient_lib_sum
            )
            & (
                recipient_lib_sum > 0
            )
        )


        if (
            int(source_keep.sum())
            < 20
        ):
            raise RuntimeError(
                f"{sid}/{cname}: "
                "too few usable source-cell units"
            )


        sk = np.flatnonzero(
            source_keep
        )


        mean_recipient_distance = (
            recipient_dist_sum[
                source_keep
            ]
            / n_recipients[
                source_keep
            ]
        )

        mean_recipient_area = (
            recipient_area_sum[
                source_keep
            ]
            / n_recipients[
                source_keep
            ]
        )

        mean_recipient_d5 = (
            recipient_d5_sum[
                source_keep
            ]
            / n_recipients[
                source_keep
            ]
        )

        mean_recipient_library = (
            recipient_lib_sum[
                source_keep
            ]
            / n_recipients[
                source_keep
            ]
        )


        source_cell_ids = (
            obs[
                source_idx[
                    source_keep
                ]
            ]
            .astype(str)
            .to_numpy()
        )


        geom = pd.DataFrame({
            "sample": sid,
            "donor": donor,
            "contrast": cname,

            "source_cell_id":
                source_cell_ids,

            "n_recipients":
                n_recipients[
                    source_keep
                ].astype(int),

            "mean_recipient_distance_um":
                mean_recipient_distance,

            "source_area_um2":
                source_area[
                    source_keep
                ],

            "mean_recipient_area_um2":
                mean_recipient_area,

            "source_d5_px":
                source_d5[
                    source_keep
                ],

            "mean_recipient_d5_px":
                mean_recipient_d5,

            "source_library":
                source_lib[
                    source_keep
                ],

            "recipient_library_sum":
                recipient_lib_sum[
                    source_keep
                ],

            "mean_recipient_library":
                mean_recipient_library,
        })


        geometry_parts.append(
            geom
        )


        print(
            "source cells total       =",
            len(source_idx),
        )

        print(
            "recipient cells total    =",
            len(recipient_idx),
        )

        print(
            "near recipients <40 um   =",
            len(ridx),
        )

        print(
            "source units with >=1 rec=",
            int(
                source_keep.sum()
            ),
        )


        # ----------------------------------------------------
        # Nuisance covariates.
        #
        # Importantly:
        # no source-expression-dependent filtering.
        # Zero-expression sources remain.
        # ----------------------------------------------------

        covariates = {
            "mean_recipient_distance_um":
                mean_recipient_distance,

            "source_area_um2":
                source_area[
                    source_keep
                ],

            "mean_recipient_area_um2":
                mean_recipient_area,

            "source_d5_px":
                source_d5[
                    source_keep
                ],

            "mean_recipient_d5_px":
                mean_recipient_d5,

            "source_library":
                source_lib[
                    source_keep
                ],

            "mean_recipient_library":
                mean_recipient_library,

            "n_recipients":
                n_recipients[
                    source_keep
                ],
        }


        # ----------------------------------------------------
        # Only already-frozen genes.
        # ----------------------------------------------------

        role_c = roles[
            roles[
                "contrast"
            ].eq(cname)
        ]

        needed_genes = list(
            role_c[
                "gene"
            ].drop_duplicates()
        )


        for gene in needed_genes:

            if gene not in genes:
                raise RuntimeError(
                    f"{sid}/{cname}: "
                    f"frozen gene missing: {gene}"
                )


            j = genes.get_loc(
                gene
            )

            gv = dense_col(
                a.X,
                j,
            )


            # --------------------------------------------
            # Source-cell dose.
            # --------------------------------------------

            source_gene_counts_all = (
                gv[
                    source_idx
                ]
            )

            source_gene_counts = (
                source_gene_counts_all[
                    source_keep
                ]
            )

            source_cpt10k = (
                1e4
                * source_gene_counts
                / source_lib[
                    source_keep
                ]
            )


            # --------------------------------------------
            # Recipient-neighborhood response:
            # pooled recipient gene molecules / pooled
            # recipient library for each nearest source.
            # --------------------------------------------

            recipient_gene_counts = (
                gv[
                    ridx
                ]
            )

            recipient_gene_sum = (
                np.bincount(
                    nearest,
                    weights=(
                        recipient_gene_counts
                    ),
                    minlength=n_source,
                )
            )

            recipient_gene_sum = (
                recipient_gene_sum[
                    source_keep
                ]
            )

            recipient_cpt10k = (
                1e4
                * recipient_gene_sum
                / recipient_lib_sum[
                    source_keep
                ]
            )


            beta, n_model = (
                adjusted_standardized_beta(
                    source_cpt10k,
                    recipient_cpt10k,
                    covariates,
                )
            )


            rho = safe_spearman(
                source_cpt10k,
                recipient_cpt10k,
            )


            role = role_c[
                role_c[
                    "gene"
                ].eq(gene)
            ]

            if len(role) != 1:
                raise RuntimeError(
                    f"{sid}/{cname}/{gene}: "
                    "gene role ambiguity"
                )

            role = role.iloc[0]


            gene_rows.append({
                "sample": sid,
                "donor": donor,
                "contrast": cname,
                "source_class":
                    c["source"],
                "recipient_class":
                    c["recipient"],

                "gene": gene,
                "gene_role":
                    role["gene_role"],
                "matched_to_marker":
                    role[
                        "matched_to_marker"
                    ],

                "n_source_units":
                    int(
                        source_keep.sum()
                    ),

                "n_model":
                    n_model,

                "source_detection":
                    float(
                        np.mean(
                            source_gene_counts
                            > 0
                        )
                    ),

                "recipient_neighborhood_detection":
                    float(
                        np.mean(
                            recipient_gene_sum
                            > 0
                        )
                    ),

                "mean_source_cpt10k":
                    float(
                        np.mean(
                            source_cpt10k
                        )
                    ),

                "mean_recipient_neighborhood_cpt10k":
                    float(
                        np.mean(
                            recipient_cpt10k
                        )
                    ),

                "adjusted_standardized_source_beta":
                    beta,

                "source_recipient_spearman":
                    rho,
            })


        print(
            "genes evaluated          =",
            len(needed_genes),
        )


    del a


# ============================================================
# Save source-neighborhood geometry
# ============================================================

geometry = pd.concat(
    geometry_parts,
    ignore_index=True,
)

geometry.to_csv(
    OUT_GEOMETRY,
    index=False,
)


gene_df = pd.DataFrame(
    gene_rows
)

gene_df.to_csv(
    OUT_GENE,
    index=False,
)


# ============================================================
# Marker minus median of 3 frozen controls
# ============================================================

cmp_rows = []

group_cols = [
    "sample",
    "donor",
    "contrast",
    "matched_to_marker",
]


for keys, g in gene_df.groupby(
    group_cols,
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
            f"{keys}: expected 1 marker, "
            f"got {len(marker)}"
        )

    if len(controls) != 3:
        raise RuntimeError(
            f"{keys}: expected 3 controls, "
            f"got {len(controls)}"
        )


    m = marker.iloc[0]

    control_beta = controls[
        "adjusted_standardized_source_beta"
    ].median()

    control_rho = controls[
        "source_recipient_spearman"
    ].median()


    cmp_rows.append({
        "sample": keys[0],
        "donor": keys[1],
        "contrast": keys[2],
        "marker": keys[3],

        "n_source_units":
            int(
                m[
                    "n_source_units"
                ]
            ),

        "marker_beta":
            m[
                "adjusted_standardized_source_beta"
            ],

        "control_median_beta":
            control_beta,

        "delta_beta":
            (
                m[
                    "adjusted_standardized_source_beta"
                ]
                - control_beta
            ),

        "marker_spearman":
            m[
                "source_recipient_spearman"
            ],

        "control_median_spearman":
            control_rho,

        "delta_spearman":
            (
                m[
                    "source_recipient_spearman"
                ]
                - control_rho
            ),

        "marker_source_detection":
            m[
                "source_detection"
            ],

        "marker_recipient_neighborhood_detection":
            m[
                "recipient_neighborhood_detection"
            ],
    })


cmp = pd.DataFrame(
    cmp_rows
)

cmp.to_csv(
    OUT_MARKER,
    index=False,
)


# ============================================================
# Donor = biological replication unit
# ============================================================

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

        median_delta_spearman=(
            "delta_spearman",
            "median",
        ),

        markers_beta_stronger=(
            "delta_beta",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        markers_spearman_stronger=(
            "delta_spearman",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        min_source_units=(
            "n_source_units",
            "min",
        ),

        n_markers=(
            "marker",
            "nunique",
        ),
    )
)


donor[
    "predeclared_donor_pass"
] = (
    (
        donor[
            "median_delta_beta"
        ] > 0
    )
    &
    (
        donor[
            "markers_beta_stronger"
        ]
        >= MIN_MARKERS_PER_DONOR
    )
)


donor.to_csv(
    OUT_DONOR,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print(
    "\n" + "=" * 125
)
print(
    "SOURCE-CELL DOSE: "
    "MARKER VS FROZEN CONTROLS"
)
print(
    "=" * 125
)

print(
    cmp.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print(
    "\n" + "=" * 125
)
print(
    "DONOR-LEVEL SOURCE-DOSE SUMMARY"
)
print(
    "=" * 125
)

print(
    donor.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print(
    "\n" + "=" * 125
)
print(
    "CROSS-DONOR DIRECTION"
)
print(
    "=" * 125
)


for cname, g in donor.groupby(
    "contrast"
):

    print(
        "\n",
        cname,
    )

    print(
        "donors median delta_beta > 0:",
        int(
            np.sum(
                g[
                    "median_delta_beta"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors >=3/5 markers beta stronger:",
        int(
            np.sum(
                g[
                    "markers_beta_stronger"
                ]
                >= MIN_MARKERS_PER_DONOR
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors satisfying BOTH donor rules:",
        int(
            np.sum(
                g[
                    "predeclared_donor_pass"
                ]
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors median delta_spearman > 0:",
        int(
            np.sum(
                g[
                    "median_delta_spearman"
                ] > 0
            )
        ),
        "/",
        len(g),
    )


# ============================================================
# Frozen primary decision
# ============================================================

primary = donor[
    donor[
        "contrast"
    ].eq(
        PRIMARY_CONTRAST
    )
]

if len(primary) != len(SAMPLES):
    raise RuntimeError(
        "Primary contrast does not contain "
        "exactly four donor rows"
    )


n_primary_pass = int(
    primary[
        "predeclared_donor_pass"
    ].sum()
)


PRIMARY_PASS = (
    n_primary_pass
    >= MIN_DONORS_PRIMARY
)


print(
    "\n" + "=" * 125
)
print(
    "PREDECLARED PRIMARY GATE"
)
print(
    "=" * 125
)

print(
    "primary contrast:",
    PRIMARY_CONTRAST,
)

print(
    "donor rule:",
    "median delta_beta > 0 "
    "AND >=3/5 markers delta_beta > 0",
)

print(
    "primary rule:",
    ">=6/7 held-out sections pass",
)

print(
    "primary donors passing:",
    n_primary_pass,
    "/",
    len(primary),
)

print(
    "PRIMARY SOURCE-DOSE GATE:",
    "PASS"
    if PRIMARY_PASS
    else "FAIL",
)


print(
    "\nSAVED:",
    OUT_GEOMETRY,
)

print(
    "SAVED:",
    OUT_GENE,
)

print(
    "SAVED:",
    OUT_MARKER,
)

print(
    "SAVED:",
    OUT_DONOR,
)


print(
    "\nLOCK SHA256:",
    hashlib.sha256(
        LOCK.read_bytes()
    ).hexdigest(),
)

print(
    "\nSOURCE-CELL-CENTRED "
    "SOURCE-DOSE AUDIT: COMPLETE"
)
