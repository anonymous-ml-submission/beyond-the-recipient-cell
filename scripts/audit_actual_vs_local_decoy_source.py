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

PRIMARY_CONTRAST = (
    "Connective_to_Inflammatory"
)

NEAR_UM = 40.0

# Fixed before seeing identity results.
N_LOCAL_DECOYS = 5

MIN_MARKERS_PER_DONOR = 3
MIN_DONORS_PRIMARY = 3


OUT_MAP = (
    ANNROOT
    / "actual_vs_local_decoy_source_map.csv"
)

OUT_GENE = (
    ANNROOT
    / "actual_vs_local_decoy_gene_summary.csv"
)

OUT_MARKER = (
    ANNROOT
    / "actual_vs_local_decoy_marker_summary.csv"
)

OUT_DONOR = (
    ANNROOT
    / "actual_vs_local_decoy_donor_summary.csv"
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


def z_required(v):

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


def z_optional(v):

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


def adjusted_beta(
    predictor,
    outcome,
    covariates,
):
    """
    Standardized coefficient:

      log1p(outcome)
          ~
      log1p(predictor)
          +
      frozen source/neighborhood covariates

    Actual and decoy models use EXACTLY the same outcome and
    nuisance-covariate matrix. Only source identity changes.
    """

    x = np.asarray(
        predictor,
        dtype=float,
    )

    y = np.asarray(
        outcome,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    for v in covariates.values():
        valid &= np.isfinite(
            np.asarray(
                v,
                dtype=float,
            )
        )

    x = x[valid]
    y = y[valid]

    cov = {
        k: np.asarray(
            v,
            dtype=float,
        )[valid]
        for k, v in covariates.items()
    }

    if len(x) < 20:
        return np.nan, int(len(x))

    zx = z_required(
        np.log1p(x)
    )

    zy = z_required(
        np.log1p(y)
    )

    if zx is None or zy is None:
        return np.nan, int(len(x))

    cols = [
        np.ones(
            len(x),
            dtype=float,
        ),
        zx,
    ]

    for name, v in cov.items():

        zv = z_optional(
            np.log1p(v)
        )

        if zv is not None:
            cols.append(zv)

    X = np.column_stack(
        cols
    )

    beta = np.linalg.lstsq(
        X,
        zy,
        rcond=None,
    )[0]

    return (
        float(beta[1]),
        int(len(x)),
    )


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


def finite_median(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    v = v[
        np.isfinite(v)
    ]

    if len(v) == 0:
        return np.nan

    return float(
        np.median(v)
    )


# ============================================================
# Frozen gene-role table
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
                "expected exactly 3 controls"
            )

        for ctrl in controls:

            role_rows.append({
                "contrast": cname,
                "gene": ctrl,
                "gene_role": "control",
                "matched_to_marker": marker,
            })


roles = pd.DataFrame(
    role_rows
)


if roles.duplicated(
    [
        "contrast",
        "gene",
    ]
).any():

    raise RuntimeError(
        "Frozen role ambiguity"
    )


# ============================================================
# Run
# ============================================================

map_parts = []
gene_rows = []


for sid, donor in SAMPLES:

    print(
        "\n" + "=" * 115
    )

    print(
        sid,
        donor,
    )

    print(
        "=" * 115
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
        cells[
            "cell_id"
        ].astype(str)
    )

    if cells[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{sid}: duplicate cells"
        )

    cells = (
        cells
        .set_index("cell_id")
        .reindex(obs)
    )


    required = [
        "x_px",
        "y_px",
        "polygon_area_um2",
    ]

    if cells[
        required
    ].isna().any().any():

        raise RuntimeError(
            f"{sid}: missing prepared-cell geometry"
        )


    ann = pd.read_parquet(
        ANNROOT
        / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann[
            "cell_id"
        ].astype(str)
    )

    if ann[
        "cell_id"
    ].duplicated().any():

        raise RuntimeError(
            f"{sid}: duplicate annotations"
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


    library = np.asarray(
        a.X.sum(axis=1)
    ).ravel().astype(float)


    xy_all = cells[
        ["x_px", "y_px"]
    ].to_numpy(float)


    area_all = pd.to_numeric(
        cells[
            "polygon_area_um2"
        ],
        errors="coerce",
    ).to_numpy(float)


    # Same density quantity used previously.
    tree_all = cKDTree(
        xy_all
    )

    d5_all, _ = tree_all.query(
        xy_all,
        k=6,
        workers=-1,
    )

    d5_px_all = (
        d5_all[:, 5]
    )


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
                "empty source/recipient class"
            )


        source_xy = (
            xy_all[
                source_idx
            ]
        )

        recipient_xy = (
            xy_all[
                recipient_idx
            ]
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


        # ----------------------------------------------------
        # Frozen recipient -> ACTUAL nearest source relation.
        # ----------------------------------------------------

        source_tree = cKDTree(
            source_xy
        )

        dist_px, nearest_local = (
            source_tree.query(
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


        valid_recipient = (
            np.isfinite(
                dist_um
            )
            & (
                dist_um < NEAR_UM
            )
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
        )


        ridx = (
            recipient_idx[
                valid_recipient
            ]
        )

        nearest = (
            nearest_local[
                valid_recipient
            ].astype(int)
        )

        rdist = (
            dist_um[
                valid_recipient
            ]
        )

        rlib = (
            recipient_lib[
                valid_recipient
            ]
        )

        rarea = (
            recipient_area[
                valid_recipient
            ]
        )

        rd5 = (
            recipient_d5[
                valid_recipient
            ]
        )


        n_source = len(
            source_idx
        )


        # ----------------------------------------------------
        # Aggregate recipients BY ACTUAL source.
        # ----------------------------------------------------

        n_rec = np.bincount(
            nearest,
            minlength=n_source,
        ).astype(float)


        rec_lib_sum = np.bincount(
            nearest,
            weights=rlib,
            minlength=n_source,
        )


        rec_dist_sum = np.bincount(
            nearest,
            weights=rdist,
            minlength=n_source,
        )


        rec_area_sum = np.bincount(
            nearest,
            weights=rarea,
            minlength=n_source,
        )


        rec_d5_sum = np.bincount(
            nearest,
            weights=rd5,
            minlength=n_source,
        )


        # Valid ACTUAL source cells.
        source_valid_base = (
            np.isfinite(
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
        )


        source_keep = (
            source_valid_base
            & (
                n_rec > 0
            )
            & np.isfinite(
                rec_lib_sum
            )
            & (
                rec_lib_sum > 0
            )
        )


        sk = np.flatnonzero(
            source_keep
        )


        if len(sk) < 20:
            raise RuntimeError(
                f"{sid}/{cname}: "
                "too few source units"
            )


        # ----------------------------------------------------
        # Geometry / nuisance quantities for ACTUAL source
        # neighborhoods.
        # ----------------------------------------------------

        mean_rec_dist = (
            rec_dist_sum[sk]
            / n_rec[sk]
        )

        mean_rec_area = (
            rec_area_sum[sk]
            / n_rec[sk]
        )

        mean_rec_d5 = (
            rec_d5_sum[sk]
            / n_rec[sk]
        )

        mean_rec_lib = (
            rec_lib_sum[sk]
            / n_rec[sk]
        )


        # ====================================================
        # LOCAL DECOY SOURCE MAP
        #
        # Candidate decoys:
        #   same H&E morphology class
        #   exact one-nucleus/one-cell cells
        #   valid source geometry/library
        #
        # Selection:
        #   5 nearest OTHER source cells to the actual source.
        #
        # Critically, NO expression values are used.
        # ====================================================

        candidate_local = np.flatnonzero(
            source_valid_base
        )

        if len(candidate_local) <= (
            N_LOCAL_DECOYS
        ):
            raise RuntimeError(
                f"{sid}/{cname}: "
                "not enough valid local-decoy sources"
            )


        candidate_xy = (
            source_xy[
                candidate_local
            ]
        )


        decoy_tree = cKDTree(
            candidate_xy
        )


        # Extra neighbors make self-exclusion robust even
        # if source centroids are duplicated.
        k_query = min(
            len(candidate_local),
            N_LOCAL_DECOYS + 8,
        )


        dd, ii = decoy_tree.query(
            source_xy[sk],
            k=k_query,
            workers=-1,
        )


        if dd.ndim == 1:
            dd = dd[:, None]
            ii = ii[:, None]


        decoy_local = np.full(
            (
                len(sk),
                N_LOCAL_DECOYS,
            ),
            -1,
            dtype=int,
        )


        decoy_dist_um = np.full(
            (
                len(sk),
                N_LOCAL_DECOYS,
            ),
            np.nan,
            dtype=float,
        )


        for row in range(
            len(sk)
        ):

            actual_local = int(
                sk[row]
            )

            chosen = []
            chosen_dist = []
            seen = set()

            for dpx, cand_pos in zip(
                dd[row],
                ii[row],
            ):

                cand_local = int(
                    candidate_local[
                        int(cand_pos)
                    ]
                )

                if (
                    cand_local
                    == actual_local
                ):
                    continue

                if cand_local in seen:
                    continue

                seen.add(
                    cand_local
                )

                chosen.append(
                    cand_local
                )

                chosen_dist.append(
                    float(dpx) * mpp
                )

                if (
                    len(chosen)
                    == N_LOCAL_DECOYS
                ):
                    break


            # Extremely defensive fallback.
            if (
                len(chosen)
                < N_LOCAL_DECOYS
            ):

                d2, i2 = (
                    decoy_tree.query(
                        source_xy[
                            actual_local
                        ],
                        k=min(
                            len(
                                candidate_local
                            ),
                            N_LOCAL_DECOYS
                            + 64,
                        ),
                    )
                )

                d2 = np.atleast_1d(
                    d2
                )

                i2 = np.atleast_1d(
                    i2
                )

                for dpx, cand_pos in zip(
                    d2,
                    i2,
                ):

                    cand_local = int(
                        candidate_local[
                            int(cand_pos)
                        ]
                    )

                    if (
                        cand_local
                        == actual_local
                        or cand_local
                        in seen
                    ):
                        continue

                    seen.add(
                        cand_local
                    )

                    chosen.append(
                        cand_local
                    )

                    chosen_dist.append(
                        float(dpx) * mpp
                    )

                    if (
                        len(chosen)
                        == N_LOCAL_DECOYS
                    ):
                        break


            if (
                len(chosen)
                != N_LOCAL_DECOYS
            ):
                raise RuntimeError(
                    f"{sid}/{cname}: "
                    f"could not obtain "
                    f"{N_LOCAL_DECOYS} "
                    f"local decoys for "
                    f"source unit {row}"
                )


            decoy_local[
                row,
                :
            ] = np.asarray(
                chosen,
                dtype=int,
            )

            decoy_dist_um[
                row,
                :
            ] = np.asarray(
                chosen_dist,
                dtype=float,
            )


        # ----------------------------------------------------
        # Save deterministic identity mapping.
        # ----------------------------------------------------

        actual_ids = (
            obs[
                source_idx[
                    sk
                ]
            ]
            .astype(str)
            .to_numpy()
        )


        map_df = pd.DataFrame({
            "sample": sid,
            "donor": donor,
            "contrast": cname,
            "actual_source_cell_id":
                actual_ids,
            "n_recipients":
                n_rec[
                    sk
                ].astype(int),
            "mean_recipient_distance_um":
                mean_rec_dist,
        })


        for rank in range(
            N_LOCAL_DECOYS
        ):

            didx = (
                decoy_local[
                    :,
                    rank
                ]
            )

            map_df[
                f"decoy_{rank+1}_cell_id"
            ] = (
                obs[
                    source_idx[
                        didx
                    ]
                ]
                .astype(str)
                .to_numpy()
            )

            map_df[
                f"decoy_{rank+1}_distance_um"
            ] = (
                decoy_dist_um[
                    :,
                    rank
                ]
            )


        map_parts.append(
            map_df
        )


        print(
            "source cells total        =",
            len(source_idx),
        )

        print(
            "recipient cells total     =",
            len(recipient_idx),
        )

        print(
            "near recipients <40 um    =",
            len(ridx),
        )

        print(
            "actual source units        =",
            len(sk),
        )

        print(
            "local decoys per unit      =",
            N_LOCAL_DECOYS,
        )

        print(
            "decoy-1 distance median um =",
            f"{np.median(decoy_dist_um[:,0]):.4f}",
        )

        print(
            "decoy-5 distance median um =",
            f"{np.median(decoy_dist_um[:,-1]):.4f}",
        )


        # ----------------------------------------------------
        # EXACT SAME nuisance matrix for actual and decoy
        # predictor models.
        # ----------------------------------------------------

        covariates = {
            "mean_recipient_distance_um":
                mean_rec_dist,

            "source_area_um2":
                source_area[sk],

            "mean_recipient_area_um2":
                mean_rec_area,

            "source_d5_px":
                source_d5[sk],

            "mean_recipient_d5_px":
                mean_rec_d5,

            "source_library":
                source_lib[sk],

            "mean_recipient_library":
                mean_rec_lib,

            "n_recipients":
                n_rec[sk],
        }


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


        # ====================================================
        # Gene evaluation
        # ====================================================

        for gene in needed_genes:

            if gene not in genes:
                raise RuntimeError(
                    f"{sid}/{cname}: "
                    f"missing frozen gene {gene}"
                )


            gv = dense_col(
                a.X,
                genes.get_loc(
                    gene
                ),
            )


            source_counts_all = (
                gv[
                    source_idx
                ]
            )


            # --------------------------------------------
            # ACTUAL source dose.
            # --------------------------------------------

            actual_source_counts = (
                source_counts_all[
                    sk
                ]
            )

            actual_source_cpt = (
                1e4
                * actual_source_counts
                / source_lib[
                    sk
                ]
            )


            # --------------------------------------------
            # Fixed recipient-neighborhood response.
            # --------------------------------------------

            recipient_counts = (
                gv[
                    ridx
                ]
            )


            recipient_gene_sum_all = (
                np.bincount(
                    nearest,
                    weights=recipient_counts,
                    minlength=n_source,
                )
            )


            recipient_gene_sum = (
                recipient_gene_sum_all[
                    sk
                ]
            )


            response_cpt = (
                1e4
                * recipient_gene_sum
                / rec_lib_sum[
                    sk
                ]
            )


            actual_beta, n_model = (
                adjusted_beta(
                    actual_source_cpt,
                    response_cpt,
                    covariates,
                )
            )


            actual_rho = (
                safe_spearman(
                    actual_source_cpt,
                    response_cpt,
                )
            )


            # --------------------------------------------
            # FIVE deterministic local-decoy models.
            # --------------------------------------------

            decoy_betas = []
            decoy_rhos = []
            decoy_detections = []


            for rank in range(
                N_LOCAL_DECOYS
            ):

                di = (
                    decoy_local[
                        :,
                        rank
                    ]
                )


                decoy_counts = (
                    source_counts_all[
                        di
                    ]
                )


                decoy_cpt = (
                    1e4
                    * decoy_counts
                    / source_lib[
                        di
                    ]
                )


                dbeta, dn = (
                    adjusted_beta(
                        decoy_cpt,
                        response_cpt,
                        covariates,
                    )
                )


                drho = (
                    safe_spearman(
                        decoy_cpt,
                        response_cpt,
                    )
                )


                decoy_betas.append(
                    dbeta
                )

                decoy_rhos.append(
                    drho
                )

                decoy_detections.append(
                    float(
                        np.mean(
                            decoy_counts > 0
                        )
                    )
                )


            valid_beta_n = int(
                np.sum(
                    np.isfinite(
                        decoy_betas
                    )
                )
            )


            if valid_beta_n < 3:
                raise RuntimeError(
                    f"{sid}/{cname}/{gene}: "
                    "fewer than 3 valid decoy betas"
                )


            median_decoy_beta = (
                finite_median(
                    decoy_betas
                )
            )

            median_decoy_rho = (
                finite_median(
                    decoy_rhos
                )
            )


            identity_delta_beta = (
                actual_beta
                - median_decoy_beta
            )


            identity_delta_rho = (
                actual_rho
                - median_decoy_rho
            )


            role = role_c[
                role_c[
                    "gene"
                ].eq(gene)
            ]

            if len(role) != 1:
                raise RuntimeError(
                    f"{sid}/{cname}/{gene}: "
                    "role ambiguity"
                )

            role = role.iloc[0]


            row = {
                "sample": sid,
                "donor": donor,
                "contrast": cname,
                "source_class":
                    c["source"],
                "recipient_class":
                    c["recipient"],

                "gene": gene,
                "gene_role":
                    role[
                        "gene_role"
                    ],
                "matched_to_marker":
                    role[
                        "matched_to_marker"
                    ],

                "n_source_units":
                    len(sk),

                "n_model":
                    n_model,

                "actual_beta":
                    actual_beta,

                "median_local_decoy_beta":
                    median_decoy_beta,

                "identity_delta_beta":
                    identity_delta_beta,

                "actual_spearman":
                    actual_rho,

                "median_local_decoy_spearman":
                    median_decoy_rho,

                "identity_delta_spearman":
                    identity_delta_rho,

                "actual_source_detection":
                    float(
                        np.mean(
                            actual_source_counts
                            > 0
                        )
                    ),

                "median_decoy_detection":
                    finite_median(
                        decoy_detections
                    ),

                "recipient_neighborhood_detection":
                    float(
                        np.mean(
                            recipient_gene_sum
                            > 0
                        )
                    ),
            }


            for rank in range(
                N_LOCAL_DECOYS
            ):

                row[
                    f"decoy_{rank+1}_beta"
                ] = (
                    decoy_betas[
                        rank
                    ]
                )

                row[
                    f"decoy_{rank+1}_spearman"
                ] = (
                    decoy_rhos[
                        rank
                    ]
                )


            gene_rows.append(
                row
            )


        print(
            "genes evaluated           =",
            len(needed_genes),
        )


    del a


# ============================================================
# Save deterministic local-decoy map
# ============================================================

decoy_map = pd.concat(
    map_parts,
    ignore_index=True,
)

decoy_map.to_csv(
    OUT_MAP,
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
# Marker summaries
#
# PRIMARY identity quantity:
#
# marker actual beta
#   - median beta of 5 local same-class decoys
#
# Secondary specificity quantity:
#
# marker identity advantage
#   - median identity advantage of its 3 frozen gene controls
#
# The secondary quantity does NOT alter the primary gate.
# ============================================================

marker_rows = []


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
        g[
            "gene_role"
        ].eq(
            "marker"
        )
    ]

    controls = g[
        g[
            "gene_role"
        ].eq(
            "control"
        )
    ]


    if len(marker) != 1:
        raise RuntimeError(
            f"{keys}: "
            f"expected one marker"
        )


    if len(controls) != 3:
        raise RuntimeError(
            f"{keys}: "
            f"expected three controls"
        )


    m = marker.iloc[0]


    control_identity_beta = (
        controls[
            "identity_delta_beta"
        ].median()
    )

    control_identity_rho = (
        controls[
            "identity_delta_spearman"
        ].median()
    )


    marker_rows.append({
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

        "actual_beta":
            m[
                "actual_beta"
            ],

        "median_local_decoy_beta":
            m[
                "median_local_decoy_beta"
            ],

        "identity_delta_beta":
            m[
                "identity_delta_beta"
            ],

        "actual_spearman":
            m[
                "actual_spearman"
            ],

        "median_local_decoy_spearman":
            m[
                "median_local_decoy_spearman"
            ],

        "identity_delta_spearman":
            m[
                "identity_delta_spearman"
            ],

        "control_median_identity_delta_beta":
            control_identity_beta,

        "marker_minus_control_identity_delta_beta":
            (
                m[
                    "identity_delta_beta"
                ]
                - control_identity_beta
            ),

        "control_median_identity_delta_spearman":
            control_identity_rho,

        "marker_minus_control_identity_delta_spearman":
            (
                m[
                    "identity_delta_spearman"
                ]
                - control_identity_rho
            ),
    })


marker_df = pd.DataFrame(
    marker_rows
)

marker_df.to_csv(
    OUT_MARKER,
    index=False,
)


# ============================================================
# Donor-level replication
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
        median_identity_delta_beta=(
            "identity_delta_beta",
            "median",
        ),

        median_identity_delta_spearman=(
            "identity_delta_spearman",
            "median",
        ),

        markers_actual_beta_gt_decoy=(
            "identity_delta_beta",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        markers_actual_rho_gt_decoy=(
            "identity_delta_spearman",
            lambda x: int(
                np.sum(
                    np.asarray(x) > 0
                )
            ),
        ),

        # Secondary marker-specificity diagnostic.
        median_marker_minus_control_identity_beta=(
            "marker_minus_control_identity_delta_beta",
            "median",
        ),

        markers_identity_advantage_gt_controls=(
            "marker_minus_control_identity_delta_beta",
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
    "predeclared_identity_donor_pass"
] = (
    (
        donor[
            "median_identity_delta_beta"
        ] > 0
    )
    &
    (
        donor[
            "markers_actual_beta_gt_decoy"
        ]
        >= MIN_MARKERS_PER_DONOR
    )
)


donor.to_csv(
    OUT_DONOR,
    index=False,
)


# ============================================================
# Output
# ============================================================

print(
    "\n" + "=" * 135
)
print(
    "ACTUAL SOURCE VS LOCAL-DECOY SOURCE: "
    "LOCKED MARKERS"
)
print(
    "=" * 135
)

print(
    marker_df.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print(
    "\n" + "=" * 135
)
print(
    "DONOR-LEVEL SOURCE-IDENTITY SUMMARY"
)
print(
    "=" * 135
)

print(
    donor.to_string(
        index=False,
        float_format=lambda x: f"{x:+.4f}",
    )
)


print(
    "\n" + "=" * 135
)
print(
    "CROSS-DONOR IDENTITY DIRECTION"
)
print(
    "=" * 135
)


for cname, g in donor.groupby(
    "contrast"
):

    print(
        "\n",
        cname,
    )

    print(
        "donors median actual-minus-decoy beta > 0:",
        int(
            np.sum(
                g[
                    "median_identity_delta_beta"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors >=3/5 markers actual beta > decoy:",
        int(
            np.sum(
                g[
                    "markers_actual_beta_gt_decoy"
                ]
                >= MIN_MARKERS_PER_DONOR
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors satisfying BOTH identity rules:",
        int(
            np.sum(
                g[
                    "predeclared_identity_donor_pass"
                ]
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors median actual-minus-decoy Spearman > 0:",
        int(
            np.sum(
                g[
                    "median_identity_delta_spearman"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "SECONDARY: donors median marker identity "
        "advantage > frozen controls:",
        int(
            np.sum(
                g[
                    "median_marker_minus_control_identity_beta"
                ] > 0
            )
        ),
        "/",
        len(g),
    )


# ============================================================
# PREDECLARED PRIMARY DECISION
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
        "Primary contrast must contain four donors"
    )


n_primary_pass = int(
    primary[
        "predeclared_identity_donor_pass"
    ].sum()
)


PRIMARY_PASS = (
    n_primary_pass
    >= MIN_DONORS_PRIMARY
)


print(
    "\n" + "=" * 135
)
print(
    "PREDECLARED PRIMARY SOURCE-IDENTITY GATE"
)
print(
    "=" * 135
)

print(
    "primary contrast:",
    PRIMARY_CONTRAST,
)

print(
    "local decoys:",
    N_LOCAL_DECOYS,
    "nearest OTHER valid exact source cells",
)

print(
    "donor rule:",
    "median(actual beta - median local-decoy beta) > 0 "
    "AND >=3/5 markers positive",
)

print(
    "primary rule:",
    ">=3/4 donors pass",
)

print(
    "primary donors passing:",
    n_primary_pass,
    "/",
    len(primary),
)

print(
    "PRIMARY SOURCE-IDENTITY GATE:",
    "PASS"
    if PRIMARY_PASS
    else "FAIL",
)


print(
    "\nSAVED:",
    OUT_MAP,
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
    "\nACTUAL VS LOCAL-DECOY "
    "SOURCE AUDIT: COMPLETE"
)
