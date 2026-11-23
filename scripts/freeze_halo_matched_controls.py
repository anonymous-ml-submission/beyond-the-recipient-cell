#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------
# IMPORTANT:
# Control matching uses ONLY non-TENX191 development samples.
# Halo results of candidate controls are NOT examined here.
# ------------------------------------------------------------

MATCH_SAMPLES = [
    ("TENX195", "Patient 2"),
    ("TENX197", "Patient 3"),
    ("TENX200", "Patient 4"),
]

PREP = Path("data/prepared_primary2025/samples")
ANN = Path("runs/independent_histo")

LOCK = Path(
    "runs/frozen_primary2025/"
    "leakage_marker_panel_pilot4.json"
)

OUT_JSON = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

OUT_CSV = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls_matching_table.csv"
)

cfg = json.loads(LOCK.read_text())

CONTRASTS = [
    {
        "name": "Connective_to_Inflammatory",
        "source": "Connective",
        "recipient": "Inflammatory",
        "markers": cfg["primary_source"]["genes"],
    },
    {
        "name": "Inflammatory_to_Connective",
        "source": "Inflammatory",
        "recipient": "Connective",
        "markers": cfg["orthogonal_source"]["genes"],
    },
]

ALL_LOCKED = set()

for c in CONTRASTS:
    ALL_LOCKED.update(c["markers"])


def summarize_axis(X, mask):
    n = int(mask.sum())

    if n == 0:
        raise RuntimeError("Empty cell class")

    Z = X[mask]

    if sp.issparse(Z):
        counts = np.asarray(
            Z.sum(axis=0)
        ).ravel().astype(float)

        det = (
            np.asarray(
                Z.getnnz(axis=0)
            ).ravel()
            / n
        )
    else:
        counts = np.asarray(
            Z.sum(axis=0)
        ).ravel().astype(float)

        det = np.mean(
            Z > 0,
            axis=0,
        )

    return counts, det


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-4,
        1 - 1e-4,
    )

    return np.log(
        p / (1 - p)
    )


rows = []

for sid, donor in MATCH_SAMPLES:

    print(
        "\n",
        sid,
        donor,
    )

    a = ad.read_h5ad(
        PREP / sid / "raw.h5ad"
    )

    obs = pd.Index(
        a.obs_names.astype(str),
        name="cell_id",
    )

    ann = pd.read_parquet(
        ANN / f"{sid}_cell_annotations.parquet"
    )

    ann["cell_id"] = (
        ann["cell_id"].astype(str)
    )

    if ann["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{sid}: duplicate annotation cell IDs"
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

    for c in CONTRASTS:

        source = (
            exact
            & ann["hovernet_type"]
              .eq(c["source"])
              .to_numpy()
        )

        recipient = (
            exact
            & ann["hovernet_type"]
              .eq(c["recipient"])
              .to_numpy()
        )

        if source.sum() == 0 or recipient.sum() == 0:
            raise RuntimeError(
                f"{sid}/{c['name']}: empty source or recipient"
            )

        source_counts, source_det = summarize_axis(
            a.X,
            source,
        )

        recipient_counts, recipient_det = summarize_axis(
            a.X,
            recipient,
        )

        source_lib = float(
            library[source].sum()
        )

        recipient_lib = float(
            library[recipient].sum()
        )

        source_cpt = (
            1e4
            * source_counts
            / source_lib
        )

        recipient_cpt = (
            1e4
            * recipient_counts
            / recipient_lib
        )

        eps = 1e-8

        log2_ratio = np.log2(
            (source_cpt + eps)
            /
            (recipient_cpt + eps)
        )

        for j, gene in enumerate(genes):

            rows.append({
                "sample": sid,
                "donor": donor,
                "contrast": c["name"],
                "gene": gene,

                "source_detection":
                    float(source_det[j]),

                "recipient_detection":
                    float(recipient_det[j]),

                "source_cpt10k":
                    float(source_cpt[j]),

                "recipient_cpt10k":
                    float(recipient_cpt[j]),

                "log2_source_recipient_ratio":
                    float(log2_ratio[j]),

                "total_counts":
                    float(
                        source_counts[j]
                        + recipient_counts[j]
                    ),
            })

    del a


detail = pd.DataFrame(rows)


# ------------------------------------------------------------
# Aggregate donor-level characteristics before matching.
# Matching variables deliberately include:
#
#   source abundance
#   recipient abundance
#   source detection
#   recipient detection
#   source/recipient specificity
#
# This prevents an "easy" null consisting of genes that are
# simply much rarer or less source-specific than the markers.
# ------------------------------------------------------------

agg = (
    detail
    .groupby(
        ["contrast", "gene"],
        as_index=False,
    )
    .agg(
        source_cpt10k=(
            "source_cpt10k",
            "median",
        ),
        recipient_cpt10k=(
            "recipient_cpt10k",
            "median",
        ),
        source_detection=(
            "source_detection",
            "median",
        ),
        recipient_detection=(
            "recipient_detection",
            "median",
        ),
        log2_source_recipient_ratio=(
            "log2_source_recipient_ratio",
            "median",
        ),
        total_counts=(
            "total_counts",
            "sum",
        ),
        donors=(
            "sample",
            "nunique",
        ),
    )
)


FEATURES = [
    "log_source_cpt",
    "log_recipient_cpt",
    "logit_source_detection",
    "logit_recipient_detection",
    "source_recipient_specificity",
]


matched_rows = []
frozen = {
    "status": "FROZEN_BEFORE_CONTROL_HALO_RESULTS",
    "matching_samples": [
        x[0]
        for x in MATCH_SAMPLES
    ],
    "n_controls_per_marker": 3,
    "matching_features": FEATURES,
    "contrasts": {},
}


for c in CONTRASTS:

    name = c["name"]
    markers = list(c["markers"])

    z = agg[
        agg["contrast"].eq(name)
    ].copy()

    z["log_source_cpt"] = np.log1p(
        z["source_cpt10k"]
    )

    z["log_recipient_cpt"] = np.log1p(
        z["recipient_cpt10k"]
    )

    z["logit_source_detection"] = logit(
        z["source_detection"]
    )

    z["logit_recipient_detection"] = logit(
        z["recipient_detection"]
    )

    z["source_recipient_specificity"] = (
        z["log2_source_recipient_ratio"]
    )

    marker_df = (
        z[
            z["gene"].isin(markers)
        ]
        .set_index("gene")
        .loc[markers]
        .reset_index()
    )

    if len(marker_df) != len(markers):
        raise RuntimeError(
            f"{name}: missing marker in matching table"
        )

    # Candidate controls:
    # - not one of any locked marker genes
    # - observed in all matching donors
    # - nonzero molecular support
    # - source-enriched in the SAME direction
    #
    # Requiring same direction makes the control test conservative.
    candidate = z[
        ~z["gene"].isin(ALL_LOCKED)
        & z["donors"].eq(len(MATCH_SAMPLES))
        & (z["total_counts"] > 0)
        & (
            z["log2_source_recipient_ratio"]
            > 0
        )
    ].copy()

    if len(candidate) < (
        len(markers) * 3
    ):
        raise RuntimeError(
            f"{name}: too few control candidates: "
            f"{len(candidate)}"
        )

    # Standardize using the full marker+candidate universe
    # for this contrast.
    base = pd.concat(
        [
            marker_df,
            candidate,
        ],
        ignore_index=True,
    )

    mu = (
        base[FEATURES]
        .mean(axis=0)
    )

    sd = (
        base[FEATURES]
        .std(axis=0, ddof=0)
        .replace(0, 1.0)
    )

    M = (
        marker_df[FEATURES]
        .sub(mu)
        .div(sd)
        .to_numpy(float)
    )

    C = (
        candidate[FEATURES]
        .sub(mu)
        .div(sd)
        .to_numpy(float)
    )

    # 3 unique controls per marker.
    marker_slots = np.repeat(
        np.arange(len(markers)),
        3,
    )

    cost = np.empty(
        (
            len(marker_slots),
            len(candidate),
        ),
        dtype=float,
    )

    for i, mi in enumerate(marker_slots):
        diff = (
            C
            - M[mi][None, :]
        )

        cost[i] = np.sqrt(
            np.sum(
                diff * diff,
                axis=1,
            )
        )

    row_ind, col_ind = linear_sum_assignment(
        cost
    )

    if len(row_ind) != len(marker_slots):
        raise RuntimeError(
            f"{name}: matching assignment incomplete"
        )

    control_map = {
        g: []
        for g in markers
    }

    for ri, ci in zip(
        row_ind,
        col_ind,
    ):

        marker = markers[
            marker_slots[ri]
        ]

        ctrl_row = candidate.iloc[ci]
        control = str(
            ctrl_row["gene"]
        )

        distance = float(
            cost[ri, ci]
        )

        control_map[marker].append(
            control
        )

        mr = marker_df[
            marker_df["gene"].eq(marker)
        ].iloc[0]

        matched_rows.append({
            "contrast": name,
            "marker": marker,
            "control": control,
            "matching_distance": distance,

            "marker_source_cpt10k":
                mr["source_cpt10k"],

            "control_source_cpt10k":
                ctrl_row["source_cpt10k"],

            "marker_recipient_cpt10k":
                mr["recipient_cpt10k"],

            "control_recipient_cpt10k":
                ctrl_row["recipient_cpt10k"],

            "marker_source_detection":
                mr["source_detection"],

            "control_source_detection":
                ctrl_row["source_detection"],

            "marker_recipient_detection":
                mr["recipient_detection"],

            "control_recipient_detection":
                ctrl_row["recipient_detection"],

            "marker_specificity":
                mr[
                    "log2_source_recipient_ratio"
                ],

            "control_specificity":
                ctrl_row[
                    "log2_source_recipient_ratio"
                ],
        })

    for marker in markers:

        if len(control_map[marker]) != 3:
            raise RuntimeError(
                f"{name}/{marker}: did not obtain 3 controls"
            )

    frozen["contrasts"][name] = {
        "source": c["source"],
        "recipient": c["recipient"],
        "markers": markers,
        "controls": control_map,
    }


OUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_JSON.write_text(
    json.dumps(
        frozen,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

match_table = pd.DataFrame(
    matched_rows
)

match_table.to_csv(
    OUT_CSV,
    index=False,
)


print("\n" + "=" * 100)
print("FROZEN MATCHED CONTROLS")
print("=" * 100)

for cname, z in frozen["contrasts"].items():

    print("\n", cname)

    for marker, controls in z["controls"].items():

        print(
            f"{marker:8s} -> "
            + ", ".join(controls)
        )


print("\nMATCH QUALITY")
print(
    match_table[
        [
            "contrast",
            "marker",
            "control",
            "matching_distance",
            "marker_source_cpt10k",
            "control_source_cpt10k",
            "marker_recipient_cpt10k",
            "control_recipient_cpt10k",
            "marker_specificity",
            "control_specificity",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

sha = hashlib.sha256(
    OUT_JSON.read_bytes()
).hexdigest()

print("\nSAVED:", OUT_JSON)
print("SAVED:", OUT_CSV)
print("SHA256:", sha)
print("\nMATCHED CONTROL FREEZE: PASS")
