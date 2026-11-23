#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


ROOT = Path("runs/independent_histo")

CELL_GENE = (
    ROOT
    / "source_facing_molecules_cell_gene.csv"
)

LOCK = Path(
    "runs/frozen_primary2025/"
    "halo_matched_controls.json"
)

OUT_MARKER = (
    ROOT
    / "source_facing_same_recipient_marker.csv"
)

OUT_DONOR = (
    ROOT
    / "source_facing_same_recipient_donor.csv"
)

cfg = json.loads(
    LOCK.read_text()
)

x = pd.read_csv(
    CELL_GENE,
    dtype={
        "sample": str,
        "donor": str,
        "contrast": str,
        "feature_name": str,
        "gene_role": str,
        "matched_to_marker": str,
        "cell_id": str,
    },
)


required = {
    "sample",
    "donor",
    "contrast",
    "feature_name",
    "gene_role",
    "matched_to_marker",
    "cell_id",
    "n_molecules",
    "sum_u",
    "sum_projection_norm",
    "n_source_facing",
    "mean_u",
    "mean_projection_norm",
    "source_facing_fraction",
}

missing = required - set(x.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# There should be exactly one aggregate row per
# sample/contrast/gene/cell after the previous streaming run.
dup_cols = [
    "sample",
    "contrast",
    "feature_name",
    "cell_id",
]

if x.duplicated(dup_cols).any():
    raise RuntimeError(
        "Duplicate sample/contrast/gene/cell rows"
    )


rows = []


for sid in x["sample"].drop_duplicates():

    sx = x[
        x["sample"].eq(sid)
    ]

    donor_values = sx[
        "donor"
    ].drop_duplicates()

    if len(donor_values) != 1:
        raise RuntimeError(
            f"{sid}: donor ambiguity"
        )

    donor = donor_values.iloc[0]

    print(
        "\n" + "=" * 100
    )
    print(sid, donor)
    print("=" * 100)

    for cname, c in cfg[
        "contrasts"
    ].items():

        cx = sx[
            sx["contrast"].eq(cname)
        ]

        for marker in c["markers"]:

            controls = list(
                c["controls"][marker]
            )

            if len(controls) != 3:
                raise RuntimeError(
                    f"{cname}/{marker}: "
                    "expected 3 controls"
                )

            # --------------------------------------------
            # Marker cell-level geometry.
            # --------------------------------------------

            m = cx[
                cx["feature_name"].eq(marker)
                & cx["gene_role"].eq("marker")
            ][
                [
                    "cell_id",
                    "n_molecules",
                    "sum_u",
                    "sum_projection_norm",
                    "n_source_facing",
                    "mean_u",
                    "mean_projection_norm",
                    "source_facing_fraction",
                ]
            ].copy()

            m = m.rename(
                columns={
                    "n_molecules":
                        "marker_n_molecules",

                    "sum_u":
                        "marker_sum_u",

                    "sum_projection_norm":
                        "marker_sum_projection",

                    "n_source_facing":
                        "marker_n_facing",

                    "mean_u":
                        "marker_mean_u",

                    "mean_projection_norm":
                        "marker_mean_projection",

                    "source_facing_fraction":
                        "marker_facing_fraction",
                }
            )


            # --------------------------------------------
            # Pool the marker's THREE FROZEN controls
            # WITHIN EACH RECIPIENT CELL.
            #
            # This is important:
            # marker and controls are now evaluated on
            # exactly the same recipient cells.
            # --------------------------------------------

            cc = cx[
                cx["feature_name"].isin(
                    controls
                )
                & cx[
                    "matched_to_marker"
                ].eq(marker)
                & cx[
                    "gene_role"
                ].eq("control")
            ].copy()

            cpool = (
                cc.groupby(
                    "cell_id",
                    as_index=False,
                )
                .agg(
                    control_n_molecules=(
                        "n_molecules",
                        "sum",
                    ),

                    control_sum_u=(
                        "sum_u",
                        "sum",
                    ),

                    control_sum_projection=(
                        "sum_projection_norm",
                        "sum",
                    ),

                    control_n_facing=(
                        "n_source_facing",
                        "sum",
                    ),

                    control_genes_observed=(
                        "feature_name",
                        "nunique",
                    ),
                )
            )

            cpool[
                "control_mean_u"
            ] = (
                cpool["control_sum_u"]
                / cpool[
                    "control_n_molecules"
                ]
            )

            cpool[
                "control_mean_projection"
            ] = (
                cpool[
                    "control_sum_projection"
                ]
                / cpool[
                    "control_n_molecules"
                ]
            )

            cpool[
                "control_facing_fraction"
            ] = (
                cpool[
                    "control_n_facing"
                ]
                / cpool[
                    "control_n_molecules"
                ]
            )


            # --------------------------------------------
            # INNER JOIN:
            #
            # every comparison below is therefore on
            # the SAME recipient cell.
            # --------------------------------------------

            z = m.merge(
                cpool,
                on="cell_id",
                how="inner",
                validate="one_to_one",
            )

            if len(z) == 0:
                raise RuntimeError(
                    f"{sid}/{cname}/{marker}: "
                    "zero same-recipient pairs"
                )


            z["delta_u"] = (
                z["marker_mean_u"]
                - z["control_mean_u"]
            )

            z[
                "delta_projection"
            ] = (
                z[
                    "marker_mean_projection"
                ]
                - z[
                    "control_mean_projection"
                ]
            )

            z[
                "delta_facing_fraction"
            ] = (
                z[
                    "marker_facing_fraction"
                ]
                - z[
                    "control_facing_fraction"
                ]
            )


            # Equal recipient-cell weighting.
            # Molecule-rich cells do NOT dominate.
            row = {
                "sample": sid,
                "donor": donor,
                "contrast": cname,
                "marker": marker,

                "controls":
                    ",".join(controls),

                "marker_cells_total":
                    int(len(m)),

                "control_cells_total":
                    int(len(cpool)),

                "paired_cells":
                    int(len(z)),

                "paired_fraction_of_marker_cells":
                    float(
                        len(z) / len(m)
                    ),

                "mean_delta_u":
                    float(
                        z["delta_u"].mean()
                    ),

                "median_delta_u":
                    float(
                        z["delta_u"].median()
                    ),

                "fraction_cells_delta_u_positive":
                    float(
                        np.mean(
                            z["delta_u"] > 0
                        )
                    ),

                "mean_delta_projection":
                    float(
                        z[
                            "delta_projection"
                        ].mean()
                    ),

                "median_delta_projection":
                    float(
                        z[
                            "delta_projection"
                        ].median()
                    ),

                "mean_delta_facing_fraction":
                    float(
                        z[
                            "delta_facing_fraction"
                        ].mean()
                    ),

                "median_delta_facing_fraction":
                    float(
                        z[
                            "delta_facing_fraction"
                        ].median()
                    ),

                "marker_molecules_in_pairs":
                    int(
                        z[
                            "marker_n_molecules"
                        ].sum()
                    ),

                "control_molecules_in_pairs":
                    int(
                        z[
                            "control_n_molecules"
                        ].sum()
                    ),
            }

            rows.append(row)

            print(
                f"{cname:32s} "
                f"{marker:8s} "
                f"paired={len(z):5d} "
                f"mean_du="
                f"{row['mean_delta_u']:+.4f} "
                f"median_du="
                f"{row['median_delta_u']:+.4f} "
                f"P(du>0)="
                f"{row['fraction_cells_delta_u_positive']:.4f}"
            )


marker = pd.DataFrame(
    rows
)

marker.to_csv(
    OUT_MARKER,
    index=False,
)


# ------------------------------------------------------------
# Donor = biological replication unit.
#
# Primary directional quantity:
# median across the five locked markers of each marker's
# SAME-RECIPIENT mean delta_u.
# ------------------------------------------------------------

donor = (
    marker.groupby(
        [
            "sample",
            "donor",
            "contrast",
        ],
        as_index=False,
    )
    .agg(
        median_marker_mean_delta_u=(
            "mean_delta_u",
            "median",
        ),

        median_marker_median_delta_u=(
            "median_delta_u",
            "median",
        ),

        median_marker_mean_delta_projection=(
            "mean_delta_projection",
            "median",
        ),

        median_marker_mean_delta_facing_fraction=(
            "mean_delta_facing_fraction",
            "median",
        ),

        markers_mean_u_positive=(
            "mean_delta_u",
            lambda v: int(
                np.sum(
                    np.asarray(v) > 0
                )
            ),
        ),

        markers_mean_projection_positive=(
            "mean_delta_projection",
            lambda v: int(
                np.sum(
                    np.asarray(v) > 0
                )
            ),
        ),

        markers_mean_facing_positive=(
            "mean_delta_facing_fraction",
            lambda v: int(
                np.sum(
                    np.asarray(v) > 0
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

donor.to_csv(
    OUT_DONOR,
    index=False,
)


print(
    "\n" + "=" * 120
)
print(
    "SAME-RECIPIENT MARKER VS FROZEN-CONTROL"
)
print(
    "=" * 120
)

print(
    marker.to_string(
        index=False,
        float_format=lambda v: f"{v:+.4f}",
    )
)


print(
    "\n" + "=" * 120
)
print(
    "DONOR-LEVEL SAME-RECIPIENT SUMMARY"
)
print(
    "=" * 120
)

print(
    donor.to_string(
        index=False,
        float_format=lambda v: f"{v:+.4f}",
    )
)


print(
    "\n" + "=" * 120
)
print(
    "CROSS-DONOR DIRECTION"
)
print(
    "=" * 120
)

for cname, g in donor.groupby(
    "contrast"
):

    print("\n", cname)

    print(
        "donors paired mean delta_u > 0:",
        int(
            np.sum(
                g[
                    "median_marker_mean_delta_u"
                ] > 0
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors >=3/5 markers delta_u > 0:",
        int(
            np.sum(
                g[
                    "markers_mean_u_positive"
                ] >= 3
            )
        ),
        "/",
        len(g),
    )

    print(
        "donors paired projection > 0:",
        int(
            np.sum(
                g[
                    "median_marker_mean_delta_projection"
                ] > 0
            )
        ),
        "/",
        len(g),
    )


print("\nSAVED:", OUT_MARKER)
print("SAVED:", OUT_DONOR)

print(
    "INPUT SHA256:",
    hashlib.sha256(
        CELL_GENE.read_bytes()
    ).hexdigest()
)

print(
    "LOCK SHA256:",
    hashlib.sha256(
        LOCK.read_bytes()
    ).hexdigest()
)

print(
    "\nSAME-RECIPIENT ANGULAR REANALYSIS: COMPLETE"
)
