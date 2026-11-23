#!/usr/bin/env python3

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd

from scipy.stats import binomtest


PRIMARY = Path(
    "runs/primary2025_context_v1"
)

ROT = Path(
    "runs/primary2025_donor_rotation_v1"
)


PATHS = {
    "Patient 1": {
        24:
            PRIMARY
            / "A_raw_ctx24"
            / "eval",

        128:
            PRIMARY
            / "A_raw_ctx128"
            / "eval",
    },

    "Patient 2": {
        24:
            ROT
            / "test_P2"
            / "raw_ctx24"
            / "eval",

        128:
            ROT
            / "test_P2"
            / "raw_ctx128"
            / "eval",
    },

    "Patient 3": {
        24:
            ROT
            / "test_P3"
            / "raw_ctx24"
            / "eval",

        128:
            ROT
            / "test_P3"
            / "raw_ctx128"
            / "eval",
    },

    "Patient 4": {
        24:
            ROT
            / "test_P4"
            / "raw_ctx24"
            / "eval",

        128:
            ROT
            / "test_P4"
            / "raw_ctx128"
            / "eval",
    },
}


def fisher_z_score(
    path,
):

    d = pd.read_csv(
        path
    )

    assert (
        d["gene"].nunique()
        == 280
    )

    donors = sorted(
        d[
            "donor_id"
        ]
        .astype(str)
        .unique()
    )

    assert len(donors) == 1

    x = pd.to_numeric(
        d["pcc"],
        errors="coerce",
    ).dropna().clip(
        -1 + 1e-6,
        1 - 1e-6,
    )

    return (
        float(
            np.arctanh(x)
            .mean()
        ),
        donors[0],
    )


rows = []

for patient, pp in PATHS.items():

    row = {
        "patient":
            patient,
    }

    actual_donor = None

    for target in [
        "raw",
        "clean",
    ]:

        for ctx in [
            24,
            128,
        ]:

            z, donor = fisher_z_score(
                pp[ctx]
                / (
                    f"metrics_{target}"
                    "_by_donor.csv"
                )
            )

            if actual_donor is None:
                actual_donor = donor

            elif donor != actual_donor:
                raise RuntimeError(
                    f"{patient}: donor mismatch"
                )

            row[
                f"{target}_z_ctx{ctx}"
            ] = z

        row[
            f"{target}_delta_z"
        ] = (
            row[
                f"{target}_z_ctx128"
            ]
            -
            row[
                f"{target}_z_ctx24"
            ]
        )

    row[
        "raw_minus_clean_delta_z"
    ] = (
        row[
            "raw_delta_z"
        ]
        -
        row[
            "clean_delta_z"
        ]
    )

    row["donor_id"] = (
        actual_donor
    )

    rows.append(row)


D = pd.DataFrame(
    rows
)

D.to_csv(
    ROT
    / "patient_level_context_effects.csv",
    index=False,
)


def one_sided_sign_test(
    x,
):

    x = np.asarray(
        x,
        float,
    )

    nz = x[
        x != 0
    ]

    if len(nz) == 0:
        return np.nan

    k = int(
        (nz > 0).sum()
    )

    return float(
        binomtest(
            k,
            n=len(nz),
            p=0.5,
            alternative="greater",
        ).pvalue
    )


def exact_signflip_two_sided(
    x,
):

    x = np.asarray(
        x,
        float,
    )

    obs = abs(
        x.mean()
    )

    vals = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(x),
    ):

        vals.append(
            abs(
                np.mean(
                    x
                    * np.asarray(signs)
                )
            )
        )

    vals = np.asarray(
        vals
    )

    return float(
        np.mean(
            vals
            >=
            obs - 1e-15
        )
    )


summary = {}

for col in [
    "raw_delta_z",
    "clean_delta_z",
    "raw_minus_clean_delta_z",
]:

    x = D[
        col
    ].to_numpy(float)

    summary[col] = {
        "n_patients":
            int(len(x)),

        "mean":
            float(
                np.mean(x)
            ),

        "median":
            float(
                np.median(x)
            ),

        "min":
            float(
                np.min(x)
            ),

        "max":
            float(
                np.max(x)
            ),

        "n_positive":
            int(
                (x > 0).sum()
            ),

        "one_sided_sign_test_p":
            one_sided_sign_test(
                x
            ),

        "two_sided_exact_signflip_p":
            exact_signflip_two_sided(
                x
            ),
    }


record = {
    "inference_unit":
        "patient",

    "n_independent_patients":
        4,

    "note":
        (
            "section-level metrics are descriptive; "
            "the four patient-level deltas are the "
            "biological replication units"
        ),

    "effects":
        summary,
}


(
    ROT
    / "patient_level_summary.json"
).write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n"
)


print(
    "\nPATIENT-LEVEL CONTEXT EFFECTS"
)

print(
    D.to_string(
        index=False
    )
)

print(
    "\nSUMMARY"
)

print(
    json.dumps(
        record,
        indent=2,
    )
)

print()
print(
    "IMPORTANT: with only four patients, "
    "even 4/4 positive gives one-sided "
    "sign-test p=0.0625. "
    "Report consistency/effect sizes; "
    "do not pretend the patient-level N is large."
)
