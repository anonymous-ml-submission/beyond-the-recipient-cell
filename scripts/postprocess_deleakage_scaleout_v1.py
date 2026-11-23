#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np


G_EXPECTED = 280
K_EXPECTED = 3

RETAINED_BLOCKS = list(
    range(
        5500,
        10001,
        500,
    )
)

REFERENCE_POSTPROCESS_SCRIPT = Path(
    "scripts/postprocess_deleakage_tenx199_v1.py"
)

REFERENCE_POSTPROCESS_SCRIPT_SHA = (
    "c6b0b4230d8269ed09d2ef2436a4f05"
    "cc474da036932f749b6c23ea8e02ff825"
)

REFERENCE_PROD_MANIFEST_SHA = (
    "5507923201e9664efa31dab8faad4bbcd"
    "4b44012c939dc0eedefaaf7cea0df0b"
)


def fail(msg):
    raise RuntimeError(msg)


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def verify_output_manifest(
    prod: Path,
):

    m = (
        prod
        / "outputs.sha256"
    )

    if not m.is_file():
        fail(
            f"missing production output manifest: {m}"
        )

    expected_names = set()

    for s in range(
        500,
        10001,
        500,
    ):

        expected_names |= {
            f"mu_{s}.txt",
            f"gamma_{s}.txt",
            f"beta_{s}.txt",
            f"I_beta_{s}.txt",
            f"I_gamma_{s}.txt",
            f"sigma_sq_{s}.txt",
            f"Z_{s // 25}.txt",
            f"pi_{s}.txt",
        }

    seen = set()

    raw_lines = [
        x.strip()
        for x in m.read_text().splitlines()
        if x.strip()
    ]

    # Historical TENX199 freeze used shell redirection directly into
    # outputs.sha256 while `find` enumerated the directory. Therefore
    # that manifest contains one self-entry for outputs.sha256 in
    # addition to the 160 actual MCMC output files.
    #
    # Later scale-out freezes explicitly excluded outputs.sha256.
    # Preserve both frozen formats; the self-entry is metadata, not a
    # posterior output and is intentionally not re-hashed here.
    lines = []
    self_entries = []

    for line in raw_lines:

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            fail(
                f"bad hash-manifest line: {line}"
            )

        name = parts[1].strip()

        if name.startswith("./"):
            name = name[2:]

        if name == "outputs.sha256":
            self_entries.append(line)
        else:
            lines.append(line)

    if len(self_entries) > 1:
        fail(
            f"{prod}: multiple outputs.sha256 self-entries"
        )

    if len(lines) != 160:
        fail(
            f"{prod}: expected 160 posterior hashes after "
            f"excluding manifest self-entry, found {len(lines)}"
        )

    if len(raw_lines) not in (160, 161):
        fail(
            f"{prod}: unexpected total manifest lines={len(raw_lines)}"
        )

    for line in lines:

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            fail(
                f"bad hash-manifest line: {line}"
            )

        expected_sha, name = parts

        name = name.strip()

        if name.startswith("./"):
            name = name[2:]

        p = (
            prod
            / name
        )

        if not p.is_file():
            fail(
                f"posterior file missing: {p}"
            )

        got = sha256(p)

        if got != expected_sha:
            fail(
                f"posterior byte mismatch: {p}\n"
                f"expected={expected_sha}\n"
                f"actual={got}"
            )

        seen.add(name)

    if seen != expected_names:
        fail(
            f"{prod}: posterior filename set mismatch\n"
            f"missing={sorted(expected_names - seen)}\n"
            f"extra={sorted(seen - expected_names)}"
        )


def verify_protocol(
    protocol: Path,
    sid: str,
):

    if not protocol.is_file():
        fail(
            f"missing production protocol: {protocol}"
        )

    p = json.loads(
        protocol.read_text()
    )

    if p["sample"] != sid:
        fail(
            f"{sid}: production protocol sample mismatch"
        )

    r = p["runtime"]

    if int(
        r["n_record"]
    ) != 10000:
        fail(
            f"{sid}: n_record changed"
        )

    if int(
        r["burn_in_iterations"]
    ) != 5000:
        fail(
            f"{sid}: burn-in changed"
        )

    if int(
        r["retained_iterations"]
    ) != 5000:
        fail(
            f"{sid}: retained-iteration contract changed"
        )

    if int(
        r["seed"]
    ) != 123:
        fail(
            f"{sid}: seed changed"
        )

    if int(
        r["OMP_NUM_THREADS"]
    ) != 24:
        fail(
            f"{sid}: OMP contract changed"
        )


def load_contracts(
    sid: str,
    structural: Path,
    labels: Path,
):

    sm = json.loads(
        (
            structural
            / "structural_manifest.json"
        ).read_text()
    )

    lm = json.loads(
        (
            labels
            / "label_manifest.json"
        ).read_text()
    )

    if sm["sample"] != sid:
        fail(
            f"{sid}: structural sample mismatch"
        )

    if lm["sample"] != sid:
        fail(
            f"{sid}: label sample mismatch"
        )

    G = int(
        sm["G"]
    )

    N = int(
        sm["N"]
    )

    K = int(
        lm["K"]
    )

    if G != G_EXPECTED:
        fail(
            f"{sid}: G={G}"
        )

    if K != K_EXPECTED:
        fail(
            f"{sid}: K={K}"
        )

    if int(
        lm["N"]
    ) != N:
        fail(
            f"{sid}: label N mismatch"
        )

    if int(
        lm["G"]
    ) != G:
        fail(
            f"{sid}: label G mismatch"
        )

    genes = (
        structural
        / "genes.txt"
    ).read_text().splitlines()

    cell_ids = (
        structural
        / "cell_ids.txt"
    ).read_text().splitlines()

    label_cells = (
        labels
        / "cell_ids.txt"
    ).read_text().splitlines()

    if len(genes) != G:
        fail(
            f"{sid}: gene count mismatch"
        )

    if len(cell_ids) != N:
        fail(
            f"{sid}: cell count mismatch"
        )

    if cell_ids != label_cells:
        fail(
            f"{sid}: label/structural cell order mismatch"
        )

    return (
        G,
        N,
        K,
        genes,
        cell_ids,
    )


def load_posterior(
    prod: Path,
    N: int,
    G: int,
    K: int,
):

    mu_blocks = []
    gamma_blocks = []
    beta_blocks = []

    z_counts = np.zeros(
        (N, K),
        dtype=np.uint16,
    )

    z_counts_first = np.zeros(
        (N, K),
        dtype=np.uint16,
    )

    z_counts_second = np.zeros(
        (N, K),
        dtype=np.uint16,
    )

    gamma_eq1 = 0
    gamma_gt1 = 0

    z_samples = 0

    for block_idx, s in enumerate(
        RETAINED_BLOCKS
    ):

        print(
            "reading retained block",
            s,
        )

        mu = np.loadtxt(
            prod
            / f"mu_{s}.txt",
            dtype=np.float64,
        )

        gamma = np.loadtxt(
            prod
            / f"gamma_{s}.txt",
            dtype=np.float64,
        )

        beta = np.loadtxt(
            prod
            / f"beta_{s}.txt",
            dtype=np.float64,
        )

        z = np.loadtxt(
            prod
            / f"Z_{s // 25}.txt",
            dtype=np.int16,
        )

        if mu.shape != (
            G * K,
            500,
        ):
            fail(
                f"mu_{s} shape={mu.shape}"
            )

        if gamma.shape != (
            G,
            500,
        ):
            fail(
                f"gamma_{s} shape={gamma.shape}"
            )

        if beta.shape != (
            500,
        ):
            fail(
                f"beta_{s} shape={beta.shape}"
            )

        if z.shape != (
            N,
            20,
        ):
            fail(
                f"Z_{s // 25} shape={z.shape}"
            )

        if not np.isfinite(
            mu
        ).all():
            fail(
                f"mu_{s} non-finite"
            )

        if not np.isfinite(
            gamma
        ).all():
            fail(
                f"gamma_{s} non-finite"
            )

        if not np.isfinite(
            beta
        ).all():
            fail(
                f"beta_{s} non-finite"
            )

        if np.any(
            gamma <= 0
        ):
            fail(
                f"gamma_{s} contains non-positive values"
            )

        if np.any(
            gamma > 1
        ):
            fail(
                f"gamma_{s} contains serialized values > 1"
            )

        if np.any(
            beta <= 0
        ):
            fail(
                f"beta_{s} contains non-positive values"
            )

        if np.any(
            (z < 0)
            |
            (z >= K)
        ):
            fail(
                f"Z_{s // 25} invalid labels"
            )

        gamma_eq1 += int(
            np.count_nonzero(
                gamma == 1
            )
        )

        gamma_gt1 += int(
            np.count_nonzero(
                gamma > 1
            )
        )

        mu_blocks.append(
            mu
        )

        gamma_blocks.append(
            gamma
        )

        beta_blocks.append(
            beta
        )

        for k in range(K):

            c = np.count_nonzero(
                z == k,
                axis=1,
            ).astype(
                np.uint16
            )

            z_counts[
                :,
                k
            ] += c

            if block_idx < 5:
                z_counts_first[
                    :,
                    k
                ] += c
            else:
                z_counts_second[
                    :,
                    k
                ] += c

        z_samples += 20

    mu_all = np.concatenate(
        mu_blocks,
        axis=1,
    )

    gamma_all = np.concatenate(
        gamma_blocks,
        axis=1,
    )

    beta_all = np.concatenate(
        beta_blocks,
    )

    if mu_all.shape != (
        G * K,
        5000,
    ):
        fail(
            f"retained mu shape={mu_all.shape}"
        )

    if gamma_all.shape != (
        G,
        5000,
    ):
        fail(
            f"retained gamma shape={gamma_all.shape}"
        )

    if beta_all.shape != (
        5000,
    ):
        fail(
            f"retained beta shape={beta_all.shape}"
        )

    if z_samples != 200:
        fail(
            f"retained Z snapshots={z_samples}"
        )

    mu_hat = (
        mu_all
        .mean(axis=1)
        .reshape(
            G,
            K,
        )
    )

    gamma_hat = (
        gamma_all
        .mean(axis=1)
    )

    beta_hat = float(
        beta_all.mean()
    )

    beta_sd = float(
        beta_all.std()
    )

    Z_hat = np.argmax(
        z_counts,
        axis=1,
    ).astype(
        np.int16
    )

    top = z_counts.max(
        axis=1,
        keepdims=True,
    )

    Z_ties = int(
        np.count_nonzero(
            np.count_nonzero(
                z_counts == top,
                axis=1,
            )
            > 1
        )
    )

    Z_first = np.argmax(
        z_counts_first,
        axis=1,
    )

    Z_second = np.argmax(
        z_counts_second,
        axis=1,
    )

    Z_split_disagreement = int(
        np.count_nonzero(
            Z_first
            != Z_second
        )
    )

    mu_first = (
        mu_all[
            :,
            :2500
        ]
        .mean(axis=1)
    )

    mu_second = (
        mu_all[
            :,
            2500:
        ]
        .mean(axis=1)
    )

    gamma_first = (
        gamma_all[
            :,
            :2500
        ]
        .mean(axis=1)
    )

    gamma_second = (
        gamma_all[
            :,
            2500:
        ]
        .mean(axis=1)
    )

    mu_split_corr = float(
        np.corrcoef(
            mu_first,
            mu_second,
        )[0, 1]
    )

    gamma_split_corr = float(
        np.corrcoef(
            gamma_first,
            gamma_second,
        )[0, 1]
    )

    return {
        "mu_hat":
            mu_hat,

        "gamma_hat":
            gamma_hat,

        "beta_hat":
            beta_hat,

        "beta_sd":
            beta_sd,

        "Z_hat":
            Z_hat,

        "Z_mode_counts":
            z_counts,

        "Z_ties":
            Z_ties,

        "Z_split_disagreement":
            Z_split_disagreement,

        "mu_split_corr":
            mu_split_corr,

        "gamma_split_corr":
            gamma_split_corr,

        "beta_first_half_mean":
            float(
                beta_all[
                    :2500
                ].mean()
            ),

        "beta_second_half_mean":
            float(
                beta_all[
                    2500:
                ].mean()
            ),

        "gamma_serialized_eq1":
            gamma_eq1,

        "gamma_serialized_gt1":
            gamma_gt1,
    }


def build_spatial_operator(
    structural: Path,
    N: int,
    beta_hat: float,
):

    cell_size = np.loadtxt(
        structural
        / "cell_size.txt",
        dtype=np.float64,
    )

    nei = np.loadtxt(
        structural
        / "nei_list.txt",
        dtype=np.int32,
    )

    dist = np.loadtxt(
        structural
        / "nei_dist.txt",
        dtype=np.float64,
    )

    if cell_size.shape != (
        N,
    ):
        fail(
            f"cell_size shape={cell_size.shape}"
        )

    if nei.shape[0] != N:
        fail(
            f"neighbor N mismatch: {nei.shape}"
        )

    if dist.shape != nei.shape:
        fail(
            "neighbor/distance shape mismatch"
        )

    if not np.isfinite(
        cell_size
    ).all():
        fail(
            "non-finite cell_size"
        )

    if np.any(
        cell_size < 0
    ):
        fail(
            "negative cell_size"
        )

    valid = (
        nei >= 0
    )

    if np.any(
        nei[
            valid
        ] >= N
    ):
        fail(
            "neighbor index outside section"
        )

    if np.any(
        dist[
            valid
        ] < 0
    ):
        fail(
            "negative valid distance"
        )

    N_nei = int(
        nei.shape[1]
    )

    flat_valid = (
        valid.ravel()
    )

    source = (
        nei.ravel()[
            flat_valid
        ]
        .astype(
            np.int32
        )
    )

    target = np.repeat(
        np.arange(
            N,
            dtype=np.int32,
        ),
        N_nei,
    )[
        flat_valid
    ]

    distance = (
        dist.ravel()[
            flat_valid
        ]
    )

    weight = (
        np.exp(
            -beta_hat
            * distance
        )
        *
        cell_size[
            source
        ]
    )

    if not np.isfinite(
        weight
    ).all():
        fail(
            "non-finite spatial weights"
        )

    if np.any(
        weight < 0
    ):
        fail(
            "negative spatial weights"
        )

    return (
        source,
        target,
        weight,
        len(weight),
    )


def correction_pass(
    structural: Path,
    genes,
    posterior,
    N: int,
    out_memmap=None,
    reference=None,
):

    mu_hat = np.asarray(
        posterior[
            "mu_hat"
        ],
        dtype=np.float64,
    )

    gamma_hat = np.asarray(
        posterior[
            "gamma_hat"
        ],
        dtype=np.float64,
    )

    beta_hat = float(
        posterior[
            "beta_hat"
        ]
    )

    Z_hat = np.asarray(
        posterior[
            "Z_hat"
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------------
    # Frozen release-compatible endpoint:
    #
    # Keep raw posterior mean mu unchanged for diagnostics.
    # For the correction operator only:
    #
    #     mu_effective = max(mu_hat, 0)
    #
    # No gamma clipping.
    # --------------------------------------------------------------

    mu_effective = np.maximum(
        mu_hat,
        0.0,
    )

    (
        source,
        target,
        weight,
        n_edges,
    ) = build_spatial_operator(
        structural,
        N,
        beta_hat,
    )

    YOBS = (
        structural
        / "Y_obs.txt"
    )

    qc_rows = []

    total_observed = 0.0
    total_corrected = 0.0

    total_decreased = 0
    total_corrected_zero = 0

    corrected_above_observed = 0
    zero_to_positive = 0

    min_leakage = np.inf
    max_leakage = -np.inf
    max_corrected_minus_observed = 0.0

    reference_max_abs = 0.0

    with YOBS.open() as f:

        rows = 0

        for g, line in enumerate(f):

            y = np.fromstring(
                line,
                sep=" ",
                dtype=np.float64,
            )

            if y.shape != (
                N,
            ):
                fail(
                    f"Y_obs gene {g}: shape={y.shape}"
                )

            if not np.isfinite(
                y
            ).all():
                fail(
                    f"Y_obs gene {g}: non-finite"
                )

            if np.any(
                y < 0
            ):
                fail(
                    f"Y_obs gene {g}: negative"
                )

            source_mu = (
                mu_effective[
                    g,
                    Z_hat,
                ]
            )

            edge_values = (
                weight
                *
                source_mu[
                    source
                ]
            )

            leakage_base = np.bincount(
                target,
                weights=edge_values,
                minlength=N,
            )

            leakage = (
                gamma_hat[
                    g
                ]
                *
                leakage_base
            )

            if not np.isfinite(
                leakage
            ).all():
                fail(
                    f"gene {g}: non-finite leakage"
                )

            if np.any(
                leakage < -1e-12
            ):
                fail(
                    f"gene {g}: negative leakage "
                    f"{float(leakage.min())}"
                )

            # Tiny numerical negatives are not expected, but prevent
            # sub-epsilon signs from violating the endpoint contract.
            leakage = np.maximum(
                leakage,
                0.0,
            )

            raw_corrected = (
                y
                -
                leakage
            )

            corrected = np.maximum(
                raw_corrected,
                0.0,
            )

            above = int(
                np.count_nonzero(
                    corrected
                    >
                    y + 1e-12
                )
            )

            ztp = int(
                np.count_nonzero(
                    (y == 0)
                    &
                    (corrected > 0)
                )
            )

            if above:
                fail(
                    f"gene {g}: corrected exceeds observed "
                    f"in {above} cells"
                )

            if ztp:
                fail(
                    f"gene {g}: raw-zero became positive "
                    f"in {ztp} cells"
                )

            corr32 = corrected.astype(
                np.float32
            )

            if reference is not None:

                ref_row = np.asarray(
                    reference[
                        g,
                        :
                    ],
                    dtype=np.float32,
                )

                d = float(
                    np.max(
                        np.abs(
                            corr32
                            -
                            ref_row
                        )
                    )
                )

                reference_max_abs = max(
                    reference_max_abs,
                    d,
                )

                if not np.allclose(
                    corr32,
                    ref_row,
                    rtol=0.0,
                    atol=1e-6,
                ):
                    fail(
                        f"REFERENCE GATE FAILED: corrected "
                        f"gene {g}, max_abs={d}"
                    )

            if out_memmap is not None:
                out_memmap[
                    g,
                    :
                ] = corr32

            observed_sum = float(
                y.sum()
            )

            corrected_sum = float(
                corrected.sum()
            )

            removed_sum = (
                observed_sum
                -
                corrected_sum
            )

            total_observed += (
                observed_sum
            )

            total_corrected += (
                corrected_sum
            )

            total_decreased += int(
                np.count_nonzero(
                    corrected
                    <
                    y
                )
            )

            total_corrected_zero += int(
                np.count_nonzero(
                    corrected
                    == 0
                )
            )

            corrected_above_observed += (
                above
            )

            zero_to_positive += (
                ztp
            )

            min_leakage = min(
                min_leakage,
                float(
                    leakage.min()
                ),
            )

            max_leakage = max(
                max_leakage,
                float(
                    leakage.max()
                ),
            )

            max_corrected_minus_observed = max(
                max_corrected_minus_observed,
                float(
                    np.max(
                        corrected
                        -
                        y
                    )
                ),
            )

            qc_rows.append({
                "gene_index":
                    g,

                "gene":
                    genes[g],

                "mu_hat_raw_min":
                    float(
                        mu_hat[
                            g
                        ].min()
                    ),

                "mu_hat_raw_max":
                    float(
                        mu_hat[
                            g
                        ].max()
                    ),

                "mu_hat_negative_clusters":
                    int(
                        np.count_nonzero(
                            mu_hat[
                                g
                            ]
                            < 0
                        )
                    ),

                "mu_effective_min":
                    float(
                        mu_effective[
                            g
                        ].min()
                    ),

                "gamma_hat":
                    float(
                        gamma_hat[
                            g
                        ]
                    ),

                "observed_sum":
                    observed_sum,

                "corrected_sum":
                    corrected_sum,

                "removed_sum":
                    removed_sum,

                "removed_fraction":
                    (
                        removed_sum
                        /
                        observed_sum
                        if observed_sum > 0
                        else 0.0
                    ),

                "leakage_min":
                    float(
                        leakage.min()
                    ),

                "leakage_max":
                    float(
                        leakage.max()
                    ),

                "corrected_zero_entries":
                    int(
                        np.count_nonzero(
                            corrected == 0
                        )
                    ),

                "entries_decreased":
                    int(
                        np.count_nonzero(
                            corrected < y
                        )
                    ),
            })

            rows += 1

    if rows != len(
        genes
    ):
        fail(
            f"Y_obs row count={rows}"
        )

    if total_observed <= 0:
        fail(
            "non-positive total observed expression"
        )

    removed = (
        total_observed
        -
        total_corrected
    )

    return {
        "mu_effective":
            mu_effective,

        "qc_rows":
            qc_rows,

        "valid_directed_edges":
            int(
                n_edges
            ),

        "observed_total":
            float(
                total_observed
            ),

        "corrected_total":
            float(
                total_corrected
            ),

        "removed_total":
            float(
                removed
            ),

        "removed_fraction":
            float(
                removed
                /
                total_observed
            ),

        "entries_decreased":
            int(
                total_decreased
            ),

        "corrected_zero_entries":
            int(
                total_corrected_zero
            ),

        "corrected_above_observed":
            int(
                corrected_above_observed
            ),

        "raw_zero_to_corrected_positive":
            int(
                zero_to_positive
            ),

        "min_modeled_leakage":
            float(
                min_leakage
            ),

        "max_modeled_leakage":
            float(
                max_leakage
            ),

        "max_corrected_minus_observed":
            float(
                max_corrected_minus_observed
            ),

        "reference_max_abs":
            float(
                reference_max_abs
            ),
    }


def verify_reference():

    root = Path.cwd()

    ref_script = (
        root
        / REFERENCE_POSTPROCESS_SCRIPT
    )

    if sha256(
        ref_script
    ) != REFERENCE_POSTPROCESS_SCRIPT_SHA:
        fail(
            "frozen TENX199 final postprocess script changed"
        )

    prod = (
        root
        / "runs/deleakage_production/"
        "TENX199_10000_v1"
    )

    if sha256(
        prod
        / "outputs.sha256"
    ) != REFERENCE_PROD_MANIFEST_SHA:
        fail(
            "TENX199 production output manifest changed"
        )

    verify_output_manifest(
        prod
    )

    structural = (
        root
        / "runs/deleakage_preprocess/"
        "TENX199_structural_v1"
    )

    labels = (
        root
        / "runs/deleakage_preprocess/"
        "TENX199_labels_K3_v1"
    )

    (
        G,
        N,
        K,
        genes,
        cell_ids,
    ) = load_contracts(
        "TENX199",
        structural,
        labels,
    )

    p = load_posterior(
        prod,
        N,
        G,
        K,
    )

    ref_dir = (
        root
        / "runs/deleakage_postprocess/"
        "TENX199_v1"
    )

    ref_p = np.load(
        ref_dir
        / "posterior_summary.npz",
        allow_pickle=False,
    )

    if not np.allclose(
        p["mu_hat"],
        np.asarray(
            ref_p[
                "mu_hat"
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "REFERENCE GATE FAILED: mu_hat differs"
        )

    if not np.allclose(
        p["gamma_hat"],
        np.asarray(
            ref_p[
                "gamma_hat"
            ],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        fail(
            "REFERENCE GATE FAILED: gamma_hat differs"
        )

    ref_beta = float(
        np.asarray(
            ref_p[
                "beta_hat"
            ]
        ).reshape(-1)[0]
    )

    if not np.isclose(
        p["beta_hat"],
        ref_beta,
        rtol=0.0,
        atol=1e-15,
    ):
        fail(
            "REFERENCE GATE FAILED: beta_hat differs"
        )

    if not np.array_equal(
        p["Z_hat"],
        np.asarray(
            ref_p[
                "Z_hat"
            ],
            dtype=np.int16,
        ),
    ):
        fail(
            "REFERENCE GATE FAILED: Z_hat differs"
        )

    ref_corrected = np.load(
        ref_dir
        / "corrected_GxN.float32.npy",
        mmap_mode="r",
    )

    if ref_corrected.shape != (
        G,
        N,
    ):
        fail(
            "TENX199 frozen corrected matrix shape changed"
        )

    if ref_corrected.dtype != np.float32:
        fail(
            "TENX199 frozen corrected matrix dtype changed"
        )

    q = correction_pass(
        structural=structural,
        genes=genes,
        posterior=p,
        N=N,
        reference=ref_corrected,
    )

    # Frozen TENX199 diagnostic landmarks.
    if not np.isclose(
        p["beta_hat"],
        0.06565242678000001,
        rtol=0.0,
        atol=1e-15,
    ):
        fail(
            "TENX199 beta landmark changed"
        )

    if int(
        np.count_nonzero(
            p["mu_hat"]
            < 0
        )
    ) != 29:
        fail(
            "TENX199 raw-negative-mu landmark changed"
        )

    if q[
        "corrected_above_observed"
    ] != 0:
        fail(
            "TENX199 corrected>observed reference invariant failed"
        )

    if q[
        "raw_zero_to_corrected_positive"
    ] != 0:
        fail(
            "TENX199 raw-zero reference invariant failed"
        )

    print()
    print("=" * 78)
    print("TENX199 GENERIC POSTPROCESS REFERENCE GATE: PASS")
    print("=" * 78)
    print(
        "beta_hat =",
        p["beta_hat"],
    )
    print(
        "raw mu negative entries =",
        int(
            np.count_nonzero(
                p["mu_hat"]
                < 0
            )
        ),
    )
    print(
        "reference corrected max abs diff =",
        q["reference_max_abs"],
    )
    print(
        "corrected > observed =",
        q["corrected_above_observed"],
    )
    print(
        "raw zero -> corrected positive =",
        q["raw_zero_to_corrected_positive"],
    )
    print("NO FILES WRITTEN")
    print("=" * 78)


def process(
    sid: str,
    structural: Path,
    labels: Path,
    prod: Path,
    protocol: Path,
    out: Path,
):

    verify_protocol(
        protocol,
        sid,
    )

    verify_output_manifest(
        prod
    )

    (
        G,
        N,
        K,
        genes,
        cell_ids,
    ) = load_contracts(
        sid,
        structural,
        labels,
    )

    if out.exists():
        fail(
            f"refusing overwrite: {out}"
        )

    tmp = out.with_name(
        out.name
        + ".tmp"
    )

    if tmp.exists():
        fail(
            f"refusing existing temp directory: {tmp}"
        )

    tmp.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp.mkdir()

    posterior = load_posterior(
        prod,
        N,
        G,
        K,
    )

    initial = np.loadtxt(
        labels
        / "Y_label.txt",
        dtype=np.int16,
    )

    if initial.shape != (
        N,
    ):
        fail(
            f"{sid}: initial-label shape mismatch"
        )

    posterior_counts = np.bincount(
        posterior[
            "Z_hat"
        ],
        minlength=K,
    ).astype(
        int
    )

    changed_from_initial = int(
        np.count_nonzero(
            posterior[
                "Z_hat"
            ]
            != initial
        )
    )

    CORRECTED = (
        tmp
        / "corrected_GxN.float32.npy"
    )

    corrected_mm = (
        np.lib.format.open_memmap(
            CORRECTED,
            mode="w+",
            dtype=np.float32,
            shape=(
                G,
                N,
            ),
        )
    )

    q = correction_pass(
        structural=structural,
        genes=genes,
        posterior=posterior,
        N=N,
        out_memmap=corrected_mm,
    )

    corrected_mm.flush()
    del corrected_mm

    POSTERIOR = (
        tmp
        / "posterior_summary.npz"
    )

    np.savez_compressed(
        POSTERIOR,

        mu_hat=
            posterior[
                "mu_hat"
            ],

        mu_hat_effective=
            q[
                "mu_effective"
            ],

        gamma_hat=
            posterior[
                "gamma_hat"
            ],

        beta_hat=np.asarray(
            posterior[
                "beta_hat"
            ],
            dtype=np.float64,
        ),

        beta_sd=np.asarray(
            posterior[
                "beta_sd"
            ],
            dtype=np.float64,
        ),

        Z_hat=
            posterior[
                "Z_hat"
            ],

        Z_mode_counts=
            posterior[
                "Z_mode_counts"
            ],
    )

    QC = (
        tmp
        / "per_gene_qc.tsv"
    )

    fields = [
        "gene_index",
        "gene",
        "mu_hat_raw_min",
        "mu_hat_raw_max",
        "mu_hat_negative_clusters",
        "mu_effective_min",
        "gamma_hat",
        "observed_sum",
        "corrected_sum",
        "removed_sum",
        "removed_fraction",
        "leakage_min",
        "leakage_max",
        "corrected_zero_entries",
        "entries_decreased",
    ]

    with QC.open(
        "w",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
        )

        w.writeheader()

        w.writerows(
            q[
                "qc_rows"
            ]
        )

    check = np.load(
        CORRECTED,
        mmap_mode="r",
    )

    if check.shape != (
        G,
        N,
    ):
        fail(
            f"{sid}: saved corrected shape mismatch"
        )

    if check.dtype != np.float32:
        fail(
            f"{sid}: corrected dtype mismatch"
        )

    for g in range(G):

        row = check[
            g,
            :
        ]

        if not np.isfinite(
            row
        ).all():
            fail(
                f"{sid}: saved corrected gene {g} non-finite"
            )

        if np.any(
            row < 0
        ):
            fail(
                f"{sid}: saved corrected gene {g} negative"
            )

    del check

    manifest = {
        "method":
            "DeLeakage",

        "stage":
            "post_burn_in_summary_and_release_compatible_correction",

        "sample":
            sid,

        "split_role":
            "train",

        "dimensions": {
            "G":
                G,
            "N":
                N,
            "K":
                K,
        },

        "posterior": {
            "burn_in_iterations":
                5000,

            "retained_parameter_iterations":
                5000,

            "retained_Z_snapshots":
                200,

            "parameter_block_suffixes":
                RETAINED_BLOCKS,

            "Z_block_suffixes": [
                x // 25
                for x in RETAINED_BLOCKS
            ],

            "beta_hat":
                float(
                    posterior[
                        "beta_hat"
                    ]
                ),

            "beta_sd":
                float(
                    posterior[
                        "beta_sd"
                    ]
                ),

            "gamma_hat_min":
                float(
                    posterior[
                        "gamma_hat"
                    ].min()
                ),

            "gamma_hat_max":
                float(
                    posterior[
                        "gamma_hat"
                    ].max()
                ),

            "serialized_gamma_eq_1":
                int(
                    posterior[
                        "gamma_serialized_eq1"
                    ]
                ),

            "serialized_gamma_gt_1":
                int(
                    posterior[
                        "gamma_serialized_gt1"
                    ]
                ),

            "raw_mu_min":
                float(
                    posterior[
                        "mu_hat"
                    ].min()
                ),

            "raw_mu_max":
                float(
                    posterior[
                        "mu_hat"
                    ].max()
                ),

            "raw_mu_negative_entries":
                int(
                    np.count_nonzero(
                        posterior[
                            "mu_hat"
                        ]
                        < 0
                    )
                ),

            "effective_mu_min":
                float(
                    q[
                        "mu_effective"
                    ].min()
                ),

            "posterior_Z_counts":
                posterior_counts.tolist(),

            "posterior_Z_mode_ties":
                int(
                    posterior[
                        "Z_ties"
                    ]
                ),

            "changed_from_initial_label":
                changed_from_initial,

            "changed_fraction":
                float(
                    changed_from_initial
                    /
                    N
                ),

            "split_half_Z_disagreement":
                int(
                    posterior[
                        "Z_split_disagreement"
                    ]
                ),

            "split_half_Z_disagreement_fraction":
                float(
                    posterior[
                        "Z_split_disagreement"
                    ]
                    /
                    N
                ),

            "mu_split_half_correlation":
                float(
                    posterior[
                        "mu_split_corr"
                    ]
                ),

            "gamma_split_half_correlation":
                float(
                    posterior[
                        "gamma_split_corr"
                    ]
                ),

            "beta_first_half_mean":
                float(
                    posterior[
                        "beta_first_half_mean"
                    ]
                ),

            "beta_second_half_mean":
                float(
                    posterior[
                        "beta_second_half_mean"
                    ]
                ),
        },

        "correction": {
            "formula":
                "corrected=max(Yobs-gamma*sum_j("
                "exp(-beta*d_ij)*cell_size_j*"
                "mu_effective[g,Z_j]),0)",

            "raw_mu_preserved_in_diagnostics":
                True,

            "mu_effective_rule":
                "max(posterior_mean_mu,0)_for_correction_only",

            "gamma_clipped":
                False,

            "I_beta_or_I_gamma_used_as_correction_gate":
                False,

            "release_compatibility_status":
                "operational_convention_inferred_from_released_artifacts",

            "valid_directed_edges":
                q[
                    "valid_directed_edges"
                ],

            "observed_total":
                q[
                    "observed_total"
                ],

            "corrected_total":
                q[
                    "corrected_total"
                ],

            "removed_total":
                q[
                    "removed_total"
                ],

            "removed_fraction":
                q[
                    "removed_fraction"
                ],

            "corrected_above_observed":
                q[
                    "corrected_above_observed"
                ],

            "raw_zero_to_corrected_positive":
                q[
                    "raw_zero_to_corrected_positive"
                ],

            "corrected_zero_entries":
                q[
                    "corrected_zero_entries"
                ],

            "entries_decreased":
                q[
                    "entries_decreased"
                ],

            "min_modeled_leakage":
                q[
                    "min_modeled_leakage"
                ],

            "max_modeled_leakage":
                q[
                    "max_modeled_leakage"
                ],

            "max_corrected_minus_observed":
                q[
                    "max_corrected_minus_observed"
                ],
        },

        "upstream": {
            "production_protocol":
                str(
                    protocol
                ),

            "production_protocol_sha256":
                sha256(
                    protocol
                ),

            "production_output_manifest":
                str(
                    prod
                    / "outputs.sha256"
                ),

            "production_output_manifest_sha256":
                sha256(
                    prod
                    / "outputs.sha256"
                ),

            "structural_manifest":
                str(
                    structural
                    / "structural_manifest.json"
                ),

            "structural_manifest_sha256":
                sha256(
                    structural
                    / "structural_manifest.json"
                ),

            "initial_label":
                str(
                    labels
                    / "Y_label.txt"
                ),

            "initial_label_sha256":
                sha256(
                    labels
                    / "Y_label.txt"
                ),

            "TENX199_reference_postprocess_script":
                str(
                    REFERENCE_POSTPROCESS_SCRIPT
                ),

            "TENX199_reference_postprocess_script_sha256":
                REFERENCE_POSTPROCESS_SCRIPT_SHA,
        },

        "generator": {
            "path":
                str(
                    Path(__file__)
                ),

            "sha256":
                sha256(
                    Path(__file__)
                ),
        },

        "artifacts": {
            "posterior_summary":
                "posterior_summary.npz",

            "posterior_summary_sha256":
                sha256(
                    POSTERIOR
                ),

            "corrected_GxN_float32":
                "corrected_GxN.float32.npy",

            "corrected_GxN_float32_sha256":
                sha256(
                    CORRECTED
                ),

            "per_gene_qc":
                "per_gene_qc.tsv",

            "per_gene_qc_sha256":
                sha256(
                    QC
                ),
        },

        "clean_h5ad_created":
            False,

        "model_training_authorized":
            False,
    }

    MANIFEST = (
        tmp
        / "postprocess_manifest.json"
    )

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    # Manifest self-consistency after serialization.
    m2 = json.loads(
        MANIFEST.read_text()
    )

    for key, rel in [
        (
            "posterior_summary_sha256",
            "posterior_summary.npz",
        ),
        (
            "corrected_GxN_float32_sha256",
            "corrected_GxN.float32.npy",
        ),
        (
            "per_gene_qc_sha256",
            "per_gene_qc.tsv",
        ),
    ]:

        if (
            m2[
                "artifacts"
            ][key]
            != sha256(
                tmp
                / rel
            )
        ):
            fail(
                f"{sid}: artifact manifest mismatch: {rel}"
            )

    tmp.rename(
        out
    )

    print()
    print("=" * 78)
    print(
        f"{sid} DELEAKAGE POSTPROCESS: PASS"
    )
    print("=" * 78)
    print(
        "beta_hat =",
        posterior[
            "beta_hat"
        ],
    )
    print(
        "raw mu min/max =",
        float(
            posterior[
                "mu_hat"
            ].min()
        ),
        float(
            posterior[
                "mu_hat"
            ].max()
        ),
    )
    print(
        "raw negative mu entries =",
        int(
            np.count_nonzero(
                posterior[
                    "mu_hat"
                ]
                < 0
            )
        ),
    )
    print(
        "posterior Z counts =",
        posterior_counts.tolist(),
    )
    print(
        "posterior Z ties =",
        posterior[
            "Z_ties"
        ],
    )
    print(
        "corrected total =",
        q[
            "corrected_total"
        ],
    )
    print(
        "removed fraction =",
        q[
            "removed_fraction"
        ],
    )
    print(
        "corrected > observed =",
        q[
            "corrected_above_observed"
        ],
    )
    print(
        "raw zero -> corrected positive =",
        q[
            "raw_zero_to_corrected_positive"
        ],
    )
    print(
        "output =",
        out,
    )
    print("=" * 78)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--verify-reference",
        action="store_true",
    )

    ap.add_argument(
        "--sample",
    )

    ap.add_argument(
        "--structural",
    )

    ap.add_argument(
        "--labels",
    )

    ap.add_argument(
        "--production",
    )

    ap.add_argument(
        "--protocol",
    )

    ap.add_argument(
        "--out",
    )

    args = ap.parse_args()

    if args.verify_reference:

        if any([
            args.sample,
            args.structural,
            args.labels,
            args.production,
            args.protocol,
            args.out,
        ]):
            fail(
                "--verify-reference must be used alone"
            )

        verify_reference()
        return

    required = [
        args.sample,
        args.structural,
        args.labels,
        args.production,
        args.protocol,
        args.out,
    ]

    if not all(
        required
    ):
        fail(
            "generation requires --sample --structural "
            "--labels --production --protocol --out"
        )

    process(
        sid=args.sample,
        structural=Path(
            args.structural
        ),
        labels=Path(
            args.labels
        ),
        prod=Path(
            args.production
        ),
        protocol=Path(
            args.protocol
        ),
        out=Path(
            args.out
        ),
    )


if __name__ == "__main__":
    main()
