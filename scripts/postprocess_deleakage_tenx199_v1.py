#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np


ROOT = Path.cwd().resolve()

PROD = (
    ROOT
    / "runs"
    / "deleakage_production"
    / "TENX199_10000_v1"
)

STRUCT = (
    ROOT
    / "runs"
    / "deleakage_preprocess"
    / "TENX199_structural_v1"
)

LABELS = (
    ROOT
    / "runs"
    / "deleakage_preprocess"
    / "TENX199_labels_K3_v1"
)

PROTOCOL = (
    ROOT
    / "runs"
    / "frozen_primary2025"
    / "deleakage_TENX199_production_protocol_v1.json"
)

FORMULA_AUDIT = (
    ROOT
    / "runs"
    / "frozen_primary2025"
    / "deleakage_released_beta_formula_check_v2.log"
)

OUT = (
    ROOT
    / "runs"
    / "deleakage_postprocess"
    / "TENX199_v1"
)

TMP = Path(
    str(OUT) + ".tmp"
)


EXPECTED_PROTOCOL_SHA = (
    "14d1ad77febdcc4984664da9dbc72d526"
    "9cc062112654aa85c610bc9502a8bef"
)

EXPECTED_OUTPUT_MANIFEST_SHA = (
    "5507923201e9664efa31dab8faad4bbcd"
    "4b44012c939dc0eedefaaf7cea0df0b"
)

EXPECTED_STRUCT_SHA = (
    "88600b36f3cf8640b9e265146e6dcfe3"
    "83499384b24935830fc8226aae72842b"
)

EXPECTED_LABEL_SHA = (
    "556af213de67c9d21e7963e598925ab6"
    "0d0c268e279761e6eaff232e42a26dd1"
)

EXPECTED_FORMULA_AUDIT_SHA = (
    "53534c290ddebc700d0d5c2bc5704312"
    "52f44e383c0590ab85d359090e8350af"
)


G = 280
K = 3
N = 150145

BURN_IN = 5000

RETAINED_BLOCKS = [
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    8500,
    9000,
    9500,
    10000,
]

N_PARAM_SAMPLES = 5000
N_Z_SAMPLES = 200


def fail(msg):
    raise RuntimeError(
        msg
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(
                1 << 20
            ),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


def require_sha(
    path: Path,
    expected: str,
):
    if not path.exists():
        fail(
            f"missing frozen artifact: {path}"
        )

    got = sha256(
        path
    )

    if got != expected:
        fail(
            "SHA mismatch:\n"
            f"path={path}\n"
            f"expected={expected}\n"
            f"actual={got}"
        )


def expected_posterior_files():
    x = set()

    for end in range(
        500,
        10001,
        500,
    ):
        x.update({
            f"mu_{end}.txt",
            f"gamma_{end}.txt",
            f"beta_{end}.txt",
            f"I_beta_{end}.txt",
            f"I_gamma_{end}.txt",
            f"sigma_sq_{end}.txt",
            f"pi_{end}.txt",
            f"Z_{end // 25}.txt",
        })

    return x


print("=" * 78)
print("TENX199 DELEAKAGE POSTPROCESS V1")
print("POST-BURN-IN SUMMARY + CORRECTION")
print("NO NEW MCMC")
print("=" * 78)


# ======================================================================
# 1. STRICT FROZEN PROVENANCE
# ======================================================================

require_sha(
    PROTOCOL,
    EXPECTED_PROTOCOL_SHA,
)

require_sha(
    PROD / "outputs.sha256",
    EXPECTED_OUTPUT_MANIFEST_SHA,
)

require_sha(
    STRUCT / "structural_manifest.json",
    EXPECTED_STRUCT_SHA,
)

require_sha(
    LABELS / "Y_label.txt",
    EXPECTED_LABEL_SHA,
)

require_sha(
    FORMULA_AUDIT,
    EXPECTED_FORMULA_AUDIT_SHA,
)


protocol = json.loads(
    PROTOCOL.read_text()
)

assert (
    protocol[
        "stochastic_runtime"
    ]["n_record"]
    == 10000
)

assert (
    protocol[
        "posterior_policy"
    ]["burn_in_iterations"]
    == 5000
)


# ======================================================================
# 2. VERIFY ALL 160 PRODUCTION POSTERIOR FILES
# ======================================================================

expected = expected_posterior_files()

actual = {
    p.name
    for p in PROD.iterdir()
    if (
        p.is_file()
        and
        p.name
        != "outputs.sha256"
    )
}

if actual != expected:
    fail(
        "production file set changed:\n"
        f"missing={sorted(expected-actual)}\n"
        f"extra={sorted(actual-expected)}"
    )


manifest_entries = {}

for line in (
    PROD / "outputs.sha256"
).read_text().splitlines():

    line = line.strip()

    if not line:
        continue

    parts = line.split(
        maxsplit=1
    )

    if len(parts) != 2:
        fail(
            f"bad output hash line: {line}"
        )

    digest = parts[0]

    name = parts[1].strip()

    if name.startswith("*"):
        name = name[1:]

    if name.startswith("./"):
        name = name[2:]

    # A redirect-created SHA manifest can theoretically see itself.
    # It is never accepted as a posterior file.
    if name == "outputs.sha256":
        continue

    manifest_entries[
        name
    ] = digest


if set(
    manifest_entries
) != expected:
    fail(
        "output SHA manifest does not describe "
        "the exact 160 posterior files"
    )


for i, name in enumerate(
    sorted(expected),
    start=1,
):
    p = PROD / name

    got = sha256(
        p
    )

    expected_sha = (
        manifest_entries[
            name
        ]
    )

    if got != expected_sha:
        fail(
            "posterior file changed:\n"
            f"{p}\n"
            f"expected={expected_sha}\n"
            f"actual={got}"
        )

    if i % 20 == 0:
        print(
            f"verified posterior hashes: "
            f"{i}/160"
        )


print(
    "PRODUCTION POSTERIOR BYTE CONTRACT: PASS"
)


# ======================================================================
# 3. VERIFY STATIC INPUT DIMENSIONS
# ======================================================================

sm = json.loads(
    (
        STRUCT
        / "structural_manifest.json"
    ).read_text()
)

lm = json.loads(
    (
        LABELS
        / "label_manifest.json"
    ).read_text()
)

assert int(
    sm["G"]
) == G

assert int(
    sm["N"]
) == N

assert int(
    lm["K"]
) == K

assert int(
    lm["N"]
) == N

N_nei = int(
    sm["N_nei"]
)

initial_labels = np.loadtxt(
    LABELS / "Y_label.txt",
    dtype=np.int16,
)

assert initial_labels.shape == (
    N,
)

assert np.array_equal(
    np.unique(
        initial_labels
    ),
    np.arange(K),
)


# ======================================================================
# 3B. RELEASED ENDPOINT OPERATIONAL CONTRACT
#
# The released DeLeakage endpoint has:
#   - no negative summarized mu
#   - exact-zero summarized mu values
#   - corrected expression never above observed expression
#   - no observed-zero -> corrected-positive transitions
#
# This does NOT establish the undocumented internal implementation.
# It establishes the released operational endpoint that this
# postprocessor must reproduce.
# ======================================================================

_RELEASE_MU = np.loadtxt(
    ROOT
    / "third_party"
    / "DeLeakage"
    / "post"
    / "mu.txt",
    dtype=np.float64,
)

_RELEASE_Y = np.loadtxt(
    ROOT
    / "third_party"
    / "DeLeakage"
    / "data"
    / "Y_obs.txt",
    dtype=np.float64,
)

_RELEASE_D = np.loadtxt(
    ROOT
    / "third_party"
    / "DeLeakage"
    / "post"
    / "decon_fact_vol.txt",
    dtype=np.float64,
)

assert _RELEASE_MU.shape == (100, 5)
assert _RELEASE_Y.shape == (100, 3234)
assert _RELEASE_D.shape == (100, 3234)

_RELEASE_TOL = 1e-12

release_negative_mu = int(
    np.sum(
        _RELEASE_MU < -_RELEASE_TOL
    )
)

release_zero_mu = int(
    np.sum(
        np.abs(_RELEASE_MU)
        <= _RELEASE_TOL
    )
)

release_above_observed = int(
    np.sum(
        _RELEASE_D
        >
        _RELEASE_Y
        + _RELEASE_TOL
    )
)

release_zero_to_positive = int(
    np.sum(
        (_RELEASE_Y == 0)
        &
        (_RELEASE_D > _RELEASE_TOL)
    )
)

release_negative_corrected = int(
    np.sum(
        _RELEASE_D < -_RELEASE_TOL
    )
)

assert release_negative_mu == 0
assert release_zero_mu == 71
assert release_above_observed == 0
assert release_zero_to_positive == 0
assert release_negative_corrected == 0

print()
print("=" * 78)
print("RELEASED DELEAKAGE ENDPOINT CONTRACT: PASS")
print("=" * 78)
print("released negative mu =", release_negative_mu)
print("released exact-zero mu =", release_zero_mu)
print(
    "released corrected > observed =",
    release_above_observed,
)
print(
    "released raw-zero -> corrected-positive =",
    release_zero_to_positive,
)
print(
    "released corrected <0 =",
    release_negative_corrected,
)
print("=" * 78)

del (
    _RELEASE_MU,
    _RELEASE_Y,
    _RELEASE_D,
)


# ======================================================================
# 4. CREATE TEMPORARY OUTPUT DIRECTORY
# ======================================================================

if OUT.exists():
    fail(
        f"refusing overwrite: {OUT}"
    )

if TMP.exists():
    fail(
        f"partial temp directory exists: {TMP}"
    )

TMP.mkdir(
    parents=True,
    exist_ok=False,
)


# ======================================================================
# 5. POSTERIOR SUMMARY
#
# Continuous parameters:
#   iterations 5001..10000 = 5000 samples.
#
# Z:
#   stored every 25 iterations.
#   retained files therefore contain 200 post-burn-in snapshots.
# ======================================================================

mu_sum = np.zeros(
    (G, K),
    dtype=np.float64,
)

mu_sumsq = np.zeros_like(
    mu_sum
)

gamma_sum = np.zeros(
    G,
    dtype=np.float64,
)

gamma_sumsq = np.zeros_like(
    gamma_sum
)

sigma_sum = np.zeros(
    G,
    dtype=np.float64,
)

pi_sum = np.zeros(
    K,
    dtype=np.float64,
)

beta_sum = 0.0
beta_sumsq = 0.0


mu_first = np.zeros_like(
    mu_sum
)

mu_second = np.zeros_like(
    mu_sum
)

gamma_first = np.zeros_like(
    gamma_sum
)

gamma_second = np.zeros_like(
    gamma_sum
)

beta_first = 0.0
beta_second = 0.0


z_counts = np.zeros(
    (N, K),
    dtype=np.uint16,
)

z_counts_first = np.zeros_like(
    z_counts
)

z_counts_second = np.zeros_like(
    z_counts
)


I_beta_sum = np.zeros(
    G,
    dtype=np.float64,
)

I_gamma_sum = 0.0


retained_gamma_eq1 = 0
retained_gamma_gt1 = 0

parameter_samples_seen = 0
z_samples_seen = 0


for block_index, end in enumerate(
    RETAINED_BLOCKS
):

    print(
        f"reading retained block "
        f"{end}"
    )

    mu_raw = np.loadtxt(
        PROD
        / f"mu_{end}.txt",
        dtype=np.float64,
    )

    gamma = np.loadtxt(
        PROD
        / f"gamma_{end}.txt",
        dtype=np.float64,
    )

    beta = np.loadtxt(
        PROD
        / f"beta_{end}.txt",
        dtype=np.float64,
    )

    sigma = np.loadtxt(
        PROD
        / f"sigma_sq_{end}.txt",
        dtype=np.float64,
    )

    pi = np.loadtxt(
        PROD
        / f"pi_{end}.txt",
        dtype=np.float64,
    )

    I_beta = np.loadtxt(
        PROD
        / f"I_beta_{end}.txt",
        dtype=np.int16,
    )

    I_gamma = np.loadtxt(
        PROD
        / f"I_gamma_{end}.txt",
        dtype=np.int16,
    )

    z = np.loadtxt(
        PROD
        / f"Z_{end // 25}.txt",
        dtype=np.int16,
    )


    assert mu_raw.shape == (
        G * K,
        500,
    )

    # Frozen C++ serialization is gene-major,
    # cluster-minor, iteration along columns.
    mu = mu_raw.reshape(
        G,
        K,
        500,
    )

    assert gamma.shape == (
        G,
        500,
    )

    assert beta.shape == (
        500,
    )

    assert sigma.shape == (
        G,
        500,
    )

    assert pi.shape == (
        K,
        500,
    )

    assert I_beta.shape == (
        G,
        500,
    )

    assert I_gamma.shape == (
        500,
    )

    assert z.shape == (
        N,
        20,
    )


    for name, x in [
        ("mu", mu),
        ("gamma", gamma),
        ("beta", beta),
        ("sigma", sigma),
        ("pi", pi),
    ]:
        if not np.isfinite(
            x
        ).all():
            fail(
                f"{end}: non-finite {name}"
            )


    if np.any(
        gamma <= 0
    ):
        fail(
            f"{end}: gamma <= 0"
        )

    gt1 = int(
        np.sum(
            gamma > 1
        )
    )

    if gt1:
        fail(
            f"{end}: serialized gamma >1 "
            f"count={gt1}"
        )

    retained_gamma_gt1 += gt1

    retained_gamma_eq1 += int(
        np.sum(
            gamma == 1
        )
    )


    if np.any(
        beta <= 0
    ):
        fail(
            f"{end}: beta <=0"
        )

    if np.any(
        sigma <= 0
    ):
        fail(
            f"{end}: sigma_sq <=0"
        )

    if not np.all(
        np.isin(
            I_beta,
            [0, 1],
        )
    ):
        fail(
            f"{end}: bad I_beta"
        )

    if not np.all(
        np.isin(
            I_gamma,
            [0, 1],
        )
    ):
        fail(
            f"{end}: bad I_gamma"
        )

    if not np.all(
        (z >= 0)
        &
        (z < K)
    ):
        fail(
            f"{end}: invalid Z"
        )

    if np.any(
        pi < 0
    ):
        fail(
            f"{end}: negative pi"
        )

    if not np.allclose(
        pi.sum(
            axis=0
        ),
        1.0,
        rtol=1e-5,
        atol=1e-5,
    ):
        fail(
            f"{end}: invalid pi"
        )


    # --------------------------------------------------------------
    # Continuous posterior accumulation.
    # --------------------------------------------------------------

    block_mu_sum = mu.sum(
        axis=2,
        dtype=np.float64,
    )

    block_gamma_sum = gamma.sum(
        axis=1,
        dtype=np.float64,
    )

    block_beta_sum = float(
        beta.sum(
            dtype=np.float64
        )
    )


    mu_sum += block_mu_sum

    mu_sumsq += np.square(
        mu,
        dtype=np.float64,
    ).sum(
        axis=2,
        dtype=np.float64,
    )

    gamma_sum += block_gamma_sum

    gamma_sumsq += np.square(
        gamma,
        dtype=np.float64,
    ).sum(
        axis=1,
        dtype=np.float64,
    )

    beta_sum += block_beta_sum

    beta_sumsq += float(
        np.square(
            beta,
            dtype=np.float64,
        ).sum(
            dtype=np.float64
        )
    )

    sigma_sum += sigma.sum(
        axis=1,
        dtype=np.float64,
    )

    pi_sum += pi.sum(
        axis=1,
        dtype=np.float64,
    )

    I_beta_sum += I_beta.sum(
        axis=1,
        dtype=np.float64,
    )

    I_gamma_sum += float(
        I_gamma.sum(
            dtype=np.float64
        )
    )


    # Retained split-half diagnostic.
    if block_index < 5:
        mu_first += (
            block_mu_sum
        )

        gamma_first += (
            block_gamma_sum
        )

        beta_first += (
            block_beta_sum
        )

    else:
        mu_second += (
            block_mu_sum
        )

        gamma_second += (
            block_gamma_sum
        )

        beta_second += (
            block_beta_sum
        )


    # --------------------------------------------------------------
    # Z posterior counts.
    # --------------------------------------------------------------

    for k in range(K):

        c = np.sum(
            z == k,
            axis=1,
        ).astype(
            np.uint16
        )

        z_counts[
            :,
            k
        ] += c

        if block_index < 5:
            z_counts_first[
                :,
                k
            ] += c

        else:
            z_counts_second[
                :,
                k
            ] += c


    parameter_samples_seen += 500
    z_samples_seen += 20


assert (
    parameter_samples_seen
    == N_PARAM_SAMPLES
)

assert (
    z_samples_seen
    == N_Z_SAMPLES
)


mu_hat = (
    mu_sum
    / N_PARAM_SAMPLES
)

# Raw Bayesian posterior mean is preserved exactly.
# DeLeakage's released corrected endpoint uses a nonnegative
# effective signal. Apply that operational convention only at
# deterministic postprocessing; never alter posterior samples.
mu_hat_effective = np.maximum(
    mu_hat,
    0.0,
)

negative_mu_hat_count = int(
    np.sum(
        mu_hat < 0
    )
)

negative_mu_hat_min = float(
    mu_hat.min()
)

effective_negative_mu_count = int(
    np.sum(
        mu_hat_effective < 0
    )
)

assert effective_negative_mu_count == 0

gamma_hat = (
    gamma_sum
    / N_PARAM_SAMPLES
)

beta_hat = (
    beta_sum
    / N_PARAM_SAMPLES
)

sigma_hat = (
    sigma_sum
    / N_PARAM_SAMPLES
)

pi_hat = (
    pi_sum
    / N_PARAM_SAMPLES
)


mu_var = (
    mu_sumsq
    / N_PARAM_SAMPLES
    - mu_hat * mu_hat
)

gamma_var = (
    gamma_sumsq
    / N_PARAM_SAMPLES
    - gamma_hat * gamma_hat
)

beta_var = (
    beta_sumsq
    / N_PARAM_SAMPLES
    - beta_hat * beta_hat
)

mu_sd = np.sqrt(
    np.maximum(
        mu_var,
        0.0,
    )
)

gamma_sd = np.sqrt(
    np.maximum(
        gamma_var,
        0.0,
    )
)

beta_sd = float(
    np.sqrt(
        max(
            beta_var,
            0.0,
        )
    )
)


max_z_count = z_counts.max(
    axis=1,
    keepdims=True,
)

tie_mask = (
    (
        z_counts
        == max_z_count
    ).sum(
        axis=1
    )
    > 1
)

z_hat = np.argmax(
    z_counts,
    axis=1,
).astype(
    np.int16
)

# np.argmax chooses the smallest cluster index
# for an exact posterior-mode tie.
tie_count = int(
    tie_mask.sum()
)


z_first_hat = np.argmax(
    z_counts_first,
    axis=1,
).astype(
    np.int16
)

z_second_hat = np.argmax(
    z_counts_second,
    axis=1,
).astype(
    np.int16
)


posterior_cluster_counts = np.bincount(
    z_hat,
    minlength=K,
)

changed_from_initial = int(
    np.sum(
        z_hat
        != initial_labels
    )
)

split_z_disagree = int(
    np.sum(
        z_first_hat
        != z_second_hat
    )
)


mu_first_mean = (
    mu_first
    / 2500.0
)

mu_second_mean = (
    mu_second
    / 2500.0
)

gamma_first_mean = (
    gamma_first
    / 2500.0
)

gamma_second_mean = (
    gamma_second
    / 2500.0
)

beta_first_mean = (
    beta_first
    / 2500.0
)

beta_second_mean = (
    beta_second
    / 2500.0
)


def safe_corr(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):
        return None

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


mu_split_corr = safe_corr(
    mu_first_mean,
    mu_second_mean,
)

gamma_split_corr = safe_corr(
    gamma_first_mean,
    gamma_second_mean,
)


print()
print("=" * 78)
print("POSTERIOR SUMMARY")
print("=" * 78)

print(
    "retained parameter samples =",
    N_PARAM_SAMPLES,
)

print(
    "retained Z samples =",
    N_Z_SAMPLES,
)

print(
    "beta_hat =",
    beta_hat,
)

print(
    "beta_sd =",
    beta_sd,
)

print(
    "raw posterior mu negative count =",
    negative_mu_hat_count,
)

print(
    "raw posterior mu min =",
    negative_mu_hat_min,
)

print(
    "effective mu negative count =",
    effective_negative_mu_count,
)

print(
    "gamma_hat min/max =",
    float(
        gamma_hat.min()
    ),
    float(
        gamma_hat.max()
    ),
)

print(
    "retained serialized gamma ==1 =",
    retained_gamma_eq1,
)

print(
    "retained serialized gamma >1 =",
    retained_gamma_gt1,
)

print(
    "posterior Z counts =",
    posterior_cluster_counts.tolist(),
)

print(
    "posterior Z mode ties =",
    tie_count,
)

print(
    "changed from initial label =",
    changed_from_initial,
)

print(
    "changed fraction =",
    changed_from_initial / N,
)

print(
    "split-half Z disagreement =",
    split_z_disagree,
)

print(
    "split-half Z disagreement fraction =",
    split_z_disagree / N,
)

print(
    "mu split-half correlation =",
    mu_split_corr,
)

print(
    "gamma split-half correlation =",
    gamma_split_corr,
)

print(
    "beta first-half mean =",
    beta_first_mean,
)

print(
    "beta second-half mean =",
    beta_second_mean,
)


# ======================================================================
# 6. SAVE POSTERIOR SUMMARY
# ======================================================================

np.savez_compressed(
    TMP
    / "posterior_summary.npz",

    mu_hat=mu_hat,
    mu_hat_effective=mu_hat_effective,
    mu_sd=mu_sd,

    gamma_hat=gamma_hat,
    gamma_sd=gamma_sd,

    beta_hat=np.array(
        [beta_hat],
        dtype=np.float64,
    ),

    beta_sd=np.array(
        [beta_sd],
        dtype=np.float64,
    ),

    sigma_hat=sigma_hat,
    pi_hat=pi_hat,

    Z_hat=z_hat,
    Z_counts=z_counts,

    Z_first_half_mode=z_first_hat,
    Z_second_half_mode=z_second_hat,

    I_beta_frequency=(
        I_beta_sum
        / N_PARAM_SAMPLES
    ),

    I_gamma_frequency=np.array(
        [
            I_gamma_sum
            / N_PARAM_SAMPLES
        ],
        dtype=np.float64,
    ),
)


# ======================================================================
# 7. BUILD EXACT SPATIAL LEAKAGE OPERATOR
# ======================================================================

cell_size = np.loadtxt(
    STRUCT
    / "cell_size.txt",
    dtype=np.float64,
)

nei = np.loadtxt(
    STRUCT
    / "nei_list.txt",
    dtype=np.int32,
)

dist = np.loadtxt(
    STRUCT
    / "nei_dist.txt",
    dtype=np.float64,
)


assert cell_size.shape == (
    N,
)

assert nei.shape == (
    N,
    N_nei,
)

assert dist.shape == (
    N,
    N_nei,
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

padding = ~valid


if np.any(
    nei[valid] >= N
):
    fail(
        "neighbor index outside cell range"
    )

if np.any(
    dist[valid] < 0
):
    fail(
        "negative valid distance"
    )

if np.any(
    nei[padding] != -2
):
    fail(
        "unexpected neighbor padding"
    )

if np.any(
    dist[padding] != -1
):
    fail(
        "unexpected distance padding"
    )


flat_valid = valid.ravel()

source = (
    nei.ravel()[
        flat_valid
    ]
).astype(
    np.int32,
    copy=False,
)

distance = (
    dist.ravel()[
        flat_valid
    ]
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


edge_weight = (
    np.exp(
        -beta_hat
        * distance
    )
    * cell_size[
        source
    ]
)


if not np.isfinite(
    edge_weight
).all():
    fail(
        "non-finite spatial edge weight"
    )

if np.any(
    edge_weight < 0
):
    fail(
        "negative spatial edge weight"
    )


n_edges = int(
    len(
        edge_weight
    )
)


del (
    nei,
    dist,
    valid,
    padding,
    flat_valid,
    distance,
)


print()
print(
    "valid directed edges =",
    n_edges,
)


# ======================================================================
# 8. CORRECT EXPRESSION
#
# leakage[g,i] =
#   gamma_hat[g] *
#   sum_j(
#       exp(-beta_hat*d_ij)
#       * cell_size[j]
#       * max(mu_hat[g,Z_hat[j]], 0)
#   )
#
# corrected = max(Y_obs - leakage, 0)
#
# Raw posterior mu_hat is retained unchanged for inference/QC.
# Only the released-endpoint effective signal is nonnegative.
#
# This is the already-validated released DeLeakage reconstruction.
# ======================================================================

genes = (
    STRUCT
    / "genes.txt"
).read_text().splitlines()

assert len(
    genes
) == G


CORRECTED = (
    TMP
    / "corrected_GxN.float32.npy"
)


corrected_mm = np.lib.format.open_memmap(
    CORRECTED,
    mode="w+",
    dtype=np.float32,
    shape=(
        G,
        N,
    ),
)


gene_qc = []


observed_total = 0.0
corrected_total = 0.0

total_preclip_negative = 0
total_corrected_zero = 0
total_changed = 0

max_corrected_minus_observed = 0.0
min_leakage_seen = float(
    "inf"
)


YOBS = (
    STRUCT
    / "Y_obs.txt"
)


with YOBS.open() as f:

    rows_seen = 0

    for g, line in enumerate(
        f
    ):

        if g >= G:
            fail(
                "Y_obs has too many rows"
            )

        y = np.fromstring(
            line,
            dtype=np.float64,
            sep=" ",
        )

        if y.shape != (
            N,
        ):
            fail(
                f"Y_obs row {g}: "
                f"{y.shape} != {(N,)}"
            )

        if not np.isfinite(
            y
        ).all():
            fail(
                f"Y_obs row {g}: non-finite"
            )

        if np.any(
            y < 0
        ):
            fail(
                f"Y_obs row {g}: negative"
            )


        source_expression = (
            mu_hat_effective[
                g,
                z_hat,
            ]
        )


        edge_contribution = (
            edge_weight
            * source_expression[
                source
            ]
        )


        leakage_base = np.bincount(
            target,
            weights=edge_contribution,
            minlength=N,
        )


        leakage = (
            gamma_hat[g]
            * leakage_base
        )


        if not np.isfinite(
            leakage
        ).all():
            fail(
                f"gene {g}: non-finite leakage"
            )


        gene_min_leakage = float(
            leakage.min()
        )

        min_leakage_seen = min(
            min_leakage_seen,
            gene_min_leakage,
        )


        # A negative modeled leakage term has no physical
        # interpretation under the validated DeLeakage formula.
        if gene_min_leakage < -1e-10:
            fail(
                f"gene {g}: negative leakage "
                f"{gene_min_leakage}"
            )


        raw_corrected = (
            y
            - leakage
        )


        preclip_negative = int(
            np.sum(
                raw_corrected < 0
            )
        )


        corrected = np.maximum(
            raw_corrected,
            0.0,
        )


        delta_above = float(
            np.max(
                corrected - y
            )
        )


        max_corrected_minus_observed = max(
            max_corrected_minus_observed,
            delta_above,
        )


        if delta_above > 1e-8:
            fail(
                f"gene {g}: corrected exceeds "
                f"observed by {delta_above}"
            )


        zero_raw = (
            y == 0
        )

        if np.any(
            corrected[
                zero_raw
            ] != 0
        ):
            fail(
                f"gene {g}: zero observed "
                "became positive"
            )


        corrected_mm[
            g,
            :
        ] = corrected.astype(
            np.float32
        )


        obs_sum = float(
            y.sum(
                dtype=np.float64
            )
        )

        corr_sum = float(
            corrected.sum(
                dtype=np.float64
            )
        )


        if (
            corr_sum
            >
            obs_sum
            + 1e-7
        ):
            fail(
                f"gene {g}: corrected sum > raw sum"
            )


        removed = (
            obs_sum
            - corr_sum
        )

        removed_fraction = (
            removed / obs_sum
            if obs_sum > 0
            else 0.0
        )


        corrected_zero_count = int(
            np.sum(
                corrected == 0
            )
        )

        changed_count = int(
            np.sum(
                corrected
                <
                y - 1e-12
            )
        )


        observed_total += obs_sum
        corrected_total += corr_sum

        total_preclip_negative += (
            preclip_negative
        )

        total_corrected_zero += (
            corrected_zero_count
        )

        total_changed += (
            changed_count
        )


        gene_qc.append({
            "gene_index": g,
            "gene": genes[g],
            "gamma_hat": float(
                gamma_hat[g]
            ),
            "observed_sum": obs_sum,
            "corrected_sum": corr_sum,
            "removed_sum": removed,
            "removed_fraction": (
                removed_fraction
            ),
            "preclip_negative_count": (
                preclip_negative
            ),
            "corrected_zero_count": (
                corrected_zero_count
            ),
            "changed_count": (
                changed_count
            ),
            "max_leakage": float(
                leakage.max()
            ),
        })


        rows_seen += 1


        if (
            rows_seen % 20 == 0
            or
            rows_seen == G
        ):
            print(
                "corrected genes:",
                rows_seen,
                "/",
                G,
            )


    if rows_seen != G:
        fail(
            f"Y_obs rows={rows_seen}, expected={G}"
        )


corrected_mm.flush()

del corrected_mm


# ======================================================================
# 9. GLOBAL CORRECTION QC
# ======================================================================

if observed_total <= 0:
    fail(
        "non-positive total observed expression"
    )


removed_total = (
    observed_total
    - corrected_total
)

removed_fraction_total = (
    removed_total
    / observed_total
)


if corrected_total < 0:
    fail(
        "negative corrected total"
    )

if (
    corrected_total
    >
    observed_total
    + 1e-6
):
    fail(
        "global corrected total > observed total"
    )


removed_fractions = np.array([
    x[
        "removed_fraction"
    ]
    for x in gene_qc
])


print()
print("=" * 78)
print("CORRECTED EXPRESSION AUDIT")
print("=" * 78)

print(
    "observed total =",
    observed_total,
)

print(
    "corrected total =",
    corrected_total,
)

print(
    "removed total =",
    removed_total,
)

print(
    "overall removed fraction =",
    removed_fraction_total,
)

print(
    "per-gene removed fraction min/median/max =",
    float(
        removed_fractions.min()
    ),
    float(
        np.median(
            removed_fractions
        )
    ),
    float(
        removed_fractions.max()
    ),
)

print(
    "preclip negative entries =",
    total_preclip_negative,
)

print(
    "final corrected zero entries =",
    total_corrected_zero,
)

print(
    "entries decreased =",
    total_changed,
)

print(
    "min modeled leakage =",
    min_leakage_seen,
)

print(
    "max(corrected-observed) =",
    max_corrected_minus_observed,
)


# ======================================================================
# 10. PER-GENE QC TABLE
# ======================================================================

QC = (
    TMP
    / "per_gene_qc.tsv"
)


with QC.open(
    "w"
) as f:

    columns = [
        "gene_index",
        "gene",
        "gamma_hat",
        "observed_sum",
        "corrected_sum",
        "removed_sum",
        "removed_fraction",
        "preclip_negative_count",
        "corrected_zero_count",
        "changed_count",
        "max_leakage",
    ]

    f.write(
        "\t".join(
            columns
        )
        + "\n"
    )

    for x in gene_qc:
        f.write(
            "\t".join(
                str(
                    x[c]
                )
                for c in columns
            )
            + "\n"
        )


# ======================================================================
# 11. POSTPROCESS MANIFEST
# ======================================================================

POSTERIOR = (
    TMP
    / "posterior_summary.npz"
)


manifest = {
    "method": "DeLeakage",

    "sample": "TENX199",

    "stage": (
        "post_burn_in_summary_"
        "and_corrected_expression"
    ),

    "status": (
        "STAGED_NOT_YET_ATTACHED_AS_CLEAN_TARGET"
    ),

    "dimensions": {
        "G": G,
        "N": N,
        "K": K,
        "N_nei": N_nei,
        "valid_directed_edges": n_edges,
    },

    "mcmc": {
        "total_iterations": 10000,
        "burn_in": BURN_IN,
        "retained_iterations": [
            5001,
            10000,
        ],
        "retained_parameter_samples": (
            N_PARAM_SAMPLES
        ),
        "Z_storage_interval": 25,
        "retained_Z_samples": (
            N_Z_SAMPLES
        ),
        "retained_blocks": (
            RETAINED_BLOCKS
        ),
    },

    "posterior_estimator": {
        "mu": "mean",
        "gamma": "mean",
        "beta": "mean",
        "sigma_sq": "mean",
        "pi": "mean",
        "Z": "per_cell_mode",
        "Z_mode_tie_break": (
            "smallest_zero_based_cluster_index"
        ),
        "Z_mode_tie_count": (
            tie_count
        ),
    },

    "correction_formula": (
        "max(Y_gi - gamma_g * "
        "sum_j(exp(-beta*d_ij) * "
        "cell_size_j * max(mu_g,Z_j,0)), 0)"
    ),

    "effective_mu_policy": {
        "posterior_mu_preserved_unmodified": True,
        "correction_mu": "max(posterior_mean_mu, 0)",
        "basis": (
            "release-compatible operational convention inferred "
            "from authors' released endpoint"
        ),
        "released_mu_negative_count": 0,
        "released_mu_exact_zero_count": 71,
        "released_corrected_above_observed_count": 0,
        "released_raw_zero_to_corrected_positive_count": 0,
        "TENX199_raw_posterior_mu_negative_count": (
            negative_mu_hat_count
        ),
        "TENX199_raw_posterior_mu_min": (
            negative_mu_hat_min
        ),
    },

    "formula_validation": {
        "released_formula_audit": str(
            FORMULA_AUDIT.relative_to(
                ROOT
            )
        ),
        "released_formula_audit_sha256": (
            sha256(
                FORMULA_AUDIT
            )
        ),
    },

    "posterior_summary": {
        "raw_mu_negative_count": (
            negative_mu_hat_count
        ),

        "raw_mu_min": (
            negative_mu_hat_min
        ),

        "effective_mu_negative_count": (
            effective_negative_mu_count
        ),

        "beta_hat": float(
            beta_hat
        ),
        "beta_sd": float(
            beta_sd
        ),

        "gamma_hat_min": float(
            gamma_hat.min()
        ),
        "gamma_hat_median": float(
            np.median(
                gamma_hat
            )
        ),
        "gamma_hat_max": float(
            gamma_hat.max()
        ),

        "retained_serialized_gamma_eq_1": (
            retained_gamma_eq1
        ),

        "retained_serialized_gamma_gt_1": (
            retained_gamma_gt1
        ),

        "posterior_Z_cluster_counts": (
            posterior_cluster_counts
            .astype(int)
            .tolist()
        ),

        "changed_from_initial_label": (
            changed_from_initial
        ),

        "changed_from_initial_fraction": (
            changed_from_initial
            / N
        ),
    },

    "retained_split_half_diagnostics": {
        "mu_mean_correlation": (
            mu_split_corr
        ),

        "gamma_mean_correlation": (
            gamma_split_corr
        ),

        "beta_first_half_mean": (
            beta_first_mean
        ),

        "beta_second_half_mean": (
            beta_second_mean
        ),

        "Z_mode_disagreement_cells": (
            split_z_disagree
        ),

        "Z_mode_disagreement_fraction": (
            split_z_disagree
            / N
        ),

        "note": (
            "descriptive convergence/stability QC; "
            "not used to retune the frozen MCMC"
        ),
    },

    "correction_qc": {
        "observed_total": (
            observed_total
        ),

        "corrected_total": (
            corrected_total
        ),

        "removed_total": (
            removed_total
        ),

        "overall_removed_fraction": (
            removed_fraction_total
        ),

        "per_gene_removed_fraction_min": float(
            removed_fractions.min()
        ),

        "per_gene_removed_fraction_median": float(
            np.median(
                removed_fractions
            )
        ),

        "per_gene_removed_fraction_max": float(
            removed_fractions.max()
        ),

        "preclip_negative_entries": (
            total_preclip_negative
        ),

        "corrected_zero_entries": (
            total_corrected_zero
        ),

        "entries_decreased": (
            total_changed
        ),

        "min_modeled_leakage": (
            min_leakage_seen
        ),

        "max_corrected_minus_observed": (
            max_corrected_minus_observed
        ),
    },

    "upstream": {
        "production_protocol": str(
            PROTOCOL.relative_to(
                ROOT
            )
        ),
        "production_protocol_sha256": (
            sha256(
                PROTOCOL
            )
        ),

        "production_output_manifest": str(
            (
                PROD
                / "outputs.sha256"
            ).relative_to(
                ROOT
            )
        ),

        "production_output_manifest_sha256": (
            sha256(
                PROD
                / "outputs.sha256"
            )
        ),

        "structural_manifest_sha256": (
            sha256(
                STRUCT
                / "structural_manifest.json"
            )
        ),

        "initial_label_sha256": (
            sha256(
                LABELS
                / "Y_label.txt"
            )
        ),
    },

    "artifacts": {
        "posterior_summary": (
            "posterior_summary.npz"
        ),

        "posterior_summary_sha256": (
            sha256(
                POSTERIOR
            )
        ),

        "corrected_GxN_float32": (
            "corrected_GxN.float32.npy"
        ),

        "corrected_GxN_float32_sha256": (
            sha256(
                CORRECTED
            )
        ),

        "per_gene_qc": (
            "per_gene_qc.tsv"
        ),

        "per_gene_qc_sha256": (
            sha256(
                QC
            )
        ),
    },

    "clean_h5ad_created": False,
    "model_training_authorized": False,
}


MANIFEST = (
    TMP
    / "postprocess_manifest.json"
)


MANIFEST.write_text(
    json.dumps(
        manifest,
        indent=2,
    )
    + "\n"
)


# ======================================================================
# 12. FINAL INTERNAL RELOAD OF STAGED MATRIX
# ======================================================================

check = np.load(
    CORRECTED,
    mmap_mode="r",
)

assert check.shape == (
    G,
    N,
)

assert check.dtype == np.float32


# Scan row-wise instead of materializing again.
for g in range(G):

    row = check[
        g,
        :
    ]

    if not np.isfinite(
        row
    ).all():
        fail(
            f"saved corrected row {g} "
            "contains non-finite values"
        )

    if np.any(
        row < 0
    ):
        fail(
            f"saved corrected row {g} "
            "contains negative values"
        )


del check


print()
print("=" * 78)
print("TENX199 POSTPROCESS INTERNAL CONTRACT: PASS")
print("=" * 78)

print(
    "posterior summary =",
    POSTERIOR,
)

print(
    "corrected matrix =",
    CORRECTED,
)

print(
    "gene QC =",
    QC,
)

print(
    "manifest =",
    MANIFEST,
)


# Atomic promotion only after every check above passes.
TMP.rename(
    OUT
)


print()
print("=" * 78)
print("TENX199 DELEAKAGE POSTPROCESS: PASS")
print("STATUS: STAGED — NOT CLEAN.H5AD YET")
print("=" * 78)
