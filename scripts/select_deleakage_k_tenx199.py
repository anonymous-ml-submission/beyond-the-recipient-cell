#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import itertools

import anndata as ad
import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)


SID = "TENX199"

RAW = Path(
    "data/prepared_primary2025/samples/"
    "TENX199/raw.h5ad"
)

STRUCT = Path(
    "runs/deleakage_preprocess/"
    "TENX199_structural_v1/"
    "structural_manifest.json"
)

OUT = Path(
    "runs/deleakage_preprocess/"
    "TENX199_k_selection_v1.json"
)

# Predeclared before observing any DeLeakage correction.
K_VALUES = list(range(3, 13))

# Independent K-means initializations.
KM_SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

# Fixed development subsample.
SAMPLE_SEED = 2026
N_FIT = 30000

# Silhouette is quadratic, so score on a fixed subset.
N_SIL = 5000

# Predeclared stability condition.
STABILITY_MIN = 0.90


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


print("=" * 78)
print("DELEAKAGE K-SELECTION — TENX199 TRAIN-ONLY")
print("NO H&E LABELS / NO MCMC / NO CORRECTED EXPRESSION")
print("=" * 78)

if OUT.exists():
    raise RuntimeError(
        f"Refusing overwrite: {OUT}"
    )

m = json.loads(
    STRUCT.read_text()
)

assert m["sample"] == SID
assert m["G"] == 280
assert m["N"] == 150145
assert m["K"] is None
assert m["Y_label_written"] is False

a = ad.read_h5ad(RAW)

if a.n_obs != 150145:
    raise RuntimeError(
        f"Unexpected N={a.n_obs}"
    )

if a.n_vars != 280:
    raise RuntimeError(
        f"Unexpected G={a.n_vars}"
    )

X = a.X

if sp.issparse(X):
    X = X.tocsr().astype(np.float32)

    totals = np.asarray(
        X.sum(axis=1)
    ).reshape(-1).astype(np.float64)
else:
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    totals = (
        X.sum(axis=1)
        .astype(np.float64)
    )

if np.any(totals < 0):
    raise RuntimeError(
        "Negative raw total found"
    )

zero = totals == 0

print()
print("RAW CONTRACT")
print("N =", len(totals))
print("G =", a.n_vars)
print("zero rows =", int(zero.sum()))
print(
    "median raw total =",
    float(np.median(totals)),
)

# ------------------------------------------------------------
# Frozen normalization convention
#
# Equivalent to section-level total-count normalization
# to the section median followed by log1p(base=2).
#
# Zero-total rows remain exactly zero.
# ------------------------------------------------------------

target_sum = float(
    np.median(totals)
)

if target_sum <= 0:
    raise RuntimeError(
        "Non-positive normalization target"
    )

rng = np.random.default_rng(
    SAMPLE_SEED
)

fit_n = min(
    N_FIT,
    a.n_obs,
)

fit_idx = np.sort(
    rng.choice(
        a.n_obs,
        size=fit_n,
        replace=False,
    )
)

if sp.issparse(X):
    Xfit = (
        X[fit_idx]
        .toarray()
        .astype(np.float32)
    )
else:
    Xfit = np.asarray(
        X[fit_idx],
        dtype=np.float32,
    ).copy()

fit_totals = totals[fit_idx]

positive = fit_totals > 0

Xfit[positive] *= (
    target_sum
    / fit_totals[positive]
)[:, None].astype(np.float32)

# zero-total cells remain all-zero.
Xfit = (
    np.log1p(Xfit)
    / np.log(2.0)
).astype(np.float32)

if not np.isfinite(Xfit).all():
    raise RuntimeError(
        "Non-finite normalized matrix"
    )

print()
print("NORMALIZATION")
print("target_sum =", target_sum)
print("log =", "log2(1+x)")
print("fit cells =", len(fit_idx))
print(
    "zero rows in fit sample =",
    int(np.sum(~positive)),
)

# Fixed silhouette evaluation subset.
sil_rng = np.random.default_rng(
    SAMPLE_SEED + 1
)

sil_n = min(
    N_SIL,
    len(fit_idx),
)

sil_local_idx = np.sort(
    sil_rng.choice(
        len(fit_idx),
        size=sil_n,
        replace=False,
    )
)

results = []

for K in K_VALUES:

    print()
    print("-" * 78)
    print("K =", K)
    print("-" * 78)

    run_labels = []
    runs = []

    for seed in KM_SEEDS:

        km = KMeans(
            n_clusters=K,
            init="k-means++",
            n_init=1,
            max_iter=300,
            tol=1e-4,
            random_state=seed,
            algorithm="lloyd",
        )

        labels = km.fit_predict(
            Xfit
        )

        counts = np.bincount(
            labels,
            minlength=K,
        )

        cluster_fraction = (
            counts / len(labels)
        )

        sil = float(
            silhouette_score(
                Xfit[sil_local_idx],
                labels[sil_local_idx],
                metric="euclidean",
            )
        )

        print(
            f"seed={seed:3d} "
            f"inertia={km.inertia_:.6e} "
            f"silhouette={sil:.6f} "
            f"min_cluster_frac="
            f"{cluster_fraction.min():.6f}"
        )

        run_labels.append(
            labels
        )

        runs.append({
            "seed": seed,
            "inertia": float(
                km.inertia_
            ),
            "silhouette": sil,
            "cluster_counts": (
                counts.astype(int)
                .tolist()
            ),
            "cluster_fractions": (
                cluster_fraction
                .astype(float)
                .tolist()
            ),
        })

    aris = []

    for i, j in itertools.combinations(
        range(len(run_labels)),
        2,
    ):
        aris.append(
            adjusted_rand_score(
                run_labels[i],
                run_labels[j],
            )
        )

    mean_sil = float(
        np.mean([
            x["silhouette"]
            for x in runs
        ])
    )

    sd_sil = float(
        np.std([
            x["silhouette"]
            for x in runs
        ])
    )

    median_ari = float(
        np.median(aris)
    )

    min_ari = float(
        np.min(aris)
    )

    print(
        "SUMMARY:",
        "mean_silhouette=",
        f"{mean_sil:.6f}",
        "median_pairwise_ARI=",
        f"{median_ari:.6f}",
        "min_pairwise_ARI=",
        f"{min_ari:.6f}",
    )

    results.append({
        "K": K,
        "runs": runs,
        "mean_silhouette": mean_sil,
        "sd_silhouette": sd_sil,
        "median_pairwise_ARI": (
            median_ari
        ),
        "min_pairwise_ARI": (
            min_ari
        ),
    })


# ------------------------------------------------------------
# Predeclared recommendation rule:
#
# 1. K must have median pairwise initialization ARI >= 0.90.
# 2. Among stable K, maximize mean silhouette.
# 3. Exact ties go to smaller K.
#
# This is a project-specific train-only rule, NOT claimed to be
# an official DeLeakage K-selection procedure.
# ------------------------------------------------------------

eligible = [
    x for x in results
    if (
        x["median_pairwise_ARI"]
        >= STABILITY_MIN
    )
]

if eligible:
    best = sorted(
        eligible,
        key=lambda x: (
            -x["mean_silhouette"],
            x["K"],
        ),
    )[0]

    recommended_k = int(
        best["K"]
    )

    decision = (
        "RECOMMEND_K_BY_PREDECLARED_"
        "TRAIN_ONLY_RULE"
    )
else:
    recommended_k = None
    decision = (
        "NO_K_MEETS_STABILITY_RULE"
    )


payload = {
    "method": "DeLeakage",
    "stage": "K_selection_before_correction",
    "sample": SID,
    "model_split_role": "train",

    "heldout_sections_used": False,
    "H_and_E_annotations_used": False,
    "corrected_expression_used": False,
    "downstream_model_performance_used": False,

    "input": {
        "raw_h5ad": str(RAW),
        "raw_h5ad_sha256": sha256(
            RAW
        ),
        "structural_manifest": str(
            STRUCT
        ),
        "structural_manifest_sha256": (
            sha256(STRUCT)
        ),
    },

    "normalization": {
        "method": (
            "total_count_to_section_median_"
            "then_log1p_base2"
        ),
        "target_sum": target_sum,
        "zero_total_policy": (
            "remain_zero"
        ),
    },

    "selection_data": {
        "N_total": int(a.n_obs),
        "G": int(a.n_vars),
        "N_fit": int(len(fit_idx)),
        "fit_sample_seed": (
            SAMPLE_SEED
        ),
        "N_silhouette": int(
            sil_n
        ),
    },

    "K_candidates": K_VALUES,
    "kmeans_seeds": KM_SEEDS,

    "selection_rule": {
        "minimum_median_pairwise_ARI": (
            STABILITY_MIN
        ),
        "primary_metric": (
            "maximum_mean_silhouette_"
            "among_stable_K"
        ),
        "tie_break": "smaller_K",
        "note": (
            "project-specific train-only "
            "selection rule; not an official "
            "DeLeakage K-selection algorithm"
        ),
    },

    "results": results,

    "decision": decision,
    "recommended_K": recommended_k,
}

OUT.write_text(
    json.dumps(
        payload,
        indent=2,
    ) + "\n"
)

print()
print("=" * 78)
print("K-SELECTION RESULT")
print("=" * 78)
print("decision =", decision)
print(
    "recommended_K =",
    recommended_k,
)

for x in results:
    print(
        f"K={x['K']:2d} "
        f"sil={x['mean_silhouette']:.6f} "
        f"medianARI="
        f"{x['median_pairwise_ARI']:.6f}"
    )

print()
print("RESULT:", OUT)
print("=" * 78)
