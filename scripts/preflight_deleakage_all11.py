#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("data/prepared_primary2025/samples")
GENES_FILE = Path("runs/frozen_primary2025/gene_axis_280.txt")
OUT = Path("runs/frozen_primary2025/deleakage_all11_preflight.json")

SAMPLES = [
    "TENX191", "TENX192", "TENX193",
    "TENX195", "TENX196",
    "TENX197", "TENX198", "TENX199",
    "TENX200", "TENX201", "TENX202",
]

EXPECTED_SPLIT = {
    "TENX191": "test",
    "TENX192": "test",
    "TENX193": "test",
    "TENX195": "train",
    "TENX196": "train",
    "TENX197": "train",
    "TENX198": "train",
    "TENX199": "train",
    "TENX200": "val",
    "TENX201": "val",
    "TENX202": "val",
}

EXPECTED_DONOR = {
    "TENX191": "Patient 1",
    "TENX192": "Patient 1",
    "TENX193": "Patient 1",
    "TENX195": "Patient 2",
    "TENX196": "Patient 2",
    "TENX197": "Patient 3",
    "TENX198": "Patient 3",
    "TENX199": "Patient 3",
    "TENX200": "Patient 4",
    "TENX201": "Patient 4",
    "TENX202": "Patient 4",
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def matrix_stats(X, chunk=20000):
    n = X.shape[0]

    min_v = np.inf
    max_v = -np.inf
    total = 0.0
    nnz = 0
    nonfinite = 0
    negative = 0
    noninteger = 0

    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        q = X[a:b]

        if sp.issparse(q):
            vals = q.data
            block_nnz = q.nnz
        else:
            vals = np.asarray(q).ravel()
            block_nnz = np.count_nonzero(vals)

        vals = np.asarray(vals)

        if vals.size:
            finite = np.isfinite(vals)
            nonfinite += int((~finite).sum())

            fv = vals[finite]

            if fv.size:
                min_v = min(min_v, float(fv.min()))
                max_v = max(max_v, float(fv.max()))
                total += float(fv.sum())
                negative += int((fv < 0).sum())

                # Raw Xenium target is expected to be count-valued.
                noninteger += int(
                    (~np.isclose(fv, np.rint(fv), atol=1e-6)).sum()
                )

        nnz += int(block_nnz)

    if min_v is np.inf:
        min_v = np.nan
        max_v = np.nan

    return {
        "min": min_v,
        "max": max_v,
        "sum": total,
        "nnz": nnz,
        "nonfinite_values": nonfinite,
        "negative_values": negative,
        "noninteger_values": noninteger,
    }


genes = [
    x.strip()
    for x in GENES_FILE.read_text().splitlines()
    if x.strip()
]

if len(genes) != 280:
    raise RuntimeError(
        f"Frozen gene axis must contain 280 genes, got {len(genes)}"
    )

if len(set(genes)) != 280:
    raise RuntimeError("Frozen 280-gene axis contains duplicates")

results = {}
global_ok = True

for sid in SAMPLES:
    print()
    print("=" * 72)
    print(sid)
    print("=" * 72)

    d = ROOT / sid
    cp = d / "cells.parquet"
    rp = d / "raw.h5ad"

    if not cp.exists() or not rp.exists():
        raise RuntimeError(
            f"{sid}: missing cells.parquet or raw.h5ad"
        )

    cells = pd.read_parquet(
        cp,
        columns=[
            "sample_id",
            "donor_id",
            "split",
            "cell_id",
            "x_um",
            "y_um",
            "polygon_area_um2",
        ],
    )

    raw = ad.read_h5ad(rp, backed="r")

    ids = cells.cell_id.astype(str).to_numpy()
    obs = raw.obs_names.astype(str).to_numpy()
    var = raw.var_names.astype(str).tolist()

    xy = cells[["x_um", "y_um"]].to_numpy(np.float64)
    area = cells["polygon_area_um2"].to_numpy(np.float64)

    donors = sorted(cells.donor_id.astype(str).unique())
    splits = sorted(cells.split.astype(str).unique())
    sample_ids = sorted(cells.sample_id.astype(str).unique())

    checks = {
        "one_sample_id":
            sample_ids == [sid],

        "expected_donor":
            donors == [EXPECTED_DONOR[sid]],

        "expected_split":
            splits == [EXPECTED_SPLIT[sid]],

        "row_count_exact":
            len(cells) == raw.n_obs,

        "cell_ids_unique":
            len(np.unique(ids)) == len(ids),

        "raw_obs_unique":
            len(np.unique(obs)) == len(obs),

        "cell_order_exact":
            np.array_equal(ids, obs),

        "gene_count_280":
            raw.n_vars == 280,

        "gene_axis_exact":
            var == genes,

        "xy_finite":
            bool(np.isfinite(xy).all()),

        "xy_unique":
            len(np.unique(xy, axis=0)) == len(xy),

        "area_finite":
            bool(np.isfinite(area).all()),

        "area_positive":
            bool((area > 0).all()),
    }

    ms = matrix_stats(raw.X)

    checks.update({
        "counts_finite":
            ms["nonfinite_values"] == 0,

        "counts_nonnegative":
            ms["negative_values"] == 0,

        "counts_integer_valued":
            ms["noninteger_values"] == 0,

        "counts_nonzero":
            ms["sum"] > 0,
    })

    ok = all(checks.values())
    global_ok &= ok

    rec = {
        "sample_id": sid,
        "donor_id": donors,
        "split": splits,
        "n_cells": int(raw.n_obs),
        "n_genes": int(raw.n_vars),

        "x_um_min": float(xy[:, 0].min()),
        "x_um_max": float(xy[:, 0].max()),
        "y_um_min": float(xy[:, 1].min()),
        "y_um_max": float(xy[:, 1].max()),

        "cell_area_um2_min": float(area.min()),
        "cell_area_um2_median": float(np.median(area)),
        "cell_area_um2_max": float(area.max()),

        "raw_matrix": ms,

        "raw_h5ad_sha256": sha256(rp),
        "cells_parquet_sha256": sha256(cp),

        "checks": checks,
        "pass": ok,
    }

    results[sid] = rec

    print(
        f"cells={raw.n_obs} genes={raw.n_vars} "
        f"donor={donors} split={splits}"
    )

    print(
        "area_um2 "
        f"min={rec['cell_area_um2_min']:.6g} "
        f"median={rec['cell_area_um2_median']:.6g} "
        f"max={rec['cell_area_um2_max']:.6g}"
    )

    print(
        "raw counts "
        f"min={ms['min']:.6g} "
        f"max={ms['max']:.6g} "
        f"sum={ms['sum']:.6g} "
        f"nnz={ms['nnz']}"
    )

    for k, v in checks.items():
        print(f"{k:26s} = {v}")

    print("SAMPLE PASS =", ok)

    raw.file.close()


# Cross-section identity contract.
# cell_id is section-local; the globally meaningful identity is
# the composite (sample_id, cell_id). Per-section uniqueness was
# already verified above.
composite_seen = set()
composite_duplicates = []

for sid in SAMPLES:
    cells = pd.read_parquet(
        ROOT / sid / "cells.parquet",
        columns=["cell_id"],
    )

    for cid in cells.cell_id.astype(str):
        key = (sid, cid)

        if key in composite_seen:
            composite_duplicates.append(key)
            if len(composite_duplicates) >= 20:
                break

        composite_seen.add(key)

    if len(composite_duplicates) >= 20:
        break

composite_ids_unique = not composite_duplicates
global_ok &= composite_ids_unique

manifest = {
    "method_target": "DeLeakage",
    "gene_axis_file": str(GENES_FILE),
    "gene_axis_sha256": sha256(GENES_FILE),
    "n_samples": len(SAMPLES),
    "composite_sample_cell_ids_unique": composite_ids_unique,
    "composite_duplicate_examples": composite_duplicates,
    "samples": results,
    "all_11_pass": bool(global_ok),
}

OUT.write_text(
    json.dumps(manifest, indent=2) + "\n"
)

print()
print("=" * 72)
print(
    "DELEAKAGE ALL-11 INPUT PREFLIGHT:",
    "PASS" if global_ok else "FAIL"
)
print("RESULT:", OUT)
print("=" * 72)

if not global_ok:
    raise SystemExit(1)
