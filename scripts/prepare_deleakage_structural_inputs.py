#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


GENE_AXIS = Path(
    "runs/frozen_primary2025/gene_axis_280.txt"
)

CONTRACT = Path(
    "runs/frozen_primary2025/"
    "deleakage_preprocess_contract_v1.json"
)

PREP_ROOT = Path(
    "data/prepared_primary2025/samples"
)

EXPECTED_GENE_SHA = (
    "14947279372641dafffe7ae755b9da7ec3b361786c4274ec554e226fdde98713"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def query_ball(tree, points, radius):
    try:
        return tree.query_ball_point(
            points,
            r=radius,
            workers=-1,
        )
    except TypeError:
        return tree.query_ball_point(
            points,
            r=radius,
        )


def write_yobs(X, out: Path):
    """
    DeLeakage requires gene x cell.

    Input raw.h5ad is cell x gene.
    Write without hidden normalization/transformation.
    """
    n_cells, n_genes = X.shape

    if sp.issparse(X):
        Xc = X.tocsc()

        with out.open("w") as f:
            for g in range(n_genes):
                row = np.asarray(
                    Xc[:, g].toarray()
                ).reshape(-1)

                np.savetxt(
                    f,
                    row.reshape(1, n_cells),
                    fmt="%.10g",
                )
    else:
        Xd = np.asarray(X)

        with out.open("w") as f:
            for g in range(n_genes):
                row = Xd[:, g]

                np.savetxt(
                    f,
                    row.reshape(1, n_cells),
                    fmt="%.10g",
                )


def build_graph(coords):
    """
    Select a single fixed Euclidean radius for this section such that
    median neighbor count is exactly 10.

    Start from the median 10th-nearest-neighbor distance.
    """
    n = len(coords)

    if n < 12:
        raise RuntimeError(
            f"Need >=12 cells, got {n}"
        )

    tree = cKDTree(coords)

    try:
        d, _ = tree.query(
            coords,
            k=12,
            workers=-1,
        )
    except TypeError:
        d, _ = tree.query(
            coords,
            k=12,
        )

    # column 0 is self; column 10 is 10th non-self neighbor.
    d10 = d[:, 10]

    radius = float(
        np.nextafter(
            np.median(d10),
            np.inf,
        )
    )

    balls = query_ball(
        tree,
        coords,
        radius,
    )

    neighbor_lists = []

    for i, b in enumerate(balls):
        # Official bundle stores 0-based cell IDs.
        # Sort by index for deterministic serialization.
        q = sorted(
            int(j)
            for j in b
            if int(j) != i
        )
        neighbor_lists.append(q)

    degree = np.asarray(
        [len(x) for x in neighbor_lists],
        dtype=np.int64,
    )

    med = float(np.median(degree))

    if med != 10.0:
        raise RuntimeError(
            "Fixed-radius construction did not produce "
            f"median degree 10: radius={radius}, "
            f"median={med}. "
            "Refusing silent radius adjustment."
        )

    width = int(degree.max())

    nei = np.full(
        (n, width),
        -2,
        dtype=np.int64,
    )

    dist = np.full(
        (n, width),
        -1.0,
        dtype=np.float64,
    )

    for i, q in enumerate(neighbor_lists):
        if not q:
            continue

        qarr = np.asarray(q, dtype=np.int64)

        dd = np.linalg.norm(
            coords[qarr] - coords[i],
            axis=1,
        )

        nei[i, :len(qarr)] = qarr
        dist[i, :len(qarr)] = dd

    return radius, degree, nei, dist


def build_mb_layers(
    nei,
    seed=123,
    stop_threshold=100,
):
    """
    Published DeLeakage Methods policy.

    For each layer:
      - randomly order currently unassigned cells;
      - greedily accept a cell if it has not been blocked by an
        already accepted cell in the same layer;
      - block that cell and its neighbors for the current layer.

    The first generated layer with size <100 is retained as the
    final parallel layer, matching the paper Methods description.
    All cells still unassigned after removing it become RMB.
    """
    n = nei.shape[0]

    adjacency = [
        row[row >= 0].astype(np.int64)
        for row in nei
    ]

    rng = np.random.default_rng(seed)

    unassigned = np.ones(
        n,
        dtype=bool,
    )

    layers = []

    while np.any(unassigned):
        ids = np.flatnonzero(unassigned)

        order = rng.permutation(ids)

        blocked = np.zeros(
            n,
            dtype=bool,
        )

        chosen = []

        for i in order:
            i = int(i)

            if not unassigned[i]:
                continue

            if blocked[i]:
                continue

            chosen.append(i)

            blocked[i] = True

            q = adjacency[i]

            if len(q):
                blocked[q] = True

        layer = np.asarray(
            sorted(chosen),
            dtype=np.int64,
        )

        if len(layer) == 0:
            raise RuntimeError(
                "Generated empty MB layer"
            )

        # Exact independent-set guard.
        layer_set = set(layer.tolist())

        for i in layer:
            if any(
                int(j) in layer_set
                for j in adjacency[int(i)]
            ):
                raise RuntimeError(
                    "MB layer violates independent-set property"
                )

        layers.append(layer)

        # These cells have now been scheduled.
        unassigned[layer] = False

        # Published stopping rule.
        if len(layer) < stop_threshold:
            break

    tail = np.flatnonzero(
        unassigned
    ).astype(np.int64)

    # Strict partition guard.
    parts = layers + [tail]

    combined = np.concatenate(parts)

    if len(combined) != n:
        raise RuntimeError(
            f"MB/RMB length mismatch: "
            f"{len(combined)} != {n}"
        )

    if len(np.unique(combined)) != n:
        raise RuntimeError(
            "MB/RMB do not form a unique partition"
        )

    if not np.array_equal(
        np.sort(combined),
        np.arange(n),
    ):
        raise RuntimeError(
            "MB/RMB partition does not cover 0..N-1 exactly"
        )

    return layers, tail


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sample",
        required=True,
    )

    ap.add_argument(
        "--out",
        required=True,
    )

    ap.add_argument(
        "--mb-seed",
        type=int,
        default=123,
    )

    args = ap.parse_args()

    sid = args.sample
    out = Path(args.out)

    print("=" * 72)
    print("DELEAKAGE STRUCTURAL INPUT ADAPTER")
    print("NO CORRECTION / NO MODEL TRAINING")
    print("=" * 72)
    print("sample =", sid)
    print("out    =", out)

    if out.exists():
        raise RuntimeError(
            f"Output already exists; refusing overwrite: {out}"
        )

    # ------------------------------------------------------------------
    # Frozen contracts
    # ------------------------------------------------------------------

    if sha256(GENE_AXIS) != EXPECTED_GENE_SHA:
        raise RuntimeError(
            "Frozen gene-axis SHA mismatch"
        )

    contract = json.loads(
        CONTRACT.read_text()
    )

    if (
        contract["development_policy"]["development_pilot"]
        == sid
    ):
        if (
            contract["development_policy"]["model_split_role"]
            != "train"
        ):
            raise RuntimeError(
                "TENX199 must remain train in frozen contract"
            )

    genes = [
        x.strip()
        for x in GENE_AXIS.read_text().splitlines()
        if x.strip()
    ]

    if len(genes) != 280:
        raise RuntimeError(
            f"Expected 280 genes, got {len(genes)}"
        )

    # ------------------------------------------------------------------
    # Input data
    # ------------------------------------------------------------------

    d = PREP_ROOT / sid

    raw_path = d / "raw.h5ad"
    cells_path = d / "cells.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    if not cells_path.exists():
        raise FileNotFoundError(cells_path)

    cells = pd.read_parquet(
        cells_path,
        columns=[
            "cell_id",
            "x_um",
            "y_um",
        ],
    )

    a = ad.read_h5ad(
        raw_path
    )

    obs = a.obs_names.astype(str).to_numpy()
    cell_ids = cells["cell_id"].astype(str).to_numpy()

    if not np.array_equal(
        obs,
        cell_ids,
    ):
        raise RuntimeError(
            "raw.h5ad obs_names and cells.parquet cell_id "
            "are not identical in order"
        )

    if a.var_names.astype(str).tolist() != genes:
        raise RuntimeError(
            "raw.h5ad gene axis differs from frozen 280 genes"
        )

    if a.n_obs != len(cells):
        raise RuntimeError(
            "raw/cells row-count mismatch"
        )

    X = a.X

    if sp.issparse(X):
        X = X.tocsr()
        vals = X.data
    else:
        X = np.asarray(X)
        vals = X.reshape(-1)

    if not np.isfinite(vals).all():
        raise RuntimeError(
            "raw expression contains non-finite values"
        )

    if np.any(vals < 0):
        raise RuntimeError(
            "raw expression contains negative values"
        )

    if not np.allclose(
        vals,
        np.rint(vals),
        atol=1e-6,
    ):
        raise RuntimeError(
            "raw expression is not count-valued"
        )

    # ------------------------------------------------------------------
    # Cell-size fallback
    # ------------------------------------------------------------------

    if sp.issparse(X):
        totals = np.asarray(
            X.sum(axis=1)
        ).reshape(-1).astype(np.float64)
    else:
        totals = (
            np.asarray(X)
            .sum(axis=1)
            .astype(np.float64)
        )

    if np.any(~np.isfinite(totals)):
        raise RuntimeError(
            "Non-finite per-cell totals"
        )

    # DeLeakage cell-size fallback:
    #
    #   delta_i = S_i / median_j(S_j)
    #
    # For this prepared cohort, S_i is the total retained count over
    # the frozen 280-gene DeLeakage expression axis.
    #
    # The TENX199 source-transcript audit proved that these row sums
    # exactly equal the complete retained transcript totals under the
    # frozen preparation filter.
    #
    # A retained segmentation cell with S_i == 0 therefore receives
    # delta_i == 0. No epsilon is added and the cell is not removed.
    if np.any(totals < 0):
        bad = int(
            np.sum(totals < 0)
        )

        raise RuntimeError(
            f"{bad} cells have negative total count"
        )

    zero_total_count = int(
        np.sum(totals == 0)
    )

    zero_total_fraction = float(
        zero_total_count / len(totals)
    )

    med_total = float(
        np.median(totals)
    )

    if med_total <= 0:
        raise RuntimeError(
            "Median total count <=0"
        )

    cell_size = totals / med_total

    # ------------------------------------------------------------------
    # Coordinates + neighbor graph
    # ------------------------------------------------------------------

    coords = cells[
        ["x_um", "y_um"]
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(coords).all():
        raise RuntimeError(
            "Coordinates contain non-finite values"
        )

    (
        radius,
        degree,
        nei,
        dist,
    ) = build_graph(coords)

    print()
    print("NEIGHBOR GRAPH")
    print("radius_um =", radius)
    print(
        "degree min/median/max =",
        int(degree.min()),
        float(np.median(degree)),
        int(degree.max()),
    )

    # ------------------------------------------------------------------
    # MB scheduling
    # ------------------------------------------------------------------

    layers, tail = build_mb_layers(
        nei,
        seed=args.mb_seed,
        stop_threshold=100,
    )

    print()
    print("MB")
    print("N_MB =", len(layers))
    print(
        "layer sizes =",
        [len(x) for x in layers],
    )
    print("N_tail =", len(tail))

    # ------------------------------------------------------------------
    # Serialize only after every in-memory guard has passed.
    # ------------------------------------------------------------------

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    mbdir = out / "MB"
    mbdir.mkdir()

    # Keep exact row identities beside the numeric files.
    np.savetxt(
        out / "cell_ids.txt",
        cell_ids,
        fmt="%s",
    )

    np.savetxt(
        out / "genes.txt",
        np.asarray(genes),
        fmt="%s",
    )

    print()
    print("writing Y_obs.txt ...")

    write_yobs(
        X,
        out / "Y_obs.txt",
    )

    np.savetxt(
        out / "cell_size.txt",
        cell_size,
        fmt="%.12g",
    )

    np.savetxt(
        out / "nei_list.txt",
        nei,
        fmt="%d",
    )

    np.savetxt(
        out / "nei_dist.txt",
        dist,
        fmt="%.12g",
    )

    lengths = np.asarray(
        [len(x) for x in layers],
        dtype=np.int64,
    )

    np.savetxt(
        mbdir / "length_vec.txt",
        lengths,
        fmt="%d",
    )

    for i, layer in enumerate(
        layers,
        start=1,
    ):
        np.savetxt(
            mbdir / f"MB_{i}.txt",
            layer,
            fmt="%d",
        )

    np.savetxt(
        mbdir / "RMB.txt",
        tail,
        fmt="%d",
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "method": "DeLeakage",
        "stage": "structural_preprocessing_before_K",
        "sample": sid,

        "G": int(a.n_vars),
        "N": int(a.n_obs),

        "K": None,
        "Y_label_written": False,

        "coordinate_columns": [
            "x_um",
            "y_um",
        ],

        "radius_um": radius,

        "degree": {
            "min": int(degree.min()),
            "median": float(np.median(degree)),
            "max": int(degree.max()),
        },

        "N_nei": int(nei.shape[1]),

        "neighbor_indexing": "zero_based",
        "neighbor_padding": -2,
        "distance_padding": -1,

        "cell_size": {
            "rule": (
                "row_sum_raw_X_over_280_genes/"
                "section_median_row_sum"
            ),
            "median_raw_total": med_total,
            "zero_total_cells": zero_total_count,
            "zero_total_fraction": zero_total_fraction,
            "zero_total_policy": "retain_with_cell_size_zero",
            "epsilon_added": False,
            "cells_removed": False,
            "min_factor": float(cell_size.min()),
            "median_factor": float(
                np.median(cell_size)
            ),
            "max_factor": float(cell_size.max()),
        },

        "MB": {
            "policy": "paper_methods_random_greedy_v1",
            "seed": int(args.mb_seed),
            "stop_threshold": 100,
            "N_MB": len(layers),
            "layer_sizes": [
                int(len(x))
                for x in layers
            ],
            "N_tail": int(len(tail)),
        },

        "input_contract": {
            "gene_axis_sha256": sha256(GENE_AXIS),
            "preprocess_contract_sha256": sha256(CONTRACT),
        },

        "inputs": {
            "raw_h5ad": str(raw_path),
            "cells_parquet": str(cells_path),
        },
    }

    # Hash serialized numeric contract files.
    files = [
        out / "Y_obs.txt",
        out / "cell_size.txt",
        out / "nei_list.txt",
        out / "nei_dist.txt",
        out / "cell_ids.txt",
        out / "genes.txt",
        mbdir / "length_vec.txt",
        mbdir / "RMB.txt",
    ]

    files += sorted(
        mbdir.glob("MB_*.txt")
    )

    manifest["output_sha256"] = {
        str(p.relative_to(out)): sha256(p)
        for p in files
    }

    mp = out / "structural_manifest.json"

    mp.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 72)
    print("DELEAKAGE STRUCTURAL PREPROCESSING: PASS")
    print("sample =", sid)
    print("G =", a.n_vars)
    print("N =", a.n_obs)
    print("radius_um =", radius)
    print("median_degree =", float(np.median(degree)))
    print("N_nei =", nei.shape[1])
    print("N_MB =", len(layers))
    print("N_tail =", len(tail))
    print("manifest =", mp)
    print("=" * 72)


if __name__ == "__main__":
    main()
