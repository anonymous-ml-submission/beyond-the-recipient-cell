#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile

import anndata as ad
import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans


ROOT = Path(".")
PREP = Path(
    "data/prepared_primary2025/samples"
)

KJSON = Path(
    "runs/deleakage_preprocess/"
    "TENX199_k_selection_v1.json"
)

K_SCRIPT = Path(
    "scripts/select_deleakage_k_tenx199.py"
)

REF_RAW = PREP / "TENX199" / "raw.h5ad"

REF_STRUCT = Path(
    "runs/deleakage_preprocess/"
    "TENX199_structural_v1/"
    "structural_manifest.json"
)

REF_LABEL_DIR = Path(
    "runs/deleakage_preprocess/"
    "TENX199_labels_K3_v1"
)

EXPECTED_KJSON_SHA = (
    "837e058fffb716df2f89328c2febaed04"
    "f9a0701b021819e8e9e6e94854c96a7"
)

EXPECTED_K_SCRIPT_SHA = (
    "d76ab152e3e96b7fdcbfb13eda55a758"
    "341d97f7c4daa2c40d28627c04d344f2"
)

EXPECTED_REF_STRUCT_SHA = (
    "88600b36f3cf8640b9e265146e6dcfe38"
    "3499384b24935830fc8226aae72842b"
)

EXPECTED_REF_LABEL_SHA = (
    "556af213de67c9d21e7963e598925ab60"
    "d0c268e279761e6eaff232e42a26dd1"
)

EXPECTED_REF_CENTER_SHA = (
    "9a9fc2edabb04d2783b872c287e057b78"
    "305251fccdd5c9ba849e6843a2d1f5b"
)

EXPECTED_REF_CELL_SHA = (
    "daf70533fcfac3947265d9c514f689aa6"
    "99fae82896fd80d14088e3c12bd6b52"
)

EXPECTED_REF_MANIFEST_SHA = (
    "2803d18416dc5408852ba7ef72785689"
    "b1b381b14707cb9f3c608f7fd7a8da37"
)

EXPECTED_GENE_SHA = (
    "14947279372641dafffe7ae755b9da7ec"
    "3b361786c4274ec554e226fdde98713"
)

K = 3
KM_SEED = 101


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def labels_as_bytes(
    labels: np.ndarray,
) -> bytes:

    b = io.StringIO()

    np.savetxt(
        b,
        labels,
        fmt="%d",
    )

    return b.getvalue().encode()


def npy_as_bytes(
    x: np.ndarray,
) -> bytes:

    b = io.BytesIO()

    np.save(
        b,
        x,
    )

    return b.getvalue()


def cells_as_bytes(
    cell_ids,
) -> bytes:

    return (
        "\n".join(
            map(str, cell_ids)
        )
        + "\n"
    ).encode()


def read_and_normalize(
    raw_path: Path,
):
    """
    EXACT numerical convention recovered from the frozen TENX199
    K-selection implementation:

      X -> float32
      totals -> float64
      target_sum = median(totals)
      positive rows *= target_sum / total
      zero rows unchanged
      log2(1+x)
      final normalized matrix float32
    """

    a = ad.read_h5ad(
        raw_path
    )

    if a.n_vars != 280:
        raise RuntimeError(
            f"{raw_path}: G={a.n_vars}, expected 280"
        )

    obs_names = (
        a.obs_names
        .astype(str)
        .tolist()
    )

    var_names = (
        a.var_names
        .astype(str)
        .tolist()
    )

    X = a.X

    if sp.issparse(X):

        X = (
            X.tocsr()
            .astype(np.float32)
        )

        totals = np.asarray(
            X.sum(axis=1)
        ).reshape(-1).astype(
            np.float64
        )

        Xn = (
            X.toarray()
            .astype(np.float32)
        )

    else:

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        totals = (
            X.sum(axis=1)
            .astype(np.float64)
        )

        Xn = np.asarray(
            X,
            dtype=np.float32,
        ).copy()

    if not np.isfinite(
        totals
    ).all():
        raise RuntimeError(
            f"{raw_path}: non-finite totals"
        )

    if np.any(
        totals < 0
    ):
        raise RuntimeError(
            f"{raw_path}: negative total"
        )

    zero = (
        totals == 0
    )

    target_sum = float(
        np.median(
            totals
        )
    )

    if not (
        np.isfinite(target_sum)
        and target_sum > 0
    ):
        raise RuntimeError(
            f"{raw_path}: invalid target_sum={target_sum}"
        )

    positive = (
        totals > 0
    )

    Xn[
        positive
    ] *= (
        target_sum
        / totals[
            positive
        ]
    )[:, None].astype(
        np.float32
    )

    # Zero-total rows remain exact zeros.

    Xn = (
        np.log1p(
            Xn
        )
        / np.log(
            2.0
        )
    ).astype(
        np.float32
    )

    if not np.isfinite(
        Xn
    ).all():
        raise RuntimeError(
            f"{raw_path}: non-finite normalized matrix"
        )

    if np.any(
        Xn[zero] != 0
    ):
        raise RuntimeError(
            f"{raw_path}: zero rows changed during normalization"
        )

    del a
    del X

    return (
        Xn,
        totals,
        zero,
        target_sum,
        obs_names,
        var_names,
    )


def fit_k3(
    Xn: np.ndarray,
):

    km = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=KM_SEED,
        algorithm="lloyd",
    )

    labels = km.fit_predict(
        Xn
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    if not np.array_equal(
        np.unique(labels),
        np.arange(K),
    ):
        raise RuntimeError(
            "KMeans did not produce labels 0,1,2"
        )

    return km, labels


def verify_frozen_contracts():

    checks = {
        KJSON:
            EXPECTED_KJSON_SHA,

        K_SCRIPT:
            EXPECTED_K_SCRIPT_SHA,

        REF_STRUCT:
            EXPECTED_REF_STRUCT_SHA,

        REF_LABEL_DIR / "Y_label.txt":
            EXPECTED_REF_LABEL_SHA,

        REF_LABEL_DIR / "cluster_centers.npy":
            EXPECTED_REF_CENTER_SHA,

        REF_LABEL_DIR / "cell_ids.txt":
            EXPECTED_REF_CELL_SHA,

        REF_LABEL_DIR / "label_manifest.json":
            EXPECTED_REF_MANIFEST_SHA,
    }

    for path, expected in checks.items():

        if not path.is_file():
            raise RuntimeError(
                f"missing frozen reference: {path}"
            )

        got = sha256(
            path
        )

        if got != expected:
            raise RuntimeError(
                f"frozen reference changed: {path}\n"
                f"expected={expected}\n"
                f"actual={got}"
            )

    k = json.loads(
        KJSON.read_text()
    )

    if (
        k["decision"]
        !=
        "RECOMMEND_K_BY_PREDECLARED_TRAIN_ONLY_RULE"
    ):
        raise RuntimeError(
            "TENX199 K decision changed"
        )

    if int(
        k["recommended_K"]
    ) != 3:
        raise RuntimeError(
            "TENX199 recommended K is not 3"
        )

    if (
        k["sample"]
        != "TENX199"
        or
        k["model_split_role"]
        != "train"
    ):
        raise RuntimeError(
            "TENX199 K-selection split contract changed"
        )


def verify_reference_exact():
    """
    Critical gate.

    Reproduce the frozen TENX199 full-section labels using the
    recovered numerical convention. NOTHING is written here.
    """

    verify_frozen_contracts()

    (
        Xn,
        totals,
        zero,
        target_sum,
        obs,
        genes,
    ) = read_and_normalize(
        REF_RAW
    )

    if len(obs) != 150145:
        raise RuntimeError(
            "TENX199 N changed"
        )

    if len(genes) != 280:
        raise RuntimeError(
            "TENX199 G changed"
        )

    if target_sum != 168.0:
        raise RuntimeError(
            f"TENX199 target_sum={target_sum}, expected 168"
        )

    if int(
        zero.sum()
    ) != 73:
        raise RuntimeError(
            "TENX199 zero-cell count changed"
        )

    km, labels = fit_k3(
        Xn
    )

    frozen_labels = np.loadtxt(
        REF_LABEL_DIR
        / "Y_label.txt",
        dtype=np.int64,
    )

    frozen_centers = np.load(
        REF_LABEL_DIR
        / "cluster_centers.npy",
        allow_pickle=False,
    )

    if not np.array_equal(
        labels,
        frozen_labels,
    ):
        diff = int(
            np.sum(
                labels
                != frozen_labels
            )
        )

        raise RuntimeError(
            "REFERENCE GATE FAILED: regenerated TENX199 "
            f"labels differ at {diff} cells"
        )

    # cluster_centers.npy is NOT a DeLeakage runtime input.
    # Historical production consumes Y_label.txt, not these centers.
    # Keep centers only as a numerical diagnostic.
    center_max_abs = float(
        np.max(
            np.abs(
                km.cluster_centers_
                -
                frozen_centers
            )
        )
    )

    if not np.allclose(
        km.cluster_centers_,
        frozen_centers,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "REFERENCE GATE FAILED: TENX199 diagnostic "
            "cluster centers differ beyond atol=1e-6; "
            f"max_abs={center_max_abs}"
        )

    if sha_bytes(
        labels_as_bytes(
            labels
        )
    ) != EXPECTED_REF_LABEL_SHA:
        raise RuntimeError(
            "REFERENCE GATE FAILED: Y_label serialization differs"
        )

    # Do not byte-gate regenerated cluster centers:
    # historical DeLeakage runtime never consumes cluster_centers.npy.
    # The frozen historical file itself is still SHA-verified above.

    if sha_bytes(
        cells_as_bytes(
            obs
        )
    ) != EXPECTED_REF_CELL_SHA:
        raise RuntimeError(
            "REFERENCE GATE FAILED: cell-id serialization differs"
        )

    counts = np.bincount(
        labels,
        minlength=3,
    )

    if counts.tolist() != [
        32257,
        76927,
        40961,
    ]:
        raise RuntimeError(
            f"TENX199 cluster counts changed: {counts.tolist()}"
        )

    if int(
        km.n_iter_
    ) != 10:
        raise RuntimeError(
            f"TENX199 n_iter changed: {km.n_iter_}"
        )

    # Float32 sklearn inertia from the frozen manifest.
    if not np.isclose(
        float(km.inertia_),
        12502918.0,
        rtol=0.0,
        atol=1e-3,
    ):
        raise RuntimeError(
            f"TENX199 inertia changed: {km.inertia_}"
        )

    print("=" * 78)
    print("TENX199 EXACT LABEL REPRODUCTION GATE: PASS")
    print("=" * 78)
    print("labels exact             = PASS")
    print("cluster centers diagnostic= PASS")
    print("center max abs difference =", center_max_abs)
    print("Y_label byte SHA          = PASS")
    print("cell-id byte SHA         = PASS")
    print("cluster counts           =", counts.tolist())
    print("n_iter                    =", int(km.n_iter_))
    print("inertia                   =", float(km.inertia_))
    print("NO FILES WRITTEN")
    print("=" * 78)


def verify_structural_input(
    sid: str,
    raw_path: Path,
    struct_path: Path,
    obs,
    genes,
    target_sum: float,
    zero_count: int,
):

    if not struct_path.is_file():
        raise RuntimeError(
            f"{sid}: missing structural manifest {struct_path}"
        )

    sm = json.loads(
        struct_path.read_text()
    )

    if sm["sample"] != sid:
        raise RuntimeError(
            f"{sid}: structural sample mismatch"
        )

    if int(
        sm["G"]
    ) != 280:
        raise RuntimeError(
            f"{sid}: structural G != 280"
        )

    if int(
        sm["N"]
    ) != len(obs):
        raise RuntimeError(
            f"{sid}: structural N mismatch"
        )

    if sm["K"] is not None:
        raise RuntimeError(
            f"{sid}: structural stage unexpectedly has K"
        )

    if sm[
        "Y_label_written"
    ] is not False:
        raise RuntimeError(
            f"{sid}: structural stage says labels written"
        )

    if float(
        sm["degree"]["median"]
    ) != 10.0:
        raise RuntimeError(
            f"{sid}: structural median degree != 10"
        )

    if (
        sm["input_contract"][
            "gene_axis_sha256"
        ]
        != EXPECTED_GENE_SHA
    ):
        raise RuntimeError(
            f"{sid}: gene-axis contract changed"
        )

    if float(
        sm["cell_size"][
            "median_raw_total"
        ]
    ) != target_sum:
        raise RuntimeError(
            f"{sid}: structural/raw median-total mismatch"
        )

    if int(
        sm["cell_size"][
            "zero_total_cells"
        ]
    ) != zero_count:
        raise RuntimeError(
            f"{sid}: structural/raw zero-count mismatch"
        )

    S = struct_path.parent

    struct_cells = (
        S
        / "cell_ids.txt"
    ).read_text().splitlines()

    struct_genes = (
        S
        / "genes.txt"
    ).read_text().splitlines()

    if struct_cells != list(
        map(str, obs)
    ):
        raise RuntimeError(
            f"{sid}: structural/raw cell axis mismatch"
        )

    if struct_genes != list(
        map(str, genes)
    ):
        raise RuntimeError(
            f"{sid}: structural/raw gene axis mismatch"
        )

    # Validate every structural object against the accepted manifest.
    for rel, expected in sm[
        "output_sha256"
    ].items():

        p = S / rel

        if not p.is_file():
            raise RuntimeError(
                f"{sid}: structural file missing: {rel}"
            )

        got = sha256(
            p
        )

        if got != expected:
            raise RuntimeError(
                f"{sid}: structural SHA mismatch: {rel}"
            )

    return sm


def generate(
    sid: str,
    struct_path: Path,
    out: Path,
):

    verify_frozen_contracts()

    if out.exists() or out.is_symlink():
        raise RuntimeError(
            f"Refusing overwrite: {out}"
        )

    raw_path = (
        PREP
        / sid
        / "raw.h5ad"
    )

    if not raw_path.is_file():
        raise RuntimeError(
            f"{sid}: raw.h5ad missing"
        )

    (
        Xn,
        totals,
        zero,
        target_sum,
        obs,
        genes,
    ) = read_and_normalize(
        raw_path
    )

    sm = verify_structural_input(
        sid=sid,
        raw_path=raw_path,
        struct_path=struct_path,
        obs=obs,
        genes=genes,
        target_sum=target_sum,
        zero_count=int(
            zero.sum()
        ),
    )

    km, labels = fit_k3(
        Xn
    )

    counts = np.bincount(
        labels,
        minlength=K,
    )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            f"{sid}: empty KMeans cluster"
        )

    zero_cluster_counts = np.bincount(
        labels[zero],
        minlength=K,
    )

    if int(
        zero_cluster_counts.sum()
    ) != int(
        zero.sum()
    ):
        raise RuntimeError(
            f"{sid}: zero-cell accounting failed"
        )

    # Identical zero vectors should receive one deterministic label.
    if (
        int(zero.sum()) > 0
        and
        int(
            zero_cluster_counts.max()
        ) != int(
            zero.sum()
        )
    ):
        raise RuntimeError(
            f"{sid}: identical zero vectors split across clusters"
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        tempfile.mkdtemp(
            prefix=out.name + ".tmp.",
            dir=out.parent,
        )
    )

    try:

        y_path = (
            tmp
            / "Y_label.txt"
        )

        centers_path = (
            tmp
            / "cluster_centers.npy"
        )

        cells_path = (
            tmp
            / "cell_ids.txt"
        )

        manifest_path = (
            tmp
            / "label_manifest.json"
        )

        np.savetxt(
            y_path,
            labels,
            fmt="%d",
        )

        np.save(
            centers_path,
            km.cluster_centers_,
        )

        cells_path.write_text(
            "\n".join(
                map(str, obs)
            )
            + "\n"
        )

        manifest = {
            "method":
                "DeLeakage",

            "stage":
                "initial_labels_before_MCMC",

            "sample":
                sid,

            "model_split_role":
                "train",

            "K":
                3,

            "K_selection": {
                "source":
                    str(KJSON),

                "source_sha256":
                    sha256(KJSON),

                "source_script":
                    str(K_SCRIPT),

                "source_script_sha256":
                    sha256(K_SCRIPT),

                "decision":
                    "FIXED_GLOBAL_K3_FROM_"
                    "TENX199_TRAIN_ONLY_SELECTION",

                "recommended_K":
                    3,

                "no_section_specific_K_search":
                    True,
            },

            "normalization": {
                "method":
                    "total_count_to_section_"
                    "median_then_log1p_base2",

                "target_sum":
                    float(target_sum),

                "zero_total_policy":
                    "remain_exact_zero",
            },

            "kmeans": {
                "init":
                    "k-means++",

                "n_init":
                    1,

                "max_iter":
                    300,

                "tol":
                    0.0001,

                "algorithm":
                    "lloyd",

                "random_state":
                    101,

                "seed_policy":
                    "fixed_from_TENX199_"
                    "train_only_contract",

                "inertia":
                    float(km.inertia_),

                "n_iter":
                    int(km.n_iter_),
            },

            "N":
                int(len(labels)),

            "G":
                280,

            "cluster_counts":
                counts.astype(
                    int
                ).tolist(),

            "cluster_fractions":
                (
                    counts
                    / len(labels)
                ).astype(
                    float
                ).tolist(),

            "zero_total_cells":
                int(zero.sum()),

            "zero_total_cluster_counts":
                zero_cluster_counts.astype(
                    int
                ).tolist(),

            "independence": {
                "H_and_E_used":
                    False,

                "corrected_expression_used":
                    False,

                "heldout_section_used":
                    False,

                "downstream_model_metric_used":
                    False,
            },

            "reference_reproduction_gate": {
                "sample":
                    "TENX199",

                "required":
                    True,

                "reference_Y_label_sha256":
                    EXPECTED_REF_LABEL_SHA,

                "reference_cluster_centers_sha256":
                    EXPECTED_REF_CENTER_SHA,
            },

            "inputs": {
                "raw_h5ad":
                    str(raw_path),

                "raw_h5ad_sha256":
                    sha256(raw_path),

                "structural_manifest":
                    str(struct_path),

                "structural_manifest_sha256":
                    sha256(struct_path),

                "structural_radius_um":
                    float(
                        sm["radius_um"]
                    ),

                "structural_N_nei":
                    int(
                        sm["N_nei"]
                    ),

                "structural_N_MB":
                    int(
                        sm["MB"]["N_MB"]
                    ),

                "structural_N_tail":
                    int(
                        sm["MB"]["N_tail"]
                    ),
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

            "outputs": {
                "Y_label.txt":
                    sha256(y_path),

                "cluster_centers.npy":
                    sha256(
                        centers_path
                    ),

                "cell_ids.txt":
                    sha256(
                        cells_path
                    ),
            },
        }

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            )
            + "\n"
        )

        # Independent reread before promotion.
        y2 = np.loadtxt(
            y_path,
            dtype=np.int64,
        )

        if not np.array_equal(
            y2,
            labels,
        ):
            raise RuntimeError(
                f"{sid}: serialized labels differ"
            )

        if y2.shape != (
            len(obs),
        ):
            raise RuntimeError(
                f"{sid}: Y_label shape incorrect"
            )

        if (
            cells_path
            .read_text()
            .splitlines()
            != list(
                map(str, obs)
            )
        ):
            raise RuntimeError(
                f"{sid}: serialized cell IDs differ"
            )

        os.rename(
            tmp,
            out,
        )

    except Exception:

        shutil.rmtree(
            tmp,
            ignore_errors=True,
        )

        raise

    print()
    print("=" * 78)
    print(
        f"{sid} FIXED-K3 LABEL GENERATION: PASS"
    )
    print("=" * 78)
    print("N =", len(labels))
    print("G = 280")
    print("K = 3")
    print("target_sum =", target_sum)
    print(
        "zero_total_cells =",
        int(zero.sum()),
    )
    print(
        "cluster_counts =",
        counts.tolist(),
    )
    print(
        "zero_total_cluster_counts =",
        zero_cluster_counts.tolist(),
    )
    print(
        "inertia =",
        float(km.inertia_),
    )
    print(
        "n_iter =",
        int(km.n_iter_),
    )
    print(
        "Y_label SHA256 =",
        sha256(
            out
            / "Y_label.txt"
        ),
    )
    print(
        "manifest SHA256 =",
        sha256(
            out
            / "label_manifest.json"
        ),
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
        "--struct",
    )

    ap.add_argument(
        "--out",
    )

    args = ap.parse_args()

    if args.verify_reference:

        if (
            args.sample
            or args.struct
            or args.out
        ):
            raise RuntimeError(
                "--verify-reference must be used alone"
            )

        verify_reference_exact()
        return

    if not (
        args.sample
        and args.struct
        and args.out
    ):
        raise RuntimeError(
            "generation requires --sample --struct --out"
        )

    generate(
        sid=args.sample,
        struct_path=Path(
            args.struct
        ),
        out=Path(
            args.out
        ),
    )


if __name__ == "__main__":
    main()
