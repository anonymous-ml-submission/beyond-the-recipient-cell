#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import deContamination as dc


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--output",
        required=True,
    )

    args = p.parse_args()

    project = Path.cwd().resolve()

    data = (
        project
        / "third_party"
        / "DeLeakage"
        / "data"
    )

    out = Path(
        args.output
    ).resolve()

    if out.exists():
        raise RuntimeError(
            f"Refusing overwrite: {out}"
        )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Exact bundled-example contract established from released files.
    G = 100
    K = 5
    N_nei = 21
    N = 3234

    # MB_1..MB_14 + RMB is the exact released runtime partition.
    N_MB = 14
    N_tail = 522

    # Smallest complete C++ output block.
    n_record = 500

    seed = 123

    output_list = str(out) + "/"

    data_name = str(
        data / "Y_obs.txt"
    )

    nei_name = str(
        data / "nei_list.txt"
    )

    dist_name = str(
        data / "nei_dist.txt"
    )

    label_name = str(
        data / "Y_labels.txt"
    )

    cell_size_name = str(
        data / "cell_size.txt"
    )

    MB_dir = str(
        data / "MB"
    ) + "/"

    print("=" * 72)
    print("DELEAKAGE BUNDLED 500-ITERATION SMOKE")
    print("=" * 72)

    print("python      =", sys.executable)
    print("G           =", G)
    print("K           =", K)
    print("N_nei       =", N_nei)
    print("N           =", N)
    print("N_MB        =", N_MB)
    print("N_tail      =", N_tail)
    print("n_record    =", n_record)
    print("seed        =", seed)
    print("output      =", output_list)

    ret = dc.run(
        G,
        K,
        N_nei,
        N,
        N_MB,
        N_tail,
        n_record,
        seed,
        output_list,
        data_name,
        nei_name,
        dist_name,
        label_name,
        cell_size_name,
        MB_dir,
    )

    print()
    print("return =", ret)
    print("DELEAKAGE BUNDLED RUNTIME CALL: PASS")


if __name__ == "__main__":
    main()
