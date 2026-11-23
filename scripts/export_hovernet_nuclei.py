#!/usr/bin/env python3

from pathlib import Path
import argparse
from collections import Counter

import pandas as pd

from tiatoolbox.annotation.storage import SQLiteStore


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-dir",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    a = p.parse_args()

    root = Path(a.input_dir)

    dbs = sorted(root.rglob("*.db"))

    if len(dbs) != 1:
        raise RuntimeError(
            f"Expected exactly one .db, found {len(dbs)}:\n"
            + "\n".join(map(str, dbs))
        )

    db = dbs[0]

    print("AnnotationStore:", db)

    store = SQLiteStore(
        str(db)
    )

    print("Number of predicted nuclei:", len(store))

    rows = []

    first_props = None

    for key, ann in store.items():

        props = dict(
            ann.properties
            or {}
        )

        if first_props is None:
            first_props = props

        geom = ann.geometry

        c = geom.centroid

        rows.append(
            {
                "nucleus_id": str(key),
                "x_px": float(c.x),
                "y_px": float(c.y),
                "histo_type": props.get(
                    "type"
                ),
                "histo_prob": props.get(
                    "prob"
                ),
                "geometry_wkb": bytes(
                    geom.wkb
                ),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No HoVer-Net nuclei found"
        )

    print("\nFIRST ANNOTATION PROPERTIES:")
    print(first_props)

    print("\nCLASS COUNTS:")
    print(
        df["histo_type"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nPROBABILITY SUMMARY:")
    print(
        pd.to_numeric(
            df["histo_prob"],
            errors="coerce",
        )
        .describe()
        .to_string()
    )

    out = Path(a.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        out,
        index=False,
    )

    print("\nSAVED:", out)
    print(
        "\nHOVERNET NUCLEUS EXPORT: PASS"
    )


if __name__ == "__main__":
    main()
