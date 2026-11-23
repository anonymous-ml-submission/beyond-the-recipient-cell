#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
from shapely.ops import unary_union


def _polygon_parts(g):
    """Yield all positive-area Polygon components."""
    if g is None or g.is_empty:
        return

    if g.geom_type == "Polygon":
        if g.area > 0:
            yield g
        return

    if g.geom_type in {
        "MultiPolygon",
        "GeometryCollection",
    }:
        for part in g.geoms:
            yield from _polygon_parts(part)


def _polygonal_only(g):
    """Remove line/point artifacts while preserving all 2-D area."""
    parts = list(_polygon_parts(g))

    if not parts:
        return None

    out = unary_union(parts)

    if (
        out.is_empty
        or out.area <= 0
        or out.geom_type not in {
            "Polygon",
            "MultiPolygon",
        }
    ):
        return None

    return out


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--sample-dir",
        required=True,
    )

    p.add_argument(
        "--nuclei",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    a = p.parse_args()

    sdir = Path(a.sample_dir)

    cells = pd.read_parquet(
        sdir / "cells.parquet",
        columns=[
            "cell_id",
            "x_px",
            "y_px",
        ],
    )

    verts = pd.read_parquet(
        sdir / "cell_polygons.parquet"
    )

    nuclei = pd.read_parquet(
        a.nuclei
    )

    cells["cell_id"] = (
        cells["cell_id"]
        .astype(str)
    )

    verts["cell_id"] = (
        verts["cell_id"]
        .astype(str)
    )

    if "vertex_order" in verts.columns:
        verts = verts.sort_values(
            [
                "cell_id",
                "vertex_order",
            ]
        )

    nuclei_points = [
        Point(float(x), float(y))
        for x, y in zip(
            nuclei.x_px,
            nuclei.y_px,
        )
    ]

    tree = STRtree(
        nuclei_points
    )

    poly_ids = []
    polygons = []

    print(
        "Building Xenium cell polygons...",
        flush=True,
    )

    for cid, g in verts.groupby(
        "cell_id",
        sort=False,
    ):

        xy = g[
            ["x_px", "y_px"]
        ].to_numpy(
            dtype=float
        )

        if len(xy) < 3:
            continue

        poly = Polygon(xy)

        if not poly.is_valid:
            poly = make_valid(poly)

        # make_valid() may legitimately return a GeometryCollection
        # containing polygonal area plus zero-area line/point artifacts.
        # Keep every 2-D polygonal component; discard only lower-dimensional
        # artifacts. MultiPolygon remains valid for point containment.
        poly = _polygonal_only(poly)

        if poly is None:
            continue

        poly_ids.append(
            str(cid)
        )

        polygons.append(
            poly
        )

    print(
        "valid polygons:",
        len(polygons),
    )

    if len(polygons) != len(cells):
        raise RuntimeError(
            f"Expected one valid polygon per cell: "
            f"cells={len(cells)}, polygons={len(polygons)}"
        )

    print(
        "Point-in-polygon query...",
        flush=True,
    )

    pairs = tree.query(
        polygons,
        predicate="intersects",
    )

    # pairs[0] -> Xenium polygon index
    # pairs[1] -> HoVer nucleus index

    poly_idx = pairs[0]
    nuc_idx = pairs[1]

    n_poly_hits = np.bincount(
        poly_idx,
        minlength=len(polygons),
    )

    n_nuc_hits = np.bincount(
        nuc_idx,
        minlength=len(nuclei),
    )

    candidate_by_poly = {}

    for pi, ni in zip(
        poly_idx,
        nuc_idx,
    ):
        candidate_by_poly.setdefault(
            int(pi),
            [],
        ).append(
            int(ni)
        )

    result = pd.DataFrame(
        {
            "cell_id": cells.cell_id,
        }
    )

    result["cell_type"] = (
        "HISTO_Unassigned"
    )

    result["annotation_status"] = (
        "no_histo_nucleus"
    )

    result["hovernet_nucleus_id"] = pd.NA
    result["hovernet_type"] = pd.NA
    result["hovernet_prob"] = np.nan

    cell_row = {
        cid: i
        for i, cid
        in enumerate(
            result.cell_id
        )
    }

    for pi, cid in enumerate(
        poly_ids
    ):

        ri = cell_row[cid]

        hits = candidate_by_poly.get(
            pi,
            [],
        )

        if len(hits) == 0:
            continue

        if len(hits) > 1:
            result.at[
                ri,
                "annotation_status",
            ] = "multiple_histo_nuclei"

            continue

        ni = hits[0]

        # Enforce one-to-one mapping.
        if n_nuc_hits[ni] != 1:
            result.at[
                ri,
                "annotation_status",
            ] = "histo_nucleus_overlaps_multiple_cells"

            continue

        typ = nuclei.iloc[
            ni
        ]["histo_type"]

        prob = nuclei.iloc[
            ni
        ]["histo_prob"]

        nid = nuclei.iloc[
            ni
        ]["nucleus_id"]

        if pd.isna(typ):
            result.at[
                ri,
                "annotation_status",
            ] = "histo_type_missing"

            continue

        typ = str(typ)

        result.at[
            ri,
            "hovernet_nucleus_id",
        ] = str(nid)

        result.at[
            ri,
            "hovernet_type",
        ] = typ

        result.at[
            ri,
            "hovernet_prob",
        ] = prob

        result.at[
            ri,
            "annotation_status",
        ] = "unique_nucleus_inside_cell"

        result.at[
            ri,
            "cell_type",
        ] = (
            "HISTO_"
            + typ.replace(
                " ",
                "_",
            )
        )

    print(
        "\nANNOTATION STATUS:"
    )

    print(
        result[
            "annotation_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nCELL TYPE COUNTS:"
    )

    print(
        result[
            "cell_type"
        ]
        .value_counts()
        .to_string()
    )

    exact = (
        result[
            "annotation_status"
        ]
        == "unique_nucleus_inside_cell"
    ).mean()

    print(
        f"\nExact unique containment coverage: "
        f"{exact:.4%}"
    )

    out = Path(a.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        out,
        index=False,
    )

    print("\nSAVED:", out)


if __name__ == "__main__":
    main()
