#!/usr/bin/env python3

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

from shapely import wkb


BIO_CLASSES = [
    "Neoplastic",
    "Non-Neoplastic Epithelial",
    "Inflammatory",
    "Connective",
]


def draw_geometry(ax, geom, x0, y0):
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        xy = np.asarray(geom.exterior.coords)
        ax.plot(
            xy[:, 0] - x0,
            xy[:, 1] - y0,
            linewidth=1.5,
        )
        return

    if geom.geom_type in {
        "MultiPolygon",
        "GeometryCollection",
    }:
        for g in geom.geoms:
            draw_geometry(ax, g, x0, y0)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--sample-dir", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--nuclei", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, default=8)
    p.add_argument("--context-um", type=float, default=70.0)
    p.add_argument("--seed", type=int, default=2026)

    a = p.parse_args()

    sdir = Path(a.sample_dir)

    image_path = Path(
        (sdir / "image_path.txt").read_text().strip()
    )

    mpp = float(
        (sdir / "pixel_size_um.txt").read_text().strip()
    )

    cells = pd.read_parquet(
        sdir / "cells.parquet",
        columns=["cell_id", "x_px", "y_px"],
    )

    verts = pd.read_parquet(
        sdir / "cell_polygons.parquet"
    )

    ann = pd.read_parquet(a.annotations)

    nuclei = pd.read_parquet(
        a.nuclei,
        columns=[
            "nucleus_id",
            "geometry_wkb",
        ],
    )

    cells["cell_id"] = cells["cell_id"].astype(str)
    verts["cell_id"] = verts["cell_id"].astype(str)
    ann["cell_id"] = ann["cell_id"].astype(str)

    nuclei["nucleus_id"] = (
        nuclei["nucleus_id"].astype(str)
    )

    ann["hovernet_nucleus_id"] = (
        ann["hovernet_nucleus_id"]
        .astype("string")
    )

    exact = ann[
        ann["annotation_status"]
        .eq("unique_nucleus_inside_cell")
        &
        ann["hovernet_type"]
        .isin(BIO_CLASSES)
    ].copy()

    exact = exact.merge(
        cells,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )

    exact = exact.merge(
        nuclei,
        left_on="hovernet_nucleus_id",
        right_on="nucleus_id",
        how="inner",
        validate="one_to_one",
    )

    print("Exact biological matches:", len(exact))

    chosen = []

    for i, typ in enumerate(BIO_CLASSES):
        g = exact[
            exact["hovernet_type"].eq(typ)
        ].copy()

        if len(g) < a.n_per_class:
            raise RuntimeError(
                f"{typ}: only {len(g)} matches"
            )

        q = g.sample(
            n=a.n_per_class,
            random_state=a.seed + i,
            replace=False,
        ).copy()

        q["qa_class"] = typ
        chosen.append(q)

    chosen = pd.concat(
        chosen,
        ignore_index=True,
    )

    chosen.to_csv(
        Path(a.out).with_suffix(".csv"),
        index=False,
    )

    if "vertex_order" in verts.columns:
        verts = verts.sort_values(
            ["cell_id", "vertex_order"]
        )

    selected_ids = set(chosen["cell_id"])

    vp = verts[
        verts["cell_id"].isin(selected_ids)
    ].copy()

    polygon_lookup = {
        cid: g[["x_px", "y_px"]].to_numpy(float)
        for cid, g in vp.groupby(
            "cell_id",
            sort=False,
        )
    }

    patch_px = int(
        round(a.context_um / mpp)
    )

    if patch_px % 2:
        patch_px += 1

    slide = openslide.OpenSlide(
        str(image_path)
    )

    fig, axes = plt.subplots(
        len(BIO_CLASSES),
        a.n_per_class,
        figsize=(
            a.n_per_class * 2.4,
            len(BIO_CLASSES) * 2.5,
        ),
        squeeze=False,
    )

    for ri, typ in enumerate(BIO_CLASSES):

        subset = chosen[
            chosen["qa_class"].eq(typ)
        ].reset_index(drop=True)

        for ci, row in subset.iterrows():

            ax = axes[ri, ci]

            cx = float(row["x_px"])
            cy = float(row["y_px"])

            x0 = int(
                round(cx - patch_px / 2)
            )
            y0 = int(
                round(cy - patch_px / 2)
            )

            patch = (
                slide.read_region(
                    (x0, y0),
                    0,
                    (patch_px, patch_px),
                )
                .convert("RGB")
            )

            ax.imshow(patch)

            cell_xy = polygon_lookup[
                row["cell_id"]
            ]

            ax.plot(
                cell_xy[:, 0] - x0,
                cell_xy[:, 1] - y0,
                linewidth=1.8,
            )

            geom = wkb.loads(
                bytes(
                    row["geometry_wkb"]
                )
            )

            draw_geometry(
                ax,
                geom,
                x0,
                y0,
            )

            ax.plot(
                cx - x0,
                cy - y0,
                marker="+",
                markersize=6,
            )

            prob = float(
                row["hovernet_prob"]
            )

            if ri == 0:
                ax.set_title(
                    f"p={prob:.3f}",
                    fontsize=8,
                )
            else:
                ax.set_title(
                    f"p={prob:.3f}",
                    fontsize=8,
                )

            if ci == 0:
                ax.set_ylabel(
                    typ,
                    fontsize=9,
                )

            ax.set_xticks([])
            ax.set_yticks([])

    slide.close()

    fig.suptitle(
        "TENX191: H&E + Xenium cell polygon + independent HoVer-Net nucleus",
        fontsize=13,
    )

    fig.tight_layout()

    out = Path(a.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("SAVED:", out)
    print(
        "SAVED:",
        out.with_suffix(".csv"),
    )


if __name__ == "__main__":
    main()
