#!/usr/bin/env python3
"""Prepare true cell-level HEST Xenium data.

Paper protocol:
- Cell universe/centres come from aligned Xenium segmentation.
- Never derive image centres from transcript positions.
- Reconstruct cell x gene counts from assigned Xenium molecules.
- Explicit gene/control + QV filtering.
- Stream transcript parquet; never load the full 60M+ row table into RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from leakage_vso.geometry import polygon_summary
from leakage_vso.splits import assign_donor_splits
from leakage_vso.wsi import TiffWSI


CONTROL_PREFIXES = (
    "negcontrolprobe_",
    "negcontrolcodeword_",
    "blank_",
    "unassignedcodeword_",
    "deprecatedcodeword_",
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(
            path,
            sep="\t" if path.suffix.lower() == ".tsv" else ",",
        )

    raise ValueError(path)


def _cell_id_col(df):
    return next(
        (
            c
            for c in ["cell_id", "cell", "cellid", "barcode"]
            if c in df.columns
        ),
        None,
    )


def _trusted_aligned_segmentation_path(path: Path) -> bool:
    """
    HEST publishes H&E-aligned Xenium segmentation with names such as:
      TENX191_xenium_cell_seg.parquet
    which do not necessarily contain 'he' or 'aligned'.
    """
    name = path.name.lower()

    return (
        "he" in name
        or "aligned" in name
        or "xenium_cell_seg" in name
        or "xenium_nucleus_seg" in name
    )


def _xy_cols(df, path: Path):
    pairs = [
        ("he_x", "he_y"),
        ("x_px", "y_px"),
        ("vertex_x_px", "vertex_y_px"),
        ("x_centroid_px", "y_centroid_px"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            return x, y

    if _trusted_aligned_segmentation_path(path):
        for x, y in [
            ("x", "y"),
            ("vertex_x", "vertex_y"),
            ("x_centroid", "y_centroid"),
        ]:
            if x in df.columns and y in df.columns:
                return x, y

    return None, None


def _geometry_from_wkb(df,path: Path):
    if "geometry" not in df.columns or not _trusted_aligned_segmentation_path(path):
        return None, None

    try:
        from shapely import wkb, wkt
    except Exception as e:
        raise RuntimeError(
            f"{path}: geometry column requires shapely>=2.0"
        ) from e

    work = df
    cid = _cell_id_col(work)

    # HEST Xenium GeoParquet stores the biological cell identifier in
    # the pandas/GeoParquet index.  pd.read_parquet() therefore exposes
    # only `geometry` as a normal column for these files.
    if cid is None:
        # A default RangeIndex contains no biological identifier.
        if isinstance(work.index, pd.RangeIndex):
            return None, None

        work = work.copy()
        cid = "__hest_cell_id_from_index__"
        work[cid] = [str(x) for x in work.index]

    # Some valid HEST GeoParquets contain a very small number of
    # duplicate index IDs (e.g. TENX195).  Do not duplicate cells
    # downstream.  Retain the largest valid polygon for each cell ID,
    # consistent with the existing MultiPolygon policy below.
    best_geom = {}

    for _, r in work.iterrows():
        cell_id = str(r[cid]).strip()

        if cell_id.casefold() in {
            "", "nan", "none", "na", "n/a"
        }:
            continue

        g = r["geometry"]

        if g is None:
            continue

        try:
            if isinstance(g, (bytes, bytearray, memoryview)):
                geom = wkb.loads(bytes(g))
            else:
                geom = wkt.loads(str(g))
        except Exception:
            continue

        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "MultiPolygon":
            geom = max(
                geom.geoms,
                key=lambda x: x.area
            )

        if geom.geom_type != "Polygon":
            continue

        old = best_geom.get(cell_id)

        if old is None or geom.area > old.area:
            best_geom[cell_id] = geom

    if not best_geom:
        return None, None

    cents = []
    polys = []

    for cell_id, geom in best_geom.items():
        cc = geom.centroid

        cents.append({
            "cell_id": cell_id,
            "x_px": float(cc.x),
            "y_px": float(cc.y),
            "polygon_area_px2": float(geom.area),
        })

        for vertex_order, (x, y) in enumerate(
            geom.exterior.coords
        ):
            polys.append({
                "cell_id": cell_id,
                "x_px": float(x),
                "y_px": float(y),
                "vertex_order": int(vertex_order),
            })

    return pd.DataFrame(cents), pd.DataFrame(polys)

def load_aligned_cell_geometry(seg_root: Path, sid: str):
    cands=[p for p in seg_root.rglob(f"*{sid}*") if p.is_file() and p.suffix.lower() in {".parquet",".csv",".tsv"}]
    # prefer H&E-aligned cell (not nucleus) files
    cands=sorted(cands,key=lambda p:("nucleus" in p.name.lower(), "cell" not in p.name.lower(), "he" not in p.name.lower() and "aligned" not in p.name.lower(),len(str(p))))
    best_cent=None; best_poly=None; used=None
    for path in cands:
        try: df=_read_table(path)
        except Exception: continue
        # Try trusted HEST GeoParquet/WKB geometry first.  These files
        # may store cell_id in the DataFrame index rather than a column.
        cents,polys=_geometry_from_wkb(df,path)
        if cents is not None and len(cents):
            if best_cent is None or len(cents)>len(best_cent):
                best_cent,best_poly,used=cents,polys,path
            continue

        # Ordinary tabular x/y fallback still requires an explicit
        # cell-id column.
        cid=_cell_id_col(df)
        if cid is None:
            continue

        x,y=_xy_cols(df,path)
        if x is None: continue
        q=df[[cid,x,y]+(["vertex_order"] if "vertex_order" in df.columns else [])].copy()
        q=q.rename(columns={cid:"cell_id",x:"x_px",y:"y_px"}); q["cell_id"]=q.cell_id.astype(str)
        q=q[np.isfinite(pd.to_numeric(q.x_px,errors="coerce")) & np.isfinite(pd.to_numeric(q.y_px,errors="coerce"))]
        q.x_px=pd.to_numeric(q.x_px); q.y_px=pd.to_numeric(q.y_px)
        sizes=q.groupby("cell_id").size()
        if len(sizes)==0: continue
        if sizes.median()>=3:
            if "vertex_order" not in q: q["vertex_order"]=q.groupby("cell_id").cumcount()
            summ=polygon_summary(q[["cell_id","x_px","y_px","vertex_order"]])
            if best_cent is None or len(summ)>len(best_cent): best_cent,best_poly,used=summ,q[["cell_id","x_px","y_px","vertex_order"]],path
        elif sizes.max()==1:
            summ=q.drop_duplicates("cell_id")[['cell_id','x_px','y_px']].copy(); summ['polygon_area_px2']=np.nan
            if best_cent is None or len(summ)>len(best_cent): best_cent,best_poly,used=summ,None,path
    if best_cent is None:
        raise RuntimeError(f"{sid}: could not find an H&E-aligned cell centroid/boundary table under {seg_root}. Refusing transcript-derived centroids.")
    if best_cent.cell_id.duplicated().any(): raise RuntimeError(f"{sid}: duplicate segmentation cell IDs in {used}")
    return best_cent.reset_index(drop=True), (best_poly.reset_index(drop=True) if best_poly is not None else None), used


def gene_mask(
    t: pd.DataFrame,
    gene_col: str,
):
    if "is_gene" in t.columns:

        v = t["is_gene"]

        if v.dtype == bool:
            return v.to_numpy(), "is_gene"

        return (
            v.astype(str)
            .str.lower()
            .isin(
                [
                    "1",
                    "true",
                    "t",
                    "yes",
                ]
            )
            .to_numpy(),
            "is_gene",
        )

    cat = next(
        (
            c
            for c in [
                "codeword_category",
                "feature_type",
                "feature_category",
            ]
            if c in t.columns
        ),
        None,
    )

    if cat is not None:

        s = t[cat].astype(str).str.lower()

        m = (
            s.str.contains("gene")
            & ~s.str.contains(
                "control|blank|negative|unassigned"
            )
        )

        if m.any():
            return m.to_numpy(), cat

    names = (
        t[gene_col]
        .astype(str)
        .str.lower()
    )

    m = ~names.str.startswith(
        CONTROL_PREFIXES
    )

    warnings.warn(
        "No is_gene/category field found; "
        "falling back to Xenium control-name prefix exclusion"
    )

    return (
        m.to_numpy(),
        "control_prefix",
    )


def release_memory():
    gc.collect()

    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def build_transcript_matrix_streaming(
    path: Path,
    seg_ids: pd.Index,
    min_qv: float | None,
    allow_missing_qv: bool,
    max_unmatched_frac: float,
    batch_rows: int = 500_000,
):
    """
    Build cell x gene counts without loading
    the complete transcript parquet into RAM.
    """

    pf = pq.ParquetFile(path)

    names = pf.schema_arrow.names
    n_input = pf.metadata.num_rows

    gene_col = next(
        (
            c
            for c in [
                "feature_name",
                "gene",
                "gene_name",
            ]
            if c in names
        ),
        None,
    )

    cell_col = next(
        (
            c
            for c in [
                "cell_id",
                "cell",
                "cellid",
            ]
            if c in names
        ),
        None,
    )

    if gene_col is None or cell_col is None:
        raise RuntimeError(
            f"{path}: transcript table lacks "
            f"gene/cell columns; columns={names}"
        )

    qcol = next(
        (
            c
            for c in [
                "qv",
                "qv_score",
                "quality_value",
            ]
            if c in names
        ),
        None,
    )

    if (
        min_qv is not None
        and qcol is None
        and not allow_missing_qv
    ):
        raise RuntimeError(
            f"{path}: --min-qv={min_qv} requested "
            "but no QV column exists"
        )

    extra = []

    if "is_gene" in names:
        extra.append("is_gene")
    else:
        cat = next(
            (
                c
                for c in [
                    "codeword_category",
                    "feature_type",
                    "feature_category",
                ]
                if c in names
            ),
            None,
        )

        if cat is not None:
            extra.append(cat)

    if qcol is not None:
        extra.append(qcol)

    read_cols = list(
        dict.fromkeys(
            [cell_col, gene_col] + extra
        )
    )

    seg_ids = pd.Index(
        seg_ids.astype(str),
        name="cell_id",
    )

    seg_set = set(seg_ids)

    def filtered_batch(batch):

        df = batch.to_pandas()

        cid = df[cell_col].astype(str)

        assigned = ~cid.str.lower().isin(
            [
                "0",
                "-1",
                "none",
                "nan",
                "unassigned",
                "background",
            ]
        )

        gm, mode = gene_mask(
            df,
            gene_col,
        )

        keep = (
            assigned.to_numpy()
            & gm
        )

        if (
            min_qv is not None
            and qcol is not None
        ):

            qv = (
                pd.to_numeric(
                    df[qcol],
                    errors="coerce",
                )
                .fillna(-np.inf)
                .to_numpy()
            )

            keep &= qv >= min_qv

        out = df.loc[
            keep,
            [
                cell_col,
                gene_col,
            ],
        ].copy()

        out[cell_col] = (
            out[cell_col]
            .astype(str)
        )

        out[gene_col] = (
            out[gene_col]
            .astype(str)
        )

        return out, mode

    total_batches = max(
        1,
        math.ceil(
            n_input / batch_rows
        ),
    )

    # --------------------------------------------------
    # PASS 1:
    # filtered genes + QC statistics
    # --------------------------------------------------

    gene_set = set()
    assigned_kept = 0
    unmatched = 0
    filter_mode = None

    print(
        f"  PASS 1/2: {path.name}"
        f" rows={n_input:,}"
        f" batches~{total_batches}",
        flush=True,
    )

    for bi, batch in enumerate(
        pf.iter_batches(
            batch_size=batch_rows,
            columns=read_cols,
            use_threads=False,
        ),
        start=1,
    ):

        b, mode = filtered_batch(
            batch
        )

        if filter_mode is None:
            filter_mode = mode

        elif mode != filter_mode:
            raise RuntimeError(
                f"{path}: gene-filter mode changed "
                f"across batches: "
                f"{filter_mode} -> {mode}"
            )

        assigned_kept += len(b)

        matched = (
            b[cell_col]
            .isin(seg_set)
        )

        unmatched += int(
            (~matched).sum()
        )

        if matched.any():

            gene_set.update(
                b.loc[
                    matched,
                    gene_col,
                ]
                .unique()
                .tolist()
            )

        if (
            bi == 1
            or bi % 10 == 0
        ):
            print(
                f"    pass1 {bi}/{total_batches}"
                f" kept={assigned_kept:,}"
                f" unmatched={unmatched:,}"
                f" genes={len(gene_set)}",
                flush=True,
            )

        del b, batch

    frac = (
        unmatched
        / max(
            1,
            assigned_kept,
        )
    )

    if frac > max_unmatched_frac:
        raise RuntimeError(
            f"{path}: {frac:.3%} of filtered assigned "
            "transcripts reference cells absent "
            "from segmentation"
        )

    if not gene_set:
        raise RuntimeError(
            f"{path}: no filtered gene transcripts "
            "match segmentation cells"
        )

    genes = pd.Index(
        sorted(gene_set),
        name="gene",
    )

    release_memory()

    # --------------------------------------------------
    # BOUNDED accumulator
    #
    # 400k x 280 x uint32 ~= 427 MiB
    # --------------------------------------------------

    counts = np.zeros(
        (
            len(seg_ids),
            len(genes),
        ),
        dtype=np.uint32,
    )

    cell_to_row = pd.Series(
        np.arange(
            len(seg_ids),
            dtype=np.int32,
        ),
        index=seg_ids,
    )

    gene_to_col = pd.Series(
        np.arange(
            len(genes),
            dtype=np.int32,
        ),
        index=genes,
    )

    flat_counts = counts.reshape(-1)

    n_genes = len(genes)

    # --------------------------------------------------
    # PASS 2:
    # accumulate counts
    # --------------------------------------------------

    print(
        f"  PASS 2/2: {path.name}",
        flush=True,
    )

    for bi, batch in enumerate(
        pf.iter_batches(
            batch_size=batch_rows,
            columns=read_cols,
            use_threads=False,
        ),
        start=1,
    ):

        b, _ = filtered_batch(
            batch
        )

        matched = (
            b[cell_col]
            .isin(seg_set)
        )

        b = b.loc[
            matched
        ]

        if len(b):

            rows = (
                b[cell_col]
                .map(cell_to_row)
                .to_numpy(
                    dtype=np.int64
                )
            )

            cols = (
                b[gene_col]
                .map(gene_to_col)
                .to_numpy(
                    dtype=np.int64
                )
            )

            flat = (
                rows * n_genes
                + cols
            )

            idx, cnt = np.unique(
                flat,
                return_counts=True,
            )

            flat_counts[idx] += (
                cnt.astype(
                    np.uint32
                )
            )

        if (
            bi == 1
            or bi % 10 == 0
        ):
            print(
                f"    pass2 {bi}/{total_batches}",
                flush=True,
            )

        del b, batch

    X = sp.csr_matrix(
        counts,
        dtype=np.float32,
    )

    del (
        counts,
        flat_counts,
        cell_to_row,
        gene_to_col,
    )

    release_memory()

    return (
        X,
        genes,
        n_input,
        assigned_kept,
        unmatched,
        frac,
        qcol,
        filter_mode,
    )


def pixel_size_from_metadata_or_wsi(
    root: Path,
    sid: str,
    wsi_path: Path,
):
    px = None

    mp = list(
        (root / "metadata")
        .glob(f"{sid}*.json")
    )

    if mp:

        md = json.loads(
            mp[0].read_text()
        )

        for k in [
            "pixel_size_um_estimated",
            "pixel_size_um",
            "pixel_size",
        ]:

            if (
                k in md
                and md[k] is not None
            ):

                try:
                    px = float(md[k])
                    break
                except Exception:
                    pass

    # Important:
    # avoid opening a multi-GB TIFF
    # when HEST metadata already gives MPP.
    if px is None:

        w = TiffWSI(
            wsi_path
        )

        try:
            px = w.pixel_size_um_x
        finally:
            try:
                w.close()
            except Exception:
                pass

            del w
            gc.collect()

    if (
        px is None
        or px <= 0
    ):
        raise RuntimeError(
            f"No valid H&E pixel size "
            f"for HEST sample {sid}"
        )

    return float(px)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--hest-root",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--selected",
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--donor-col",
        default=None,
    )

    p.add_argument(
        "--allow-slide-as-donor",
        action="store_true",
    )

    p.add_argument(
        "--min-qv",
        type=float,
        default=20.0,
    )

    p.add_argument(
        "--allow-missing-qv",
        action="store_true",
    )

    p.add_argument(
        "--allow-centroid-only",
        action="store_true",
        help=(
            "Engineering only; "
            "paper source-intervention "
            "protocol requires polygons"
        ),
    )

    p.add_argument(
        "--max-unmatched-assigned-frac",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--transcript-batch-rows",
        type=int,
        default=500_000,
    )

    a = p.parse_args()

    root = Path(
        a.hest_root
    )

    outroot = Path(
        a.out
    )

    meta_path = (
        Path(a.selected)
        if a.selected
        else root
        / "selected_xenium_samples.csv"
    )

    meta = pd.read_csv(
        meta_path
    )

    if "id" not in meta.columns:
        raise RuntimeError(
            f"Selected manifest lacks "
            f"id column: {meta_path}"
        )

    if (
        meta["id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Selected manifest contains "
            f"duplicate sample IDs: {meta_path}"
        )

    donor_col = (
        a.donor_col
        or next(
            (
                c
                for c in [
                    "patient",
                    "patient_id",
                    "donor_id",
                    "donor",
                ]
                if c in meta.columns
            ),
            None,
        )
    )

    if donor_col is None:

        if not a.allow_slide_as_donor:
            raise RuntimeError(
                "No donor/patient column found. "
                "Refusing slide-as-donor paper splits."
            )

        meta["donor_id"] = (
            meta.id.astype(str)
        )

        donor_col = "donor_id"

    donor_raw = meta[
        donor_col
    ]

    bad_donor = (
        donor_raw.isna()
        | donor_raw.astype(str)
        .str.strip()
        .eq("")
        | donor_raw.astype(str)
        .str.strip()
        .str.casefold()
        .isin(
            [
                "nan",
                "none",
                "na",
                "n/a",
                "unknown",
            ]
        )
    )

    if bad_donor.any():
        raise RuntimeError(
            "Selected manifest contains "
            "missing/invalid donor labels:\n"
            + meta.loc[
                bad_donor,
                [
                    "id",
                    donor_col,
                ],
            ].to_string(
                index=False
            )
        )

    # ------------------------------------------
    # Assign split from SMALL sample manifest.
    # Do not concatenate all cell tables in RAM.
    # ------------------------------------------

    sample_split = pd.DataFrame(
        {
            "sample_id": (
                meta["id"]
                .astype(str)
            ),
            "donor_id": (
                meta[donor_col]
                .astype(str)
                .str.strip()
            ),
        }
    )

    sample_split = (
        assign_donor_splits(
            sample_split,
            seed=a.seed,
        )
    )

    split_map = dict(
        zip(
            sample_split.sample_id,
            sample_split.split,
        )
    )

    print(
        "\nDONOR SPLIT TABLE"
    )

    print(
        sample_split
        .drop_duplicates()
        .sort_values(
            [
                "split",
                "donor_id",
                "sample_id",
            ]
        )
        .to_string(
            index=False
        )
    )

    sample_stats = []

    for _, r in meta.iterrows():

        sid = str(
            r.id
        )

        donor_id = str(
            r[donor_col]
        ).strip()

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"PREPARING {sid}"
            f" donor={donor_id}"
            f" split={split_map[sid]}"
        )

        print(
            "=" * 100,
            flush=True,
        )

        cent, poly, seg_path = (
            load_aligned_cell_geometry(
                root / "xenium_seg",
                sid,
            )
        )

        if (
            poly is None
            and not a.allow_centroid_only
        ):
            raise RuntimeError(
                f"{sid}: aligned cell centroids found "
                "but no cell polygons. "
                "Paper protocol requires polygons."
            )

        tcands = sorted(
            (
                root
                / "transcripts"
            )
            .glob(
                f"{sid}*.parquet"
            )
        )

        if not tcands:
            raise FileNotFoundError(
                f"No aligned HEST transcript "
                f"parquet for {sid}"
            )

        seg_ids = pd.Index(
            cent.cell_id.astype(str),
            name="cell_id",
        )

        seg_set = set(
            seg_ids
        )

        (
            X,
            genes,
            n0,
            assigned_kept,
            unmatched,
            frac,
            qcol,
            filter_mode,
        ) = build_transcript_matrix_streaming(
            tcands[0],
            seg_ids=seg_ids,
            min_qv=a.min_qv,
            allow_missing_qv=a.allow_missing_qv,
            max_unmatched_frac=(
                a.max_unmatched_assigned_frac
            ),
            batch_rows=(
                a.transcript_batch_rows
            ),
        )

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                index=seg_ids
            ),
            var=pd.DataFrame(
                index=genes
            ),
        )

        wsi_candidates = sorted(
            (
                root
                / "wsis"
            )
            .glob(
                f"{sid}*"
            )
        )

        if not wsi_candidates:
            raise FileNotFoundError(
                f"No HEST WSI for {sid}"
            )

        wsi_path = (
            wsi_candidates[0]
        )

        px = (
            pixel_size_from_metadata_or_wsi(
                root,
                sid,
                wsi_path,
            )
        )

        cent = (
            cent
            .set_index(
                cent.cell_id.astype(str)
            )
            .loc[seg_ids]
            .reset_index(
                drop=True
            )
        )

        cells = pd.DataFrame(
            {
                "cell_id": (
                    seg_ids.astype(str)
                ),
                "sample_id": sid,
                "donor_id": donor_id,
                "split": (
                    split_map[sid]
                ),
                "x_px": (
                    cent.x_px
                    .to_numpy(float)
                ),
                "y_px": (
                    cent.y_px
                    .to_numpy(float)
                ),
            }
        )

        cells["x_um"] = (
            cells.x_px * px
        )

        cells["y_um"] = (
            cells.y_px * px
        )

        if "polygon_area_px2" in cent:

            cells[
                "polygon_area_um2"
            ] = (
                cent.polygon_area_px2
                .to_numpy(float)
                * (px**2)
            )

        sdir = (
            outroot
            / "samples"
            / sid
        )

        sdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        adata.write_h5ad(
            sdir / "raw.h5ad",
            compression="gzip",
        )

        cells.to_parquet(
            sdir / "cells.parquet",
            index=False,
        )

        if poly is not None:

            poly = poly[
                poly.cell_id
                .astype(str)
                .isin(seg_set)
            ].copy()

            poly.cell_id = (
                poly.cell_id.astype(str)
            )

            poly.to_parquet(
                sdir
                / "cell_polygons.parquet",
                index=False,
            )

        (
            sdir / "image_path.txt"
        ).write_text(
            str(
                wsi_path.resolve()
            )
            + "\n"
        )

        (
            sdir
            / "pixel_size_um.txt"
        ).write_text(
            str(px)
            + "\n"
        )

        prov = {
            "sample_id": sid,
            "segmentation_source": str(
                seg_path
            ),
            "transcript_source": str(
                tcands[0]
            ),
            "n_transcripts_input": int(
                n0
            ),
            "n_transcripts_filtered_assigned_gene_qv": int(
                assigned_kept
            ),
            "n_unmatched_to_segmentation": int(
                unmatched
            ),
            "unmatched_fraction": float(
                frac
            ),
            "n_segmentation_cells": int(
                len(seg_ids)
            ),
            "n_genes_observed": int(
                len(genes)
            ),
            "min_qv": a.min_qv,
            "qv_column": qcol,
            "gene_filter_mode": filter_mode,
            "has_cell_polygons": (
                poly is not None
            ),
            "pixel_size_um": px,
            "transcript_loader": (
                "pyarrow_two_pass_streaming_"
                "dense_uint32_to_csr"
            ),
            "transcript_batch_rows": int(
                a.transcript_batch_rows
            ),
        }

        (
            sdir
            / "preparation_provenance.json"
        ).write_text(
            json.dumps(
                prov,
                indent=2,
            )
        )

        sample_stats.append(
            {
                "sample_id": sid,
                "donor_id": donor_id,
                "split": split_map[sid],
                "n_cells": len(cells),
                "n_genes": len(genes),
            }
        )

        # Free sample-scale objects
        # before processing next slide.
        del (
            adata,
            X,
            cells,
            cent,
            poly,
            seg_set,
            seg_ids,
            genes,
        )

        release_memory()

    stats = pd.DataFrame(
        sample_stats
    )

    stats.to_csv(
        outroot
        / "preparation_summary.csv",
        index=False,
    )

    print(
        "\nPREPARATION SUMMARY"
    )

    print(
        stats.to_string(
            index=False
        )
    )

    print(
        "\nCELL COUNTS BY SPLIT"
    )

    print(
        stats.groupby(
            "split"
        )["n_cells"]
        .sum()
        .to_string()
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()
