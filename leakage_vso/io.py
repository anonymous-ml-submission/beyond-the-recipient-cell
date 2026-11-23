from __future__ import annotations
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet" or p.name.endswith(".parquet.gz"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def read_10x_h5_as_adata(path: str | Path) -> ad.AnnData:
    import scanpy as sc
    x = sc.read_10x_h5(str(path), gex_only=True)
    x.var_names_make_unique()
    return x


def read_mtx_bundle(folder: str | Path, prefix: str) -> ad.AnnData:
    folder = Path(folder)
    mtx = folder / f"{prefix}.mtx.gz"
    genes = folder / f"{prefix}.genes.tsv.gz"
    cells = folder / f"{prefix}.cells.tsv.gz"
    if not (mtx.exists() and genes.exists() and cells.exists()):
        raise FileNotFoundError(f"Missing MatrixMarket bundle for prefix={prefix} in {folder}")
    with gzip.open(mtx, "rb") as f:
        X_gxc = scipy.io.mmread(f).tocsr()
    gene_names = pd.read_csv(genes, sep="\t", header=None)[0].astype(str).to_numpy()
    cell_names = pd.read_csv(cells, sep="\t", header=None)[0].astype(str).to_numpy()
    if X_gxc.shape != (len(gene_names), len(cell_names)):
        raise ValueError("MatrixMarket dimensions do not match genes/cells files")
    return ad.AnnData(X=X_gxc.T.tocsr(), obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))


def write_h5ad_safe(adata: ad.AnnData, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")


def intersect_genes(a: ad.AnnData, b: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    genes = a.var_names.intersection(b.var_names)
    if len(genes) == 0:
        raise ValueError("No shared genes")
    return a[:, genes].copy(), b[:, genes].copy()
