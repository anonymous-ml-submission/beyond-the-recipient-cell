#!/usr/bin/env Rscript
# Real Xenium -> RCTD -> SPLIT purification wrapper.
# The default reference URLs are the exact Janesick/10x reference used by the
# official SPLIT Xenium vignette (Bilous et al., Nature Methods 2026).
suppressPackageStartupMessages({
  library(optparse); library(Matrix); library(Seurat); library(spacexr); library(SPLIT)
  library(readxl); library(httr); library(dplyr)
})
opts <- list(
  make_option("--xenium-dir", type="character"),
  make_option("--out", type="character"),
  make_option("--cores", type="integer", default=4),
  make_option("--min-umi", type="integer", default=10)
)
a <- parse_args(OptionParser(option_list=opts))
if (is.null(a$`xenium-dir`) || is.null(a$out)) stop("--xenium-dir and --out are required")
dir.create(a$out, recursive=TRUE, showWarnings=FALSE)
xd <- a$`xenium-dir`
# Find old/new standard Xenium files without guessing an alternative modality.
h5 <- file.path(xd,"cell_feature_matrix.h5")
if (!file.exists(h5)) {
  cc <- list.files(xd, pattern="cell_feature_matrix.*\\.h5$", recursive=TRUE, full.names=TRUE)
  if(length(cc)!=1) stop("Cannot uniquely locate cell_feature_matrix.h5")
  h5 <- cc[[1]]
}
cf <- list.files(xd, pattern="^cells\\.(csv\\.gz|parquet)$", recursive=TRUE, full.names=TRUE)
if(length(cf)<1) stop("Cannot locate cells.csv.gz/cells.parquet")
# Prefer CSV to avoid arrow dependency.
cf_csv <- cf[grepl("csv",cf)]; if(length(cf_csv)) cf <- cf_csv[[1]] else stop("Parquet-only cells file: convert to CSV or install an Arrow-aware wrapper")
counts <- Read10X_h5(h5)
if (is.list(counts)) {
  if("Gene Expression" %in% names(counts)) counts <- counts[["Gene Expression"]] else counts <- counts[[1]]
}
cells <- read.csv(cf, stringsAsFactors=FALSE)
if(!all(c("cell_id","x_centroid","y_centroid") %in% colnames(cells))) stop("Xenium cells file missing required columns")
rownames(cells) <- as.character(cells$cell_id)
common_cells <- intersect(colnames(counts), rownames(cells))
if(length(common_cells)<100) stop(paste("Too few cell IDs match H5 and cells file:",length(common_cells)))
counts <- counts[,common_cells,drop=FALSE]; cells <- cells[common_cells,,drop=FALSE]
# Download exact matched Chromium reference and annotation used in SPLIT's official vignette.
meta_url <- "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-43458-x/MediaObjects/41467_2023_43458_MOESM4_ESM.xlsx"
ref_url <- "https://cf.10xgenomics.com/samples/cell-exp/7.0.1/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5"
meta_file <- tempfile(fileext=".xlsx"); resp <- GET(meta_url, write_disk(meta_file, overwrite=TRUE)); stop_for_status(resp)
chrom_metadata <- read_excel(meta_file, sheet=1) %>% as.data.frame(); rownames(chrom_metadata)<-chrom_metadata$Barcode
ref_file <- tempfile(fileext=".h5"); resp <- GET(ref_url, write_disk(ref_file, overwrite=TRUE)); stop_for_status(resp)
chrom_counts <- Read10X_h5(ref_file); if(is.list(chrom_counts)) chrom_counts <- if("Gene Expression" %in% names(chrom_counts)) chrom_counts[["Gene Expression"]] else chrom_counts[[1]]
chrom <- CreateSeuratObject(counts=chrom_counts, assay="RNA", meta.data=chrom_metadata)
chrom$QC <- !is.na(chrom$Annotation); chrom$is_hybrid <- grepl("Hybrid",chrom$Annotation,ignore.case=TRUE)
chrom <- subset(chrom, subset=QC==TRUE & is_hybrid==FALSE)
cell_type_to_class <- c(
 "B_Cells"="B cell","CD4+_T_Cells"="T cell","CD8+_T_Cells"="T cell",
 "IRF7+_DCs"="Myeloid","LAMP3+_DCs"="Myeloid","Macrophages_1"="Myeloid","Macrophages_2"="Myeloid","Mast_Cells"="Myeloid",
 "DCIS 1"="Epithelial","DCIS 2"="Epithelial","Invasive_Tumor"="Epithelial","Prolif_Invasive_Tumor"="Epithelial",
 "Myoepi_ACTA2+"="Myoepithelial","Myoepi_KRT15+"="Myoepithelial","Stromal"="Stromal","Perivascular-Like"="Stromal","Endothelial"="Endothelial")
class_df <- data.frame(class=cell_type_to_class)
common_genes <- intersect(rownames(counts), rownames(chrom))
# Remove Xenium controls by retaining genes shared with the scRNA reference.
xcounts <- counts[common_genes,,drop=FALSE]
ref_labels <- as.factor(chrom$Annotation)
ref.obj <- Reference(GetAssayData(chrom,"RNA","counts")[common_genes,],cell_types=ref_labels,min_UMI=10,require_int=TRUE)
coords <- data.frame(x=cells$x_centroid,y=cells$y_centroid,row.names=common_cells)
test.obj <- SpatialRNA(coords=coords,counts=xcounts,require_int=TRUE)
rctd <- create.RCTD(test.obj,ref.obj,UMI_min=a$`min-umi`,counts_MIN=a$`min-umi`,UMI_min_sigma=100,max_cores=a$cores,CELL_MIN_INSTANCE=25,class_df=class_df)
rctd <- run.RCTD(rctd,doublet_mode="doublet")
rctd <- SPLIT::run_post_process_RCTD(rctd)
res <- SPLIT::purify(counts=xcounts,rctd=rctd,DO_purify_singlets=TRUE,DO_remove_residual_contamination=TRUE)
# Exact cell order contract.
clean <- res$purified_counts
shared <- intersect(colnames(xcounts),colnames(clean)); xcounts<-xcounts[,shared,drop=FALSE]; clean<-clean[,shared,drop=FALSE]; cells<-cells[shared,,drop=FALSE]
write_bundle <- function(mat,prefix){
  tf<-file.path(a$out,paste0(prefix,".mtx")); writeMM(mat,tf); system2("gzip",c("-f",tf))
  gz1<-gzfile(file.path(a$out,paste0(prefix,".genes.tsv.gz")),"wt"); writeLines(rownames(mat),gz1); close(gz1)
  gz2<-gzfile(file.path(a$out,paste0(prefix,".cells.tsv.gz")),"wt"); writeLines(colnames(mat),gz2); close(gz2)
}
write_bundle(xcounts,"raw_counts"); write_bundle(clean,"clean_counts")
write.csv(data.frame(cell_id=shared,x_centroid=cells$x_centroid,y_centroid=cells$y_centroid),file.path(a$out,"coords.csv"),row.names=FALSE)
meta <- res$cell_meta; meta$cell_id <- rownames(meta); meta <- meta[match(shared,meta$cell_id),,drop=FALSE]
write.csv(meta,file.path(a$out,"cell_meta.csv"),row.names=FALSE)
saveRDS(rctd,file.path(a$out,"rctd.rds")); saveRDS(res,file.path(a$out,"split_result.rds"))
message(sprintf("DONE: %d cells x %d genes",ncol(clean),nrow(clean)))
