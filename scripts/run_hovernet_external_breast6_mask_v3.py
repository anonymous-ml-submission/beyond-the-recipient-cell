#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import torch
import tiatoolbox
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader

from tiatoolbox.models.engine.multi_task_segmentor import (
    MultiTaskSegmentor,
)


CLASSES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Non-Neoplastic Epithelial",
}


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--prepared-root",
        required=True,
    )

    p.add_argument(
        "--sample",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    a = p.parse_args()

    root = Path(a.prepared_root)
    sdir = root / "samples" / a.sample

    if not sdir.exists():
        raise RuntimeError(
            f"Missing prepared sample: {sdir}"
        )

    image_path = Path(
        (sdir / "image_path.txt")
        .read_text()
        .strip()
    )

    mpp = float(
        (sdir / "pixel_size_um.txt")
        .read_text()
        .strip()
    )

    if not image_path.exists():
        raise RuntimeError(
            f"Missing WSI: {image_path}"
        )

    if not (0.20 <= mpp <= 0.40):
        raise RuntimeError(
            f"Unexpected native MPP: {mpp}"
        )

    out = Path(a.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("INDEPENDENT H&E NUCLEAR ANNOTATION")
    print("=" * 90)
    print("sample:", a.sample)
    print("WSI:", image_path)
    print("native MPP:", mpp)
    print("model: hovernet_fast-pannuke")
    print("tiatoolbox:", tiatoolbox.__version__)
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("classes:", CLASSES)

    assert torch.cuda.is_available()

    engine = MultiTaskSegmentor(
        model="hovernet_fast-pannuke",
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        device="cuda",
        verbose=True,
    )

    result = engine.run(
        images=[OpenSlideWSIReader(str(image_path), mpp=(mpp, mpp))],
        patch_mode=False,
        save_dir=out,
        output_type="annotationstore",
        overwrite=True,
        masks=[str(sdir / "tissue_mask.png")],
        auto_get_mask=False,

        # Do not trust TIFF metadata implicitly.
        # Use the already audited HEST MPP.
        wsireader_kwargs={
            "mpp": mpp,
        },

        # Prevent the WSI engine consuming the host.
        memory_threshold=50,
    )

    print("\nRESULT:")
    print(result)

    dbs = sorted(
        out.rglob("*.db")
    )

    if not dbs:
        raise RuntimeError(
            "HoVer-Net completed but produced no "
            "AnnotationStore .db file"
        )

    prov = {
        "sample_id": a.sample,
        "source_modality": "H&E_only",
        "reader_backend": "OpenSlideWSIReader",
        "xenium_expression_used": False,
        "model": "hovernet_fast-pannuke",
        "training_dataset": "PanNuke",
        "tiatoolbox_version": tiatoolbox.__version__,
        "torch_version": torch.__version__,
        "native_mpp_um_per_px": mpp,
        "model_input_mpp_um_per_px": 0.25,
        "classes": CLASSES,
        "wsi": str(image_path),
        "tissue_mask": str(sdir / "tissue_mask.png"),
        "mask_method": "H&E saturation delta>=18; 7x7 closing; fill holes; min region 300px",
        "annotationstores": [
            str(x.resolve())
            for x in dbs
        ],
    }

    (
        out / "provenance.json"
    ).write_text(
        json.dumps(
            prov,
            indent=2,
        )
    )

    print("\nANNOTATION STORES:")
    for x in dbs:
        print(x)

    print("\nH&E-ONLY HOVERNET INFERENCE: PASS")


if __name__ == "__main__":
    main()
