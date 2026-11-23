#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch

from torch.utils.data import DataLoader, Subset

from leakage_vso.dataset import CellImageDataset
from leakage_vso.models import ContextRegressor
from leakage_vso.losses import poisson_loss


ROOT = "data/prepared_primary2025"

TRAIN_IDS = [
    "TENX195",
    "TENX196",
    "TENX197",
    "TENX198",
    "TENX199",
]

VAL_IDS = [
    "TENX200",
    "TENX201",
    "TENX202",
]

EXPECTED_TRAIN_N = 1571084
EXPECTED_VAL_N = 730562

GENE_FILE = Path(
    "runs/frozen_primary2025/gene_axis_280.txt"
)

OUT = Path(
    "runs/model_preflight/"
    "primary2025_context_throughput_v1.json"
)

SEED = 2026
BATCH = 48
WORKERS = 0

# Enough batches to pass initial TIFF-open/cache overhead while keeping
# this a smoke test rather than a scientific training run.
N_SMOKE_CELLS = 1152

CONTEXTS = [
    24,
    64,
    128,
]


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(b)

    return h.hexdigest()


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable. Do not freeze a GPU training budget "
        "from a CPU throughput smoke."
    )


device = torch.device(
    "cuda:0"
)

torch.manual_seed(
    SEED
)

np.random.seed(
    SEED
)

genes = [
    x.strip()
    for x in GENE_FILE.read_text().splitlines()
    if x.strip()
]

assert len(genes) == 280
assert len(set(genes)) == 280


def deterministic_subset(
    n,
    take,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    take = min(
        int(take),
        int(n),
    )

    idx = rng.choice(
        n,
        size=take,
        replace=False,
    )

    return np.sort(
        idx
    ).astype(
        np.int64
    )


def check_visible_mask(
    tensor,
    context_um,
):

    # tensor shape C,H,W, already ImageNet normalized.
    assert tensor.shape == (
        3,
        256,
        256,
    )

    if context_um == 128:
        return

    visible_px = int(
        round(
            256
            * context_um
            / 128.0
        )
    )

    y0 = (
        256
        - visible_px
    ) // 2

    x0 = y0

    mask = torch.ones(
        (256, 256),
        dtype=torch.bool,
    )

    mask[
        y0:y0 + visible_px,
        x0:x0 + visible_px,
    ] = False

    outside = tensor[
        :,
        mask,
    ]

    if not torch.equal(
        outside,
        torch.zeros_like(
            outside
        ),
    ):
        raise RuntimeError(
            f"context={context_um}: "
            "pixels outside visible FOV are not exact zero "
            "after normalization"
        )


def make_dataset(
    ids,
    split,
    context_um,
):

    return CellImageDataset(
        prepared_root=ROOT,
        sample_ids=ids,
        genes=genes,
        target="raw",
        context_um=float(
            context_um
        ),
        center_um=24.0,
        model_mpp=0.5,
        canvas_um=128.0,
        split=split,
        augment=False,
        ring_shuffle=False,
    )


def make_loader(
    ds,
    indices,
):

    subset = Subset(
        ds,
        indices.tolist(),
    )

    return DataLoader(
        subset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )


results = []


for context in CONTEXTS:

    print()
    print("=" * 78)
    print(
        f"CONTEXT {context} um"
    )
    print("=" * 78)

    train_ds = make_dataset(
        TRAIN_IDS,
        "train",
        context,
    )

    val_ds = make_dataset(
        VAL_IDS,
        "val",
        context,
    )

    if len(train_ds) != EXPECTED_TRAIN_N:
        raise RuntimeError(
            f"train N changed: {len(train_ds)}"
        )

    if len(val_ds) != EXPECTED_VAL_N:
        raise RuntimeError(
            f"val N changed: {len(val_ds)}"
        )

    train_idx = deterministic_subset(
        len(train_ds),
        N_SMOKE_CELLS,
        SEED + context,
    )

    val_idx = deterministic_subset(
        len(val_ds),
        N_SMOKE_CELLS,
        SEED + 1000 + context,
    )

    # Direct geometry/target checks.
    item = train_ds[
        int(
            train_idx[0]
        )
    ]

    assert item[
        "outer"
    ].shape == (
        3,
        256,
        256,
    )

    assert item[
        "center"
    ].shape == (
        3,
        48,
        48,
    )

    assert item[
        "raw"
    ].shape == (
        280,
    )

    assert torch.isfinite(
        item[
            "outer"
        ]
    ).all()

    assert torch.isfinite(
        item[
            "raw"
        ]
    ).all()

    assert torch.all(
        item[
            "raw"
        ]
        >= 0
    )

    check_visible_mask(
        item[
            "outer"
        ],
        context,
    )

    train_loader = make_loader(
        train_ds,
        train_idx,
    )

    val_loader = make_loader(
        val_ds,
        val_idx,
    )

    model = ContextRegressor(
        n_genes=280,
        backbone="resnet18",

        # Throughput only. Avoid a network/download dependency.
        # Weight initialization does not affect tensor shapes or
        # forward/backward throughput materially.
        pretrained=False,

        freeze_backbone=False,
    ).to(
        device
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------------
    # Train-path smoke: exact current engine semantics, no AMP.
    # --------------------------------------------------------------

    model.train()

    train_times = []
    train_cells = 0

    for step, b in enumerate(
        train_loader
    ):

        x = b[
            "outer"
        ].to(
            device,
            non_blocking=True,
        )

        y = b[
            "raw"
        ].to(
            device,
            non_blocking=True,
        )

        assert x.shape[1:] == (
            3,
            256,
            256,
        )

        assert y.shape[1] == 280

        torch.cuda.synchronize()

        t0 = time.perf_counter()

        opt.zero_grad(
            set_to_none=True
        )

        pred = model(
            x
        )

        loss = poisson_loss(
            pred,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

        torch.cuda.synchronize()

        dt = (
            time.perf_counter()
            - t0
        )

        # First four measured batches are warmup.
        if step >= 4:
            train_times.append(
                dt
            )

            train_cells += int(
                x.shape[0]
            )

        if step >= 15:
            break

    if len(train_times) < 8:
        raise RuntimeError(
            "insufficient measured training batches"
        )

    train_sec = float(
        sum(
            train_times
        )
    )

    train_cps = float(
        train_cells
        /
        train_sec
    )

    # --------------------------------------------------------------
    # Validation forward path.
    # --------------------------------------------------------------

    model.eval()

    val_times = []
    val_cells = 0

    with torch.no_grad():

        for step, b in enumerate(
            val_loader
        ):

            x = b[
                "outer"
            ].to(
                device,
                non_blocking=True,
            )

            torch.cuda.synchronize()

            t0 = time.perf_counter()

            pred = model(
                x
            )

            # Force output realization.
            _ = float(
                pred[
                    0,
                    0
                ].item()
            )

            torch.cuda.synchronize()

            dt = (
                time.perf_counter()
                - t0
            )

            if step >= 4:
                val_times.append(
                    dt
                )

                val_cells += int(
                    x.shape[0]
                )

            if step >= 15:
                break

    if len(val_times) < 8:
        raise RuntimeError(
            "insufficient measured validation batches"
        )

    val_sec = float(
        sum(
            val_times
        )
    )

    val_cps = float(
        val_cells
        /
        val_sec
    )

    projected_full_train_sec = (
        EXPECTED_TRAIN_N
        /
        train_cps
    )

    projected_full_val_sec = (
        EXPECTED_VAL_N
        /
        val_cps
    )

    projected_epoch_sec = (
        projected_full_train_sec
        +
        projected_full_val_sec
    )

    peak_mb = float(
        torch.cuda.max_memory_allocated(
            device
        )
        /
        (1024 ** 2)
    )

    result = {
        "context_um":
            context,

        "canvas_um":
            128,

        "model_mpp":
            0.5,

        "tensor_shape":
            [
                3,
                256,
                256,
            ],

        "n_genes":
            280,

        "batch_size":
            BATCH,

        "workers":
            WORKERS,

        "measured_train_cells":
            train_cells,

        "measured_train_seconds":
            train_sec,

        "train_cells_per_second":
            train_cps,

        "measured_val_cells":
            val_cells,

        "measured_val_seconds":
            val_sec,

        "val_cells_per_second":
            val_cps,

        "projected_full_train_seconds":
            projected_full_train_sec,

        "projected_full_val_seconds":
            projected_full_val_sec,

        "projected_full_epoch_seconds":
            projected_epoch_sec,

        "projected_full_epoch_hours":
            projected_epoch_sec
            /
            3600.0,

        "cuda_peak_allocated_mb":
            peak_mb,
    }

    results.append(
        result
    )

    print(
        "geometry = PASS"
    )

    print(
        "train N =",
        len(
            train_ds
        ),
    )

    print(
        "val N =",
        len(
            val_ds
        ),
    )

    print(
        "train cells/sec =",
        train_cps,
    )

    print(
        "val cells/sec =",
        val_cps,
    )

    print(
        "projected full epoch hours =",
        projected_epoch_sec
        /
        3600.0,
    )

    print(
        "peak CUDA allocated MB =",
        peak_mb,
    )

    del model
    del opt
    del train_loader
    del val_loader
    del train_ds
    del val_ds

    torch.cuda.empty_cache()


payload = {
    "stage":
        "primary2025_pretraining_throughput_smoke",

    "status":
        "PASS",

    "scientific_training_performed":
        False,

    "test_sections_used":
        False,

    "seed":
        SEED,

    "gene_policy": {
        "policy":
            "all_frozen_280_genes",

        "path":
            str(
                GENE_FILE
            ),

        "sha256":
            sha(
                GENE_FILE
            ),
    },

    "splits": {
        "train":
            TRAIN_IDS,

        "val":
            VAL_IDS,

        "test_used":
            [],
    },

    "expected_dataset_sizes": {
        "train":
            EXPECTED_TRAIN_N,

        "val":
            EXPECTED_VAL_N,
    },

    "model_smoke": {
        "method":
            "ContextRegressor",

        "backbone":
            "resnet18",

        "pretrained_weights_loaded":
            False,

        "reason_pretrained_false":
            "throughput smoke must not depend on external weight download",

        "optimizer":
            "AdamW",

        "lr":
            1e-4,

        "weight_decay":
            1e-4,

        "loss":
            "poisson_loss",

        "amp":
            False,
    },

    "results":
        results,

    "provenance": {
        "script":
            str(
                Path(
                    __file__
                )
            ),

        "script_sha256":
            sha(
                Path(
                    __file__
                )
            ),

        "dataset_py_sha256":
            sha(
                Path(
                    "leakage_vso/dataset.py"
                )
            ),

        "train_engine_py_sha256":
            sha(
                Path(
                    "leakage_vso/train_engine.py"
                )
            ),

        "train_entrypoint_sha256":
            sha(
                Path(
                    "scripts/train.py"
                )
            ),

        "split_routing_freeze_sha256":
            sha(
                Path(
                    "runs/frozen_primary2025/"
                    "model_split_routing_v1.json"
                )
            ),

        "all11_clean_target_freeze_sha256":
            sha(
                Path(
                    "runs/frozen_primary2025/"
                    "deleakage_all11_clean_targets_v1.json"
                )
            ),
    },
}


OUT.write_text(
    json.dumps(
        payload,
        indent=2,
    )
    + "\n"
)

print()
print("=" * 78)
print("PRIMARY2025 CONTEXT THROUGHPUT SMOKE: PASS")
print("=" * 78)

for r in results:

    print(
        "context",
        r[
            "context_um"
        ],
        "train_cells/sec=",
        round(
            r[
                "train_cells_per_second"
            ],
            3,
        ),
        "val_cells/sec=",
        round(
            r[
                "val_cells_per_second"
            ],
            3,
        ),
        "full_epoch_hours=",
        round(
            r[
                "projected_full_epoch_hours"
            ],
            3,
        ),
    )

print(
    "output =",
    OUT,
)

print(
    "output SHA256 =",
    sha(
        OUT
    ),
)

print("NO SCIENTIFIC TRAINING PERFORMED")
print("=" * 78)
