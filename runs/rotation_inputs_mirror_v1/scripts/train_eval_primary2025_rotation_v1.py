#!/usr/bin/env python3

from pathlib import Path
import argparse

import torch
import yaml

from train_primary2025_context_v1 import (
    RawContextDataset,
)

from leakage_vso.dataset import (
    _dense_rows,
)

from leakage_vso.train_engine import (
    train_model,
)

from leakage_vso.evaluate import (
    evaluate_checkpoint,
)


COUNTS = {
    "TENX191": 209467,
    "TENX192": 375529,
    "TENX193": 254480,

    "TENX195": 380739,
    "TENX196": 358519,

    "TENX197": 308531,
    "TENX198": 373150,
    "TENX199": 150145,

    "TENX200": 192898,
    "TENX201": 335709,
    "TENX202": 201955,
}


class RotationEvalDataset(
    RawContextDataset
):

    def __getitem__(
        self,
        idx,
    ):

        item = super().__getitem__(
            idx
        )

        si, li = self.rows[idx]

        s, gidx = self.samples[si]

        if s.clean is None:
            raise RuntimeError(
                f"{s.sample_id}: missing clean"
            )

        clean = _dense_rows(
            s.clean.X,
            [li],
            gidx,
        )[0]

        item["clean"] = (
            torch.from_numpy(
                clean
            )
        )

        return item


def close_dataset(ds):

    for w in ds._wsi.values():
        try:
            w.close()
        except Exception:
            pass


def expected(ids):

    return sum(
        COUNTS[x]
        for x in ids
    )


def load_cfg(path):

    cfg = yaml.safe_load(
        Path(path).read_text()
    )

    assert cfg["seed"] == 2026
    assert cfg["data"]["target"] == "raw"

    assert int(
        cfg["data"]["context_um"]
    ) in {
        24,
        128,
    }

    assert float(
        cfg["data"]["canvas_um"]
    ) == 128.0

    assert float(
        cfg["data"]["model_mpp"]
    ) == 0.5

    assert cfg["model"]["method"] == "context"

    for role in [
        "train_sample_ids",
        "val_sample_ids",
        "test_sample_ids",
    ]:
        assert (
            role
            in cfg["data"]
        )

    tr = set(
        cfg["data"][
            "train_sample_ids"
        ]
    )

    va = set(
        cfg["data"][
            "val_sample_ids"
        ]
    )

    te = set(
        cfg["data"][
            "test_sample_ids"
        ]
    )

    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)

    return cfg


def gene_axis(cfg):

    genes = [
        x.strip()
        for x in Path(
            cfg["data"]["genes"]
        ).read_text().splitlines()
        if x.strip()
    ]

    assert len(genes) == 280

    return genes


def make_train_or_val(
    cfg,
    genes,
    ids,
    augment,
):

    return RawContextDataset(
        prepared_root=
            cfg["data"]["prepared_root"],

        sample_ids=
            ids,

        genes=
            genes,

        target=
            "raw",

        context_um=
            cfg["data"]["context_um"],

        center_um=
            cfg["data"]["center_um"],

        model_mpp=
            cfg["data"]["model_mpp"],

        canvas_um=
            cfg["data"]["canvas_um"],

        # CRITICAL:
        # external donor-rotation roles;
        # ignore original frozen split column.
        split=
            None,

        augment=
            augment,

        ring_shuffle=
            False,
    )


def make_test(
    cfg,
    genes,
    ids,
):

    return RotationEvalDataset(
        prepared_root=
            cfg["data"]["prepared_root"],

        sample_ids=
            ids,

        genes=
            genes,

        target=
            "both",

        context_um=
            cfg["data"]["context_um"],

        center_um=
            cfg["data"]["center_um"],

        model_mpp=
            cfg["data"]["model_mpp"],

        canvas_um=
            cfg["data"]["canvas_um"],

        split=
            None,

        augment=
            False,

        ring_shuffle=
            False,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--run-dir",
        required=True,
    )

    ap.add_argument(
        "--stage",
        required=True,
        choices=[
            "train",
            "eval",
        ],
    )

    a = ap.parse_args()

    cfg = load_cfg(
        a.config
    )

    genes = gene_axis(
        cfg
    )

    run = Path(
        a.run_dir
    )

    train_ids = (
        cfg["data"][
            "train_sample_ids"
        ]
    )

    val_ids = (
        cfg["data"][
            "val_sample_ids"
        ]
    )

    test_ids = (
        cfg["data"][
            "test_sample_ids"
        ]
    )

    if a.stage == "train":

        out = (
            run
            / "train"
        )

        if out.exists():
            raise RuntimeError(
                f"refusing overwrite: {out}"
            )

        tr = make_train_or_val(
            cfg,
            genes,
            train_ids,
            True,
        )

        va = make_train_or_val(
            cfg,
            genes,
            val_ids,
            False,
        )

        assert len(tr) == expected(
            train_ids
        )

        assert len(va) == expected(
            val_ids
        )

        print(
            "ROTATION TRAIN",
            "context=",
            cfg["data"]["context_um"],
            "train=",
            train_ids,
            "val=",
            val_ids,
            "test=UNUSED",
            flush=True,
        )

        train_model(
            cfg,
            tr,
            va,
            genes,
            out,
        )

        close_dataset(tr)
        close_dataset(va)

        assert (
            out
            / "best.pt"
        ).is_file()

        assert (
            out
            / "history.json"
        ).is_file()

        print(
            "ROTATION TRAIN: PASS",
            flush=True,
        )

    else:

        ck = (
            run
            / "train"
            / "best.pt"
        )

        if not ck.is_file():
            raise RuntimeError(
                f"missing checkpoint: {ck}"
            )

        out = (
            run
            / "eval"
        )

        if out.exists():
            raise RuntimeError(
                f"refusing overwrite: {out}"
            )

        te = make_test(
            cfg,
            genes,
            test_ids,
        )

        assert len(te) == expected(
            test_ids
        )

        print(
            "ROTATION TEST",
            "context=",
            cfg["data"]["context_um"],
            "test=",
            test_ids,
            flush=True,
        )

        evaluate_checkpoint(
            checkpoint=
                ck,

            dataset=
                te,

            outdir=
                out,

            batch_size=
                64,

            workers=
                6,

            device=
                "auto",
        )

        close_dataset(te)

        for f in [
            "metrics_raw.csv",
            "metrics_raw_by_donor.csv",
            "metrics_clean.csv",
            "metrics_clean_by_donor.csv",
            "predictions.npz",
        ]:
            assert (
                out
                / f
            ).is_file()

        print(
            "ROTATION TEST: PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
