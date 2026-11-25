#!/usr/bin/env python
"""Lay out the released L training rows (train/lm/ of the Read My Lips data) the way configs/training/lm_stage*.yaml
read them, and print the RML_* variables to export.

    python scripts/train/prepare_lm_data.py --lm data/read-my-lips-data/train/lm --out data/lm_train

Writes under --out:
  stage1_2/, stage3/     train.jsonl and val.jsonl with each row's `feats` path, keep_meta.json (linked);
                         stage3/readers_denoise/<reader>.json (linked to readers/<reader>_denoise.json)
  readers/               the reader n-best files under the names of the configs (linked)
  feats/<window>.npz     zero arrays [n_frames, dim] (key feats): L uses no lip features, only their length
                         (docs/TRAINING.md, the AV-HuBERT feature-length dependency of L)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

READERS = {"clean": "autoavsr_vsr_clean_train_shard0.json", "down4": "autoavsr_vsr_down4_train_shard0.json",
           "drop50": "autoavsr_vsr_drop50_train_shard0.json", "avhubert": "avhubert_vsr_train.json"}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lm", type=Path, required=True, help="the downloaded train/lm/ directory")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    feats = out / "feats"
    feats.mkdir(parents=True, exist_ok=True)
    for line in (a.lm / "feat_lengths.tsv").read_text().splitlines()[1:]:
        w, n, d = line.split("\t")
        p = feats / f"{w}.npz"
        if not p.exists():
            np.savez_compressed(p, feats=np.zeros((int(n), int(d)), np.float32))
    for stage in ("stage1_2", "stage3"):
        (out / stage).mkdir(exist_ok=True)
        for split in ("train", "val"):
            rows = [json.loads(x) for x in (a.lm / stage / f"{split}.jsonl").read_text().splitlines() if x.strip()]
            with open(out / stage / f"{split}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps({**r, "feats": str(feats / f"{r['window']}.npz")}, ensure_ascii=False) + "\n")
        link(a.lm / stage / "keep_meta.json", out / stage / "keep_meta.json")
    for r, name in READERS.items():
        link(a.lm / "readers" / f"{r}_inpaint.json", out / "readers" / name)
        link(a.lm / "readers" / f"{r}_denoise.json", out / "stage3" / "readers_denoise" / f"{r}.json")
    print(f"export RML_LM_STAGE12_DATA={out / 'stage1_2'} RML_LM_STAGE3_DATA={out / 'stage3'} "
          f"RML_LM_READERS={out / 'readers'} RML_AVH_FEATS={feats}")


if __name__ == "__main__":
    main()
