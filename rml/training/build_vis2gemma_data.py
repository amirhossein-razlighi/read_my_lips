#!/usr/bin/env python
"""Language-model (L) rows: paired (gap feature slice, whisper context, target words) manifests.

Sources (all existing, no GPU work):
  - span-root metadata: id, masked_words, mask_segments_s, language, k
  - conditions_span/_provenance.json: the whisper context per row
    (WhisperX large-v3 on the MUTED audio, ___ at the detected silence — no ground truth)
  - cached AV-HuBERT features per base window (referenced, not copied; with
    --evidence none the trainer only uses their frame count, see docs/TRAINING.md)

Output JSONL rows:
  {id, feats, f0, f1, context, target, k, language}
where [f0,f1) are the gap frames with the same ±margin as the span decoder.

Train split: --val-videos distinct video ids (deterministic, seed-sorted) are
held out as val — speaker-disjoint early-stopping set. Test roots: single file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FPS = 25


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output dir")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--language", default="en",
                    help="single code, comma list, or 'all'")
    ap.add_argument("--metadata", default="metadata.json",
                    help="metadata filename under --root")
    ap.add_argument("--val-videos", type=int, default=8,
                    help="train only: videos held out for val PER LANGUAGE")
    ap.add_argument("--frame-margin", type=int, default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    langs = (None if args.language == "all"
             else set(args.language.split(",")))

    rows = json.loads((args.root / args.metadata).read_text())["clips"]
    prov = json.loads((args.root / "conditions_span" /
                       "_provenance.json").read_text())

    items, missing_feat, missing_prov = [], 0, 0
    for r in rows:
        if langs is not None and r.get("language", "en") not in langs:
            continue
        rid = r["id"]
        p = prov.get(rid)
        if p is None:
            missing_prov += 1
            continue
        base = rid.split("__k")[0]
        fpz = args.feats / f"{base}.npz"
        if not fpz.exists():
            fpz = args.feats / f"{rid}.npz"
            if not fpz.exists():
                missing_feat += 1
                continue
        (g0, g1) = r["mask_segments_s"][0]
        f0 = max(0, int(np.floor(g0 * FPS)) - args.frame_margin)
        f1 = min(121, int(np.ceil(g1 * FPS)) + args.frame_margin)
        if f1 - f0 < 3:
            continue
        ctx = re.sub(r"_{4,}", " ", p["context"]).strip()
        ctx = re.sub(r"\s{2,}", " ", ctx)
        items.append({
            "id": rid,
            "feats": str(fpz),
            "f0": f0, "f1": f1,
            "context": ctx,
            "target": " ".join(w.lower() for w in r["masked_words"]),
            "k": r["n_masked_words"],
            "language": r.get("language", "en"),
        })

    def dump(path: Path, rs: list) -> None:
        with path.open("w") as f:
            for x in rs:
                f.write(json.dumps(x) + "\n")
        ks = {}
        for x in rs:
            ks[x["k"]] = ks.get(x["k"], 0) + 1
        print(f"{path.name}: {len(rs)} rows | per-k "
              f"{dict(sorted(ks.items()))} | videos "
              f"{len({x['id'].split('_NA_')[0] for x in rs})}")

    if args.split == "test":
        dump(args.out / "test.jsonl", items)
    else:
        val_vids: set[str] = set()
        for lg in sorted({x["language"] for x in items}):
            vids = sorted({x["id"].split("_NA_")[0] for x in items
                           if x["language"] == lg})
            val_vids |= set(vids[:: max(1, len(vids) // args.val_videos)]
                            [: args.val_videos])
        train = [x for x in items if x["id"].split("_NA_")[0] not in val_vids]
        val = [x for x in items if x["id"].split("_NA_")[0] in val_vids]
        dump(args.out / "train.jsonl", train)
        dump(args.out / "val.jsonl", val)
        print("val videos:", sorted(val_vids))
    if missing_prov or missing_feat:
        print(f"skipped: {missing_prov} without provenance, "
              f"{missing_feat} without features")


if __name__ == "__main__":
    main()
