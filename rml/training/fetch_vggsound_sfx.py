#!/usr/bin/env python3
"""Download the VGGSound sound-effect clips of the generator's noise bank (build_noise_bank.py).

The bank uses 9 "seen" classes (train + test clips) and 3 held-out classes (church_bell, sneezing, train_horn;
test only). YouTube availability changes over time: the exact clips used for the paper are listed in
rml/training/manifests/noise_bank_sources.json.

VGGSound clips are 10 s YouTube segments curated so the sound SOURCE IS
VISIBLE in frame — exactly the audio-visual correspondence this project needs.
Downloads are muxed <=720p single streams via yt-dlp --download-sections
(may need cookies, depending on YouTube's bot checks).

Classes (visible source, temporally localizable):
  dog_barking, door_slamming, sliding_door, car_horn, train_horn, church_bell,
  hammering, chopping_wood, clapping, sneezing (human non-speech), cat_meowing,
  fireworks.

Layout:  <out>/raw/{train,test}/<slug>/<ytid>_<start>.mp4  + manifest.json
Resumable (skips existing files). Needs internet and yt-dlp.

  python rml/training/fetch_vggsound_sfx.py \
      --csv <path>/vggsound.csv --out <sfx> \
      --per-class-train 40 --per-class-test 10 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLASSES = {  # csv label -> slug
    "dog barking": "dog_barking",
    "door slamming": "door_slamming",
    "sliding door": "sliding_door",
    "vehicle horn, car horn, honking": "car_horn",
    "train horning": "train_horn",
    "church bell ringing": "church_bell",
    "hammering nails": "hammering",
    "chopping wood": "chopping_wood",
    "people clapping": "clapping",
    "people sneezing": "sneezing",
    "cat meowing": "cat_meowing",
    "fireworks banging": "fireworks",
}
SEG_S = 10  # VGGSound segment length


def fetch_one(ytid: str, start: int, out: Path) -> str:
    if out.exists():
        return "cached"
    tmp = out.with_suffix(".part.mp4")
    r = subprocess.run(
        ["yt-dlp", "-q", "--no-warnings", "--no-playlist",
         "-f", "b[height<=720]/b",
         "--download-sections", f"*{start}-{start + SEG_S}",
         "-o", str(tmp), f"https://www.youtube.com/watch?v={ytid}"],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        err = (r.stderr or "").strip().splitlines()
        return f"fail: {err[-1][:80] if err else 'unknown'}"
    tmp.rename(out)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class-train", type=int, default=40)
    ap.add_argument("--per-class-test", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows_by = {("train", s): [] for s in CLASSES.values()}
    rows_by.update({("test", s): [] for s in CLASSES.values()})
    with args.csv.open() as f:
        for row in csv.reader(f):
            if len(row) < 4 or row[2] not in CLASSES:
                continue
            ytid, start, label, split = row[0], int(row[1]), row[2], row[3]
            rows_by[(split, CLASSES[label])].append((ytid, start))

    rng = random.Random(args.seed)
    jobs = []
    for (split, slug), pool in sorted(rows_by.items()):
        rng.shuffle(pool)
        want = args.per_class_train if split == "train" else args.per_class_test
        for ytid, start in pool[:want]:
            d = args.out / "raw" / split / slug
            d.mkdir(parents=True, exist_ok=True)
            jobs.append((split, slug, ytid, start, d / f"{ytid}_{start}.mp4"))

    print(f"{len(jobs)} downloads queued "
          f"({args.per_class_train}+{args.per_class_test} per class x {len(CLASSES)})",
          flush=True)
    manifest, counts = [], {"ok": 0, "cached": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool_ex:
        futs = {pool_ex.submit(fetch_one, y, s, o): (sp, sl, y, s, o)
                for sp, sl, y, s, o in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            sp, sl, y, s, o = futs[fut]
            try:
                status = fut.result()
            except Exception as e:  # noqa: BLE001
                status = f"fail: {type(e).__name__}"
            counts["fail" if status.startswith("fail") else status] += 1
            manifest.append({"split": sp, "class": sl, "ytid": y, "start": s,
                             "file": str(o), "status": status})
            if i % 25 == 0 or status.startswith("fail"):
                print(f"[{i}/{len(jobs)}] {sl}/{y}: {status} "
                      f"(ok={counts['ok']} cached={counts['cached']} "
                      f"fail={counts['fail']}, {i / max(1, time.time() - t0):.1f}/s)",
                      flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE ok={counts['ok']} cached={counts['cached']} fail={counts['fail']} "
          f"-> {args.out}/raw", flush=True)


if __name__ == "__main__":
    main()
