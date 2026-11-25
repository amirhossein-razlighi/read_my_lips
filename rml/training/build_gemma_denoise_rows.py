"""Language model (L) stage-3 data: select-then-edit rows for DENOISING next to the existing inpainting rows.

A denoise row is a whole-window "gap" (f0 0, f1 120; the frame range only matters for the feature slice that the
trainer still loads, see docs/TRAINING.md), the transcript
line is '<denoise> ' + Whisper on the noisy audio (no ___), and the lettered candidate list interleaves the lip
reader's distinct readings with Auto-AVSR AV's n-best and Whisper's line on the noisy audio
(reader 1, AV 1, Whisper, reader 2, AV 2, reader 3, AV 3, reader 4, ...; distinct, at most 10). The target is the
usual 'Pick: <closest>' + 'Corrected: <reference window transcript>' (train_vis2gemma --select-edit; truth from
--keep-meta, where the denoise row's "masked words" are the whole window).

  train  -> <out>/train.jsonl (stage-1/2 inpaint rows + English denoise rows), <out>/val.jsonl (inpaint monitor),
            <out>/readers_denoise/<reader>.json (per pool reader: its readings + AV + Whisper for each denoise id),
            <out>/keep_meta.json (generator train-root metadata + one whole-window entry per denoise id)
Paths are command-line arguments (defaults from RML_* environment variables, docs/TRAINING.md).
"""
import argparse
import glob
import json
import re
from pathlib import Path

# set in __main__ from the command line
V4: Path            # stage-1/2 language-model rows (train.jsonl / val.jsonl)
TRAIN_META: Path    # generator stage-1 train root metadata.json (truth for the inpaint cuts)
DN_TRAIN: Path      # metadata.json of the multi-task generator root (its English denoise samples)
REC: Path           # dir with whisper_noisy.json + autoavsr_av_noisy.json for those denoise samples
FEATS: Path         # AV-HuBERT feature dir (<window>.npz); see docs/TRAINING.md
POOL: dict = {}     # reader name -> [glob of its train n-best jsons]
FRAMES = 120


def clean(s):
    return " ".join(re.sub(r"[^\w' ]+", " ", (s or "").lower()).split())


def merged(reader, av, whisper, cap=10):
    order = []
    r, a = list(reader), list(av)
    for i in range(max(len(r), len(a), 1)):
        if i < len(r): order.append(r[i])
        if i < len(a): order.append(a[i])
        if i == 0 and whisper: order.append(whisper)
    out, seen = [], set()
    for h in order:
        h = clean(h)
        if h and h not in seen:
            seen.add(h); out.append(h)
        if len(out) >= cap:
            break
    return out


def load_pool(paths):
    d = {}
    for pat in paths:
        for p in sorted(glob.glob(pat)):
            j = json.load(open(p))
            for r in (j["rows"] if isinstance(j, dict) and "rows" in j else j):
                nb = r.get("nbest") or []
                nb = json.loads(nb) if isinstance(nb, str) else nb
                d.setdefault(r["id"].rsplit("__k", 1)[0], list(nb))
    return d


def window_text(meta_row):
    return " ".join(clean(w["word"]) for w in meta_row["window_words"] if clean(w["word"]))


def build_train(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "readers_denoise").mkdir(exist_ok=True)
    cuts = {c["id"]: c for c in json.load(open(TRAIN_META))["clips"]}
    dn = [r for r in json.load(open(DN_TRAIN))["clips"] if r.get("task") == "denoise" and r.get("language") == "en"]
    wh = json.load(open(REC / "whisper_noisy.json"))
    av = {r["id"]: r.get("nbest") or [] for r in json.load(open(REC / "autoavsr_av_noisy.json"))["rows"]}
    pools = {name: load_pool(p) for name, p in POOL.items()}
    rows, metas, skipped = [], [], 0
    readers = {name: [] for name in POOL}
    for r in dn:
        sid, cut = r["id"], r["source_cut"]
        win = cut.rsplit("__k", 1)[0]
        base = cuts.get(cut)
        feats = FEATS / f"{win}.npz"
        if base is None or sid not in wh or sid not in av or not feats.exists() or win not in pools["clean"]:
            skipped += 1
            continue
        words = [w["word"] for w in base["window_words"]]
        truth = window_text(base)
        rows.append({"id": sid, "feats": str(feats), "f0": 0, "f1": FRAMES, "context": f"<denoise> {wh[sid]}".strip(),
                     "target": truth, "k": len(truth.split()), "language": "en", "task": "denoise"})
        metas.append({**base, "id": sid, "masked_words": words, "n_masked_words": len(words), "span_start_index": 0,
                      "mask_segments_s": [], "task": "denoise"})
        for name, pool in pools.items():
            if win in pool:
                readers[name].append({"id": sid, "nbest": merged(pool[win], av[sid], wh[sid])})
    inp = (V4 / "train.jsonl").read_text()
    (out / "train.jsonl").write_text(inp + "".join(json.dumps(x) + "\n" for x in rows))
    (out / "val.jsonl").write_text((V4 / "val.jsonl").read_text())
    for name, rr in readers.items():
        json.dump({"rows": rr}, open(out / "readers_denoise" / f"{name}.json", "w"))
    json.dump({"clips": list(cuts.values()) + metas}, open(out / "keep_meta.json", "w"))
    json.dump({"purpose": "Gemma multi-task (inpaint + denoise) select-then-edit; build_gemma_denoise_rows.py train",
               "inpaint_rows": inp.count("\n"), "denoise_rows": len(rows), "skipped_denoise": skipped,
               "readers_denoise": {k: len(v) for k, v in readers.items()}}, open(out / "README.json", "w"), indent=1)
    print(f"denoise rows {len(rows)} (skipped {skipped}); readers {[(k, len(v)) for k, v in readers.items()]}")
    ex = rows[0]["id"]
    print("example:", rows[0]["context"][:120], "|", next(x for x in readers["clean"] if x["id"] == ex)["nbest"][:4])


if __name__ == "__main__":
    import os
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inpaint-rows", type=Path, default=env("RML_LM_STAGE12_DATA"),
                    help="dir of the stage-1/2 rows (train.jsonl, val.jsonl)")
    ap.add_argument("--train-meta", type=Path, required=True, help="generator stage-1 train root metadata.json")
    ap.add_argument("--denoise-meta", type=Path, required=True,
                    help="multi-task generator train root metadata.json (the English denoise samples)")
    ap.add_argument("--recognizers", type=Path, required=True,
                    help="dir with whisper_noisy.json and autoavsr_av_noisy.json of the denoise samples")
    ap.add_argument("--feats", type=Path, default=env("RML_AVH_FEATS"), help="AV-HuBERT feature dir")
    ap.add_argument("--reader", action="append", required=True, metavar="NAME=GLOB",
                    help="training reader pool, e.g. clean='readers/clean_train_shard*.json' (repeat; the released "
                         "recipe: clean, down4, drop50, avhubert in that order)")
    a = ap.parse_args()
    V4, TRAIN_META, DN_TRAIN, REC, FEATS = a.inpaint_rows, a.train_meta, a.denoise_meta, a.recognizers, a.feats
    POOL.update({n: [g] for n, g in (x.split("=", 1) for x in a.reader)})
    build_train(a.out)
