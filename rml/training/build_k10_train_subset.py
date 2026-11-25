#!/usr/bin/env python
"""k=1..10 LTX (generator stage 1) train subset: every model supports 1-10 silenced words; Russian is not used.

This script draws the 15,816 stage-1 cuts. Its inputs are a 1..4-word subset's condition dirs
(conditions_avsr_vsrfw / conditions_gemma_avsrline), the L rows and a YouTube-id exclusion list; the cut ids it
produces are shipped in rml/training/manifests/ (docs/TRAINING.md). Paths are command-line arguments.
k=1..4 cuts reuse the incumbent 20k subset's cuts first (conditions + n-best exist),
topped up from the rest of the root when the edge rule leaves too few.

Draws PER_K cuts for every k=1..10 from the TalkVid train root, whose latents
already exist for all 87k k=1..10 cuts; only text conditions are missing for
k=5..10. Rules: language en/es/fr at the incumbent subset's mix; span edges pass
span_rules.edge_ok; the cut has a proposer row (vis2gemma_p3/train.jsonl, whisper
context with ___). k=1..4 cuts come from the incumbent 20k subset (their
conditions and proposer n-best exist) where possible; every cut comes from a window of that
subset (the reader->LTX condition is window-level: 'Transcript: <VSR top-1>',
tensors verified bit-identical across a window's cuts -> hardlinked).

Writes
  <out>/{ids.json, README.json}
  <proposer dir>/train_k10new_shardNN.jsonl          proposer rows to decode (new cuts)
  <root>/conditions_avsr_vsrfw_k10/                  reader->LTX conditions (hardlinks)
  <root>/conditions_gemma_avsrline_k10/              k=1..4 full-pipeline conditions
                                                     (hardlinks; new cuts encoded later)
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
from rml.eval.span_rules import edge_ok  # noqa: E402

MIX = {"en": 0.66, "es": 0.32, "fr": 0.02}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-k", type=int, default=2000)
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, required=True, help="the generator train root (talkvid_mt_span/train)")
    ap.add_argument("--proposer-dir", type=Path, required=True,
                    help="dir of the proposer rows train.jsonl (whisper context with ___), build_vis2gemma_data.py")
    ap.add_argument("--exclude-ytids", type=Path, required=True,
                    help="text file of YouTube ids excluded from training (test / validation videos)")
    ap.add_argument("--vsr-windows", type=Path, required=True,
                    help="{window id: Auto-AVSR VSR top-1 transcript} json of the train windows")
    ap.add_argument("--out", type=Path, required=True, help="output dir for ids.json / README.json")
    a = ap.parse_args()
    ROOT, P3 = a.root, a.proposer_dir
    rng = random.Random(a.seed)
    meta = {c["id"]: c for c in json.load(open(ROOT / "metadata.json"))["clips"]}
    rows = {}
    for l in open(P3 / "train.jsonl"):
        if l.strip():
            r = json.loads(l)
            rows[r["id"]] = r
    casc = ROOT / "conditions_avsr_vsrfw"
    e2e = ROOT / "conditions_gemma_avsrline"
    have = {p.stem for p in casc.glob("*.pt")}
    assert have == {p.stem for p in e2e.glob("*.pt")}, "incumbent condition dirs disagree"
    win_src = {}                           # window -> one conditioned cut (condition source)
    for i in sorted(have):
        win_src.setdefault(meta[i]["base_id"], i)
    excl = set(a.exclude_ytids.read_text().split())

    drop = collections.Counter()
    pool = collections.defaultdict(list)   # (k, lang) -> ids
    for i, c in meta.items():
        k, lg = int(c["n_masked_words"]), c["language"]
        if lg not in MIX:
            drop["lang"] += 1; continue
        if i.split("_NA_")[0] in excl:
            drop["excluded_ytid"] += 1; continue
        if i not in rows:
            drop["no_proposer_row"] += 1; continue
        if c["base_id"] not in win_src:
            drop["window_without_condition"] += 1; continue
        if not edge_ok(c["masked_words"], lg):
            drop["edge_stopword"] += 1; continue
        pool[(k, lg)].append(i)

    chosen, per = [], {}
    for k in range(1, 11):
        want = {lg: round(a.per_k * f) for lg, f in MIX.items()}
        short = 0
        for lg in ("fr", "es", "en"):          # scarce languages first; en absorbs any shortfall
            p = sorted(pool[(k, lg)])
            rng.shuffle(p)
            p.sort(key=lambda i: i not in have)   # already-conditioned cuts first (stable: keeps the shuffle)
            n = min(len(p), want[lg] + (short if lg == "en" else 0))
            short += want[lg] - n if lg != "en" else 0
            chosen += p[:n]
            per[f"k{k}_{lg}"] = n
    chosen.sort()
    # windows whose Auto-AVSR transcript is empty: the proposer's transcript line is
    # required, so they get no full-pipeline condition (the reader->LTX set keeps them,
    # 'Transcript: ' as in the incumbent recipe). Applied AFTER sampling and after the
    # shard assignment below, so every other row keeps its shard
    vsr = json.load(open(a.vsr_windows))
    empty = {i for i in chosen if not vsr.get(meta[i]["base_id"], "").strip()}
    drop["empty_vsr_transcript"] = len(empty)
    empty_ids = sorted(empty)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    json.dump([i for i in chosen if i not in empty], open(out / "ids.json", "w"))

    # reader->LTX conditions: hardlink the window's tensor; texts carried over
    ct = json.load(open(casc / "_final_conditions.json"))
    cd = ROOT / "conditions_avsr_vsrfw_k10"
    ed = ROOT / "conditions_gemma_avsrline_k10"
    for d, src in ((cd, casc), (ed, e2e)):
        d.mkdir(exist_ok=True)
        for f in ("_encoder.json", "_provenance.json"):
            if (src / f).exists():
                shutil.copy(src / f, d / f)
    texts, new = {}, []
    for i in chosen:
        s = win_src[meta[i]["base_id"]] if i not in have else i
        if i in empty:             # reader->LTX: 'Transcript: ' (empty) as in the incumbent recipe;
            dst = cd / f"{i}.pt"    # full pipeline: no proposer prompt -> left out
            if not dst.exists():
                os.link(casc / f"{s}.pt", dst)
            texts[i] = ct[s]
            if i not in have:
                new.append(i)      # holds its shard slot; skipped when the shards are written
            continue
        dst = cd / f"{i}.pt"
        if not dst.exists():
            os.link(casc / f"{s}.pt", dst)
        texts[i] = ct[s]
        if i in have:
            dst = ed / f"{i}.pt"
            if not dst.exists():
                os.link(e2e / f"{i}.pt", dst)
        else:
            new.append(i)
    json.dump(texts, open(cd / "_final_conditions.json", "w"), ensure_ascii=False)
    keep = set(chosen) - empty
    for d, kd in ((cd, set(chosen)), (ed, keep)):   # stale links from an earlier build
        for f in d.glob("*.pt"):
            if f.stem not in kd:
                f.unlink()

    # proposer rows for the new cuts, sharded for parallel decoding
    for f in P3.glob("train_k10new_shard*.jsonl"):
        f.unlink()
    for s in range(a.shards):
        with open(P3 / f"train_k10new_shard{s:02d}.jsonl", "w") as fh:
            for i in [j for j in new[s::a.shards] if j not in empty]:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    readme = {"per_k": a.per_k, "seed": a.seed, "mix": MIX, "n": len(keep), "n_reused": len(keep) - len(set(new) - empty),
              "n_new": len(set(new) - empty), "n_new_by_k": dict(collections.Counter(int(meta[i]["n_masked_words"]) for i in new)), "per_k_lang": per, "dropped": dict(drop), "empty_vsr_ids": empty_ids,
              "pool": {f"k{k}_{lg}": len(v) for (k, lg), v in sorted(pool.items())}}
    json.dump(readme, open(out / "README.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in readme.items() if k != "pool"}, indent=1))


if __name__ == "__main__":
    main()
