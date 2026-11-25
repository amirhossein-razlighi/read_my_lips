#!/usr/bin/env python
"""Regenerate the paper's main tables from per-sample results (rml/eval/tables.py has the format):
Tab. 2 (silenced speech), Tab. 3 (noisy speech), Tab. 4 (hard-to-read lips), Tab. 5a (components) and Fig. 4
(condition grid: WER per condition and the configurations chosen).

  python scripts/make_tables.py --results outputs/results [--out outputs/tables.txt]
Results files: <results>/<benchmark>.json for the benchmarks of benchmarks/index.json that were run
(silenced_talkvid_en, silenced_voxmm, silenced_hard_en, noisy_talkvid_en, noisy_voxmm, noisy_hard_en,
grid_talkvid_en, grid_voxmm); missing ones are skipped. Baseline transcripts: <results>/<benchmark>.baselines.json.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rml.eval import tables as T  # noqa: E402


def load(results, bench, name):
    p = results / f"{name}.json"
    if not p.exists():
        return None, None, None
    b = results / f"{name}.baselines.json"
    return (json.loads(p.read_text()), T.load_manifest(bench / f"{name}.jsonl"),
            json.loads(b.read_text()) if b.exists() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--benchmarks", type=Path, default=REPO / "benchmarks")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    L = []
    # Tab. 2 / 5a / 4 (silenced)
    for metric, label in (("g_exact", "exact span recovery %"), ("g_acc", "word accuracy in the span %")):
        sets, hard = {}, {}
        for st, nm in (("talkvid_en", "TalkVid"), ("voxmm", "VoxMM"), ("hard_en", "Hard")):
            res, man, bl = load(a.results, a.benchmarks, f"silenced_{st}")
            if res is not None:
                (hard if st == "hard_en" else sets)[nm] = T.silenced_scores(res, man, bl, metric)
        if sets or hard:
            L += [f"=== Silenced speech (Tab. 2, 4, 5a): {label}, ASR-verified cuts ===",
                  T.table_silenced(sets, hard, label), ""]
    # Tab. 3 / 5a / 4 (noisy)
    pooled = {}
    for st in ("talkvid_en", "voxmm", "hard_en"):
        res, man, bl = load(a.results, a.benchmarks, f"noisy_{st}")
        if res is None:
            continue
        per = T.window_wers(res, man, bl)
        L += [f"=== Noisy speech, {st}: window WER % ===", T.table_conditions(per, T.NOISY_CONDS), ""]
        if st != "hard_en":
            pooled.update(per)
    if pooled:
        L += ["=== Noisy speech (Tab. 3, 5a): window WER %, pooled TalkVid + VoxMM ===",
              T.table_conditions(pooled, T.NOISY_CONDS, "Avg"), ""]
    # Fig. 4 (condition grid)
    grid = {}
    for st in ("talkvid_en", "voxmm"):
        res, man, _ = load(a.results, a.benchmarks, f"grid_{st}")
        if res is not None:
            grid.update(T.grid_wers(res, man))
    if grid:
        cells = sorted({v[1] for v in grid.values()})
        L += ["=== Condition grid (Fig. 4): window WER %, pooled TalkVid + VoxMM ===",
              T.table_conditions(grid, cells, "All"), "", "configurations chosen per condition (Fig. 4, bottom):",
              T.grid_choices(grid), ""]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
