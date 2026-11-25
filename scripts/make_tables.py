#!/usr/bin/env python
"""Regenerate the paper's main tables from per-sample results (rml/eval/tables.py has the format):
Tab. 2 (silenced speech), Tab. 3 (noisy speech), Tab. 4 (hard-to-read lips), Tab. 5a (components), Tab. 5b (Spanish
and French) and Fig. 4 (condition grid: WER per condition and the configurations chosen). Results of the pipeline with
USR 2.0 as R (<benchmark>.usr2.json, scripts/run_benchmark.py --reader usr2) give the "Ours (USR 2.0 as R)" rows of
Tab. 2 and 3.

  python scripts/make_tables.py --results outputs/results [--out outputs/tables.txt]
Results files: <results>/<benchmark>.json for the benchmarks of benchmarks/index.json that were run
(silenced_talkvid_en, silenced_voxmm, silenced_hard_en, noisy_talkvid_en, noisy_voxmm, noisy_hard_en,
grid_talkvid_en, grid_voxmm, silenced_talkvid_esfr) and <results>/<benchmark>.usr2.json; missing ones are skipped.
Baseline transcripts: <results>/<benchmark>.baselines.json.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rml.eval import tables as T  # noqa: E402

READERS = (("", "R = Auto-AVSR"), (".usr2", "R = USR 2.0"))


def load(results, bench, name, reader=""):
    p = results / f"{name}{reader}.json"
    if not p.exists():
        return None, None, None
    b = results / f"{name}.baselines.json"
    return (json.loads(p.read_text()), T.load_manifest(bench / f"{name}.jsonl"),
            json.loads(b.read_text()) if b.exists() and not reader else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--benchmarks", type=Path, default=REPO / "benchmarks")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    L = []
    for rd, rname in READERS:
        which = "Tab. 2, 4, 5a" if not rd else "Tab. 2, Ours (USR 2.0 as R)"
        # Tab. 2 / 5a / 4 (silenced)
        for metric, label in (("g_exact", "exact span recovery %"), ("g_acc", "word accuracy in the span %")):
            sets, hard = {}, {}
            for st, nm in (("talkvid_en", "TalkVid"), ("voxmm", "VoxMM"), ("hard_en", "Hard")):
                res, man, bl = load(a.results, a.benchmarks, f"silenced_{st}", rd)
                if res is not None:
                    (hard if st == "hard_en" else sets)[nm] = T.silenced_scores(res, man, bl, metric)
            if sets or hard:
                L += [f"=== Silenced speech ({which}), {rname}: {label}, ASR-verified cuts ===",
                      T.table_silenced(sets, hard, label), ""]
        # Tab. 3 / 5a / 4 (noisy)
        which = "Tab. 3, 5a" if not rd else "Tab. 3, Ours (USR 2.0 as R)"
        pooled = {}
        for st in ("talkvid_en", "voxmm", "hard_en"):
            res, man, bl = load(a.results, a.benchmarks, f"noisy_{st}", rd)
            if res is None:
                continue
            per = T.window_wers(res, man, bl)
            L += [f"=== Noisy speech, {st}, {rname}: window WER % ===", T.table_conditions(per, T.NOISY_CONDS), ""]
            if st != "hard_en":
                pooled.update(per)
        if pooled:
            L += [f"=== Noisy speech ({which}), {rname}: window WER %, pooled TalkVid + VoxMM ===",
                  T.table_conditions(pooled, T.NOISY_CONDS, "Avg"), ""]
    # Tab. 5b (Spanish and French: R = Ma et al., L and G unchanged)
    res, man, _ = load(a.results, a.benchmarks, "silenced_talkvid_esfr")
    if res is not None:
        for metric, label in (("g_exact", "exact span recovery %"), ("g_acc", "word accuracy in the span %")):
            L += [f"=== Spanish and French (Tab. 5b), R = Ma et al.: {label}, ASR-verified cuts ===",
                  T.table_languages(T.fixed_scores(res, man, metric, T.word_by_word_scorer())), ""]
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
