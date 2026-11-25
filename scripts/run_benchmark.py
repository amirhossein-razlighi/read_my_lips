#!/usr/bin/env python
"""Run Read My Lips on a test set of benchmarks/ and write its per-sample results for scripts/make_tables.py.

  python scripts/run_benchmark.py --benchmark silenced_talkvid_en --windows data/windows/talkvid_en \
      --work data/runs/silenced_talkvid_en --results outputs/results

--windows holds the 4.8 s face windows <base_id>.mp4 of the set (docs/EVALUATION.md: rebuild them from the source
videos with rml.data.windows.make_window at window_start_s). Noisy and grid sets also need the noise bank
(configs/paths.yaml: noise_bank). Recipes: tables for silenced_/noisy_ TalkVid and VoxMM (Tab. 2, 3), hard for the
hard set (Tab. 4), grid for grid_* (Fig. 4); the default follows the benchmark name.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rml.paths import load_paths  # noqa: E402
from rml.pipeline import CONFIGS, STAGES, Pipeline, add_muted_twins, kind, twin_id  # noqa: E402


def root_rows(bench, rows):
    """manifest rows -> pipeline sample rows (clips/<base_id>.mp4)"""
    out = []
    for r in rows:
        if r.get("gt_review") == "removed":
            continue
        base = {"id": r["id"], "clip": f"clips/{r['base_id']}.mp4", "base_id": r["base_id"],
                "language": r.get("language", "en")}
        if bench.startswith("silenced"):
            # the reference span words feed only the audible-context leak guard (as in the paper's runs)
            out.append({**base, "task": "inpaint", "mask_segments_s": r["mask_segments_s"],
                        "masked_words": r["masked_words"]})
        elif bench.startswith("noisy"):
            out.append({**base, "task": "denoise", "mask_segments_s": [], "noise": r["noise"], "snr_db": r["snr_db"],
                        "source_cut": r["source_cut"]})
        elif r["noise"] == "silence" and r["span"] == "full":
            out.append({**base, "task": "inpaint", "input": "muted", "evaluate": True, "mask_segments_s": [[0.0, 4.8]],
                        "masked_words": []})
        elif r["noise"] == "silence":
            out.append({**base, "task": "inpaint", "mask_segments_s": r["mask_segments_s"],
                        "masked_words": r.get("masked_words") or []})
        else:
            out.append({**base, "task": "denoise", "mask_segments_s": [], "span": r["span"], "noise": r["noise"],
                        "noise_segments_s": r.get("noise_segments_s") or []})
    return add_muted_twins(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="a manifest name of benchmarks/index.json")
    ap.add_argument("--windows", type=Path, required=True, help="dir of <base_id>.mp4 face windows")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--recipe", choices=["tables", "hard", "grid"], default=None)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    P = load_paths(a.paths)
    P.export_env()
    recipe = a.recipe or ("grid" if a.benchmark.startswith("grid") else "hard" if a.benchmark.endswith("hard_en")
                          else "tables")
    rows = [json.loads(line) for line in (REPO / "benchmarks" / f"{a.benchmark}.jsonl").read_text().splitlines() if line]
    root = a.work
    (root / "clips").mkdir(parents=True, exist_ok=True)
    meta = root_rows(a.benchmark, rows)
    if not (root / "metadata.json").exists():
        (root / "metadata.json").write_text(json.dumps({"clips": meta}, indent=1))
    for w in sorted({r["base_id"] for r in meta}):
        dst, src = root / "clips" / f"{w}.mp4", a.windows / f"{w}.mp4"
        if not dst.exists():
            if not src.exists():
                raise SystemExit(f"missing window {src}")
            os.symlink(src.resolve(), dst)
    stages = [s for s in a.stages.split(",") if s]
    if "latents" in stages and not a.benchmark.startswith("silenced"):
        den = [r for r in meta if r["task"] == "denoise"]
        if a.benchmark.startswith("noisy"):
            from rml.eval.benchmarks.noisy import build
            build(root, den, P.noise_bank, P.ltx_audio_vae)
        else:
            from rml.eval.benchmarks.grid import build
            build(root, den, P.noise_bank, P.ltx_audio_vae)
    pipe = Pipeline(root, P, recipe=recipe)
    pipe.run(stages)
    # results: the selection's input plus the unnormalised transcripts
    sc, tx, voters = pipe.jload("output_scores.json"), pipe.jload("transcripts.json"), pipe.jload("text_voters.json")
    cfg = {(v[0], v[1]): k for k, v in CONFIGS.items()}
    res = {}
    for r in pipe.evaluated():
        sid = r["id"]
        if sid not in sc:
            continue
        e = dict(sc[sid])
        raw = dict(voters[sid])
        for key in e["hyp"]:
            if key.startswith("text|"):
                continue
            arm, mode = key.split("|")
            muted = mode == "mute" or kind(r) == "muted"
            name = cfg[(f"{arm}|mute" if muted else key, "muted" if muted else r["task"])]
            t = tx.get(f"{name}|{twin_id(r['base_id']) if muted else sid}")
            if t is not None:
                raw[key] = t
        if r["task"] == "denoise" and kind(r) != "muted":
            import soundfile as sf

            from rml.eval.asr import transcribe_16k
            y, _ = sf.read(pipe.input_wav(sid), dtype="float32")
            raw["input|keep"] = transcribe_16k(y)
        e["raw"] = raw
        res[sid] = e
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"{a.benchmark}.json").write_text(json.dumps(res))
    print(f"{len(res)} samples -> {a.results / (a.benchmark + '.json')}")


if __name__ == "__main__":
    main()
