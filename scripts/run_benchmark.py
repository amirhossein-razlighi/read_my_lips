#!/usr/bin/env python
"""Run Read My Lips on a test set of benchmarks/ and write its per-sample results for scripts/make_tables.py.

  python scripts/run_benchmark.py --benchmark silenced_talkvid_en --work data/runs/silenced_talkvid_en \
      --results outputs/results

The inputs come from the Read My Lips data (scripts/download_data.sh; --data, default rml_data of configs/paths.yaml):
test_sets/<benchmark>/video/<base_id>.mp4 (the 4.8 s face windows) and, for the noisy and grid sets,
test_sets/<benchmark>/input/<id>.wav (the noisy inputs of the paper). Without the data, give --windows (windows rebuilt
from the source videos with rml.data.windows.make_window at window_start_s, docs/EVALUATION.md); the noisy and grid
inputs are then mixed from the noise bank (configs/paths.yaml: noise_bank).
Recipes: tables for silenced_/noisy_ TalkVid and VoxMM (Tab. 2, 3), hard for the hard set (Tab. 4), grid for grid_*
(Fig. 4), multilingual for silenced_talkvid_esfr (Tab. 5b); the default follows the benchmark name.

--reader replaces the lip reader R (Sec. 5.5): usr2 gives the "USR 2.0 as R" rows of Tab. 2 and 3
(<results>/<benchmark>.usr2.json); silenced_talkvid_esfr reads with vsr_multilang (the readers of Ma et al.).
  python scripts/run_benchmark.py --benchmark noisy_voxmm --reader usr2 --work data/runs/noisy_voxmm_usr2 \
      --results outputs/results
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rml.paths import load_paths  # noqa: E402
from rml.pipeline import CONFIGS, READERS, STAGES, Pipeline, add_muted_twins, kind, twin_id  # noqa: E402

MULTILINGUAL = ("silenced_talkvid_esfr",)


def results_name(bench, reader):
    """<benchmark>.json for the benchmark's own reader, <benchmark>.<reader>.json for a swapped one"""
    own = "vsr_multilang" if bench in MULTILINGUAL else "auto_avsr"
    return f"{bench}.json" if reader == own else f"{bench}.{reader}.json"


def root_rows(bench, rows, twins=True):
    """manifest rows -> pipeline sample rows (clips/<base_id>.mp4)"""
    out = []
    for r in rows:
        if r.get("gt_review") == "removed":
            continue
        base = {"id": r["id"], "clip": f"clips/{r['base_id']}.mp4", "base_id": r["base_id"],
                "language": r.get("language", "en")}
        if bench.startswith("silenced"):
            # the reference span words feed the audible-context leak guard
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
    return add_muted_twins(out) if twins else out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="a manifest name of benchmarks/index.json")
    ap.add_argument("--data", type=Path, default=None,
                    help="the downloaded Read My Lips data (default: rml_data of configs/paths.yaml)")
    ap.add_argument("--windows", type=Path, default=None,
                    help="dir of <base_id>.mp4 face windows (default: <data>/test_sets/<benchmark>/video)")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--recipe", choices=["tables", "hard", "grid", "multilingual"], default=None)
    ap.add_argument("--reader", choices=READERS, default=None,
                    help="lip reader R (default: auto_avsr; vsr_multilang for silenced_talkvid_esfr)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--ids", default=None, help="comma list of sample ids: run only these (a quick check)")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    P = load_paths(a.paths)
    P.export_env()
    recipe = a.recipe or ("grid" if a.benchmark.startswith("grid") else "hard" if a.benchmark.endswith("hard_en")
                          else "multilingual" if a.benchmark in MULTILINGUAL else "tables")
    reader = a.reader or ("vsr_multilang" if a.benchmark in MULTILINGUAL else "auto_avsr")
    rows = [json.loads(line) for line in (REPO / "benchmarks" / f"{a.benchmark}.jsonl").read_text().splitlines() if line]
    if a.ids:
        keep = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    data = a.data
    if data is None:
        try:
            data = P.rml_data
        except KeyError:
            data = None
    shipped = data / "test_sets" / a.benchmark if data is not None else None
    if shipped is not None and not shipped.is_dir():
        shipped = None
    windows = a.windows or (shipped / "video" if shipped is not None else None)
    if windows is None:
        raise SystemExit(f"no windows for {a.benchmark}: run scripts/download_data.sh test_sets or give --windows")
    root = a.work
    (root / "clips").mkdir(parents=True, exist_ok=True)
    meta = root_rows(a.benchmark, rows, twins=recipe != "multilingual")
    if not (root / "metadata.json").exists():
        (root / "metadata.json").write_text(json.dumps({"clips": meta}, indent=1))
    for w in sorted({r["base_id"] for r in meta}):
        dst, src = root / "clips" / f"{w}.mp4", windows / f"{w}.mp4"
        if not dst.exists():
            if not src.exists():
                raise SystemExit(f"missing window {src}")
            os.symlink(src.resolve(), dst)
    stages = [s for s in a.stages.split(",") if s]
    if "latents" in stages and not a.benchmark.startswith("silenced"):
        den = [r for r in meta if r["task"] == "denoise"]
        if shipped is not None and (shipped / "input").is_dir():
            # the paper's noisy inputs: rml.data.latents encodes G's input from noisy_wavs_16k/<id>.wav
            (root / "noisy_wavs_16k").mkdir(exist_ok=True)
            for r in den:
                dst, src = root / "noisy_wavs_16k" / f"{r['id']}.wav", shipped / "input" / f"{r['id']}.wav"
                if not dst.exists():
                    if not src.exists():
                        raise SystemExit(f"missing input {src}")
                    os.symlink(src.resolve(), dst)
        elif a.benchmark.startswith("noisy"):
            from rml.eval.benchmarks.noisy import build
            build(root, den, P.noise_bank, P.ltx_audio_vae)
        else:
            from rml.eval.benchmarks.grid import build
            build(root, den, P.noise_bank, P.ltx_audio_vae)
    pipe = Pipeline(root, P, recipe=recipe, reader=reader)
    pipe.run(stages)
    a.results.mkdir(parents=True, exist_ok=True)
    out = a.results / results_name(a.benchmark, reader)
    if recipe == "multilingual":
        # the fixed configurations of Tab. 5b: R's top reading, L's transcript, R -> L -> G
        res = {sid: {"hyp": h, "raw": dict(h)} for sid, h in pipe.hypotheses().items()}
        out.write_text(json.dumps(res, ensure_ascii=False))
        print(f"{len(res)} samples -> {out}")
        return
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
    out.write_text(json.dumps(res))
    print(f"{len(res)} samples -> {out}")


if __name__ == "__main__":
    main()
