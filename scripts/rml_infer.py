#!/usr/bin/env python
"""Restore the speech of one video with Read My Lips (Sec. 4): R -> L -> G under the six configurations, then the
evidence-weighted selection. Output: <out>/<name>_restored.mp4 (the 4.8 s face window with the selected speech) and
<out>/<name>_result.json (transcript, chosen configuration, every candidate).

The task is detected from the audio (Appendix B.1): a stretch of digital silence with audio on both sides is a
silenced span (<inpaint>), otherwise the track is treated as noisy speech (<denoise>). --span or --task override it.

  python scripts/rml_infer.py --video talk.mp4 --start 12.0 --out outputs/talk
  python scripts/rml_infer.py --video talk.mp4 --span 1.20,2.05 --out outputs/talk      (known silenced span)
  python scripts/rml_infer.py --video talk.mp4 --task denoise --out outputs/talk        (noisy speech)

Needs one 80 GB GPU (G is a 22B transformer; L and the scorers run one at a time). Paths: configs/paths.yaml.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rml.data.audio_video import load_audio_window
from rml.data.silence import find_silenced_regions, pick_gap
from rml.data.windows import make_window
from rml.paths import load_paths
from rml.pipeline import STAGES, Pipeline, add_muted_twins, mux


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="window start in the video (s); the window is 4.8 s")
    ap.add_argument("--out", type=Path, required=True, help="work and output directory")
    ap.add_argument("--span", default=None, help="silenced span 'start,end' in seconds from the window start")
    ap.add_argument("--task", choices=["auto", "inpaint", "denoise"], default="auto")
    ap.add_argument("--no-crop", action="store_true", help="the video is already a 4.8 s 768x512 face window")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list (resume a run from a stage)")
    ap.add_argument("--no-lm-score", action="store_true",
                    help="skip the Gemma-4 language score of the candidates (it does not enter the selection rule)")
    ap.add_argument("--paths", default=None, help="paths config (default configs/paths.yaml)")
    a = ap.parse_args()
    P = load_paths(a.paths)
    P.export_env()
    name = a.video.stem
    root = a.out / "root"
    (root / "clips").mkdir(parents=True, exist_ok=True)
    clip = root / "clips" / f"{name}.mp4"
    if not clip.exists():
        if a.no_crop:
            shutil.copy2(a.video, clip)
        elif not make_window(a.video, a.start, clip, P.face_landmarker):
            raise SystemExit(f"no face found in {a.video} at {a.start} s")
    meta = root / "metadata.json"
    if not meta.exists():
        task, span = a.task, None
        if a.span:
            task, span = "inpaint", [float(x) for x in a.span.split(",")]
        elif task != "denoise":
            y = load_audio_window(clip, 0.0, 16000)
            gap = pick_gap(find_silenced_regions(y, 16000), len(y) / 16000)
            if gap is not None:
                task, span = "inpaint", [round(gap[0], 3), round(gap[1], 3)]
            elif task == "inpaint":
                raise SystemExit("--task inpaint but no silenced span was detected: give it with --span")
            else:
                task = "denoise"
        row = {"id": f"{name}__{task}", "clip": f"clips/{name}.mp4", "base_id": name, "language": "en", "task": task,
               "mask_segments_s": [span] if span else []}
        meta.write_text(json.dumps({"clips": add_muted_twins([row])}, indent=1))
        print(f"task {task}" + (f", silenced span {span[0]:.3f}-{span[1]:.3f} s" if span else ""), flush=True)
    pipe = Pipeline(root, P)
    pipe.run([s for s in a.stages.split(",") if s], with_lm=not a.no_lm_score)
    sel = pipe.jload("selection.json")
    sid, res = next(iter(sel.items()))
    out_mp4 = a.out / f"{name}_restored.mp4"
    mux(clip, Path(res["wav"]), out_mp4)
    result = {"sample": sid, "task": pipe.by_id[sid]["task"], "span_s": pipe.by_id[sid]["mask_segments_s"],
              "transcript": res["transcript"], "configuration": res["name"], "output": str(out_mp4),
              "candidates": pipe.jload("output_scores.json")[sid]["hyp"]}
    (a.out / f"{name}_result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
