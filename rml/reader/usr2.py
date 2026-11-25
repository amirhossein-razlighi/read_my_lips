"""R = USR 2.0 (Haliassos et al., third_party/usr2, Huge high-resource checkpoint), lips only (Sec. 5.5).

The official demo's pipeline, unchanged: MediaPipe face landmarks -> the repository's VideoProcess mouth crop (96 px)
-> centre crop 88, grayscale, normalisation -> encoder -> BatchBeamSearch with the repository's decoding config (beam 40,
CTC weight 0.1, length penalty 0). The finished hypotheses of the beam are kept as a list of the 10 best distinct
transcripts (lower case, best first), the input L expects; a window without a usable face gets an empty list.
USR reads the face window video itself (not the dlib mouth ROI of Auto-AVSR).

Only R changes: the audio-visual n-best for noisy inputs and the visual evidence e_R of the selection stay Auto-AVSR.

Rows: [{"id": <window>, "video": <window .mp4>}] -> [{"id", "nbest": [...]}]. The pipeline runs this module in its
own process (its `espnet` package would shadow Auto-AVSR's), with the interpreter `usr2_python` when it is set:
  python -m rml.reader.usr2 --usr2-dir third_party/usr2 --ckpt usr2_huge_high_resource.pth --rows rows.json --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NBEST = 10


class USR2:
    def __init__(self, usr2_dir, ckpt, device="cuda"):
        d, ckpt = Path(usr2_dir).resolve(), Path(ckpt).resolve()
        sys.path.insert(0, str(d))
        import demo as D  # the official demo's loading, transform, model and beam-search builders
        import torch
        from espnet.asr.asr_utils import parse_hypothesis
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from preprocessing.landmarks_detector import LandmarksDetector
        from preprocessing.video_preprocess import VideoProcess
        from utils.utils import UNIGRAM1000_LIST

        OmegaConf.register_new_resolver("len", len, replace=True)
        with initialize_config_dir(config_dir=str(d / "conf"), version_base="1.3"):
            self.cfg = compose(config_name="config", overrides=[f"model.pretrained_model_path={ckpt}"])
        self.D, self.torch, self.parse, self.tokens = D, torch, parse_hypothesis, UNIGRAM1000_LIST
        self.dev = torch.device(device)
        self.model = D.load_model(self.cfg, str(ckpt), self.dev)
        self.beam = D.build_beam_search(self.cfg, self.model)
        self.beam.to(self.dev)
        self.landmarks, self.crop = LandmarksDetector(detector="mediapipe"), VideoProcess(convert_gray=False)
        self.transform = D.build_video_transform()

    def nbest(self, video, nbest=NBEST):
        """distinct lower-case transcripts, best first (empty when no face is found in enough frames)"""
        torch = self.torch
        frames, _ = self.D.load_video_audio(str(video))
        mouth = self.crop(frames, self.landmarks(frames))
        if mouth is None:
            raise RuntimeError("no face in enough frames")
        x = self.transform(torch.from_numpy(mouth).permute(3, 0, 1, 2).float())
        with torch.no_grad():
            feat = self.model.encoder(xs_v=x.unsqueeze(0).to(self.dev))
            hyps = self.beam(x=feat.squeeze(0), modality="v", maxlenratio=self.cfg.decode.maxlenratio,
                             minlenratio=self.cfg.decode.minlenratio)
        seen = []
        for h in hyps:
            t, _, _, _ = self.parse(h.asdict(), self.tokens)
            t = " ".join(t.replace("<eos>", "").replace("▁", " ").lower().split())
            if t and t not in seen:
                seen.append(t)
            if len(seen) == nbest:
                break
        return seen


def read_rows(rows, usr2_dir, ckpt, nbest=NBEST, device="cuda"):
    r = USR2(usr2_dir, ckpt, device)
    out = []
    for row in rows:
        try:
            nb = r.nbest(row["video"], nbest)
        except Exception as e:  # noqa: BLE001 - a window without a usable face is a result (empty list), logged
            print(f"USR 2.0: {row['id']}: {type(e).__name__}: {e}"[:300], flush=True)
            nb = []
        out.append({"id": row["id"], "nbest": nb})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usr2-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=Path, required=True, help="json list of {id, video}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nbest", type=int, default=NBEST)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rows = read_rows(json.loads(a.rows.read_text()), a.usr2_dir, a.ckpt, a.nbest, a.device)
    a.out.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    print(f"USR 2.0: {sum(bool(r['nbest']) for r in rows)}/{len(rows)} windows read -> {a.out}")


if __name__ == "__main__":
    main()
