"""R for Spanish and French (Tab. 5b): the visual speech recognisers of Ma et al. (third_party/vsr_multilang), trained
on CMU-MOSEAS (Spanish CMUMOSEAS_V_ES_WER44.5, French CMUMOSEAS_V_FR_WER58.6), each with its RNN language model.

They read the same dlib mouth ROI of the whole window as Auto-AVSR (rml/reader/roi.py; 25 fps, the models' v_fps)
with the decoding parameters of the repository's config (configs/<model>.ini: beam 30, CTC weight 0.1, LM weight 0.4,
penalty 0). The 10 best distinct transcripts, best first; an empty reading is dropped and a window with none gets
["..."]. L and G are unchanged.

Rows: [{"id": <window>, "roi": <window ROI .avi>, "language": "es" | "fr"}] -> [{"id", "nbest": [...]}]. The pipeline
runs this module in its own process (its `espnet` package would shadow Auto-AVSR's):
  python -m rml.reader.vsr_multilang --vsr-dir third_party/vsr_multilang --model es=<model dir>,<lm dir> \
      --model fr=<model dir>,<lm dir> --rows rows.json --out out.json
A model dir holds model.pth and model.json (e.g. CMUMOSEAS_V_ES_WER44.5/), an LM dir the same (lm_es/).
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

NBEST = 10
CONFIGS = {"es": "CMUMOSEAS_V_ES_WER44.5", "fr": "CMUMOSEAS_V_FR_WER58.6"}


def read_roi(avi):
    """the whole ROI video as [T, 96, 96] uint8 grayscale"""
    cap = cv2.VideoCapture(str(avi))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return torch.from_numpy(np.stack(fr)) if fr else None


class MultilingualVSR:
    def __init__(self, vsr_dir, language, model_dir, lm_dir, device="cuda"):
        d = Path(vsr_dir).resolve()
        sys.path.insert(0, str(d))
        from espnet.asr.asr_utils import add_results_to_json
        from pipelines.data.transforms import VideoTransform
        from pipelines.model import AVSR
        cfg = configparser.ConfigParser()
        cfg.read(d / f"configs/{CONFIGS[language]}.ini")
        self.dev = device if torch.cuda.is_available() else "cpu"
        m, lm = Path(model_dir), Path(lm_dir)
        self.m = AVSR("video", str(m / "model.pth"), str(m / "model.json"), str(lm / "model.pth"), str(lm / "model.json"),
                      penalty=cfg.getfloat("decode", "penalty"), ctc_weight=cfg.getfloat("decode", "ctc_weight"),
                      lm_weight=cfg.getfloat("decode", "lm_weight"), beam_size=cfg.getint("decode", "beam_size"),
                      device=self.dev)
        self.tf = VideoTransform(speed_rate=1.0)   # the ROIs are at 25 fps, the models' v_fps
        self.add_results_to_json = add_results_to_json

    def nbest(self, roi_avi, nbest=NBEST):
        v = read_roi(roi_avi) if Path(roi_avi).exists() else None
        if v is None:
            return None
        x = self.tf(v)                                       # [1, T, 88, 88]
        with torch.no_grad():
            hyps = self.m.beam_search(self.m.model.encode(x.to(self.dev)))
        nb, seen = [], set()
        for h in hyps[: nbest * 2]:
            t = self.add_results_to_json([h.asdict()], self.m.token_list)
            t = t.replace("▁", " ").replace("<eos>", "").strip()
            if t and t not in seen:
                seen.add(t)
                nb.append(t)
            if len(nb) >= nbest:
                break
        return [t for t in nb if t.strip()] or ["..."]


def read_rows(rows, vsr_dir, models, nbest=NBEST, device="cuda"):
    """models: {language: (model dir, LM dir)}; rows are read by the model of their language, missing ROIs skipped"""
    out = []
    for lang in sorted({r["language"] for r in rows}):
        if lang not in models:
            raise SystemExit(f"no lip reader for language '{lang}' (have {sorted(models)})")
        r = MultilingualVSR(vsr_dir, lang, *models[lang], device=device)
        for row in rows:
            if row["language"] == lang:
                t = r.nbest(row["roi"], nbest)
                if t is not None:
                    out.append({"id": row["id"], "nbest": t})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vsr-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="<lang>=<model dir>,<LM dir>")
    ap.add_argument("--rows", type=Path, required=True, help="json list of {id, roi, language}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nbest", type=int, default=NBEST)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    models = {k: tuple(v.split(",")) for k, v in (m.split("=", 1) for m in a.model)}
    rows = read_rows(json.loads(a.rows.read_text()), a.vsr_dir, models, a.nbest, a.device)
    a.out.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    print(f"Ma et al. VSR: {len(rows)} windows read -> {a.out}")


if __name__ == "__main__":
    main()
