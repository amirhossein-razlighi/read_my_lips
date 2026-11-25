"""Mouth ROIs for the lip reader: the AV-HuBERT / Auto-AVSR preparation recipe (dlib 68 landmarks, HOG detector
with the CNN detector as fallback, landmarks interpolated over missed frames, mean-face warp on the stable points,
96 x 96 grayscale mouth crop), written as MJPEG .avi at 25 fps.

Needs dlib and the upstream preparation code (third_party/av_hubert/avhubert/preparation, scripts/fetch_third_party.sh)
plus the models shape_predictor_68_face_landmarks.dat, mmod_human_face_detector.dat and 20words_mean_face.npy
(paths: configs/paths.yaml, section roi). CPU only.
  python -m rml.reader.roi --root <root with metadata.json + clips> --out <dir> [--procs 8]
-> <out>/{landmarks,roi}/<id>.{pkl,avi}
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FPS = 25
# align_mouth.py conventions (upstream defaults)
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
WINDOW_MARGIN = 12
START_IDX, STOP_IDX = 48, 68
CROP_W = CROP_H = 96

_M = {}
_CFG = {}


def configure(prep_dir, face_predictor, cnn_detector, mean_face):
    _CFG.update(prep_dir=str(prep_dir), face_predictor=str(face_predictor), cnn_detector=str(cnn_detector),
                mean_face=str(mean_face))
    if _CFG["prep_dir"] not in sys.path:
        sys.path.insert(0, _CFG["prep_dir"])


def _models():
    if "m" not in _M:
        import dlib
        _M["m"] = (dlib.get_frontal_face_detector(),
                   dlib.cnn_face_detection_model_v1(_CFG["cnn_detector"]),
                   dlib.shape_predictor(_CFG["face_predictor"]))
    return _M["m"]


def write_roi_video(rois, target: Path) -> None:
    """png sequence -> ffmpeg MJPEG .avi (as preparation/align_mouth.py:write_video_ffmpeg)"""
    import cv2
    with tempfile.TemporaryDirectory() as td:
        for i, roi in enumerate(rois):
            cv2.imwrite(os.path.join(td, f"{i:010d}.png"), roi)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", os.path.join(td, "%010d.png"),
             "-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj420p", str(target)],
            check=True, capture_output=True)


def detect_landmarks(video_path: Path, detector, cnn_detector, predictor):
    """Per-frame dlib 68 landmarks; None where no face is found (preparation/detect_landmark.py, frames read with cv2)"""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0 and cnn_detector is not None:
            rects = [d.rect for d in cnn_detector(gray)]
        coords = None
        for rect in rects:
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        landmarks.append(coords)
    cap.release()
    return landmarks


def prep_clip(task) -> tuple[str, str]:
    clip, lm_pkl, roi_avi, cfg = task
    if not _CFG:
        configure(**cfg)
    from align_mouth import crop_patch, landmarks_interpolate
    cid = roi_avi.stem
    try:
        if roi_avi.exists():
            return cid, "ok"
        if lm_pkl.exists():
            landmarks = pickle.loads(lm_pkl.read_bytes())
        else:
            detector, cnn_detector, predictor = _models()
            landmarks = detect_landmarks(clip, detector, cnn_detector, predictor)
            lm_pkl.write_bytes(pickle.dumps(landmarks))
        if sum(1 for l in landmarks if l is not None) == 0:
            return cid, "no_face"
        interp = landmarks_interpolate(list(landmarks))
        mean_face = np.load(_CFG["mean_face"])
        rois = crop_patch(str(clip), interp, mean_face, STABLE_PNTS_IDS, STD_SIZE,
                          WINDOW_MARGIN, START_IDX, STOP_IDX, CROP_H, CROP_W)
        if rois is None:
            return cid, "crop_failed"
        write_roi_video(rois, roi_avi)
        return cid, "ok"
    except Exception as e:
        return cid, f"error: {e}"


def prep_rows(rows, root: Path, out: Path, cfg: dict, procs: int = 8) -> dict:
    """ROIs for rows [{id, clip}] (clip relative to root) -> {id: status}"""
    for d in ("landmarks", "roi"):
        (out / d).mkdir(parents=True, exist_ok=True)
    tasks = [(root / r["clip"], out / "landmarks" / f"{r['id']}.pkl", out / "roi" / f"{r['id']}.avi", cfg)
             for r in rows if (root / r["clip"]).exists()]
    status = {}
    if procs <= 1:
        for t in tasks:
            cid, st = prep_clip(t)
            status[cid] = st
    else:
        with Pool(procs) as pool:
            for cid, st in pool.imap_unordered(prep_clip, tasks):
                status[cid] = st
    print(f"ROI {dict(Counter(status.values()))}", flush=True)
    return status


def main() -> None:
    from rml.paths import load_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="dir with metadata.json (clips relative to it)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--paths", default=None, help="paths config (default configs/paths.yaml)")
    a = ap.parse_args()
    P = load_paths(a.paths)
    rows = json.loads((a.root / "metadata.json").read_text())["clips"]
    status = prep_rows(rows, a.root, a.out, P.roi_config(), a.procs)
    (a.out / "prep_status.json").write_text(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
