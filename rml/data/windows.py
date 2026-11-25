"""Cut a 4.8 s face window out of any video: the face region (union of MediaPipe face-landmark boxes over five
frames of the window, margin for the shoulders, 3:2 aspect) scaled to 768 x 512 at 25 fps, the source audio copied
along. This is the clip format every model of the pipeline was trained and tested on."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

W_OUT, H_OUT, FPS = 768, 512, 25
WINDOW_S = 4.8

_LANDMARKER = {}  # created once: per-window creation aborts MediaPipe after many calls


def video_dims(video: Path) -> tuple[float, float]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0",
                        str(video)], capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")[:2]
    return float(w), float(h)


def _landmarker(model_path):
    if "m" not in _LANDMARKER:
        from mediapipe.tasks.python import BaseOptions, vision
        _LANDMARKER["m"] = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)), num_faces=1))
    return _LANDMARKER["m"]


def face_bbox(video: Path, start_s: float, model_path, n_probe: int = 5) -> tuple[float, float, float, float] | None:
    """Union of face-landmark bboxes over n_probe frames of the window, as fractions of the frame."""
    import mediapipe as mp
    lm = _landmarker(model_path)
    xs, ys = [], []
    with tempfile.TemporaryDirectory() as td:
        for k in range(n_probe):
            ts = start_s + WINDOW_S * (k + 0.5) / n_probe
            png = Path(td) / f"f{k}.png"
            r = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-threads", "1",
                 "-ss", f"{ts:.3f}", "-i", str(video), "-frames:v", "1",
                 "-filter_threads", "1", str(png)], capture_output=True)
            if r.returncode != 0 or not png.exists():
                continue
            res = lm.detect(mp.Image.create_from_file(str(png)))
            if res.face_landmarks:
                for p in res.face_landmarks[0]:
                    xs.append(p.x)
                    ys.append(p.y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def crop_window(src: Path, start_s: float, bbox, out: Path) -> bool:
    """ffmpeg: crop the face region (margin + 3:2 aspect), scale 768x512, 25 fps, window-trimmed, source audio
    re-encoded (AAC 48 kHz). A failed crop never leaves a partial file."""
    W, H = video_dims(src)
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    fh = (y1 - y0) * H
    ch = fh * 1.9                      # face height + margin (shoulders in, scene out)
    cw = ch * (W_OUT / H_OUT)          # 3:2 aspect
    if cw > W:                         # never exceed the frame
        cw, ch = W, W * (H_OUT / W_OUT)
    x = min(max(cx - cw / 2, 0), W - cw)
    y = min(max(cy - ch / 2, 0), H - ch)
    vf = (f"crop={cw:.0f}:{ch:.0f}:{x:.0f}:{y:.0f},"
          f"scale={W_OUT}:{H_OUT},fps={FPS}")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-t", f"{WINDOW_S:.3f}",
         "-i", str(src), "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-c:a", "aac", "-ar", "48000", str(out)], capture_output=True)
    if r.returncode != 0 or not out.exists():
        out.unlink(missing_ok=True)
        return False
    return True


def make_window(src: Path, start_s: float, out: Path, landmarker_model) -> bool:
    """face window of src starting at start_s -> out (mp4); False when no face is found"""
    bbox = face_bbox(src, start_s, landmarker_model)
    return bbox is not None and crop_window(src, start_s, bbox, out)
