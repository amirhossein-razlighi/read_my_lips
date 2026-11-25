#!/usr/bin/env python
"""Noise bank for the multi-task (inpaint + denoise) LoRA: varied noise TYPES, split by source so evaluation can
use unseen noise recordings and wholly unseen noise types. Packs every (type, split) into ONE float32 array at
16 kHz (scratch file quota), plus a JSON index of the source segments inside it.

  babble       DEMAND PCAFETER cafe babble: channels 01-12 -> train, 13-16 -> test
  sfx_seen     VGGSound environmental sounds, 9 classes, the dataset's own train/test clip split
  sfx_unseen   3 further VGGSound classes, TEST ONLY (never mixed into training): a held-out noise type
  colored      white / pink / brown noise (synthetic, generated per split with different seeds)
  speech       NOT stored here: the competing speaker is another window's clean speech, drawn at mixing time
               (different source video, same split) by the denoise data builder.

Output: <out>/<type>_<split>.npy (float32, 16 kHz, loudness-normalised segments concatenated) and
<out>/index.json ({type: {split: [{"src", "start", "len"}...]}}). CPU only; resumable per file.
  python rml/training/build_noise_bank.py --out <noise_bank> --pcafeter <DEMAND>/PCAFETER --sfx-manifest <sfx>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
UNSEEN = ("church_bell", "sneezing", "train_horn")     # held-out noise TYPES: test only


def norm(y: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    return (y / r * 0.05).astype(np.float32)               # common RMS; the mixer rescales to the target SNR anyway


def decode(path: Path) -> np.ndarray | None:
    p = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    y = np.frombuffer(p.stdout, dtype=np.float32)
    return y if np.sqrt(np.mean(y ** 2)) > 1e-4 else None   # drop silent clips


def colored(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    shape = 1 / np.sqrt(f) if kind == "pink" else 1 / f
    return np.fft.irfft(np.fft.rfft(w) * shape, n).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pcafeter", type=Path, required=True,
                    help="DEMAND PCAFETER dir with ch01.wav .. ch16.wav (the 16-channel cafeteria recording)")
    ap.add_argument("--sfx-manifest", type=Path, required=True, help="manifest.json of fetch_vggsound_sfx.py")
    a = ap.parse_args()
    PCAFETER, SFX_MANIFEST = a.pcafeter, a.sfx_manifest
    a.out.mkdir(parents=True, exist_ok=True)
    index: dict = {}

    def save(typ: str, split: str, segs: list[tuple[str, np.ndarray]]) -> None:
        cat, idx, pos = [], [], 0
        for src, y in segs:
            y = norm(y)
            cat.append(y)
            idx.append({"src": src, "start": pos, "len": len(y)})
            pos += len(y)
        np.save(a.out / f"{typ}_{split}.npy", np.concatenate(cat))
        index.setdefault(typ, {})[split] = idx
        print(f"{typ}/{split}: {len(segs)} segments, {pos / SR / 60:.1f} min", flush=True)

    import librosa
    for split, chans in (("train", range(1, 13)), ("test", range(13, 17))):
        segs = []
        for c in chans:
            y, sr = sf.read(PCAFETER / f"ch{c:02d}.wav", dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            segs.append((f"PCAFETER/ch{c:02d}", librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y))
        save("babble", split, segs)

    man = [m for m in json.loads(SFX_MANIFEST.read_text()) if m.get("status") == "ok"]
    for typ, split, sel in (("sfx_seen", "train", lambda m: m["class"] not in UNSEEN and m["split"] == "train"),
                            ("sfx_seen", "test", lambda m: m["class"] not in UNSEEN and m["split"] == "test"),
                            ("sfx_unseen", "test", lambda m: m["class"] in UNSEEN)):
        segs = []
        for m in sorted(filter(sel, man), key=lambda m: m["file"]):
            y = decode(Path(m["file"]))
            if y is not None:
                segs.append((f"{m['class']}/{Path(m['file']).name}", y))
        save(typ, split, segs)

    for split, seed in (("train", 1), ("test", 2)):
        rng = np.random.default_rng(seed)
        save("colored", split, [(f"{k}/{i}", colored(k, SR * 30, rng)) for k in ("white", "pink", "brown")
                                for i in range(4)])
    (a.out / "index.json").write_text(json.dumps({"sr": SR, "unseen_sfx_classes": list(UNSEEN), "types": index},
                                                 indent=1))
    print("NOISE_BANK_DONE", a.out)


if __name__ == "__main__":
    main()
