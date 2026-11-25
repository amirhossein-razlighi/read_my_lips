"""Noise mixing of the noisy-speech sets: a noise bank (index.json + <type>_<split>.npy, built from DEMAND, MUSAN
and the other sources of docs/EVALUATION.md), mixed over the whole window at a fixed SNR; the competing speaker is
another window of a different video."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from rml.data.audio_video import resample


class Bank:
    def __init__(self, bank_dir: Path, split: str):
        bank_dir = Path(bank_dir)
        idx = json.loads((bank_dir / "index.json").read_text())
        self.sr = idx["sr"]
        self.arr, self.segs = {}, {}
        for typ, splits in idx["types"].items():
            if split in splits:
                self.arr[typ] = np.load(bank_dir / f"{typ}_{split}.npy", mmap_mode="r")
                self.segs[typ] = splits[split]

    def draw(self, typ: str, n: int, rng: random.Random) -> np.ndarray:
        segs = self.segs[typ]
        s = rng.choices(segs, weights=[x["len"] for x in segs])[0]
        y = np.asarray(self.arr[typ][s["start"]:s["start"] + s["len"]], dtype=np.float32)
        if len(y) < n:
            y = np.tile(y, n // len(y) + 1)
        o = rng.randint(0, len(y) - n)
        return y[o:o + n].copy()


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)) + 1e-9)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    return clean + noise * (rms(clean) / (10 ** (snr_db / 20.0)) / rms(noise))


def finish(y: np.ndarray) -> np.ndarray:
    p = float(np.abs(y).max())
    return (y / p * 0.95 if p > 0.99 else y).astype(np.float32)


def noise_for(typ, n, sr, bank, rng, speech_pool, own_vid, load):
    """n samples at the VAE rate sr of noise type typ; speech = another video's clean window."""
    if typ == "speech":
        while True:
            clip, vid = rng.choice(speech_pool)
            if vid != own_vid:
                break
        y = load(clip, sr)
        return np.tile(y, n // len(y) + 1)[:n] if len(y) < n else y[:n]
    m = int(np.ceil(n * bank.sr / sr)) + 1
    return resample(bank.draw(typ, m, rng), bank.sr, sr)[:n]
