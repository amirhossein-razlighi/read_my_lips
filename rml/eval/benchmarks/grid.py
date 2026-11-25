"""The condition grid of Fig. 4 (Sec. 5.3, Appendix C): every test window under babble at -5, -10 or -20 dB, or
silence, over one word, four words (the window's reviewed k = 1 and k = 4 cuts) or the whole clip, plus the clean
clip. Babble (test split of the noise bank, deterministic per sample) is mixed only inside the span for one and four
words (SNR against the speech inside the span, 8 ms fades) and over the whole window for the whole clip. Silenced
cells are inpainting samples (the cut's own mute); the whole-clip silence cell is the window's muted twin.
Writes noisy_audio_latents/ and noisy_wavs_16k/ for the babble and clean cells.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rml.data.audio_video import audio_vae_and_processor, encode_waveform, load_audio_window, resample
from rml.data.noise import Bank, finish, mix, rms

FADE_S = 0.008


def build(root: Path, rows, bank_dir, audio_vae_path, device=None) -> None:
    """rows: the grid's denoise samples ({id, clip, span, noise, noise_segments_s})"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for d in ("noisy_audio_latents", "noisy_wavs_16k"):
        (root / d).mkdir(parents=True, exist_ok=True)
    vae, proc = audio_vae_and_processor(audio_vae_path, device)
    sr = vae.sample_rate
    bank = Bank(bank_dir, "test")
    for r in rows:
        sid, n, span = r["id"], r["noise"], r["span"]
        dst = root / "noisy_audio_latents" / f"{sid}.pt"
        if dst.exists():
            continue
        clean = load_audio_window(root / r["clip"], 0.0, sr)
        if n == "clean":
            y = clean.copy()
        else:
            snr = -float(n.split("-", 1)[1])
            rr = random.Random(f"0:{sid}")
            m = int(np.ceil(len(clean) * bank.sr / sr)) + 1
            noise = resample(bank.draw("babble", m, rr), bank.sr, sr)[:len(clean)]
            if span == "full":
                y = finish(mix(clean, noise, snr))
            else:
                a, b = (int(t * sr) for t in r["noise_segments_s"][0])
                a, b = max(0, a), min(len(clean), b)
                f = int(FADE_S * sr)
                env = np.zeros(len(clean), np.float32)
                env[a:b] = 1.0
                if a - f >= 0:
                    env[a - f:a] = np.linspace(0, 1, f, dtype=np.float32)
                if b + f <= len(env):
                    env[b:b + f] = np.linspace(1, 0, f, dtype=np.float32)
                g = rms(clean[a:b]) / (10 ** (snr / 20.0)) / rms(noise[a:b])
                y = finish(clean + noise * g * env)
        torch.save(encode_waveform(vae, proc, y, sr), dst)
        sf.write(root / "noisy_wavs_16k" / f"{sid}.wav", resample(y, sr, 16000), 16000)
