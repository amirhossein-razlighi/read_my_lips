"""Noisy inputs of the noisy-speech sets (Tab. 1, 3, 4): each window x five conditions (babble -10 / -5 dB, seen and
unseen sound effects -5 dB, a competing talker -5 dB), noise from the TEST split of the noise bank only, mixed over
the whole window; the competing talker is another window of a different video of the same set. Deterministic per
sample (random.Random(f"0:{sample id}")). Writes <root>/noisy_audio_latents/<id>.pt (encoded from the mix at the VAE
rate) and <root>/noisy_wavs_16k/<id>.wav, <root>/clean_wavs_16k/<id>.wav.
"""
from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch

from rml.data.audio_video import audio_vae_and_processor, encode_waveform, load_audio_window, resample
from rml.data.noise import Bank, finish, mix, noise_for

CONDITIONS = [("babble", -10), ("babble", -5), ("sfx_seen", -5), ("sfx_unseen", -5), ("speech", -5)]


def build(root: Path, rows, bank_dir, audio_vae_path, seed: int = 0, device=None) -> None:
    """rows: the root's denoise samples ({id, clip, base_id, noise, snr_db, source_cut})"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for d in ("noisy_audio_latents", "noisy_wavs_16k", "clean_wavs_16k"):
        (root / d).mkdir(parents=True, exist_ok=True)
    vae, proc = audio_vae_and_processor(audio_vae_path, device)
    sr = vae.sample_rate
    bank = Bank(bank_dir, "test")

    def load(clip, s):
        return load_audio_window(clip, 0.0, s)
    first = {}
    for r in sorted(rows, key=lambda r: r["source_cut"]):
        first.setdefault(r["base_id"], r)
    speech_pool = [(str(root / r["clip"]), w.split("_NA_")[0]) for w, r in first.items()]
    for r in rows:
        sid = r["id"]
        dst = root / "noisy_audio_latents" / f"{sid}.pt"
        if dst.exists():
            continue
        rng = random.Random(f"{seed}:{sid}")
        clean = load(str(root / r["clip"]), sr)
        noisy = finish(mix(clean, noise_for(r["noise"], len(clean), sr, bank, rng, speech_pool,
                                            r["source_cut"].split("_NA_")[0], load), r["snr_db"]))
        torch.save(encode_waveform(vae, proc, noisy, sr), dst)
        sf.write(root / "clean_wavs_16k" / f"{sid}.wav", resample(clean, sr, 16000), 16000)
        sf.write(root / "noisy_wavs_16k" / f"{sid}.wav", resample(noisy, sr, 16000), 16000)
