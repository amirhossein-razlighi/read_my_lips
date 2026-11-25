"""Latents of a root (Sec. 4.2, G): per window the video latents (2.5 video VAE, also the reference stream), per
sample the reference audio latents and the latents of the INPUT track G receives:
  task inpaint   the window audio with mask_segments_s muted (10 ms fades)
  task denoise   the sample's noisy track (noisy_wavs_16k/<id>.wav written by the noisy-set builder, or the window
                 audio itself for a real recording)
  muted          zeros (the 'audio muted' configurations)
and the same input as 16 kHz wav under inputs_16k/ (what Whisper, Auto-AVSR AV and the evidence hear).
Clip latents of one window are encoded once and hard-linked under every sample of that window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rml.data.audio_video import (apply_mute_mask, audio_vae_and_processor, encode_video, encode_waveform,
                                  extract_video_frames, load_audio_window, resample)

DIRS = ("latents", "audio_latents", "noisy_audio_latents", "reference_latents", "inputs_16k")
SR16 = 16_000


def input_wave(root: Path, row: dict, sr: int) -> np.ndarray:
    """the input track of a sample at sr"""
    clip = root / row["clip"]
    wave = load_audio_window(clip, 0.0, sr)
    if wave is None:
        raise RuntimeError(f"{row['id']}: audio extraction failed")
    if row.get("input") == "muted":
        return np.zeros_like(wave)
    if row["task"] == "denoise":
        p = root / "noisy_wavs_16k" / f"{row['id']}.wav"
        if p.exists():
            y, s = sf.read(p, dtype="float32")
            return resample(y if y.ndim == 1 else y.mean(1), s, sr)[: len(wave)]
        return wave
    return apply_mute_mask(wave, sr, [tuple(s) for s in row["mask_segments_s"]], 10.0)


def encode_root(root: Path, video_vae_path, audio_vae_path, device=None) -> None:
    from ltx_trainer.model_loader import load_video_vae_encoder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads((root / "metadata.json").read_text())["clips"]
    for d in DIRS:
        (root / d).mkdir(exist_ok=True)
    video_vae = load_video_vae_encoder(video_vae_path, device=device)
    audio_vae, proc = audio_vae_and_processor(audio_vae_path, device)
    sr = audio_vae.sample_rate
    first = {}
    for row in rows:
        sid, w = row["id"], row["base_id"]
        outs = {d: root / d / f"{sid}.pt" for d in DIRS[:4]}
        wav16 = root / "inputs_16k" / f"{sid}.wav"
        if all(p.exists() for p in outs.values()) and wav16.exists():
            first.setdefault(w, outs)
            continue
        clip = root / row["clip"]
        if w in first:
            for d in ("latents", "audio_latents", "reference_latents"):
                if not outs[d].exists():
                    os.link(first[w][d], outs[d])
        else:
            frames = extract_video_frames(clip, 0.0)
            if frames is None:
                raise RuntimeError(f"{sid}: frame extraction failed")
            v = encode_video(video_vae, frames)
            torch.save(v, outs["latents"])
            torch.save(v, outs["reference_latents"])
            torch.save(encode_waveform(audio_vae, proc, load_audio_window(clip, 0.0, sr), sr), outs["audio_latents"])
            first[w] = outs
        if not outs["noisy_audio_latents"].exists():    # the noisy-set builder writes its own (from the VAE-rate mix)
            y = input_wave(root, row, sr)
            torch.save(encode_waveform(audio_vae, proc, y, sr), outs["noisy_audio_latents"])
        if row.get("input") == "muted":
            y16 = np.zeros(int(SR16 * 4.8), np.float32)
        elif row["task"] == "denoise" and (root / "noisy_wavs_16k" / f"{sid}.wav").exists():
            y16, _ = sf.read(root / "noisy_wavs_16k" / f"{sid}.wav", dtype="float32")
        elif row["task"] == "denoise":
            y16 = load_audio_window(clip, 0.0, SR16)
        else:
            y16 = apply_mute_mask(load_audio_window(clip, 0.0, SR16), SR16, [tuple(s) for s in row["mask_segments_s"]], 10.0)
        sf.write(wav16, y16, SR16)
        print(f"[latents] {sid}", flush=True)
