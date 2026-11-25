"""Audio-latent helpers used by build_multitask_restoration.py (load_audio_window, encode_waveform and the window
constants they use)."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# window spec (4.8 s at 25 fps, 121 frames)
TARGET_FPS = 25
TARGET_FRAMES = 121
CLIP_SECONDS = (TARGET_FRAMES - 1) / TARGET_FPS  # 4.8


def load_audio_window(video_path: Path, start_s: float, sr_target: int) -> np.ndarray | None:
    """Extract mono PCM at sr_target Hz via ffmpeg, return CLIP_SECONDS of float32 samples."""
    target_len = int(sr_target * CLIP_SECONDS)
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start_s:.3f}",
                "-i", str(video_path),
                "-t", f"{CLIP_SECONDS + 0.05:.3f}",
                "-vn", "-ac", "1", "-ar", str(sr_target),
                "-c:a", "pcm_s16le", str(tmp),
            ],
            capture_output=True,
        )
        if r.returncode != 0 or not tmp.exists():
            return None
        y, _ = sf.read(str(tmp), dtype="float32", always_2d=False)
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def encode_waveform(audio_vae, audio_processor, wave: np.ndarray, sr: int) -> dict:
    waveform = torch.from_numpy(wave).unsqueeze(0)  # [1, samples]
    device = next(audio_vae.parameters()).device
    dtype = next(audio_vae.parameters()).dtype
    waveform = waveform.unsqueeze(0).to(device=device, dtype=dtype)  # [1, 1, samples]
    from ltx_core.types import Audio
    mel = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sr)).to(dtype=dtype)
    if mel.shape[1] == 1:
        mel = torch.cat([mel, mel], dim=1)
    with torch.inference_mode():
        latents = audio_vae(mel)
    _, _, T, F = latents.shape
    return {
        "latents": latents.squeeze(0).cpu().contiguous(),  # [C, T, F]
        "num_time_steps": T,
        "frequency_bins": F,
        "duration": float(CLIP_SECONDS),
    }
