"""Window I/O shared by every stage: 4.8 s windows at 25 fps (121 frames, LTX convention) and the audio-latent
encoding. Video frames are scaled to 384 x 224 before the video VAE, as in training."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TARGET_FPS = 25
TARGET_FRAMES = 121
TARGET_W = 384
TARGET_H = 224
CLIP_SECONDS = (TARGET_FRAMES - 1) / TARGET_FPS  # 4.8


def extract_video_frames(video_path: Path, start_s: float) -> torch.Tensor | None:
    """TARGET_FRAMES frames at TARGET_FPS from start_s as a [C, F, H, W] uint8 tensor at (TARGET_H, TARGET_W)."""
    with tempfile.TemporaryDirectory() as td:
        out_mp4 = Path(td) / "clip.mp4"
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start_s:.3f}",
                "-i", str(video_path),
                "-t", f"{CLIP_SECONDS + 0.1:.3f}",
                "-vf", f"scale={TARGET_W}:{TARGET_H},fps={TARGET_FPS}",
                "-an",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                str(out_mp4),
            ],
            capture_output=True,
        )
        if r.returncode != 0:
            return None
        from torchvision.io import read_video
        frames, _, _ = read_video(str(out_mp4), pts_unit="sec", output_format="TCHW")
        if frames.shape[0] == 0:
            return None
        if frames.shape[0] < TARGET_FRAMES:
            # 4.80 s at 25 fps often decodes as 120 frames, while the 121-frame convention covers 0.00..4.80
            # inclusive: hold the last frame for the missing step
            missing = TARGET_FRAMES - frames.shape[0]
            frames = torch.cat([frames, frames[-1:].repeat(missing, 1, 1, 1)], dim=0)
        frames = frames[:TARGET_FRAMES]
        return frames.permute(1, 0, 2, 3).contiguous()


def load_audio_window(video_path: Path, start_s: float, sr_target: int) -> np.ndarray | None:
    """Mono PCM at sr_target via ffmpeg: CLIP_SECONDS of float32 samples (zero-padded)."""
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
    except Exception:  # noqa: BLE001
        return None
    finally:
        tmp.unlink(missing_ok=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def apply_mute_mask(wav: np.ndarray, sample_rate: int, segments, fade_ms: float) -> np.ndarray:
    """Zero the segments [(start_s, end_s)] with fade_ms linear fades outside them."""
    masked = wav.copy()
    fade = max(1, int(round(sample_rate * fade_ms / 1000.0)))
    for start_s, end_s in segments:
        start = max(0, int(round(start_s * sample_rate)))
        end = min(masked.shape[0], int(round(end_s * sample_rate)))
        if end <= start:
            continue
        local_fade = min(fade, max(1, (end - start) // 2))
        if start > 0 and local_fade > 0:
            fade_start = max(0, start - local_fade)
            ramp_down = np.linspace(1.0, 0.0, start - fade_start, endpoint=False, dtype=np.float32)
            masked[fade_start:start] = wav[fade_start:start] * ramp_down
        masked[start:end] = 0.0
        if end < masked.shape[0] and local_fade > 0:
            fade_end = min(masked.shape[0], end + local_fade)
            ramp_up = np.linspace(0.0, 1.0, fade_end - end, endpoint=False, dtype=np.float32)
            masked[end:fade_end] = wav[end:fade_end] * ramp_up
    return masked


def encode_video(vae, frames: torch.Tensor) -> dict:
    """frames: [C, F, H, W] uint8 -> video latent dict"""
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    x = frames.to(device=device, dtype=dtype) / 127.5 - 1.0
    x = x.unsqueeze(0)
    with torch.inference_mode():
        latents = vae(x)
    if hasattr(latents, "sample"):
        latents = latents.sample()
    latents = latents.squeeze(0)
    return {
        "latents": latents.cpu().contiguous(),
        "num_frames": latents.shape[1],
        "height": latents.shape[2],
        "width": latents.shape[3],
        "fps": TARGET_FPS,
    }


def encode_waveform(audio_vae, audio_processor, wave: np.ndarray, sr: int) -> dict:
    """mono waveform at sr -> audio latent dict"""
    waveform = torch.from_numpy(wave).unsqueeze(0)
    device = next(audio_vae.parameters()).device
    dtype = next(audio_vae.parameters()).dtype
    waveform = waveform.unsqueeze(0).to(device=device, dtype=dtype)
    from ltx_core.types import Audio
    mel = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=sr)).to(dtype=dtype)
    if mel.shape[1] == 1:
        mel = torch.cat([mel, mel], dim=1)
    with torch.inference_mode():
        latents = audio_vae(mel)
    _, _, T, F = latents.shape
    return {
        "latents": latents.squeeze(0).cpu().contiguous(),
        "num_time_steps": T,
        "frequency_bins": F,
        "duration": float(CLIP_SECONDS),
    }


def audio_vae_and_processor(audio_vae_path, device):
    """the audio VAE encoder + mel processor (2.5 audio VAE, bit-identical to 2.3)"""
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_trainer.model_loader import load_audio_vae_encoder
    vae = load_audio_vae_encoder(audio_vae_path, device=device, dtype=torch.bfloat16)
    proc = AudioProcessor(target_sample_rate=vae.sample_rate, mel_bins=vae.mel_bins,
                          mel_hop_length=vae.mel_hop_length, n_fft=vae.n_fft).to(device)
    return vae, proc


def resample(y: np.ndarray, a: int, b: int) -> np.ndarray:
    if a == b:
        return y
    import librosa
    return librosa.resample(y, orig_sr=a, target_sr=b)
