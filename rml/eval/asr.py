"""Whisper large-v3 (WhisperX, faster-whisper backend) transcription, offline-safe.

The same model and call score every audio output (generated, clean or damaged) and build the audible context of a
silenced span. Models load lazily on the first call. Model directory: RML_WHISPERX_DIR (default
checkpoints/whisperx, holding faster-whisper-large-v3/ and the wav2vec2 alignment model); the Silero VAD is read
from the torch hub cache (TORCH_HOME).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

WHISPER_MODEL_ROOT = Path(os.environ.get("RML_WHISPERX_DIR", "checkpoints/whisperx"))
WHISPER_MODEL_PATH = Path(os.environ.get("RML_WHISPER_MODEL_PATH", str(WHISPER_MODEL_ROOT / "faster-whisper-large-v3")))

_ASR_MODEL = None
_ALIGN_MODEL = None


def _resolve_whisper_model(model_size: str = "large-v3") -> Path:
    candidates = [
        Path(model_size) if Path(model_size).exists() else None,
        WHISPER_MODEL_PATH,
        WHISPER_MODEL_ROOT / model_size,
        WHISPER_MODEL_ROOT / f"faster-whisper-{model_size}",
    ]
    for c in candidates:
        if c and c.exists() and c.is_dir() and (c / "config.json").exists():
            return c
    raise FileNotFoundError(
        "No faster-whisper model found. Checked:\n" + "\n".join(f"  {c}" for c in candidates if c)
    )


def load_asr_model(model_size: str = "large-v3", device: str | None = None):
    """Load WhisperX once (offline) and cache it."""
    global _ASR_MODEL
    if device is None and os.environ.get("WER_ASR_DEVICE"):
        device = os.environ["WER_ASR_DEVICE"]
    if _ASR_MODEL is not None:
        return _ASR_MODEL

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    # Silero VAD from the local torch hub cache (no network)
    _orig_hub_load = torch.hub.load

    def _load_hub_offline(repo_or_dir, *args, **kwargs):
        if repo_or_dir == "snakers4/silero-vad":
            hub_dir = Path(torch.hub.get_dir())
            found = sorted(hub_dir.glob("snakers4_silero-vad_*"))
            if found:
                kwargs = dict(kwargs)
                kwargs["source"] = "local"
                return _orig_hub_load(str(found[-1]), *args, **kwargs)
            raise FileNotFoundError(
                "Silero VAD is required in offline mode but was not found under "
                f"{hub_dir}. Set TORCH_HOME to the cache root (see docs/INSTALL.md)."
            )
        return _orig_hub_load(repo_or_dir, *args, **kwargs)

    torch.hub.load = _load_hub_offline

    import rml.eval.torchaudio_compat  # noqa: F401, I001  (must precede whisperx)
    import whisperx  # isort: skip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    whisper_path = _resolve_whisper_model(model_size)
    from whisperx.vads import Silero as SileroVad

    silero_vad = SileroVad(vad_onset=0.5, vad_offset=0.363, chunk_size=30)
    _ASR_MODEL = whisperx.load_model(
        str(whisper_path),
        device=device,
        compute_type=compute_type,
        download_root=str(WHISPER_MODEL_ROOT),
        vad_model=silero_vad,
    )
    print(f"[asr] WhisperX loaded from {whisper_path} (device={device})", flush=True)
    return _ASR_MODEL


def transcribe_16k(audio16k: np.ndarray, batch_size: int = 8, language: str = "en") -> str:
    """Transcribe a mono 16 kHz float32 array."""
    model = load_asr_model()
    audio16k = np.ascontiguousarray(audio16k, dtype=np.float32)
    if audio16k.size < 160:  # < 10 ms
        return ""
    try:
        result = model.transcribe(audio16k, batch_size=batch_size, language=language)
    except Exception as e:
        print(f"[asr] transcription failed: {e}", flush=True)
        return ""
    return " ".join(s["text"].strip() for s in result.get("segments", []) if s.get("text", "").strip())


def load_word_align_model(device: str | None = None, language: str = "en"):
    """WhisperX forced-alignment model, once per language (offline)."""
    global _ALIGN_MODEL
    if _ALIGN_MODEL is None:
        _ALIGN_MODEL = {}
    if language in _ALIGN_MODEL:
        return _ALIGN_MODEL[language]
    load_asr_model(device=device)  # applies the offline shims first
    import torch
    import whisperx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _ALIGN_MODEL[language] = whisperx.load_align_model(
        language_code=language, device=device, model_dir=str(WHISPER_MODEL_ROOT)
    )
    return _ALIGN_MODEL[language]


def transcribe_words_16k(audio16k: np.ndarray, batch_size: int = 8, language: str = "en") -> list[dict]:
    """Transcribe in one pass and force-align word timestamps: [{word, start_s, end_s, score}] in order. Words the
    aligner could not time carry start_s = end_s = None. Raises on failure so callers can fall back explicitly."""
    import torch
    import whisperx

    model = load_asr_model()
    audio16k = np.ascontiguousarray(audio16k, dtype=np.float32)
    if audio16k.size < 160:
        return []
    result = model.transcribe(audio16k, batch_size=batch_size, language=language)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_model, align_meta = load_word_align_model(device, language=language)
    aligned = whisperx.align(result["segments"], align_model, align_meta,
                             audio16k, device, return_char_alignments=False)
    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": str(w.get("word", "")).strip(),
                "start_s": float(w["start"]) if "start" in w else None,
                "end_s": float(w["end"]) if "end" in w else None,
                "score": float(w.get("score", 0.0)),
            })
    return [w for w in words if w["word"]]


def transcribe_file(path, language: str = "en") -> str:
    """transcribe_16k on a wav file of any rate (resampled to 16 kHz with librosa), as the scorer of every render"""
    import librosa
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return transcribe_16k(librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y, language=language)


def transcribe_file_words(path, language: str = "en") -> str:
    """the aligned words of a wav file joined (resampled to 16 kHz with torchaudio), the span scorer's transcript of
    the silenced renders; Whisper's segment text if the alignment fails"""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != 16000:
        y = AF.resample(torch.from_numpy(y), sr, 16000).numpy()
    y = np.ascontiguousarray(y)
    try:
        return " ".join(w["word"] for w in transcribe_words_16k(y, language=language))
    except Exception as e:  # noqa: BLE001
        print(f"[asr] alignment failed ({e}): segment text", flush=True)
        return transcribe_16k(y, language=language)
