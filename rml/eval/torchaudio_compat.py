"""Monkey-patch torchaudio 2.11+ to restore APIs removed from the top-level namespace.

pyannote.audio (used by whisperx) calls torchaudio.info(), torchaudio.AudioMetaData,
and torchaudio.list_audio_backends(), all of which were removed in torchaudio 2.11.
Import this module before importing whisperx.
"""
import collections

import torchaudio

if not hasattr(torchaudio, "AudioMetaData"):
    torchaudio.AudioMetaData = collections.namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
    )

if not hasattr(torchaudio, "info"):
    import soundfile as sf

    def _info(path, backend=None):
        if hasattr(path, "read"):
            # file-like: fall back to load
            wav, sr = torchaudio.load(path)
            return torchaudio.AudioMetaData(sr, wav.shape[-1], wav.shape[0], 16, "PCM_S")
        i = sf.info(str(path))
        return torchaudio.AudioMetaData(
            sample_rate=i.samplerate,
            num_frames=i.frames,
            num_channels=i.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )

    torchaudio.info = _info

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
