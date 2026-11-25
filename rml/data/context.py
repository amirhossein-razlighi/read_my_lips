"""The audible context of a silenced span: Whisper large-v3 on the input with the span filled, words split left and
right of the span, written 'left ___ right'.

Whisper on audio with a silent hole drops or invents words at the hole's edges, so the known interval is filled with
low-level Gaussian noise (0.25 x the RMS of the audible audio), the window is transcribed once with word timestamps,
and whatever Whisper heard inside the interval (by word midpoint) is discarded. Leak guard: a word that normalises to
a silenced reference word and lies within 0.35 s of the span is dropped (`forbid`; empty for a real input, where the
silenced words are unknown).
"""
from __future__ import annotations

import re

import numpy as np

SR = 16_000


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower(), flags=re.UNICODE)


def build_context(muted: np.ndarray, s: float, e: float, forbid: set[str], sid: str, language: str = "en") -> str:
    from rml.eval.asr import transcribe_words_16k
    filled = muted.copy()
    i0, i1 = int(s * SR), int(e * SR)
    aud = np.concatenate([muted[:i0], muted[i1:]])
    rms = float(np.sqrt(np.mean(aud ** 2)) + 1e-8)
    # NOTE: hash() of a str is salted per process unless PYTHONHASHSEED is set; the paper's runs used the
    # interpreter default, so this noise (and rarely the context) can differ between runs
    rng = np.random.default_rng(abs(hash(sid)) % (2 ** 32))
    filled[i0:i1] = rng.standard_normal(i1 - i0).astype(np.float32) * 0.25 * rms
    words = transcribe_words_16k(np.ascontiguousarray(filled), language=language)
    left, right = [], []
    side = left
    for w in words:
        timed = w["start_s"] is not None and w["end_s"] is not None
        mid = 0.5 * (w["start_s"] + w["end_s"]) if timed else None
        if norm(w["word"]) in forbid and (not timed or s - 0.35 < mid < e + 0.35):
            continue
        if not timed:
            side.append(w["word"])
            continue
        if mid < s:
            side = left
        elif mid > e:
            side = right
        else:
            continue                       # decoded inside the noise fill
        side.append(w["word"])
    return f"{' '.join(left)} ___ {' '.join(right)}".strip()


def context_or_placeholder(muted, s, e, forbid, sid, language="en"):
    """build_context, degrading to a bare placeholder when Whisper fails (never crash a batch)"""
    try:
        return build_context(muted, s, e, forbid, sid, language=language)
    except Exception as ex:  # noqa: BLE001
        print(f"{sid}: whisper failed ({ex!r}) -> bare placeholder", flush=True)
        return "___"
