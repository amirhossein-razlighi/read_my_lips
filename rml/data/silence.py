"""Find the silenced regions of an input recording from the audio alone.

A silenced region is a stretch whose level stays below a floor far under any real recording's noise (default -70 dB
relative to the clip's peak), for at least min_dur_s. Frames of 5 ms are tested by RMS, so codec noise from a lossy
round trip (AAC) does not break a region into pieces; the edges are then refined to the sample.

  regions = find_silenced_regions(wav, sr)            # [(start_s, end_s), ...], longest first
  gap = pick_gap(regions, len(wav) / sr)              # the silenced span: speech on both sides, longest
A recording can also contain silence of its own (a noise gate, a scene cut, digital padding at the end). The span to
restore has speech on both sides, so pick_gap prefers regions that touch neither end of the window.
"""
import numpy as np


def find_silenced_regions(wav, sr, floor_db=-70.0, min_dur_s=0.05, frame_s=0.005):
    """Regions of the recording that are silent at a level no microphone produces: frame RMS below
    floor_db relative to the clip's peak (and below 1 LSB of 16-bit audio for digital zeros), lasting at least
    min_dur_s. Returns [(start_s, end_s), ...] sorted by duration, longest first."""
    x = np.asarray(wav, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return [(0.0, len(x) / sr)] if len(x) else []
    thr = max(peak * 10 ** (floor_db / 20), 1.0 / 32768)
    hop = max(1, int(round(frame_s * sr)))
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    quiet = np.concatenate([[False], rms <= thr, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    out = []
    for f0, f1 in zip(edges[::2], edges[1::2]):
        a, b = f0 * hop, f1 * hop if f1 < n else len(x)
        while a > 0 and abs(x[a - 1]) <= thr:                 # refine the edges to the sample
            a -= 1
        while b < len(x) and abs(x[b]) <= thr:
            b += 1
        if (b - a) / sr >= min_dur_s:
            out.append((a / sr, b / sr))
    return sorted(out, key=lambda r: r[0] - r[1])


def pick_gap(regions, dur_s, edge_s=0.02):
    """the silenced span among the detected regions: the longest one with audio on both sides (touching neither
    end of the window); the longest region overall if every region touches an end; None if there is none"""
    inner = [r for r in regions if r[0] > edge_s and r[1] < dur_s - edge_s]
    return max(inner or regions, key=lambda r: r[1] - r[0]) if regions else None
