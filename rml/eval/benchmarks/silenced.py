"""Silenced cuts of a 4.8 s window (Tab. 1, Appendix A): one cut per (window, k), the span position chosen per k so
that the reference is exact:
  * neither edge word is a stop word or filler (span_rules.edge_ok), whose WhisperX timings are least reliable;
  * at least one context word on each side inside the window;
  * the mute (span words +- 80 ms, clamped 20 ms away from both neighbours) covers every span word fully;
  * edge words last >= 80 ms.
Among valid positions: the one containing the window's anchor word (target_index) and closest to it, then closest to
the window centre. The released manifests (benchmarks/silenced_*.jsonl) are the result after the human review of the
references; this module documents and reproduces the construction.
"""
from __future__ import annotations

import re

from rml.eval.span_rules import edge_ok

PAD, MARGIN, WIN_S, MIN_EDGE_S, COVER_TOL = 0.08, 0.02, 4.8, 0.08, 0.03


def norm(w: str) -> str:
    return re.sub(r"[^\w']+", "", w.lower())


def mask_for(ws, start, k):
    span = ws[start:start + k]
    s = float(span[0]["start_s"]) - PAD
    e = float(span[-1]["end_s"]) + PAD
    s = max(s, float(ws[start - 1]["end_s"]) + MARGIN)
    e = min(e, float(ws[start + k]["start_s"]) - MARGIN)
    return max(0.0, s), min(WIN_S, e)


def pick(r, k):
    """(span start index, mute start, mute end) of the k-word cut of window row r, or None"""
    ws = r.get("window_words") or []
    toks = str(r.get("sentence", "")).split()
    if len(ws) != len(toks) or len(ws) < k + 2:
        return None
    if any(norm(w.get("word", "")) != norm(t) for w, t in zip(ws, toks)):
        return None
    ti = int(r.get("target_index", len(ws) // 2))
    cands = []
    for st in range(1, len(ws) - k):
        span = ws[st:st + k]
        if not edge_ok([w["word"] for w in span], r["language"]):
            continue
        if min(float(span[0]["end_s"]) - float(span[0]["start_s"]),
               float(span[-1]["end_s"]) - float(span[-1]["start_s"])) < MIN_EDGE_S:
            continue
        s, e = mask_for(ws, st, k)
        if s > float(span[0]["start_s"]) + COVER_TOL or e < float(span[-1]["end_s"]) - COVER_TOL or e - s < 0.15:
            continue
        has_anchor = st <= ti < st + k
        cands.append((not has_anchor, abs(st + (k - 1) / 2 - ti), abs(st + (k - 1) / 2 - (len(ws) - 1) / 2), st, s, e))
    if not cands:
        return None
    _, _, _, st, s, e = min(cands)
    return st, s, e


def cuts(window_row, ks=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    """the cut rows of one window row ({id, sentence, window_words, language, target_index})"""
    out = []
    for k in ks:
        p = pick(window_row, k)
        if p is None:
            continue
        st, s, e = p
        ws = window_row["window_words"]
        out.append({"id": f"{window_row['id']}__k{k}", "base_id": window_row["id"],
                    "mask_segments_s": [[round(s, 3), round(e, 3)]], "masked_words": [x["word"] for x in ws[st:st + k]],
                    "n_masked_words": k, "span_start_index": st})
    return out
