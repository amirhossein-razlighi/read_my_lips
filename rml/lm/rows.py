"""Rows of L and the use of its answers (Sec. 4.2).

silenced input ('<inpaint>'): context = the audible context 'left ___ right'; candidates = the lip reader's 10-best;
    f0/f1 = the gap's video frames with a 2-frame margin (its duration is shown).
noisy input or a muted track ('<denoise>'): context = '<denoise> ' + Whisper on the input; candidates = the lip
    reader's readings interleaved with Auto-AVSR AV's n-best, Whisper's line after the first pair, distinct, at most
    10; f0 0, f1 120 (the whole window).
"""
from __future__ import annotations

import re

import numpy as np

FPS = 25
FRAMES = 120


def clean_txt(s):
    return " ".join(re.sub(r"[^\w' ]+", " ", (s or "").lower()).split())


def inpaint_row(sid, context, gap_s, frame_margin=2, language="en"):
    g0, g1 = gap_s
    f0 = max(0, int(np.floor(g0 * FPS)) - frame_margin)
    f1 = min(121, int(np.ceil(g1 * FPS)) + frame_margin)
    ctx = re.sub(r"_{4,}", " ", context).strip()
    ctx = re.sub(r"\s{2,}", " ", ctx)
    return {"id": sid, "f0": f0, "f1": f1, "context": ctx, "language": language, "task": "inpaint"}


def merged(reader, av, whisper, cap=10):
    """reader 1, AV 1, Whisper, reader 2, AV 2, ... cleaned, distinct, at most cap"""
    order = []
    r, a = list(reader), list(av)
    for i in range(max(len(r), len(a), 1)):
        if i < len(r):
            order.append(r[i])
        if i < len(a):
            order.append(a[i])
        if i == 0 and whisper:
            order.append(whisper)
    out, seen = [], set()
    for h in order:
        h = clean_txt(h)
        if h and h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= cap:
            break
    return out


def denoise_row(sid, whisper_text, language="en"):
    return {"id": sid, "f0": 0, "f1": FRAMES, "context": f"<denoise> {clean_txt(whisper_text)}".strip(),
            "language": language, "task": "denoise"}


def inpaint_answer(dec_row, context):
    """L's full-window transcript for a silenced input (a gap-type answer is spliced into its context)"""
    fl = dec_row.get("final_list") or []
    t = fl[0] if fl else {"type": "full", "text": ""}
    txt = t["text"] if t["type"] == "full" else context.replace("___", t["text"], 1)
    return " ".join(txt.split())


def denoise_answer(dec_row, cands):
    """L's transcript in denoise mode, with the deployment guard: an empty or runaway answer (longer than 1.3 x the
    picked reading + 2 words, or a stray viseme line) falls back to the picked reading"""
    ans = (dec_row.get("final_list") or [{"text": ""}])[0]["text"]
    p = dec_row.get("pick")
    pk = cands[SW.index(p)] if p in SW[:len(cands)] else (cands[0] if cands else "")
    bad = "lip visemes" in ans.lower() or not ans.strip() or len(ans.split()) > 1.3 * max(1, len(pk.split())) + 2
    return clean_txt(pk if bad else ans), bool(bad)


SW = "ABCDEFGHIJ"
