"""Label scorer of the L trainer: it labels the training targets ("Pick:" letter / closest reading) on the fly.
Part of the training recipe; the evaluation scorer is rml/eval/.

Gap scorer.

Any hypothesis is a transcript of the WHOLE window (a lip reader's text, or
whisper's reading of an audio track: generated, clean, or muted). It is
aligned to the reference window words (Levenshtein on whisper-normalized
tokens, same normalizer both sides); the gap region of the alignment spans
from after the last left-context reference token to before the first
right-context reference token, so insertions inside the gap count against it.

Per row: g_acc (fraction of gap tokens aligned to an identical hyp token),
g_exact (every gap token matched, no substitution/insertion in the region),
g_wer (S+D+I in the gap region / gap tokens). Gap-only hypotheses (a
proposer's filling) are scored by splicing them into the reference context
(then only the gap can differ) — marked `spliced`.

Library only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_scoring import norm_tokens  # noqa: E402


def load_meta(paths):
    meta = {}
    for p in paths:
        for c in json.load(open(p))["clips"]:
            meta.setdefault(c["id"], c)
    return meta


def ref_tokens(c):
    """Reference window tokens + gap token span [a, b)."""
    lg = c.get("language", "en")
    words = [w["word"] for w in c["window_words"]]
    k = int(c["n_masked_words"])
    si = c.get("span_start_index")
    if si is None or [w.lower().strip(".,?!") for w in words[si:si + k]] != \
            [w.lower().strip(".,?!") for w in c["masked_words"]]:
        # locate the masked words in the window by exact match
        tgt = [w.lower().strip(".,?!") for w in c["masked_words"]]
        low = [w.lower().strip(".,?!") for w in words]
        si = next((i for i in range(len(low) - k + 1) if low[i:i + k] == tgt), None)
        if si is None:
            return None
    toks, a, b = [], None, None
    for i, w in enumerate(words):
        if i == si:
            a = len(toks)
        toks += norm_tokens(w, lg)
        if i == si + k - 1:
            b = len(toks)
    if a is None or b is None or b <= a:
        return None
    return toks, a, b


def align(ref, hyp):
    """Levenshtein alignment; returns list of (op, ri, hi)."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("M" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("D", i - 1, None)); i -= 1
        else:
            ops.append(("I", None, j - 1)); j -= 1
    return ops[::-1]


def score_full(c, hyp_text):
    rt = ref_tokens(c)
    if rt is None:
        return None
    ref, a, b = rt
    hyp = norm_tokens(hyp_text or "", c.get("language", "en"))
    ops = align(ref, hyp)
    # region = ops strictly after the last left-context ref op and before
    # the first right-context ref op
    idx = [t for t, (op, ri, _) in enumerate(ops) if ri is not None]
    left = max((t for t in idx if ops[t][1] < a), default=-1)
    right = min((t for t in idx if ops[t][1] >= b), default=len(ops))
    region = ops[left + 1:right]
    hits = sum(1 for op, ri, _ in region if op == "M" and ri is not None and a <= ri < b)
    errs = sum(1 for op, _, _ in region if op in ("S", "D", "I"))
    n = b - a
    return {"g_acc": hits / n, "g_exact": float(hits == n and errs == 0),
            "g_wer": errs / n, "gap_hyp": " ".join(hyp[j] for op, _, j in region if j is not None)}


def score_gap_only(c, cand, context=None):
    """A gap filling is scored as the system would emit it: spliced into the
    CONTEXT THE SYSTEM WAS GIVEN (the whisper transcript with ___), then the
    whole string is aligned to the reference like any full-window arm. (The
    whisper context can drop words next to the gap — a filling that restores
    them is right, and splicing into the reference would count them as
    insertions.) Without a context: splice into the reference (legacy)."""
    rt = ref_tokens(c)
    if rt is None:
        return None
    ref, a, b = rt
    if context and "___" in context:
        left, _, right = context.partition("___")
        return score_full(c, f"{left} {cand or ''} {right}")
    hyp = ref[:a] + norm_tokens(cand or "", c.get("language", "en")) + ref[b:]
    return _score_tokens(c, ref, a, b, hyp)


def _score_tokens(c, ref, a, b, hyp):
    ops = align(ref, hyp)
    idx = [t for t, (op, ri, _) in enumerate(ops) if ri is not None]
    left = max((t for t in idx if ops[t][1] < a), default=-1)
    right = min((t for t in idx if ops[t][1] >= b), default=len(ops))
    region = ops[left + 1:right]
    hits = sum(1 for op, ri, _ in region if op == "M" and ri is not None and a <= ri < b)
    errs = sum(1 for op, _, _ in region if op in ("S", "D", "I"))
    n = b - a
    return {"g_acc": hits / n, "g_exact": float(hits == n and errs == 0), "g_wer": errs / n,
            "gap_hyp": " ".join(hyp[j] for op, _, j in region if j is not None)}
