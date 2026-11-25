"""Label scorer of the L trainer: it labels the training targets ("Pick:" letter / closest reading) on the fly.
Part of the training recipe; the evaluation scorer is rml/eval/.

Three-tier token scoring.

Motivation: the historical scorer (lowercase + punctuation strip) is blind to
number format ("twenty four seven" vs GT "24 seven" scored 0.5 despite being
fully correct — ltx51-nbest row 890), compounds ("story teller"/"storyteller")
and contractions ("we're"/"we are"). These change token COUNTS, so exact==
and positional accuracy silently break; alignment-based WER is needed.

Tiers (always applied to BOTH ref and hyp — symmetric, fair to baselines too):
  strict — the historical scorer, unchanged (continuity with published numbers)
  norm   — whisper text normalizers: en = EnglishTextNormalizer (spelled
           numbers -> digits, contraction expansion, currency, filler removal;
           ships inside transformers, offline); es/fr/ru = BasicTextNormalizer
           (case/punct/diacritic folding; NOTE: no spelled-number handling
           outside English — whisper emits digits in those languages).
           Equality additionally accepts a no-space join match so
           segmentation-only differences count ("storyteller"=="story teller").
  lem    — Snowball stems (existing; explicitly-lenient upper bound — known
           overmerges like universe/university, keep it labeled as such).

Primary continuous metric: gap_wer (edit-distance WER on normalized tokens).
"""
from __future__ import annotations

import difflib
import re

_EN_NORM = None
_BASIC_NORM = None


def _en():
    global _EN_NORM
    if _EN_NORM is None:
        from transformers.models.whisper.english_normalizer import (
            EnglishTextNormalizer)
        _EN_NORM = EnglishTextNormalizer({})
    return _EN_NORM


def _basic():
    global _BASIC_NORM
    if _BASIC_NORM is None:
        from transformers.models.whisper.english_normalizer import (
            BasicTextNormalizer)
        _BASIC_NORM = BasicTextNormalizer()
    return _BASIC_NORM


def strict_tokens(text: str) -> list[str]:
    out = []
    for w in str(text).split():
        w = re.sub(r"[^\w']", "", w.lower(), flags=re.UNICODE)
        if w:
            out.append(w)
    return out


def norm_tokens(text: str, lang: str = "en") -> list[str]:
    """Whisper-normalized token list. Input may be raw text or a
    strict-normalized string (apostrophes survive strict, so contractions
    still expand)."""
    nm = _en() if lang == "en" else _basic()
    # a punctuation-only token is never a word: the Whisper normalizer leaves a stray "." after a trailing "..."
    return [t for t in nm(str(text)).split() if re.search(r"\w", t)]


def eq_norm(a: list[str], b: list[str]) -> bool:
    """normalized equality, accepting segmentation-only differences"""
    return a == b or "".join(a) == "".join(b)


def contains_norm(window: list[str], gt: list[str]) -> bool:
    """normalized contiguous containment (exact_span / containment scoring):
    any window slice equal to gt, incl. via no-space join (slices only —
    substring matching on the full join would cross word boundaries)."""
    if not gt:
        return False
    jg = "".join(gt)
    n = len(window)
    for i in range(n):
        for j in range(i + 1, n + 1):
            sl = window[i:j]
            if sl == gt or "".join(sl) == jg:
                return True
    return False


def pos_of(gt: list[str], pred: list[str]) -> float:
    sm = difflib.SequenceMatcher(a=gt, b=pred)
    return sum(bl.size for bl in sm.get_matching_blocks()) / max(1, len(gt))


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def best_slice_wer(window: list[str], gt: list[str]) -> float:
    """gap-WER against the best contiguous window slice (window carries ±1
    context words by construction; scoring the whole window would charge
    those as insertions)."""
    if not gt:
        return 0.0
    if not window:
        return 1.0
    best = wer(gt, window)
    n = len(window)
    for i in range(n):
        for j in range(i + 1, n + 1):
            w = wer(gt, window[i:j])
            if w < best:
                best = w
    return best
