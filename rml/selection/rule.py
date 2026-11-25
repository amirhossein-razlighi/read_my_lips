"""Test-time selection: evidence-weighted minimum Bayes risk (MBR) with nested voters.

Every configuration of the pipeline produces an output; the selection returns the output with the lowest expected
word distance to the voters, each voter weighted by exp(evidence / T). The evidence of a transcript is its pointwise
mutual information with the input audio (Whisper large-v3, input vs silence) plus its pointwise mutual information
with the lips (the lip reader, real mouth vs a frozen mouth).

Nested voters: L reads the lip reader's whole N-best list, so when L's transcript votes, the lip reader's transcript
and the output that only speaks it with the audio muted stay selectable but do not vote. T = 100 (set on the
validation grid).

An evidence entry `e` (one sample of an `output_scores.json`) is
    {"hyp": {config: transcript}, "s": {transcript: {"hear", "hear0", "vis", "vis0", "lm"}}}
"""
import numpy as np

from rml.eval.scoring import norm_tokens, wer

# the six configurations (text given to G: L's transcript / the lip reader's / none, x input audio kept / muted);
# the order is the tie-break: more components and the audio kept first
SIX = ("gemma|keep", "lip_text|keep", "av_only|keep", "gemma|mute", "lip_text|mute", "av_only|mute")
# text lines that vote: the lip reader's top reading and L's transcript
TEXTS = ("text|lip_text", "text|gemma")
SUBSUMED_BY_L = ("text|lip_text", "lip_text|mute")
NAME = {"gemma|keep": "R -> L -> G, audio kept", "lip_text|keep": "R -> G, audio kept",
        "av_only|keep": "G alone, audio kept", "gemma|mute": "R -> L -> G, audio muted",
        "lip_text|mute": "R -> G, audio muted", "av_only|mute": "G alone, audio muted",
        "text|lip_text": "lip reader (text)", "text|gemma": "R -> L (text)"}


def dist(a, b):
    """symmetric word distance: the mean of the two directed WERs on Whisper-normalised tokens"""
    a, b = norm_tokens(a, "en"), norm_tokens(b, "en")
    if not a and not b:
        return 0.0
    return 0.5 * (wer(a, b) if a else 1.0) + 0.5 * (wer(b, a) if b else 1.0)


def evidence(e, t):
    """audio PMI + lip PMI of transcript t; None when t is empty or not scored"""
    d = e["s"].get(t, {}) if t else {}
    if not t or "lm" not in d or d.get("hear") is None:
        return None
    pv = d["vis"] - d["vis0"] if d.get("vis") is not None and d.get("vis0") is not None else 0.0
    return d["hear"] - d["hear0"] + pv


def mbr(e, sel, voters, T=100.0, lips=True, audio=True):
    """The paper's selection. sel: selectable outputs; voters: candidate voters before the nesting rule.
    Outputs without a transcript (e.g. G alone on silence) neither vote nor can be selected."""
    def ev(k):
        t = e["hyp"].get(k)
        d = e["s"].get(t, {}) if t else {}
        if not t or "lm" not in d or d.get("hear") is None:
            return None
        pv = d["vis"] - d["vis0"] if d.get("vis") is not None and d.get("vis0") is not None else 0.0
        return (d["hear"] - d["hear0"] if audio else 0.0) + (pv if lips else 0.0)
    S_ = [k for k in sel if ev(k) is not None]
    V = [k for k in voters if ev(k) is not None]
    if "text|gemma" in V:
        V = [k for k in V if k not in SUBSUMED_BY_L]
    v = np.array([ev(k) for k in V])
    w = np.exp((v - v.max()) / T) if np.isfinite(T) else np.ones(len(V))
    risk = [sum(wj * dist(e["hyp"][a], e["hyp"][b]) for b, wj in zip(V, w)) for a in S_]
    return S_[int(np.argmin(risk))]


def select(e, T=100.0):
    """MBR over the six configurations with the two text voters (the deployed call)"""
    return mbr(e, list(SIX), list(SIX) + list(TEXTS), T=T)


def mbr_flat(e, sel, voters, T=100.0):
    """evidence-weighted MBR without the nesting rule (used for the two-configuration comparisons)"""
    S_ = [k for k in sel if e["hyp"].get(k) and evidence(e, e["hyp"][k]) is not None]
    V = [k for k in voters if e["hyp"].get(k) and evidence(e, e["hyp"][k]) is not None]
    v = np.array([evidence(e, e["hyp"][k]) for k in V])
    w = np.exp((v - v.max()) / T)
    risk = [sum(wj * dist(e["hyp"][a], e["hyp"][b]) for b, wj in zip(V, w)) for a in S_]
    return S_[int(np.argmin(risk))]


HARD_SIX = ["full_keep", "lip_keep", "av_keep", "full_mute", "lip_mute", "av_mute"]


def hard_mbr(cand, T=100.0):
    """the same rule on a hard-set candidate store entry ({config: {"text", "ear", "eye"}}): six configurations,
    voters + the reader's and L's lines, outputs without speech drop out, nested voters"""
    S_ = [k for k in HARD_SIX if k in cand and cand[k]["text"]]
    V = S_ + [k for k in ("text_lip", "text_full") if k in cand and cand[k]["text"]]
    if "text_full" in V:
        V = [k for k in V if k not in ("text_lip", "lip_mute")]
    ev = np.array([cand[k]["ear"] + cand[k]["eye"] for k in V])
    w = np.exp((ev - ev.max()) / T)
    risk = [sum(wj * dist(cand[a]["text"], cand[b]["text"]) for b, wj in zip(V, w)) for a in S_]
    return S_[int(np.argmin(risk))]
