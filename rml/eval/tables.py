"""The paper's main tables from per-sample results.

A results file (one per benchmark, written by scripts/run_benchmark.py) maps every sample to its candidates and their
evidence, the selection's input format plus the unnormalised transcripts:
  {sample_id: {"hyp": {key: canonical transcript}, "s": {transcript: evidence}, "raw": {key: transcript}}}
keys: the six configurations of rml.selection.rule.SIX, the text lines text|lip_text (R) and text|gemma (R -> L),
and for noisy samples optionally input|keep (Whisper on the noisy input). Each key is scored from its raw transcript
when there is one, else from the canonical one.
Optional baselines file <benchmark>.baselines.json: {method: {sample_id: transcript}}, scored the same way.
With another lip reader R the results file is <benchmark>.<reader>.json (same format). Spanish and French
(silenced_talkvid_esfr.json, Tab. 5b) have no selection: "hyp" holds the fixed configurations only.

Metrics (Sec. 5): silenced speech = exact span recovery and word accuracy in the span (rml.eval.gap_score) on the
ASR-verified cuts; noisy speech and the condition grid = window WER. 95% intervals: paired bootstrap over source
videos (4,000 resamples, seed 0).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from rml.eval.gap_score import score_full
from rml.eval.scoring import norm_tokens, wer
from rml.selection.rule import SIX, TEXTS, evidence, mbr

COMPONENTS = [("R (text)", "text|lip_text"), ("R -> L (text)", "text|gemma"), ("G without text", "av_only|keep"),
              ("R -> G", "lip_text|keep"), ("R -> L -> G", "gemma|keep")]
FIXED = [("R (text)", "text|lip_text"), ("R -> L (text)", "text|gemma"), ("R -> L -> G", "gemma|keep")]
LANGUAGES = (("es", "Spanish"), ("fr", "French"))
NOISY_CONDS = ("babble-10", "babble-5", "sfx_seen-5", "sfx_unseen-5", "speech-5")
SEL = "Selection (ours)"


def boot(vids, d, n=4000, seed=0):
    """paired mean difference with a 95% interval by resampling source videos"""
    vids, d = np.asarray(vids), np.asarray(d, float)
    u, inv = np.unique(vids, return_inverse=True)
    s, c = np.bincount(inv, d, len(u)), np.bincount(inv, None, len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load_manifest(path):
    return {r["id"]: r for r in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())}


def cut_meta(r):
    """manifest row -> the fields gap_score.score_full reads"""
    return {**r, "window_words": [{"word": w, "start_s": a, "end_s": b} for w, a, b in r["window_words"]]}


def text_of(e, k):
    t = (e.get("raw") or {}).get(k)
    return t if t is not None else e["hyp"].get(k)


def has_evidence(e, k):
    t = e["hyp"].get(k)
    return bool(t) and evidence(e, t) is not None


def choose(e):
    return mbr(e, list(SIX), list(SIX) + list(TEXTS))


# ------------------------------------------------------------------ silenced speech
def silenced_scores(results, manifest, baselines=None, metric="g_exact"):
    """{cut: {row name: score}} on the ASR-verified cuts that the selection covers (both kept routes scored)"""
    out = {}
    for i, e in results.items():
        c = manifest.get(i)
        if c is None or c.get("gt_review") == "removed" or not c.get("asr_verified"):
            continue
        if not (has_evidence(e, "gemma|keep") and has_evidence(e, "lip_text|keep")):
            continue
        m = cut_meta(c)

        def sc(t):
            v = score_full(m, t) if t is not None else None
            return v[metric] if v else 0.0
        row = {name: sc(text_of(e, k)) for name, k in COMPONENTS if text_of(e, k) is not None}
        row[SEL] = sc(text_of(e, choose(e)))
        for b, d in (baselines or {}).items():
            if i in d:
                row[b] = sc(d[i])
        out[i] = (c["video"], row)
    return out


def word_by_word_scorer():
    """score_full of the token scorer that normalises the reference word by word, without the stutter, compound and
    spelling tolerances of rml/eval/gap_score.py (rml/training/label_gap_score.py, which also labels L's training
    targets): the scorer of Tab. 5b and of the asr_verified flags of silenced_talkvid_esfr"""
    from rml.training.label_gap_score import score_full as word_by_word
    return word_by_word


def fixed_scores(results, manifest, metric="g_exact", score=score_full):
    """{cut: (video, language, {row name: score})} of the fixed configurations (Tab. 5b) on the ASR-verified cuts;
    a cut that the scorer cannot align for a row is left out of that row"""
    out = {}
    for i, e in results.items():
        c = manifest.get(i)
        if c is None or c.get("gt_review") == "removed" or not c.get("asr_verified"):
            continue
        m = cut_meta(c)
        row = {}
        for name, k in FIXED:
            t = text_of(e, k)
            v = score(m, t) if t is not None else None
            if v:
                row[name] = v[metric]
        out[i] = (c["video"], c.get("language", "en"), row)
    return out


def table_languages(per, languages=LANGUAGES, ref="R (text)"):
    """per: fixed_scores(...); columns = the languages + pooled; last column R -> L -> G - R with its 95% interval"""
    cols = [(lg, nm) for lg, nm in languages] + [(None, "Pooled")]
    L = [f"{'':28s}" + "".join(f"{nm:>12s}" for _, nm in cols) + f"   R -> L -> G - {ref}, pooled [95% CI]"]
    for n, _k in FIXED:
        cells = []
        for lg, _ in cols:
            ks = [i for i, v in per.items() if n in v[2] and (lg is None or v[1] == lg)]
            cells.append(f"{100 * sum(per[i][2][n] for i in ks) / len(ks):12.1f}" if ks else f"{'-':>12s}")
        L.append(f"{n:28s}" + "".join(cells))
    ks = [i for i, v in per.items() if "R -> L -> G" in v[2] and ref in v[2]]
    if ks:
        mu, lo, hi = boot([per[i][0] for i in ks], [per[i][2]["R -> L -> G"] - per[i][2][ref] for i in ks])
        L[-1] += f"   {100 * mu:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}] (n={len(ks)})"
    L.append("n: " + ", ".join(f"{nm} {sum(1 for v in per.values() if lg is None or v[1] == lg)}" for lg, nm in cols))
    return "\n".join(L)


# ------------------------------------------------------------------ noisy speech and the grid
def window_wers(results, manifest, baselines=None, keys=None):
    """{sample: (video, condition, {row name: window WER})}"""
    out = {}
    names = [("Noisy input", "input|keep"), *COMPONENTS] if keys is None else keys
    for sid, e in results.items():
        c = manifest.get(sid)
        if c is None or c.get("gt_review") == "removed":
            continue
        if sum(has_evidence(e, k) for k in ("av_only|keep", "lip_text|keep", "input|keep")) < 2:
            continue
        ref = norm_tokens(c["sentence"], "en")
        row = {n: wer(ref, norm_tokens(text_of(e, k), "en")) for n, k in names if text_of(e, k) is not None}
        row[SEL] = wer(ref, norm_tokens(text_of(e, choose(e)), "en"))
        for b, d in (baselines or {}).items():
            if sid in d:
                row[b] = wer(ref, norm_tokens(d[sid], "en"))
        out[sid] = (c["video"], sid.rsplit("__", 1)[1], row)
    return out


def grid_wers(results, manifest):
    """the condition grid: every sample that has a kept configuration with evidence"""
    out = {}
    for sid, e in results.items():
        c = manifest.get(sid)
        if c is None or not any(has_evidence(e, k) for k in SIX):
            continue
        ref = norm_tokens(c["sentence"], "en")
        row = {n: wer(ref, norm_tokens(text_of(e, k), "en")) for n, k in COMPONENTS if text_of(e, k) is not None}
        row[SEL] = wer(ref, norm_tokens(text_of(e, choose(e)), "en"))
        row["_choice"] = choose(e)
        out[sid] = (c["video"], f"{c['span']}|{c['noise']}", row)
    return out


# ------------------------------------------------------------------ formatting
def _rows_order(per):
    names = []
    for v in per.values():
        for n in v[-1]:
            if n not in names and not n.startswith("_"):
                names.append(n)
    first = [n for n, _ in COMPONENTS] + [SEL]
    return [n for n in ["Noisy input", *first] if n in names] + [n for n in names if n not in first + ["Noisy input"]]


def table_silenced(sets, sets_hard=None, metric_label="exact"):
    """sets: {set name: silenced_scores(...)} for TalkVid and VoxMM; pooled = their union"""
    pooled = {k: v for d in sets.values() for k, v in d.items()}
    cols = {**sets, "Pooled": pooled}
    if sets_hard:
        cols.update(sets_hard)
    names = _rows_order(pooled)
    L = [f"{'':28s}" + "".join(f"{c:>12s}" for c in cols) + "   Selection - row, pooled [95% CI]"]
    for n in names:
        cells = []
        for d in cols.values():
            ks = [i for i in d if n in d[i][1]]
            cells.append(f"{100 * np.mean([d[i][1][n] for i in ks]):12.1f}" if ks else f"{'-':>12s}")
        ks = [i for i in pooled if n in pooled[i][1]]
        ci = ""
        if n != SEL and ks:
            mu, lo, hi = boot([pooled[i][0] for i in ks], [pooled[i][1][SEL] - pooled[i][1][n] for i in ks])
            ci = f"   {100 * mu:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}] (n={len(ks)})"
        L.append(f"{n:28s}" + "".join(cells) + ci)
    L.append("n: " + ", ".join(f"{c} {len(d)}" for c, d in cols.items()) + f"  ({metric_label})")
    return "\n".join(L)


def table_conditions(per, conds, label="ALL"):
    """per: {sample: (video, condition, {row: WER})}; columns = conditions + the mean over all samples"""
    names = _rows_order(per)
    L = [f"{'':28s}" + "".join(f"{c:>14s}" for c in (*conds, label)) + "   Selection - row [95% CI]"]
    for n in names:
        cells = []
        for c in (*conds, label):
            ks = [s for s, v in per.items() if n in v[2] and (c == label or v[1] == c)]
            cells.append(f"{100 * np.mean([per[s][2][n] for s in ks]):14.1f}" if ks else f"{'-':>14s}")
        ks = [s for s, v in per.items() if n in v[2]]
        ci = ""
        if n != SEL and ks:
            mu, lo, hi = boot([per[s][0] for s in ks], [per[s][2][SEL] - per[s][2][n] for s in ks])
            ci = f"   {100 * mu:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}] (n={len(ks)})"
        L.append(f"{n:28s}" + "".join(cells) + ci)
    L.append(f"n = {len(per)}")
    return "\n".join(L)


def grid_choices(per):
    """share of each chosen configuration per grid cell (Fig. 4, bottom)"""
    by = collections.defaultdict(collections.Counter)
    for _v, cell, row in per.values():
        by[cell][row["_choice"]] += 1
    L = []
    for cell in sorted(by):
        n = sum(by[cell].values())
        L.append(f"{cell:20s} " + "  ".join(f"{k} {100 * v / n:.0f}%" for k, v in by[cell].most_common()))
    return "\n".join(L)
