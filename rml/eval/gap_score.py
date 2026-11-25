"""Span scorer for silenced-speech recovery (exact span recovery and word accuracy in the gap).

Any hypothesis is a transcript of the WHOLE window (a lip reader's text, or
whisper's reading of an audio track: generated, clean, or muted). It is
aligned to the reference window words (Levenshtein on whisper-normalized
tokens, same normalizer both sides); the gap region of the alignment spans
from after the last left-context reference token to before the first
right-context reference token, so insertions inside the gap count against it.

Tolerances (all symmetric, every system alike): the reference is normalized per chunk (left context / gap / right context) like a
hypothesis string, not word by word (word-by-word turned "I'd been" into "i would been" and "one percent" into
"1 percent" vs "1%"); a word run that repeats the run just before it (a stutter: "i i", "as a as a", "is is is")
is optional on both sides (omitting or keeping it costs nothing); two adjacent tokens whose concatenation is a
token of the other side are joined ("paint brush" = "paintbrush"). A gap that is only a repeated copy is
unscorable (nothing to recover).

Per row: g_acc (fraction of gap tokens aligned to an identical hyp token),
g_exact (every gap token matched, no substitution/insertion in the region),
g_wer (S+D+I in the gap region / gap tokens). Gap-only hypotheses (a
proposer's filling) are scored by splicing them into the reference context
(then only the gap can differ) — marked `spliced`.

Library + CLI: score arms from span_scores.json (hyp_full) or n-best json
(rows[].nbest[0] full-window text) and print a per-language / per-k table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rml.eval.scoring import norm_tokens


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
    left, gap = norm_tokens(" ".join(words[:si]), lg), norm_tokens(" ".join(words[si:si + k]), lg)
    toks = left + gap + norm_tokens(" ".join(words[si + k:]), lg)
    a, b = len(left), len(left) + len(gap)
    if b <= a or all(repeats(toks)[a:b]):
        return None
    return toks, a, b


def repeats(toks):
    """True for the 2nd+ copy of an immediately repeated 1-3 token run (stutter / disfluent restart)"""
    f = [False] * len(toks)
    for n in (1, 2, 3):
        for i in range(n, len(toks) - n + 1):
            if toks[i:i + n] == toks[i - n:i]:
                f[i:i + n] = [True] * n
    return f


def join_compounds(ref, a, b, hyp):
    """join adjacent tokens whose concatenation is a token of the other side; never across a gap edge"""
    hs = set(hyp)
    r, ra, rb, i = [], None, None, 0
    while i < len(ref):
        if i == a:
            ra = len(r)
        if i == b:
            rb = len(r)
        if i + 1 < len(ref) and i + 1 not in (a, b) and ref[i] + ref[i + 1] in hs and ref[i] not in hs:
            r.append(ref[i] + ref[i + 1])
            i += 2
            continue
        r.append(ref[i])
        i += 1
    rb = len(r) if rb is None else rb
    rs, h, j = set(r), [], 0
    while j < len(hyp):
        if j + 1 < len(hyp) and hyp[j] + hyp[j + 1] in rs and hyp[j] not in rs:
            h.append(hyp[j] + hyp[j + 1])
            j += 2
            continue
        h.append(hyp[j])
        j += 1
    return r, ra, rb, h


def align(ref, hyp, ropt=None, hopt=None):
    """Levenshtein alignment; returns list of (op, ri, hi). ropt/hopt flag optional tokens: deleting an
    optional ref token ("d") or inserting an optional hyp token ("i") costs nothing. Without flags this is
    the plain alignment."""
    n, m = len(ref), len(hyp)
    rc = [0 if ropt and ropt[i] else 1 for i in range(n)]
    hc = [0 if hopt and hopt[j] else 1 for j in range(m)]
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + rc[i - 1]
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + hc[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + rc[i - 1], D[i][j - 1] + hc[j - 1],
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("M" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + rc[i - 1]:
            ops.append(("D" if rc[i - 1] else "d", i - 1, None)); i -= 1
        else:
            ops.append(("I" if hc[j - 1] else "i", None, j - 1)); j -= 1
    return ops[::-1]


def score_full(c, hyp_text):
    rt = ref_tokens(c)
    if rt is None:
        return None
    ref, a, b = rt
    # region = ops strictly after the last left-context ref op and before the first right-context ref op
    return _score_tokens(c, ref, a, b, norm_tokens(hyp_text or "", c.get("language", "en")))


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
    ref, a, b, hyp = join_compounds(ref, a, b, hyp)
    ops = align(ref, hyp, repeats(ref), repeats(hyp))
    idx = [t for t, (op, ri, _) in enumerate(ops) if ri is not None]
    left = max((t for t in idx if ops[t][1] < a), default=-1)
    right = min((t for t in idx if ops[t][1] >= b), default=len(ops))
    region = ops[left + 1:right]
    hits = sum(1 for op, ri, _ in region if op in ("M", "d") and ri is not None and a <= ri < b)
    errs = sum(1 for op, _, _ in region if op in ("S", "D", "I"))
    n = b - a
    return {"g_acc": hits / n, "g_exact": float(hits == n and errs == 0), "g_wer": errs / n,
            "gap_hyp": " ".join(hyp[j] for op, _, j in region if j is not None)}


def load_arm(spec):
    """spec = name=kind:path ; kind in {span, nbest, gap}"""
    name, rest = spec.split("=", 1)
    kind, path = rest.split(":", 1)
    rows = {}
    import glob
    for p in sorted(glob.glob(path)):
        for r in json.load(open(p))["rows"]:
            if kind == "span":
                rows[r["id"]] = ("full", r.get("hyp_full", ""))
            elif kind == "nbest":
                rows[r["id"]] = ("full", (r.get("nbest") or [""])[0])
            elif kind == "gap":
                nb = r.get("nbest") or [r.get("pred", "")]
                t = nb[0] if nb else ""
                if "Missing words:" in t:
                    t = t.split("Missing words:", 1)[1]
                rows[r["id"]] = ("gap", t.split("\n")[0].strip())
    return name, rows


def summarize(scored, ids_by_group):
    out = {}
    for g, ids in ids_by_group.items():
        rs = [scored[i] for i in ids if i in scored]
        if not rs:
            continue
        out[g] = {"n": len(rs), **{m: round(100 * sum(r[m] for r in rs) / len(rs), 1)
                                   for m in ("g_acc", "g_exact", "g_wer")}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", nargs="+", required=True)
    ap.add_argument("--ids", required=True, help="json list of row ids (the common set)")
    ap.add_argument("--arm", action="append", required=True, help="name=kind:glob")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = load_meta(args.meta)
    ids = json.load(open(args.ids))
    groups = {"all": ids}
    for lg in ("en", "es", "fr"):
        groups[lg] = [i for i in ids if meta[i]["language"] == lg]
    for k in range(1, 11):
        g = [i for i in ids if int(meta[i]["n_masked_words"]) == k]
        if g:
            groups[f"k{k}"] = g
            groups[f"en_k{k}"] = [i for i in g if meta[i]["language"] == "en"]
    res, per_row = {}, {}
    for spec in args.arm:
        name, rows = load_arm(spec)
        sc = {}
        for i in ids:
            if i not in rows:
                continue
            kind, txt = rows[i]
            s = score_full(meta[i], txt) if kind == "full" else score_gap_only(meta[i], txt)
            if s is not None:
                sc[i] = s
        res[name] = summarize(sc, groups)
        per_row[name] = sc
        miss = len(ids) - len(sc)
        a = res[name]
        print(f"{name:34s} n={len(sc):4d} (miss {miss:3d})  all {a['all']['g_acc']:5.1f}/{a['all']['g_exact']:5.1f}  "
              + "  ".join(f"{lg} {a[lg]['g_acc']:5.1f}/{a[lg]['g_exact']:5.1f}" for lg in ("en", "es", "fr") if lg in a))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "rows": per_row}, open(args.out, "w"))


if __name__ == "__main__":
    main()
