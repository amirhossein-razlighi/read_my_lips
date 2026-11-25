#!/usr/bin/env python
"""Context-aware correction model L: LoRA SFT of Gemma on lip-reader hypotheses + transcript context.

Read My Lips trains L with --evidence none (no lip soft tokens reach Gemma, no viseme line, CTC weight 0) in three
stages: configs/training/lm_stage*.yaml, docs/TRAINING.md. The rest of this docstring describes the general
(lip-evidence) form of the code.

Original design: learned fusion — viseme head -> K learned query tokens -> Gemma
soft tokens -> LoRA SFT on the missing phrase.

Architecture (simplest form, no word-count head):
  frozen AV-HuBERT feats (cached npz)
    -> viseme head (warm-started from avh_20k_B_reg; frozen in arm B,
       trainable at tiny LR + CTC anchor in arm C)
    -> per-frame z_t = MLP([h_t ; logP_t ; entropy ; t/T])
    -> K=8 learned queries, 2-layer cross-attention resampler
    -> Linear + LN to Gemma hidden size (soft tokens, RMS-matched to the
       embedding table)
  Gemma-3 12B (bf16, frozen) + LoRA r16 on q/k/v/o
  input:  [BOS] Transcript: <whisper ctx with ___>\n [VIS x K] Missing words:
  labels: -100 everywhere except " <target words> <eos>"

Two stages in one run (resumable):
  stage 1 (steps < --warmup-steps): only adapter trains (queries, x-attn,
    frame MLP, projection). LoRA + head frozen.
  stage 2: + LoRA (1.5e-5); arm C also unfreezes head (5e-6) with CTC anchor.
Early stopping on held-out val positional accuracy (greedy decode), not loss.
TensorBoard: losses, LRs, val pos/exact, sample decodes.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules (flat imports)
from lm_support import BLANK, CLS2ID, VISEME_CLASSES, load_gemma  # noqa: E402

N_CLS = 1 + len(VISEME_CLASSES)          # blank + 13
ID2CLS = {i: c for c, i in CLS2ID.items()}
# arm G first supervised field label: "Lip visemes:" (certified recipe) or
# "Phones:" (--aux-line phones — gold = the phone recognizer's reading of the
# CLEAN gap crop, i.e. the SAME alphabet as the draft_phones emb-init tokens,
# so the emb-init copy-then-ground warmup mechanism applies to that channel;
# with the viseme-letter field it did not: CE@1000 4.14, gap -0.02)
AUX_LABEL = "Lip visemes:"


def set_aux_label(label: str) -> None:
    global AUX_LABEL
    AUX_LABEL = label

# ─────────── avsr_soft_tokens: Auto-AVSR evidence channel (--evidence avsr) ──
# The lip evidence behind the soft tokens comes from the official video-only
# Auto-AVSR (Conformer encoder 768-d + CTC over 5,049 sentencepiece units,
# frame-synchronous at 25 fps) instead of AV-HuBERT + the 12-viseme head.
# Extracted once per window by lipreading_pipeline/extract_avsr_evidence.py; the dataset
# packs the per-frame evidence into ONE float array so every existing slice /
# time-mask / collate path applies unchanged. --evidence avh (default) is
# byte-identical to the certified recipe.
AVSR_ENC_DIM = 768
AVSR_SPECIAL = ("<blank>", "<unk>", "<eos>")
AVH_DIM = 1024          # AV-HuBERT trunk feature width (the viseme head's d_in)

# ─────────── draft_phones: draft-phone evidence channel (ablation) ──────────
# The evidence is the no-text LTX draft ("mumble": lips-only acoustic
# estimate of the gap, real audio in the context) read by
# wav2vec2-lv-60-espeak-cv-ft — 392 IPA phone classes, CTC at 50 Hz
# (lipreading_pipeline/draft_extract_phones.py, one npz per ROW id, blank kept separate).
# Packed like the Auto-AVSR channel but with NO trunk columns:
# [blank_p | top-K p | top-K id]. Same AvsrAdapter (enc_dim=0), same
# emb-init mixture — unit text = the IPA symbol itself (Gemma-4 tokenizes
# every phone into 1-3 tokens, no <unk>).
PHONE_SPECIAL = ("<pad>", "<s>", "</s>", "<unk>")   # <pad> = CTC blank (id 0)
PHONE_RATE = 2          # phone frames (20 ms) per 25-fps video frame


def load_phone_vocab(ev_dir) -> tuple[list[str], int]:
    """_phone_vocab.json written by draft_extract_phones.py ->
    (phones by id, blank_id)."""
    d = json.loads((Path(ev_dir) / "_phone_vocab.json")
                   .read_text(encoding="utf8"))
    inv = {int(k): v for k, v in d["id2phone"].items()}
    return [inv[i] for i in range(len(inv))], int(d["blank_id"])


def phone_text(p: str) -> str | None:
    """IPA symbol as Gemma reads it; None for the CTC specials."""
    return None if p in PHONE_SPECIAL else p


def pack_phone_evidence(z, K: int) -> np.ndarray:
    """[T, 1 + 2K] float32 = [blank_p | top-K p | top-K id] (ids as floats).
    A time-masked (zeroed) row reads as id 0 = <pad>/blank with p 0 -> the
    frame carries no mixture mass (same convention as the Auto-AVSR pack)."""
    p = z["logits_topk_p"].astype(np.float32)
    ids = z["logits_topk_idx"].astype(np.float32)
    assert p.shape[1] >= K, (p.shape, K)
    blank = z["blank_p"].astype(np.float32)[:, None]
    assert p.shape[0] == ids.shape[0] == blank.shape[0]
    return np.concatenate([blank, p[:, :K], ids[:, :K]], 1)


def route_width(K: int) -> int:
    """lang_routed_tokens packed width: max(AVH, packed avsr) + 1 flag col
    (last column: 1 = Auto-AVSR channel, 0 = viseme-head channel)."""
    return max(AVH_DIM, AVSR_ENC_DIM + 1 + 2 * K) + 1


def load_avsr_units(units_file) -> list[str]:
    """token_list exactly as Auto-AVSR's TextTransform builds it:
    ['<blank>'] + units-file pieces + ['<eos>'] (index 0 = CTC blank)."""
    units = [ln.split()[0] for ln in
             Path(units_file).read_text(encoding="utf8").splitlines()
             if ln.strip()]
    return ["<blank>"] + units + ["<eos>"]


def avsr_unit_text(u: str) -> str | None:
    """sentencepiece piece -> the text Gemma tokenizes: '▁THE' -> ' the',
    'ING' -> 'ing'. None for blank/<unk>/<eos> (excluded from the mixture)."""
    if u in AVSR_SPECIAL:
        return None
    return (" " + u[1:] if u.startswith("▁") else u).lower()


def pack_avsr_evidence(z, K: int) -> np.ndarray:
    """[T, 768 + 1 + 2K] float32 = [enc | entropy | top-K p | top-K id].
    ids ride as floats (exact below 2^24). A time-masked (zeroed) row reads
    as id 0 = blank with p 0 -> the frame carries no mixture mass."""
    assert z["topk_p"].shape[1] >= K, (z["topk_p"].shape, K)
    enc = z["enc"].astype(np.float32)
    assert enc.shape[1] == AVSR_ENC_DIM, enc.shape
    ent = z["ent"].astype(np.float32)[:, None]
    p = z["topk_p"][:, :K].astype(np.float32)
    ids = z["topk_id"][:, :K].astype(np.float32)
    assert enc.shape[0] == ent.shape[0] == p.shape[0] == ids.shape[0]
    return np.concatenate([enc, ent, p, ids], 1)


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower(), flags=re.UNICODE)


# ───────────────────────────── data ──────────────────────────────────────────
class GapDataset(torch.utils.data.Dataset):
    """v2 augmentations (train split only; all default 0.0 = off):
    ctx_full_drop  replace the whole context with just "___" — the row becomes
                   unsolvable without lips, making the visual tokens load-bearing
    ctx_word_drop  drop each non-placeholder context word independently
    time_mask_p    zero a random time_mask_len-frame stretch of the gap feats
    feat_noise     additive gaussian noise, relative to the sample's feat std
    """

    def __init__(self, jsonl: Path, viseme_targets: bool = False,
                 ml_lexicons: Path | None = None,
                 ctx_word_drop: float = 0.0, ctx_full_drop: float = 0.0,
                 time_mask_p: float = 0.0, time_mask_len: int = 4,
                 feat_noise: float = 0.0, full_window: bool = False,
                 slice_pad: int = 0, evidence: str = "avh",
                 evidence_dirs=None, evidence_topk: int = 8,
                 route_langs=("en",), drop_langs=(),
                 aux_line: str = "visemes", phone_targets=None,
                 phone_vocab=None):
        self.full_window = full_window
        self.aux_line = aux_line
        self.slice_pad = slice_pad   # frames of context kept on each side
        self.rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l]
        # drop_langs: rows of
        # these languages leave the dataset before any selection/sampling
        self.drop_langs = set(drop_langs or ())
        if self.drop_langs:
            n0 = len(self.rows)
            self.rows = [r for r in self.rows
                         if r.get("language", "en") not in self.drop_langs]
            print(f"[dataset] drop_langs {sorted(self.drop_langs)}: "
                  f"{n0} -> {len(self.rows)} rows", flush=True)
        # avsr_soft_tokens: per-frame evidence npz looked up by window stem
        # across evidence_dirs; noise_cols limits feat_noise to the encoder
        # columns (posteriors / ids must never be perturbed)
        # route (lang_routed_tokens): rows of route_langs carry the Auto-AVSR
        # channel, every other row the viseme-head channel; both packed to
        # route_width(K) with the channel flag in the last column
        evidence = "avh" if evidence == "none" else evidence    # none: feats still load (lengths only)
        assert evidence in ("avh", "avsr", "route", "draft_phones"), evidence
        self.evidence = evidence
        self.route_langs = set(route_langs or ())
        self.evidence_dirs = [Path(d) for d in (evidence_dirs or [])]
        self.evidence_topk = evidence_topk
        self.noise_cols = AVSR_ENC_DIM if evidence == "avsr" else \
            (0 if evidence == "draft_phones" else None)
        # evidence frames per video frame: the phone stream runs at 50 Hz
        self.ev_rate = PHONE_RATE if evidence == "draft_phones" else 1
        self._evpath: dict[str, Path | None] = {}
        self.n_dropped_noev = 0
        if evidence == "draft_phones":
            # one npz per ROW (one draft per cut). Rows without a draft leave
            # the dataset: the train manifest holds every cut, drafts exist
            # for the OneCutFixedK picks only (draft_manifest)
            assert self.evidence_dirs, "--avsr-evidence-dir (phones_*) required"
            keep = [r for r in self.rows
                    if self._evidence_path(r["id"]) is not None]
            self.n_dropped_noev = len(self.rows) - len(keep)
            print(f"[dataset] draft_phones: {len(keep)}/{len(self.rows)} rows"
                  f" carry a phone npz under "
                  f"{[str(d) for d in self.evidence_dirs]} "
                  f"({self.n_dropped_noev} without -> dropped)", flush=True)
            self.rows = keep
        if evidence in ("avsr", "route"):
            assert self.evidence_dirs, "--avsr-evidence-dir required"
            miss = sorted({r["feats"] for r in self.rows
                           if self._channel(r) == "avsr"
                           and self._evidence_path(r["feats"]) is None})
            assert not miss, (f"{len(miss)} windows without Auto-AVSR "
                              f"evidence under {self.evidence_dirs}, e.g. "
                              f"{miss[:3]}")
            n_av = len({r["feats"] for r in self.rows
                        if self._channel(r) == "avsr"})
            print(f"[dataset] {evidence}: avsr evidence for {n_av}/"
                  f"{len({r['feats'] for r in self.rows})} windows "
                  f"(top-{evidence_topk}, route_langs="
                  f"{sorted(self.route_langs) if evidence == 'route' else '-'})",
                  flush=True)
        self.ctx_word_drop, self.ctx_full_drop = ctx_word_drop, ctx_full_drop
        self.time_mask_p, self.time_mask_len = time_mask_p, time_mask_len
        self.feat_noise = feat_noise
        self._cache: dict[str, np.ndarray] = {}
        self.vis_targets: list[list[int] | None] = [None] * len(self.rows)
        self.vis_words: list[list[list[int]] | None] = [None] * len(self.rows)
        if viseme_targets:                 # CTC anchor labels (arm C)
            # en: CMUdict route; es/fr/ru: LOOKUP-ONLY from the corpus
            # lexicons (no runtime espeak). Uncovered words -> row excluded
            # from the CTC term only (stays in the LM loss).
            from lm_support import Phonemizer, collapse, visemes
            ph = Phonemizer()
            lex: dict[str, dict] = {}

            def vis_of(w: str, lang: str):
                if lang == "en":
                    p = ph.phonemes(w)
                    return collapse(visemes(p)) if p else None
                if ml_lexicons is None:
                    return None
                if lang not in lex:
                    f = ml_lexicons / f"lexicon_{lang}.json"
                    lex[lang] = (json.loads(f.read_text())
                                 if f.exists() else {})
                return lex[lang].get(norm(w))

            n_ctc = 0
            for i, r in enumerate(self.rows):
                ids: list[int] = []
                per_word: list[list[int]] = []
                ok = True
                for w in r["target"].split():
                    v = vis_of(w, r.get("language", "en"))
                    if not v:
                        ok = False
                        break
                    wid = [CLS2ID[c] for c in v]
                    ids += wid
                    per_word.append(wid)
                self.vis_targets[i] = ids if ok and ids else None
                self.vis_words[i] = per_word if ok and per_word else None
                n_ctc += self.vis_targets[i] is not None
            print(f"[dataset] CTC anchor targets on {n_ctc}/{len(self.rows)} "
                  f"rows", flush=True)
        # --aux-line phones: the arm-G first field is the clean-gap phone
        # string (recognizer pseudo-labels, gap +-0.1 s / min 0.4 s crop) in
        # the evidence's own IPA alphabet; rows without a target get "?"
        # (same fallback as the viseme line)
        self.aux_strs: list[str | None] = [None] * len(self.rows)
        if aux_line == "phones":
            assert phone_targets is not None and phone_vocab is not None, \
                "--aux-line phones needs --phone-targets and a phone vocab"
            n_aux = 0
            for i, r in enumerate(self.rows):
                ids = phone_targets.get(r["id"])
                if ids:
                    self.aux_strs[i] = " ".join(phone_vocab[p] for p in ids)
                    n_aux += 1
            print(f"[dataset] aux-line phones: gold phone strings on "
                  f"{n_aux}/{len(self.rows)} rows", flush=True)

    def __len__(self):
        return len(self.rows)

    def _evidence_path(self, feats_path: str) -> Path | None:
        # draft_phones: keyed by the ROW id verbatim (ids contain dots —
        # Path.stem would truncate them); avsr/route: by the window stem
        stem = feats_path if self.evidence == "draft_phones" \
            else Path(feats_path).stem
        if stem not in self._evpath:
            hit = None
            for d in self.evidence_dirs:
                if (d / f"{stem}.npz").exists():
                    hit = d / f"{stem}.npz"
                    break
            self._evpath[stem] = hit
        return self._evpath[stem]

    def _channel(self, r: dict) -> str:
        if self.evidence == "draft_phones":
            return "phones"
        if self.evidence == "avsr":
            return "avsr"
        if self.evidence == "route" and \
                r.get("language", "en") in self.route_langs:
            return "avsr"
        return "avh"

    def _feats(self, path: str, channel: str = "avh") -> np.ndarray:
        key = (path, channel)
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            if self.evidence == "route":
                if channel == "avsr":
                    core = pack_avsr_evidence(
                        np.load(self._evidence_path(path)), self.evidence_topk)
                else:
                    core = np.load(path)["feats"].astype(np.float32)
                    assert core.shape[1] == AVH_DIM, core.shape
                out = np.zeros((core.shape[0], route_width(self.evidence_topk)),
                               np.float32)
                out[:, : core.shape[1]] = core
                out[:, -1] = 1.0 if channel == "avsr" else 0.0
                self._cache[key] = out
            elif self.evidence == "avsr":
                self._cache[key] = pack_avsr_evidence(
                    np.load(self._evidence_path(path)), self.evidence_topk)
            elif self.evidence == "draft_phones":   # path == row id here
                self._cache[key] = pack_phone_evidence(
                    np.load(self._evidence_path(path)), self.evidence_topk)
            else:
                self._cache[key] = np.load(path)["feats"]
        return self._cache[key]

    def __getitem__(self, i: int):
        r = self.rows[i]
        if self.aux_line == "phones":
            r = {**r, "aux_str": self.aux_strs[i] or "?"}
        ch = self._channel(r)
        # draft_phones evidence is per row (keyed by id) and at ev_rate x the
        # video frame rate: every frame index below is scaled by q (q == 1
        # for avh/avsr/route -> byte-identical to the certified recipe)
        src = r["id"] if self.evidence == "draft_phones" else r["feats"]
        q = self.ev_rate
        if self.full_window:
            f = self._feats(src, ch).astype(np.float32)  # whole window
        elif self.slice_pad:
            # widened slice: span +- slice_pad frames, clamped to the window.
            # sf0/sf1 = the true span's position INSIDE the slice (CTC anchor
            # + desync-tolerant soft alignment: the span is somewhere near
            # the middle, the model learns to find it)
            ff = self._feats(src, ch)
            a0 = max(0, (r["f0"] - self.slice_pad) * q)
            b0 = min(ff.shape[0], (r["f1"] + self.slice_pad) * q)
            f = ff[a0:b0].astype(np.float32)
            r = {**r, "sf0": r["f0"] * q - a0,
                 "sf1": r["f0"] * q - a0 + (r["f1"] - r["f0"]) * q}
        else:
            f = self._feats(src, ch)[r["f0"] * q: r["f1"] * q] \
                .astype(np.float32)
        if q > 1 and f.shape[0] == 0:
            # draft wav marginally shorter than the window: fall back to the
            # last evidence frame rather than an empty slice
            f = self._feats(src, ch)[-1:].astype(np.float32)
        if self.ctx_full_drop or self.ctx_word_drop:
            ctx = r["context"]
            if random.random() < self.ctx_full_drop:
                ctx = "___"
            elif self.ctx_word_drop:
                kept = [t for t in ctx.split() if t == "___"
                        or random.random() >= self.ctx_word_drop]
                ctx = " ".join(kept)
            r = {**r, "context": ctx}
        if self.time_mask_p and random.random() < self.time_mask_p \
                and f.shape[0] > self.time_mask_len + 2:
            m0 = random.randrange(0, f.shape[0] - self.time_mask_len)
            f[m0: m0 + self.time_mask_len] = 0.0
            if self.evidence == "route":      # the channel flag survives
                f[m0: m0 + self.time_mask_len, -1] = 1.0 if ch == "avsr" \
                    else 0.0
        noise_cols = self.noise_cols
        if self.evidence == "route":
            noise_cols = AVSR_ENC_DIM if ch == "avsr" else AVH_DIM
        if self.feat_noise and noise_cols is None:
            f = f + np.random.randn(*f.shape).astype(np.float32) \
                * (self.feat_noise * float(f.std()))
        elif self.feat_noise and noise_cols:  # avsr/route: trunk columns only
            nc = noise_cols                   # (draft_phones has none: no-op)
            f[:, :nc] = f[:, :nc] + np.random.randn(f.shape[0], nc) \
                .astype(np.float32) * (self.feat_noise * float(f[:, :nc].std()))
        return torch.from_numpy(f), r, self.vis_targets[i], self.vis_words[i]


class OneCutFixedK(torch.utils.data.Dataset):
    """STRICT partition: each base window contributes exactly ONE cut with a
    FIXED k for the whole run (round-robin over window index -> balanced k
    histogram, deterministic). No window ever appears with two different
    masks -> zero context duplication and zero cross-cut answer leakage, at
    the cost of ~4x less data. Delegates to the base dataset at call time."""

    def __init__(self, base: GapDataset):
        self.base = base
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(base.rows):
            groups.setdefault(r["id"].split("__k")[0], []).append(i)
        self.picks: list[int] = []
        hist: dict[int, int] = {}
        for gi, (_, idxs) in enumerate(sorted(groups.items())):
            byk = {base.rows[i]["k"]: i for i in sorted(idxs)}
            want = 1 + gi % 4
            k = want if want in byk else \
                sorted(byk, key=lambda x: (abs(x - want), x))[0]
            self.picks.append(byk[k])
            hist[k] = hist.get(k, 0) + 1
        print(f"[one-cut-fixed-k] {len(self.picks)} rows "
              f"(k hist {dict(sorted(hist.items()))})", flush=True)

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, i: int):
        return self.base[self.picks[i]]


class OneCutPerWindow(torch.utils.data.Dataset):
    """Epoch view over a GapDataset: each base window contributes ONE randomly
    drawn cut (k) per draw. Kills same-epoch context duplication and the
    cross-cut answer leakage of the 4x multi-cut layout while keeping every
    cut in rotation across epochs. Delegates to the base dataset at call time,
    so curriculum attribute changes (ctx_full_drop) propagate."""

    def __init__(self, base: GapDataset):
        self.base = base
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(base.rows):
            groups.setdefault(r["id"].split("__k")[0], []).append(i)
        self.groups = list(groups.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i: int):
        return self.base[random.choice(self.groups[i])]


def collate(batch):
    feats, rows, vis, visw = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats])
    T = int(lens.max())
    x = torch.zeros(len(feats), T, feats[0].shape[1])
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f
    return x, lens, list(rows), list(vis), list(visw)


# ───────────────────────────── model ─────────────────────────────────────────
class VisAdapter(nn.Module):
    """Viseme head (owned here for stage control) + frame MLP + resampler +
    projection to Gemma space. Two resampler modes:
      fixed-K  (default): K learned queries, 2-layer cross-attention — one
               fixed-size summary regardless of span length.
      per_frame (arm G): length-proportional tokens — stride-2 slice of the
               per-frame z (adaptive stride caps at max_pf tokens), learned
               temporal pos-emb, 2 bidirectional self-attention layers. No
               pooling: temporal structure reaches Gemma intact (LLaVA-style
               per-token projection instead of Q-Former-style queries).
    Returns (soft [B,n_max,gemma_dim], soft_lens [B], ctc_logits [B,T,14])."""

    def __init__(self, head: nn.Module, gemma_dim: int, K: int = 8,
                 d: int = 512, n_layers: int = 2, n_heads: int = 8,
                 per_frame: bool = False, max_pf: int = 40,
                 gap_channel: bool = False, min_stride: int = 2,
                 emb_init: bool = False):
        super().__init__()
        self.head = head
        # emb_init: soft token = head-posterior-weighted mixture of
        # the REAL viseme-letter token embeddings (on-manifold, readable by
        # the frozen LM from step 0) + a zero-initialized learned residual.
        # Attacks the warmup plateau at its root: the prior-only basin is no
        # longer the nearest minimum because the lip->content path is open
        # at init for every seed.
        self.emb_init = emb_init
        if emb_init:
            self.register_buffer("vis_emb", torch.zeros(N_CLS, gemma_dim))
        self.per_frame, self.max_pf, self.K = per_frame, max_pf, K
        self.min_stride = min_stride   # 2 = original behavior (repro-exact)
        self.gap_channel = gap_channel
        self.d = d
        self.stash_states = False   # xattn mode: keep pre-proj states around
        d_h = head.proj.out_features
        self.frame_mlp = nn.Sequential(
            nn.Linear(d_h + N_CLS + 2 + int(gap_channel), d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d))
        if per_frame:
            self.pos_emb = nn.Parameter(torch.randn(max_pf, d) * 0.02)
        else:
            self.queries = nn.Parameter(torch.randn(K, d) * 0.02)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, n_heads, batch_first=True)
            for _ in range(n_layers))
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_layers))
        self.ln_f = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.proj = nn.Linear(d, gemma_dim)
        self.out_ln = nn.LayerNorm(gemma_dim)
        self.out_gain = nn.Parameter(torch.tensor(1.0))  # set to emb RMS at init
        if emb_init:                       # residual starts at exactly 0
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, feats, lens, gapm=None):
        B, T, _ = feats.shape
        lens = lens.to(feats.device)
        pad = (torch.arange(T, device=feats.device)[None]
               >= lens[:, None])                            # True = padding
        h = self.head.proj(self.head.in_drop(feats)) + self.head.pos[:, :T]
        h = self.head.enc(h, src_key_padding_mask=pad)
        logits = self.head.head(h)                          # [B,T,14]
        logp = F.log_softmax(logits.float(), -1)
        ent = -(logp.exp() * logp).sum(-1, keepdim=True)
        tt = (torch.arange(T, device=feats.device).float()[None, :, None]
              / (lens.float()[:, None, None] - 1).clamp(min=1))
        parts = [h, logp, ent, tt.clamp(max=1.0)]
        if self.gap_channel:
            if gapm is None:
                gapm = torch.ones(B, T, device=feats.device)
            parts.append(gapm.to(h.dtype)[:, :, None])
        z = self.frame_mlp(torch.cat(parts, -1))
        if self.per_frame:
            # --min-stride 1 keeps every frame (stride-2 deletes narrow
            # emissions of peaky heads: +0.089 vs +0.029 in an earlier ablation);
            # default 2 = the original working-run behavior
            sel = [torch.arange(0, int(L),
                                max(self.min_stride,
                                    -(-int(L) // self.max_pf)),
                                device=z.device) for L in lens]
            n = torch.tensor([len(s) for s in sel], device=z.device)
            n_max = int(n.max())
            q = torch.zeros(B, n_max, z.shape[-1], device=z.device,
                            dtype=z.dtype)
            for i, s in enumerate(sel):
                q[i, : len(s)] = z[i, s]
            mix = None
            if self.emb_init:
                # posterior pooled over each stride window (keeps the peaky
                # CTC spikes that frame-selection drops), blank removed,
                # renormalized -> expected viseme-letter embedding
                pr = logp.exp()
                pr = pr.masked_fill(
                    F.one_hot(torch.tensor(BLANK, device=pr.device),
                              pr.shape[-1]).bool()[None, None], 0.0)
                mix = torch.zeros(B, n_max, self.vis_emb.shape[-1],
                                  device=z.device, dtype=torch.float32)
                for i, s in enumerate(sel):
                    st = int(s[1] - s[0]) if len(s) > 1 else 1
                    L_i = int(lens[i])
                    ws = torch.stack([pr[i, int(f): min(int(f) + st, L_i)]
                                      .mean(0) for f in s])      # [n_i, C]
                    ws = ws / ws.sum(-1, keepdim=True).clamp(min=1e-6)
                    mix[i, : len(s)] = ws @ self.vis_emb
            q = q + self.pos_emb[:n_max][None]
            kpm = torch.arange(n_max, device=z.device)[None] >= n[:, None]
            for at, lq, ff, lf in zip(self.attn, self.ln_q, self.ffn,
                                      self.ln_f):
                x = lq(q)
                a, _ = at(x, x, x, key_padding_mask=kpm, need_weights=False)
                q = q + a
                q = q + ff(lf(q))
            if self.stash_states:
                self.last_states = q            # [B, n_max, d] pre-projection
        else:
            n = torch.full((B,), self.K, device=z.device, dtype=torch.long)
            q = self.queries[None].expand(B, -1, -1)
            for at, lq, ff, lf in zip(self.attn, self.ln_q, self.ffn,
                                      self.ln_f):
                a, _ = at(lq(q), z, z, key_padding_mask=pad,
                          need_weights=False)
                q = q + a
                q = q + ff(lf(q))
        if self.emb_init and mix is not None:
            soft = mix + self.proj(q).float()     # on-manifold + 0-init residual
        else:
            soft = self.out_ln(self.proj(q)) * self.out_gain
        return soft, n, logits


class AvsrAdapter(nn.Module):
    """avsr_soft_tokens: the per-frame resampler + emb-init mixture of
    VisAdapter (arm G, per_frame), with the lip-evidence channel replaced by
    Auto-AVSR's video-only Conformer encoder + 5,049-unit CTC posteriors
    (top-K per frame, packed by GapDataset) instead of AV-HuBERT + the
    12-viseme head:
      E_t  = sum_k p_tk * GemmaEmb(text of unit k)   (blank/<unk>/<eos> out)
      z_t  = MLP([enc_t ; W_c E_t/m_t ; entropy_t ; max-p_t ; t/T])
      soft = stride-pooled E / m  (on-manifold expected unit embedding)
             + zero-init residual from the 2-layer self-attention resampler
    Token positions (tokens=):
      spike  (default) CTC collapse of the soft posteriorgram: frames whose
             non-blank mass >= spike_thresh, adjacent frames with the same
             argmax unit merged into ONE token (mass-weighted mixture over
             the run, renormalized -> full-magnitude expected embedding),
             capped at max_pf highest-mass runs (time order), single best
             frame as fallback. ~1.9 tokens/word; every token is evidence.
      stride the VisAdapter layout (every min_stride-th frame, posterior
             pooled per stride window). NO-GO on Auto-AVSR evidence: 87% of
             frames are blank with a non-blank residue of ~1e-3, and the
             renormalization turns that residue into full-magnitude noise
             tokens at ~2/3 of the positions (warmup plateau, grounding
             gap 0 at step 2000).
    Gathers and matmuls only (no scatter/index_add/cumsum) so --deterministic
    holds. Returns (soft [B,n,D], n [B], None): no head logits, no CTC anchor.
    enc_dim: width of the trunk columns in front of the packed posteriors
    (768 for Auto-AVSR; 0 for the draft_phones channel, whose only
    per-frame scalar is blank_p in the 'entropy' slot).
    """

    def __init__(self, gemma_dim: int, n_units: int, K: int, d: int = 512,
                 n_layers: int = 2, n_heads: int = 8, max_pf: int = 40,
                 min_stride: int = 2, cls_dim: int = 64,
                 tokens: str = "spike", spike_thresh: float = 0.3,
                 enc_dim: int = AVSR_ENC_DIM):
        super().__init__()
        assert tokens in ("spike", "stride"), tokens
        self.K, self.n_units, self.d = K, n_units, d
        self.enc_dim = enc_dim
        self.max_pf, self.min_stride = max_pf, min_stride
        self.tokens, self.spike_thresh = tokens, spike_thresh
        self.per_frame, self.emb_init, self.stash_states = True, True, False
        self.register_buffer("unit_emb", torch.zeros(n_units, gemma_dim))
        self.register_buffer("unit_ok", torch.zeros(n_units))
        self.register_buffer("out_gain", torch.tensor(1.0))   # API parity
        self.cls_proj = nn.Linear(gemma_dim, cls_dim, bias=False)
        self.frame_mlp = nn.Sequential(
            nn.Linear(enc_dim + cls_dim + 3, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d))
        self.pos_emb = nn.Parameter(torch.randn(max_pf, d) * 0.02)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, n_heads, batch_first=True)
            for _ in range(n_layers))
        self.ln_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_layers))
        self.ln_f = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.proj = nn.Linear(d, gemma_dim)
        nn.init.zeros_(self.proj.weight)          # residual starts at 0
        nn.init.zeros_(self.proj.bias)

    def split(self, feats):
        E, K = self.enc_dim, self.K
        assert feats.shape[-1] == E + 1 + 2 * K, \
            f"packed width {feats.shape[-1]} != {E + 1 + 2 * K} (K={K})"
        enc = feats[..., :E]
        ent = feats[..., E:E + 1]
        p = feats[..., E + 1:E + 1 + K]
        ids = feats[..., E + 1 + K:E + 1 + 2 * K].round().long() \
            .clamp_(0, self.n_units - 1)
        return enc, ent, p, ids

    def _spike_runs(self, mass, top1, lens):
        """CTC collapse on the soft posteriorgram. mass [B,T] non-blank mass,
        top1 [B,T] argmax unit. Per sample: frames with mass >= thresh;
        adjacent selected frames sharing the argmax unit form one run
        (start, end, peak-frame); keep the max_pf highest-mass runs in time
        order; fallback = the single highest-mass frame (mass may be 0 for
        a fully masked/zeroed slice -> the token is the residual only).
        Pure python on tiny CPU tensors -> deterministic."""
        m = mass.detach().float().cpu()
        u = top1.detach().cpu()
        out = []
        for i in range(m.shape[0]):
            L = int(lens[i])
            mi, ui = m[i, :L], u[i, :L]
            runs = []
            f = 0
            while f < L:
                if float(mi[f]) >= self.spike_thresh:
                    a, unit = f, int(ui[f])
                    while f + 1 < L and float(mi[f + 1]) >= self.spike_thresh \
                            and int(ui[f + 1]) == unit:
                        f += 1
                    seg = mi[a: f + 1]
                    runs.append((a, f + 1, a + int(seg.argmax()),
                                 float(seg.sum())))
                f += 1
            if not runs:
                pk = int(mi.argmax())
                runs = [(pk, pk + 1, pk, float(mi[pk]))]
            if len(runs) > self.max_pf:
                keep = sorted(sorted(range(len(runs)),
                                     key=lambda j: -runs[j][3])[: self.max_pf])
                runs = [runs[j] for j in keep]
            out.append([(a, b, pk) for a, b, pk, _ in runs])
        return out

    def forward(self, feats, lens, gapm=None):
        B, T, _ = feats.shape
        lens = lens.to(feats.device)
        enc, ent, p, ids = self.split(feats)
        w = p * self.unit_ok[ids]                                # [B,T,K]
        e_t = (w.unsqueeze(-1) * self.unit_emb[ids]).sum(-2)     # [B,T,D]
        m_t = w.sum(-1, keepdim=True)                            # [B,T,1]
        cls = self.cls_proj(e_t / m_t.clamp(min=1e-6))
        tt = (torch.arange(T, device=feats.device).float()[None, :, None]
              / (lens.float()[:, None, None] - 1).clamp(min=1)).clamp(max=1.0)
        z = self.frame_mlp(torch.cat([enc, cls, ent, p[..., :1], tt], -1))
        if self.tokens == "spike":
            runs = self._spike_runs(m_t[..., 0], ids[..., 0], lens)
            n = torch.tensor([len(r) for r in runs], device=z.device)
        else:
            sel = [torch.arange(0, int(L), max(self.min_stride,
                                               -(-int(L) // self.max_pf)),
                                device=z.device) for L in lens]
            n = torch.tensor([len(s) for s in sel], device=z.device)
        n_max = int(n.max())
        q = torch.zeros(B, n_max, z.shape[-1], device=z.device, dtype=z.dtype)
        mix = torch.zeros(B, n_max, self.unit_emb.shape[-1],
                          device=z.device, dtype=torch.float32)
        if self.tokens == "spike":
            for i, rs in enumerate(runs):
                # rs: list of (start, end, peak) frame runs, time-ordered
                pk = torch.tensor([r[2] for r in rs], device=z.device)
                q[i, : len(rs)] = z[i, pk]
                es = torch.stack([e_t[i, a:b].sum(0) for a, b, _ in rs])
                ms = torch.stack([m_t[i, a:b].sum(0) for a, b, _ in rs])
                mix[i, : len(rs)] = es / ms.clamp(min=1e-6)
        else:
            for i, s in enumerate(sel):
                q[i, : len(s)] = z[i, s]
                st = int(s[1] - s[0]) if len(s) > 1 else 1
                L_i = int(lens[i])
                es = torch.stack([e_t[i, int(f): min(int(f) + st, L_i)].sum(0)
                                  for f in s])                    # [n_i, D]
                ms = torch.stack([m_t[i, int(f): min(int(f) + st, L_i)].sum(0)
                                  for f in s])                    # [n_i, 1]
                mix[i, : len(s)] = es / ms.clamp(min=1e-6)
        q = q + self.pos_emb[:n_max][None]
        kpm = torch.arange(n_max, device=z.device)[None] >= n[:, None]
        for at, lq, ff, lf in zip(self.attn, self.ln_q, self.ffn, self.ln_f):
            x = lq(q)
            a, _ = at(x, x, x, key_padding_mask=kpm, need_weights=False)
            q = q + a
            q = q + ff(lf(q))
        if self.stash_states:
            self.last_states = q
        soft = mix + self.proj(q).float()
        return soft, n, None


class NullAdapter(nn.Module):
    """--evidence none (the paper recipe): NO lip evidence reaches Gemma. Same call contract
    as VisAdapter but zero soft tokens and no CTC logits; out_gain is a buffer (API parity, never trained)."""

    def __init__(self, gemma_dim: int):
        super().__init__()
        self.gemma_dim = gemma_dim
        self.per_frame, self.gap_channel = False, False
        self.register_buffer("out_gain", torch.tensor(1.0))

    def forward(self, feats, lens, gapm=None):
        B = feats.shape[0]
        return (torch.zeros(B, 0, self.gemma_dim, device=feats.device),
                torch.zeros(B, dtype=torch.long), None)


class RouteAdapter(nn.Module):
    """lang_routed_tokens: ONE proposer, TWO evidence channels chosen per row
    by language (deploy-legal: whisper's language id). Rows of the routed
    languages (en) carry Auto-AVSR spike tokens (AvsrAdapter); every other
    row carries the viseme-head tokens of the certified recipe (VisAdapter,
    per-frame + emb-init). GapDataset packs both into one [T, W] array with
    the channel flag in the LAST column (1 = avsr). Sub-batches are gathered
    with index_select and merged back by per-sample slice copies (no
    index_put/scatter) -> --deterministic holds. The viseme head's CTC anchor
    applies to avh rows only: last_channel tells the training loop which
    rows went where. Returns (soft, n, logits[B,T,N_CLS]; zeros on avsr
    rows)."""

    def __init__(self, avh: "VisAdapter", avsr: "AvsrAdapter"):
        super().__init__()
        assert avh.per_frame and avh.emb_init and not avh.gap_channel
        self.avh, self.avsr = avh, avsr
        self.per_frame, self.emb_init, self.gap_channel = True, True, False
        self.stash_states = False
        self.last_channel: list[bool] = []

    @property
    def head(self):
        return self.avh.head

    @property
    def out_gain(self):
        return self.avh.out_gain

    def forward(self, feats, lens, gapm=None):
        B, T, W = feats.shape
        assert W == route_width(self.avsr.K), (W, route_width(self.avsr.K))
        lens = lens.to(feats.device)
        flag = (feats[..., -1].amax(1) > 0.5).tolist()
        self.last_channel = flag
        ia = [i for i in range(B) if flag[i]]
        ih = [i for i in range(B) if not flag[i]]
        outs: dict[int, tuple] = {}
        logits = torch.zeros(B, T, N_CLS, device=feats.device,
                             dtype=torch.float32)
        if ih:
            idx = torch.tensor(ih, device=feats.device)
            s, n, lg = self.avh(feats[idx][..., :AVH_DIM], lens[idx])
            for j, i in enumerate(ih):
                outs[i] = (s[j, : int(n[j])], lg[j].float())
        if ia:
            idx = torch.tensor(ia, device=feats.device)
            wa = AVSR_ENC_DIM + 1 + 2 * self.avsr.K
            s, n, _ = self.avsr(feats[idx][..., :wa], lens[idx])
            for j, i in enumerate(ia):
                outs[i] = (s[j, : int(n[j])], None)
        n_all = torch.tensor([outs[i][0].shape[0] for i in range(B)],
                             device=feats.device)
        soft = torch.zeros(B, int(n_all.max()), self.avsr.unit_emb.shape[-1],
                           device=feats.device, dtype=torch.float32)
        for i in range(B):
            soft[i, : outs[i][0].shape[0]] = outs[i][0].float()
            if outs[i][1] is not None:
                logits[i] = outs[i][1]
        return soft, n_all, logits


# ─────────────── arm XA: gated cross-attention injection ─────────────────────
class GatedXAttn(nn.Module):
    """Flamingo-style gated cross-attention: the LM's residual stream attends
    to the viseme encoder states (VisAdapter pre-projection z, d=512).
    tanh(gate) starts at 0 -> the wrapped model is EXACTLY base Gemma at init;
    training opens the visual tap, and the gate magnitude is a direct
    grounding readout. Bottleneck d_inner keeps params ~4.5M/block."""

    def __init__(self, d_lm: int, d_enc: int, d_inner: int = 512,
                 n_heads: int = 8):
        super().__init__()
        self.h, self.dh = n_heads, d_inner // n_heads
        self.ln = nn.LayerNorm(d_lm)
        self.q = nn.Linear(d_lm, d_inner, bias=False)
        self.k = nn.Linear(d_enc, d_inner, bias=False)
        self.v = nn.Linear(d_enc, d_inner, bias=False)
        self.o = nn.Linear(d_inner, d_lm, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hs, enc, keep):
        # hs [B,L,d_lm]; enc [B,S,d_enc]; keep [B,S] bool, True = valid
        B, L, _ = hs.shape
        q = self.q(self.ln(hs)).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(enc).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.v(enc).view(B, -1, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep[:, None, None, :])
        a = a.transpose(1, 2).reshape(B, L, -1)
        return hs + torch.tanh(self.gate) * self.o(a)


class LayerWithXAttn(nn.Module):
    """Wraps one Gemma decoder layer; runs it, then adds the gated
    cross-attention read. Transparent to HF internals: unknown attributes
    (attention_type, self_attn, ...) delegate to the wrapped layer. Handles
    beam expansion (hidden batch = k * enc batch) via repeat_interleave,
    matching HF's _expand_inputs_for_generation ordering."""

    def __init__(self, layer, block, ctx: dict):
        super().__init__()
        self.layer = layer
        self.xattn_block = block
        object.__setattr__(self, "_xa_ctx", ctx)   # plain dict, not a module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, hidden_states, *a, **kw):
        out = self.layer(hidden_states, *a, **kw)
        hs = out[0] if isinstance(out, tuple) else out
        enc, keep = self._xa_ctx.get("enc"), self._xa_ctx.get("keep")
        if enc is not None:
            if hs.shape[0] != enc.shape[0]:
                rep = hs.shape[0] // enc.shape[0]
                enc = enc.repeat_interleave(rep, 0)
                keep = keep.repeat_interleave(rep, 0)
            hs = self.xattn_block(hs, enc.to(hs.dtype), keep)
        return ((hs,) + tuple(out[1:])) if isinstance(out, tuple) else hs


def find_decoder_layers(root: nn.Module) -> nn.ModuleList:
    for _, mod in root.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 8 \
                and "DecoderLayer" in type(mod[0]).__name__:
            return mod
    raise RuntimeError("no DecoderLayer ModuleList found in model")


def install_xattn(gemma, d_lm: int, d_enc: int, every: int, inner: int,
                  heads: int, ctx: dict, device) -> nn.ModuleList:
    layers = find_decoder_layers(gemma)
    idx = list(range(every - 1, len(layers), every))
    blocks = nn.ModuleList()
    for i in idx:
        blk = GatedXAttn(d_lm, d_enc, inner, heads).to(
            device=device, dtype=torch.bfloat16)
        layers[i] = LayerWithXAttn(layers[i], blk, ctx)
        blocks.append(blk)
    n_par = sum(p.numel() for p in blocks.parameters())
    print(f"xattn: {len(blocks)} gated blocks after layers {idx} "
          f"({n_par/1e6:.1f}M params, inner {inner})", flush=True)
    return blocks


# ─────────────────────── batch assembly for Gemma ────────────────────────────
def chat_affixes(tok, soft_label: str = "", soft_markers: str = ""):
    """(pre_ids, post_ids, end_id) wrapping user content in the model's chat
    template. Gemma-4 is template-locked: raw continuations degenerate
    (`1.\\n1.\\n...` loops), inside the template it answers cleanly. The
    generation prompt ends with an EMPTY pre-closed thought channel, so the
    supervised target follows it directly."""
    m = "\x00C\x00"
    s = tok.apply_chat_template([{"role": "user", "content": m}],
                                tokenize=False, add_generation_prompt=True)
    pre, post = s.split(m)
    end = tok.convert_tokens_to_ids("<turn|>")
    if end is None or end == getattr(tok, "unk_token_id", None):
        end = tok.eos_token_id
    lab = tok(soft_label, add_special_tokens=False).input_ids \
        if soft_label else []
    post_ids = tok(post, add_special_tokens=False).input_ids
    if soft_markers == "audio":
        # NATIVE modality framing: Gemma-4-Unified is trained to read
        # continuous embeddings between <|audio> ... <audio|> as SPEECH at
        # 25 tokens/s (40 ms frames, linear proj) — the exact rate and role
        # of our per-frame lip tokens. Unmarked vectors read as spam text.
        boa = tok.convert_tokens_to_ids("<|audio>")
        eoa = tok.convert_tokens_to_ids("<audio|>")
        assert None not in (boa, eoa)
        lab = lab + [boa]
        post_ids = [eoa] + post_ids
    return (tok(pre, add_special_tokens=False).input_ids,
            post_ids, end, lab)


# optional deploy-legal line 'Auto-AVSR transcript: <VSR full-window
# top-1>' per window (video-only Auto-AVSR; no GT). Filled by --avsr-json;
# empty dict => byte-identical prompts to the certified recipe.
_AVSR: dict = {}


def load_avsr_json(path):
    """window-or-row id -> top-1 transcript, from baseline_auto_avsr-shaped
    json(s) ({rows:[{id,nbest}]}) or a flat {id: text} dict."""
    import json as _json
    d = {}
    for pth in ([path] if not isinstance(path, (list, tuple)) else path):
        j = _json.loads(open(pth).read())
        if isinstance(j, dict) and "rows" in j:
            for r in j["rows"]:
                t = (r.get("nbest") or [""])[0]
                d[r["id"]] = t
                d[r["id"].rsplit("__k", 1)[0]] = t
        else:
            d.update(j)
    return d


# Reader-noise augmentation of the transcript line (TRAINING batches only; val and
# every decode see the clean line). Per sample, with prob p the line is corrupted at
# a rate drawn U(0, max_rate): each word is deleted / substituted by a random
# transcript word / followed by an inserted one (0.3 / 0.5 / 0.2) — the error types
# of a weaker or degraded lip reader — so Gemma learns calibrated trust in the line
# instead of a copy-or-rewrite habit. Off (None) => byte-identical prompts.
_AVSR_NOISE = None          # {"p": float, "max_rate": float, "rng": random.Random, "vocab": list}
_AVSR_NOISE_ON = False


def _corrupt(t: str) -> str:
    cfg = _AVSR_NOISE
    rng = cfg["rng"]
    if rng.random() >= cfg["p"]:
        return t
    rate, out = rng.uniform(0.0, cfg["max_rate"]), []
    for w in t.split():
        if rng.random() < rate:
            u = rng.random()
            if u < 0.3:
                continue
            if u < 0.8:
                out.append(rng.choice(cfg["vocab"]))
                continue
            out += [w, rng.choice(cfg["vocab"])]
        else:
            out.append(w)
    return " ".join(out)


# ---- Gemma-improvement options (reader-aware training) — every knob off by default ----
# reader pool (idea 1 + 6): [(name, weight, {window_id: [n-best full-window readings]})], the FIRST
# entry is the deployed reader used for val / decode; training batches draw a reader per row
# (weights) so the prompt carries REAL reader errors (clean / degraded / other-model readers).
_READERS: list = []
_READER_K = 0               # >0: prompt shows the reader's top-K DISTINCT readings (idea 1)
_READER_RNG = None          # random.Random(seed): deterministic per-row reader draws
_PLAN_WORDS = False         # target line 'Words: N' (idea 7)
_KEEP_DECISION = False      # target line 'Reader: keep|edit' (idea 1)
_MCL_SLOTS = 1              # idea 4: K hypothesis slots ('Hypothesis j of K' in the user turn)
_KEEP_CACHE: dict = {}
_KEEP_META: dict = {}       # id -> cut metadata (window_words, masked_words): TRUTH-based keep labels
# select-or-write (stage 1): the prompt lists the reader's distinct readings
# under letters A..J; the target line 'Pick: <letter>' names the first reading whose gap is exactly right
# (truth-based, --keep-meta), else 'Pick: none' (Gemma writes). Training shows a random 1..K readings
# per row (any reader, any n-best length); decode/val show K.
_SELECT_WRITE = False
SW_LETTERS = "ABCDEFGHIJ"
_SW_CACHE: dict = {}
# select-then-edit (stages 2-3): same lettered reading list; the target says
# 'Pick: <letter>' = the CLOSEST reading (fewest gap-word errors vs the reference, truth-based) and then
# 'Corrected: <reference window transcript>' — the model copies the picked reading and edits it; the output is a
# full-window transcript scored like the reader (score_full), so the whisper-context splice cap does not apply.
_SELECT_EDIT = False
_SE_CACHE: dict = {}
_NO_VIS = False      # --evidence none: no lip tokens, no viseme line in the target


def load_reader_nbest(spec):
    """'name=path[,path...]:weight' -> (name, weight, {window: [readings]}). Files are
    baseline_auto_avsr-shaped ({rows:[{id, nbest}]}, row ids may carry __k<N>)."""
    import glob as _glob
    name, rest = spec.split("=", 1)
    paths, _, w = rest.rpartition(":")
    try:
        weight = float(w)
    except ValueError:
        paths, weight = rest, 1.0
    d = {}
    for pat in paths.split(","):
        for pth in sorted(_glob.glob(pat)):
            j = json.loads(open(pth).read())
            for r in (j["rows"] if isinstance(j, dict) and "rows" in j else j):
                nb = r.get("nbest") or []
                nb = json.loads(nb) if isinstance(nb, str) else nb
                d.setdefault(r["id"].rsplit("__k", 1)[0], list(nb))
    return name, weight, d


def _reader_for(r, rd=None):
    """(name, readings) of the reader shown for row r; rd = an explicit pool index (training draw)."""
    if not _READERS:
        return None, []
    name, _, d = _READERS[rd or 0]
    w = r["id"].rsplit("__k", 1)[0]
    nb = d.get(w) or d.get(str(r.get("feats", "")).rsplit("/", 1)[-1][:-4]) or []
    return name, nb


def draw_reader(r):
    """training: weighted reader draw per row (deterministic RNG); val/decode: the deployed reader"""
    if not _READERS or not _AVSR_NOISE_ON or len(_READERS) == 1:
        return 0
    ws = [w for _, w, _ in _READERS]
    x, acc = _READER_RNG.random() * sum(ws), 0.0
    for i, w in enumerate(ws):
        acc += w
        if x < acc:
            return i if _reader_for(r, i)[1] else 0
    return 0


def shown_readings(r, rd=None):
    """the reader's distinct readings shown in the prompt (best first); select-or-write rows carry
    their own per-row count r['_rk'] (training draw), otherwise _READER_K"""
    name, nb = _reader_for(r, rd)
    k = int(r.get("_rk") or _READER_K)
    out, seen = [], set()
    for h in nb:
        key = " ".join(h.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(h.strip())
        if len(out) >= k:
            break
    return out


def _reader_lines(r, rd=None):
    out = shown_readings(r, rd)
    if not out:
        return ""
    if _SELECT_WRITE or _SELECT_EDIT:
        return "Lip reading candidates (best first):\n" + "".join(
            f"{SW_LETTERS[i]}. {h}\n" for i, h in enumerate(out))
    return "Lip reading (best first):\n" + "".join(f"{i + 1}. {h}\n" for i, h in enumerate(out))


def pick_label(r, rd=None):
    """select-or-write target: letter of the FIRST shown reading whose gap is exactly right against the
    reference window (score_full on --keep-meta: truth-based), else 'none'. Exactness is cached per
    (reader, cut, reading)."""
    name, _ = _reader_for(r, rd)
    out = shown_readings(r, rd)
    meta = _KEEP_META.get(r["id"])
    for j, h in enumerate(out):
        key = (name, r["id"], h)
        if key not in _SW_CACHE:
            if meta is not None:
                from label_gap_score import score_full
                v = score_full(meta, h)
                _SW_CACHE[key] = bool(v and v["g_exact"])
            else:           # no metadata: lossy deploy-legal extraction (measured to miss 60% of rights)
                from reader_gap import gap_of
                from label_scoring import norm_tokens
                lg = r.get("language", "en")
                g = gap_of(r["context"], h, lg)
                _SW_CACHE[key] = bool(g) and norm_tokens(g, lg) == norm_tokens(r["target"], lg)
        if _SW_CACHE[key]:
            return SW_LETTERS[j]
    return "none"


def closest_label(r, rd=None):
    """select-then-edit target: letter of the shown reading with the fewest gap errors (score_full g_wer, then
    word acc; ties -> the earlier = better-ranked reading). Needs --keep-meta (truth-based)."""
    name, _ = _reader_for(r, rd)
    out = shown_readings(r, rd)
    meta = _KEEP_META[r["id"]]
    best, best_key = 0, None
    for j, h in enumerate(out):
        key = (name, r["id"], h)
        if key not in _SE_CACHE:
            from label_gap_score import score_full
            v = score_full(meta, h)
            _SE_CACHE[key] = (v["g_wer"], -v["g_acc"]) if v else (99.0, 0.0)
        if best_key is None or _SE_CACHE[key] < best_key:
            best, best_key = j, _SE_CACHE[key]
    return SW_LETTERS[best]


def window_truth(r):
    """reference transcript of the cut's whole window, in the readers' style (lower case, no punctuation)"""
    import re as _re
    ws = [w["word"] for w in _KEEP_META[r["id"]]["window_words"]]
    return " ".join(t for t in (_re.sub(r"[^\w' -]", "", w.lower()).strip() for w in ws) if t)


def answer_text(r):
    """the supervised answer after the plan lines"""
    if _SELECT_EDIT:
        return f"Corrected: {window_truth(r)}"
    return f"Missing words: {r['target']}"


def _keep_label(r, rd=None):
    """1 if the shown reader's top-1 full-window reading gets the gap exactly right, scored against
    the reference window (label_gap_score.score_full; --keep-meta). Fallback without metadata:
    gap-extraction against the whisper context — measured to miss 60% of truly-right readings
    , so training uses the truth-based label."""
    name, nb = _reader_for(r, rd)
    key = (name, r["id"])
    if key not in _KEEP_CACHE:
        lg = r.get("language", "en")
        if r["id"] in _KEEP_META and nb:
            from label_gap_score import score_full
            v = score_full(_KEEP_META[r["id"]], nb[0])
            _KEEP_CACHE[key] = int(bool(v and v["g_exact"]))
        else:
            from reader_gap import gap_of
            from label_scoring import norm_tokens
            g = gap_of(r["context"], nb[0], lg) if nb else ""
            _KEEP_CACHE[key] = int(bool(g) and norm_tokens(g, lg) == norm_tokens(r["target"], lg))
    return _KEEP_CACHE[key]


def plan_lines(r, rd=None):
    """supervised lines between the viseme line and 'Missing words:' (ideas 7 + 1)"""
    s = ""
    if _PLAN_WORDS:
        s += f"Words: {len(r['target'].split())}\n"
    if _KEEP_DECISION:
        s += f"Reader: {'keep' if _keep_label(r, rd) else 'edit'}\n"
    if _SELECT_WRITE:
        s += f"Pick: {pick_label(r, rd)}\n"
    if _SELECT_EDIT:
        s += f"Pick: {closest_label(r, rd)}\n"
    return s


def _avsr_line(r, rd=None):
    if _READER_K > 0 and _READERS:
        return _reader_lines(r, rd)
    if not _AVSR:
        return ""
    t = _AVSR.get(r["id"]) or _AVSR.get(r["id"].rsplit("__k", 1)[0]) \
        or _AVSR.get(str(r.get("feats", "")).rsplit("/", 1)[-1][:-4])
    if t and _AVSR_NOISE is not None and _AVSR_NOISE_ON:
        t = _corrupt(t) or t
    return f"Auto-AVSR transcript: {t}\n" if t else ""


def _ctx_text(r, gap_dur=False, rd=None):
    """User-turn text. gap_dur adds the gap's length in seconds — the word-
    count cue for full-window mode (deploy-legal: the corrupted region's
    timing is known at inference). Off => byte-identical to the old prompt.
    rd = training reader draw (None = the deployed reader); r['_slot'] = MCL slot."""
    s = f"Transcript: {r['context']}\n"
    s += _avsr_line(r, rd)
    if gap_dur:
        s += f"Gap duration: {(r['f1'] - r['f0']) / 25.0:.1f} s\n"
    if _MCL_SLOTS > 1:
        s += f"Hypothesis {int(r.get('_slot', 0)) + 1} of {_MCL_SLOTS}\n"
    return s


def build_batch(tok, emb, soft, soft_lens, rows, device, vis_strs=None,
                interleaved=False, chat=None, vis_only=False, gap_dur=False):
    """vis_strs != None (arm G): the target is a two-field chain —
    'Lip visemes: <gold viseme string>\\nMissing words: <gold words>' — both
    supervised, so reducing the viseme part's loss REQUIRES reading the visual
    tokens (it is not predictable from context).
    interleaved (arm H): vis_strs already contains the full per-word chain
    '[vis] word [vis] word' — words are inside; no second field."""
    dtype = emb.weight.dtype
    segs = []
    for j, r in enumerate(rows):
        rd = draw_reader(r)
        if (_SELECT_WRITE or _SELECT_EDIT) and _AVSR_NOISE_ON and _READER_RNG is not None:
            r = dict(r, _rk=_READER_RNG.randint(1, max(1, _READER_K)))   # n-best length 1..K per row
        pl = plan_lines(r, rd)
        if chat is not None:
            # template mode: soft tokens sit inside the user turn; the cue
            # moves INTO the supervised response after the template scaffold
            pre, post, end, lab = chat
            a = pre + tok(_ctx_text(r, gap_dur, rd),
                          add_special_tokens=False).input_ids + lab
            b = list(post)
            n_sup = None
            if vis_strs is None and _NO_VIS:
                t_text = f"{pl}{answer_text(r)}"      # 'Pick: X' + 'Corrected: ...', no 'Lip visemes:' line
                t = tok(t_text, add_special_tokens=False).input_ids + [end]
            elif vis_strs is None:
                t_text = f"Missing words: {r['target']}"
                t = tok(t_text, add_special_tokens=False).input_ids + [end]
            elif interleaved:
                t_text = f"{AUX_LABEL} {vis_strs[j]}"
                t = tok(t_text, add_special_tokens=False).input_ids + [end]
            elif vis_only:
                # warmup-viseme-only: tokenize in two parts so supervision
                # ends exactly after the viseme line (+\n); the words stay
                # teacher-forced in the input but carry no loss
                tv = tok(f"{AUX_LABEL} {vis_strs[j]}\n",
                         add_special_tokens=False).input_ids
                tr = tok(f"{pl}{answer_text(r)}",
                         add_special_tokens=False).input_ids + [end]
                t, n_sup = tv + tr, len(tv)
            else:
                t_text = f"{AUX_LABEL} {vis_strs[j]}\n" \
                         f"{pl}{answer_text(r)}"
                t = tok(t_text, add_special_tokens=False).input_ids + [end]
            segs.append((a, b, t, n_sup))
            continue
        a = tok(_ctx_text(r, gap_dur, rd)).input_ids
        # Gemma-4's tokenizer does NOT auto-add <bos> (Gemma-3's did); a
        # missing BOS makes Gemma confidently wrong (CE above uniform).
        if tok.bos_token_id is not None and a[:1] != [tok.bos_token_id]:
            a = [tok.bos_token_id] + a
        n_sup = None
        if vis_strs is None:
            b = tok("Missing words:", add_special_tokens=False).input_ids
            t_text = " " + r["target"]
        elif interleaved:
            b = tok(AUX_LABEL, add_special_tokens=False).input_ids
            t_text = " " + vis_strs[j]
        elif vis_only:
            b = tok(AUX_LABEL, add_special_tokens=False).input_ids
            tv = tok(f" {vis_strs[j]}\n",
                     add_special_tokens=False).input_ids
            tr = tok(f"{pl}{answer_text(r)}",
                     add_special_tokens=False).input_ids \
                + [tok.eos_token_id]
            segs.append((a, b, tv + tr, len(tv)))
            continue
        else:
            b = tok(AUX_LABEL, add_special_tokens=False).input_ids
            t_text = f" {vis_strs[j]}\n{pl}{answer_text(r)}"
        t = tok(t_text, add_special_tokens=False).input_ids \
            + [tok.eos_token_id]
        segs.append((a, b, t, n_sup))
    L = max(len(a) + int(soft_lens[j]) + len(b) + len(t)
            for j, (a, b, t, _) in enumerate(segs))
    ie = torch.zeros(len(rows), L, emb.weight.shape[1],
                     dtype=dtype, device=device)
    am = torch.zeros(len(rows), L, dtype=torch.long, device=device)
    lb = torch.full((len(rows), L), -100, dtype=torch.long, device=device)
    for i, (a, b, t, n_sup) in enumerate(segs):
        n = int(soft_lens[i])
        ea = emb(torch.tensor(a, device=device))
        eb = emb(torch.tensor(b, device=device))
        et = emb(torch.tensor(t, device=device))
        row = torch.cat([ea, soft[i, :n].to(dtype), eb, et])
        ie[i, : row.shape[0]] = row
        am[i, : row.shape[0]] = 1
        off = len(a) + n + len(b)
        ns = len(t) if n_sup is None else n_sup
        lb[i, off: off + ns] = torch.tensor(t[:ns], device=device)
    return ie, am, lb


@torch.no_grad()
def decode_rows(rows_ds, idxs, adapter, gemma, tok, emb, device, max_new=12,
                zero_visual=False, viseme_cot=False, interleaved=False,
                xa_ctx=None, chat=None, nbest=1, gap_dur=False):
    adapter.eval()
    gemma.eval()
    out = []
    for i in idxs:
        feats, r, _, _ = rows_ds[i]
        if zero_visual:
            feats = torch.zeros_like(feats)
        gapm = None
        if getattr(adapter, "gap_channel", False):
            gapm = torch.zeros(1, len(feats), device=device)
            gapm[0, r["f0"]: r["f1"]] = 1.0
        soft, sl, _ = adapter(feats[None].to(device),
                              torch.tensor([len(feats)]), gapm=gapm)
        if xa_ctx is not None:                # visual enters via xattn, not
            xa_ctx["enc"] = adapter.last_states           # the input prefix
            xa_ctx["keep"] = (torch.arange(adapter.last_states.shape[1],
                                           device=device)[None]
                              < sl.to(device)[:, None])
            sl = torch.zeros_like(sl)
        if chat is not None:
            pre, post, end, lab = chat
            a = pre + tok(_ctx_text(r, gap_dur),
                          add_special_tokens=False).input_ids + lab
            b = list(post)
            eos_ids = [end, tok.eos_token_id]
        else:
            a = tok(_ctx_text(r, gap_dur)).input_ids
            if tok.bos_token_id is not None and a[:1] != [tok.bos_token_id]:
                a = [tok.bos_token_id] + a
            cue = AUX_LABEL if viseme_cot else "Missing words:"
            b = tok(cue, add_special_tokens=False).input_ids
            eos_ids = tok.eos_token_id
        ie = torch.cat([emb(torch.tensor(a, device=device)),
                        soft[0, : int(sl[0])].to(emb.weight.dtype),
                        emb(torch.tensor(b, device=device))])[None]
        am = torch.ones(1, ie.shape[1], dtype=torch.long, device=device)
        beam_kw = {"num_beams": nbest,
                   "num_return_sequences": nbest} if nbest > 1 else {}
        gen = gemma.generate(inputs_embeds=ie, attention_mask=am,
                             max_new_tokens=max_new, do_sample=False,
                             eos_token_id=eos_ids,
                             pad_token_id=tok.eos_token_id, use_cache=True,
                             **beam_kw)
        gt = [norm(w) for w in r["target"].split()]
        cands = []
        for g_ in gen[:nbest]:
            text = tok.decode(g_, skip_special_tokens=True).strip()
            if interleaved:
                words_text = re.sub(r"\[[^\]]*\]", " ", text)
            elif viseme_cot or (chat is not None and "Missing words:" in text):
                words_text = text.split("Missing words:")[-1]
            else:
                words_text = text
            pred = [norm(w) for w in words_text.split()][:6]
            sm = difflib.SequenceMatcher(a=gt, b=pred)
            cands.append((sum(bl.size for bl in sm.get_matching_blocks())
                          / max(1, len(gt)), pred == gt, text))
        out.append({"id": r["id"], "k": r["k"], "gt": r["target"],
                    "pred": cands[0][2], "pos": cands[0][0],
                    "exact": cands[0][1],
                    "bestpos": max(c[0] for c in cands),
                    "hit": any(c[1] for c in cands)})
    adapter.train()
    return out


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _eval_attn(gemma, impl: str):
    """Temporarily switch the LM's attention implementation for eval.
    Forward-only decoding is deterministic under SDPA/flash (only their
    BACKWARD kernels are nondeterministic), so eval can run fast attention
    without giving up the bitwise-repro guarantee of eager training."""
    if not impl:
        yield
        return
    # silence the per-toggle "sub-config vision/audio without sub-model"
    # warnings: our checkpoint instantiates no vision/audio attention —
    # the flag lands on configs nothing reads
    from transformers.utils import logging as hf_logging

    def _set(v):
        lvl = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            gemma.set_attn_implementation(v)
        finally:
            hf_logging.set_verbosity(lvl)

    prev = gemma.config._attn_implementation
    _set(impl)
    try:
        yield
    finally:
        _set(prev)


def decode_rows_batched(rows_ds, idxs, adapter, gemma, tok, emb, device,
                        max_new=12, zero_visual=False, viseme_cot=False,
                        interleaved=False, xa_ctx=None, chat=None, nbest=1,
                        batch_size=8, gap_dur=False):
    """Batched twin of decode_rows: LEFT-padded inputs_embeds, one generate
    per chunk. Same output contract; xa_ctx arms fall back to sequential."""
    if xa_ctx is not None:
        return decode_rows(rows_ds, idxs, adapter, gemma, tok, emb, device,
                           max_new=max_new, zero_visual=zero_visual,
                           viseme_cot=viseme_cot, interleaved=interleaved,
                           xa_ctx=xa_ctx, chat=chat, nbest=nbest,
                           gap_dur=gap_dur)
    adapter.eval()
    gemma.eval()
    dtype = emb.weight.dtype
    out = []
    for c0 in range(0, len(idxs), batch_size):
        chunk = idxs[c0: c0 + batch_size]
        feats_l, rows = [], []
        for i in chunk:
            f, r, _, _ = rows_ds[i]
            if zero_visual:
                f = torch.zeros_like(f)
            feats_l.append(f)
            rows.append(r)
        T = max(len(f) for f in feats_l)
        fb = torch.zeros(len(rows), T, feats_l[0].shape[-1])
        for j, f in enumerate(feats_l):
            fb[j, : len(f)] = f
        lens = torch.tensor([len(f) for f in feats_l])
        gapm = None
        if getattr(adapter, "gap_channel", False):
            gapm = torch.zeros(len(rows), T, device=device)
            for j, r in enumerate(rows):
                gapm[j, r["f0"]: r["f1"]] = 1.0
        with torch.no_grad():
            soft, sl, _ = adapter(fb.to(device), lens, gapm=gapm)
            prefixes, eos_ids = [], tok.eos_token_id
            for j, r in enumerate(rows):
                if chat is not None:
                    pre, post, end, lab = chat
                    a = pre + tok(_ctx_text(r, gap_dur),
                                  add_special_tokens=False).input_ids + lab
                    b = list(post)
                    eos_ids = [end, tok.eos_token_id]
                else:
                    a = tok(_ctx_text(r, gap_dur)).input_ids
                    if tok.bos_token_id is not None \
                            and a[:1] != [tok.bos_token_id]:
                        a = [tok.bos_token_id] + a
                    cue = AUX_LABEL if viseme_cot else "Missing words:"
                    b = tok(cue, add_special_tokens=False).input_ids
                prefixes.append(torch.cat([
                    emb(torch.tensor(a, device=device)),
                    soft[j, : int(sl[j])].to(dtype),
                    emb(torch.tensor(b, device=device))]))
            L = max(p.shape[0] for p in prefixes)
            ie = torch.zeros(len(rows), L, prefixes[0].shape[-1],
                             dtype=dtype, device=device)
            am = torch.zeros(len(rows), L, dtype=torch.long, device=device)
            for j, p in enumerate(prefixes):           # LEFT pad
                ie[j, L - p.shape[0]:] = p
                am[j, L - p.shape[0]:] = 1
            beam_kw = {"num_beams": nbest,
                       "num_return_sequences": nbest} if nbest > 1 else {}
            gen = gemma.generate(inputs_embeds=ie, attention_mask=am,
                                 max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=eos_ids,
                                 pad_token_id=tok.eos_token_id,
                                 use_cache=True, **beam_kw)
        for j, r in enumerate(rows):
            gt = [norm(w) for w in r["target"].split()]
            seqs = gen[j * nbest: (j + 1) * nbest] if nbest > 1 \
                else gen[j: j + 1]
            cands = []
            for g_ in seqs:
                text = tok.decode(g_, skip_special_tokens=True).strip()
                if interleaved:
                    words_text = re.sub(r"\[[^\]]*\]", " ", text)
                elif viseme_cot or (chat is not None
                                    and "Missing words:" in text):
                    words_text = text.split("Missing words:")[-1]
                else:
                    words_text = text
                pred = [norm(w) for w in words_text.split()][:6]
                sm = difflib.SequenceMatcher(a=gt, b=pred)
                cands.append((sum(bl.size for bl in sm.get_matching_blocks())
                              / max(1, len(gt)), pred == gt, text))
            out.append({"id": r["id"], "k": r["k"], "gt": r["target"],
                        "pred": cands[0][2], "pos": cands[0][0],
                        "exact": cands[0][1],
                        "bestpos": max(c[0] for c in cands),
                        "hit": any(c[1] for c in cands)})
    adapter.train()
    return out


# ───────────────────────────── main ──────────────────────────────────────────
# ── Recipe: options fixed to the values of the released runs ────────────────────────────────────────────────────
# Each option below has one value in all three stages of L; passing it on the command line is an error. The values
# land in args (and in args.json / the checkpoint). Internal branches guarded by other values are never taken.
PAPER_FIXED = {
    "evidence": "none", "ctc_weight": 0.0, "arm": "G", "head_ckpt": None, "K": 8,
    "avsr_evidence_dir": None, "avsr_units_file": None, "avsr_topk": 8, "avsr_cls_dim": 64,
    "avsr_tokens": "spike", "avsr_spike_thresh": 0.3, "route_langs": ["en"], "drop_langs": [],
    "aux_line": "visemes", "phone_targets": None, "val_one_cut": False, "val_lang_weights": "",
    "full_window": False, "no_gap_channel": False, "avsr_noise": None, "avsr_json": None,
    "plan_words": False, "keep_decision": False, "mcl_slots": 1, "mcl_eps": 0.05,
    "slice_pad_s": 0.0, "min_stride": 2, "max_pf": 40, "warmup_only": False, "val_k_max": 0,
    "val_en_all": False, "warmup_viseme_only": False, "validate_batched": 0, "emb_init": False,
    "lora_ramp_steps": 0, "probe_grounding": 0, "lr_head": 5e-6, "ml_lexicons": None,
    "ctx_full_drop": 0.0, "cfd_cosine": False, "cfd_start": 1.0, "cfd_floor": 0.15, "cfd_anneal": 2500,
    "soft_markers": "", "chat_soft_label": "", "one_cut_fixed_k": False, "init_adapter": None,
    "inject_mode": "prefix", "xattn_every": 8, "xattn_inner": 512, "xattn_heads": 8, "lr_xattn": 1e-4,
    "gate_lr_mult": 10.0, "lang_balance_alpha": 1.0,
}


def _fix_paper_options(ap: argparse.ArgumentParser) -> None:
    """Hide the PAPER_FIXED options from --help, reject them on the command line, set their paper values."""
    fixed_flags = set()
    for act in ap._actions:
        if act.dest in PAPER_FIXED:
            act.help = argparse.SUPPRESS
            fixed_flags.update(act.option_strings)
    bad = sorted({a.split("=", 1)[0] for a in sys.argv[1:] if a.split("=", 1)[0] in fixed_flags})
    if bad:
        ap.error(f"{', '.join(bad)}: fixed to the paper recipe in this release (see PAPER_FIXED)")
    missing = set(PAPER_FIXED) - {a.dest for a in ap._actions}
    assert not missing, missing
    ap.set_defaults(**PAPER_FIXED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="dir with train/val jsonl")
    ap.add_argument("--head-ckpt", type=Path, default=None,
                    help="viseme head (required for --evidence avh)")
    ap.add_argument("--evidence",
                    choices=["avh", "avsr", "route", "draft_phones", "none"],
                    default="avh",
                    help="avsr_soft_tokens: lip evidence = Auto-AVSR encoder "
                         "+ CTC posteriors (AvsrAdapter) instead of "
                         "AV-HuBERT + viseme head (byte-identical default); "
                         "draft_phones (ablation): IPA phone "
                         "posteriors of the lips-only LTX draft "
                         "(draft_extract_phones.py npz per row, 50 Hz) — "
                         "AvsrAdapter with no trunk columns")
    ap.add_argument("--avsr-evidence-dir", type=Path, nargs="+", default=None,
                    help="dirs of extract_avsr_evidence.py npz (train+val)")
    ap.add_argument("--avsr-units-file", type=Path, default=Path(
        "third_party/auto_avsr_v1_joint/spm/unigram/unigram5000_units.txt"))
    ap.add_argument("--avsr-topk", type=int, default=8,
                    help="CTC units per frame entering the mixture")
    ap.add_argument("--avsr-cls-dim", type=int, default=64)
    ap.add_argument("--avsr-tokens", choices=["spike", "stride"],
                    default="spike",
                    help="soft-token positions: spike = CTC collapse of the "
                         "posteriorgram (default); stride = VisAdapter "
                         "layout (NO-GO on this evidence, kept for record)")
    ap.add_argument("--avsr-spike-thresh", type=float, default=0.3,
                    help="non-blank mass that makes a frame a spike")
    ap.add_argument("--route-langs", nargs="+", default=["en"],
                    help="--evidence route: languages that get the Auto-AVSR "
                         "channel; all others keep the viseme-head channel")
    ap.add_argument("--drop-langs", nargs="*", default=[],
                    help="languages removed from train AND val before any "
                         "selection/sampling")
    ap.add_argument("--aux-line", choices=["visemes", "phones"],
                    default="visemes",
                    help="arm G first supervised field: gold viseme letters "
                         "(certified) or 'Phones:' = the phone recognizer's "
                         "reading of the CLEAN gap crop (--phone-targets; "
                         "same alphabet as the draft_phones emb-init tokens)")
    ap.add_argument("--phone-targets", type=Path, nargs="+", default=None,
                    help="build_phone_ctc_targets.py json(s) {targets: "
                         "{row_id: [phone ids]}} for --aux-line phones")
    ap.add_argument("--aux-max-new", type=int, default=44,
                    help="decode budget for the arm-G chain (44 fits the "
                         "viseme line; the IPA phone line needs ~96: 1.54 "
                         "tok/phone, p90 55 tokens)")
    ap.add_argument("--val-one-cut", action="store_true",
                    help="decode ONE cut per val window, k round-robin 1..4 "
                         "(the frozen-test protocol) instead of a random "
                         "--val-decode-n sample of k=1..10 cuts")
    ap.add_argument("--val-lang-weights", default="",
                    help="'en:0.5,es:0.1667,...': model-selection metric = "
                         "weighted mean of per-language val bestpos/pos")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=["B", "C", "G", "H"], default="B",
                    help="B: frozen head, fixed-K tokens. C: +trainable head "
                         "w/ CTC anchor. G: C + per-frame tokens + viseme "
                         "string in the supervised target (transcribe-then-"
                         "fill). H: G but INTERLEAVED per-word chunks "
                         "'[vis] word [vis] word' — each word locally "
                         "grounded, each prediction re-anchors the next")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=800)
    ap.add_argument("--full-window", action="store_true",
                    help="feed the WHOLE window's AVH features to the "
                         "adapter (not just the gap span) with a per-frame "
                         "in-gap flag channel; CTC anchor is sliced to the "
                         "gap region (its labels are gap visemes)")
    ap.add_argument("--no-gap-channel", action="store_true",
                    help="with --full-window: DROP the per-frame in-gap flag "
                         "channel (pure soft alignment — pair with "
                         "--gap-dur-prompt for the word-count cue)")
    ap.add_argument("--avsr-noise", type=str, default=None,
                    help="reader-noise augmentation of the transcript line in TRAINING "
                         "batches, 'p,max_rate' (e.g. '0.5,0.5'); off by default")
    ap.add_argument("--avsr-json", type=Path, nargs="+", default=None,
                    help="json(s) with the Auto-AVSR VSR full-window "
                         "top-1 transcript per window -> extra prompt line "
                         "'Auto-AVSR transcript: ...' (deploy-legal, video-only)")
    ap.add_argument("--gap-dur-prompt", action="store_true",
                    help="append 'Gap duration: X.X s' to the user turn "
                         "(deploy-legal length cue for wide/full inputs)")
    # ---- Gemma-improvement options (reader-aware training) ----
    ap.add_argument("--reader-pool", nargs="+", default=None,
                    help="'name=path_glob[,glob]:weight' window-level reader n-best jsons; the FIRST is "
                         "the deployed reader (val/decode); training draws one per row by weight (idea 6)")
    ap.add_argument("--reader-nbest-k", type=int, default=0,
                    help="prompt shows the drawn reader's top-K distinct full-window readings (idea 1); "
                         "0 = old single 'Auto-AVSR transcript:' line from --avsr-json")
    ap.add_argument("--plan-words", action="store_true", help="target line 'Words: N' (idea 7)")
    ap.add_argument("--keep-decision", action="store_true",
                    help="target line 'Reader: keep|edit' = is the shown reader's top-1 gap reading right (idea 1)")
    ap.add_argument("--keep-meta", type=Path, nargs="+", default=None,
                    help="metadata.json(s) of the training cuts: truth-based 'Reader: keep|edit' labels")
    ap.add_argument("--mcl-slots", type=int, default=1,
                    help="idea 4: K hypothesis slots, relaxed winner-take-all loss (min over slots + eps*mean)")
    ap.add_argument("--mcl-eps", type=float, default=0.05)
    ap.add_argument("--select-edit", action="store_true",
                    help="select-then-edit: lettered reader readings, target 'Pick: <closest reading>' + 'Corrected: "
                         "<reference window transcript>'. Needs --reader-pool, --reader-nbest-k <= 10, --keep-meta")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="warm start adapter (+ head) and LoRA from a checkpoint, step 0 and a fresh optimizer "
                         "(--resume takes precedence when it exists)")
    ap.add_argument("--select-write", action="store_true",
                    help="select-or-write: reader readings listed under letters A..J, target line 'Pick: <letter>|none' "
                         "(first exactly-right reading, truth-based via --keep-meta); training shows 1..K readings "
                         "per row. Needs --reader-pool, --reader-nbest-k <= 10, --keep-meta")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save last.pt (+ optimizer state) every N steps between evals: chunked training "
                         "in 45-min debug slots keeps its progress")
    ap.add_argument("--keep-ckpts", default="",
                    help="comma list of steps whose checkpoint is kept as ckpt_s<step>.pt (validation decodes)")
    ap.add_argument("--slice-pad-s", type=float, default=0.0,
                    help="widen the feature slice to span +- this many "
                         "seconds (desync-tolerant middle ground between "
                         "exact span and --full-window); CTC anchor stays "
                         "on the true span via sf0/sf1")
    ap.add_argument("--min-stride", type=int, default=2,
                    help="per-frame resampler minimum stride; 2 = original "
                    "working-run behavior, 1 = keep every frame")
    ap.add_argument("--max-pf", type=int, default=40,
                    help="per-frame token cap (use 64 for --full-window: "
                         "120-frame windows at stride 2)")
    ap.add_argument("--warmup-only", action="store_true",
                    help="never leave the adapter-only stage: Gemma+LoRA and "
                         "head stay frozen for the whole run (prompt-tuning "
                         "regime — the frozen LM cannot memorize, so any "
                         "grounding is pure visual-channel steering)")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-decode-n", type=int, default=150)
    ap.add_argument("--val-nbest", type=int, default=1,
                    help=">1: beam n-best val decode; checkpoint selection "
                         "moves to mean best-pos over the n-best (downstream "
                         "LTX consumes the n-best, not top-1)")
    ap.add_argument("--val-k-max", type=int, default=0,
                    help="restrict the val-selection pool to rows with "
                         "k<=N (0 = no filter); frozen test is k1-4")
    ap.add_argument("--val-en-all", action="store_true",
                    help="put every en row of the (filtered) val pool in "
                         "first, fill the remainder with a seeded shuffle "
                         "of the other languages")
    ap.add_argument("--warmup-viseme-only", action="store_true",
                    help="during the adapter warmup: context fully dropped "
                         "(cfd=1.0 via the ___ mechanism) and loss only on "
                         "the viseme line — the frozen LM's CE can then "
                         "drop ONLY through lip information, turning the "
                         "warmup into a well-conditioned alignment task")
    ap.add_argument("--decode-batch", type=int, default=1,
                    help=">1: batched val/gap decoding (left-padded "
                         "inputs_embeds, decode_rows_batched); 1 = original "
                         "sequential path")
    ap.add_argument("--eval-attn", type=str, default="",
                    help="attention implementation for eval decodes only "
                         "(e.g. sdpa). Forward-only, so fast attention here "
                         "keeps bitwise repro; training attn is untouched")
    ap.add_argument("--validate-batched", type=int, default=0,
                    help="run N val rows through sequential vs batched "
                         "decoders (greedy + beam), report agreement/"
                         "repeatability/speed, then exit. Use with --resume "
                         "on a trained ckpt")
    ap.add_argument("--emb-init", action="store_true",
                    help="soft tokens = head-posterior mixture of real "
                         "viseme-letter embeddings + zero-init residual "
                         "(on-manifold from step 0; targets the warmup "
                         "plateau / seed lottery at its root)")
    ap.add_argument("--lora-ramp-steps", type=int, default=0,
                    help="linear LR warmup of the LoRA group over N steps "
                         "after the joint switch (0 = off, legacy full-LR "
                         "shock); blunts the grounded-vs-shortcut knife "
                         "edge at joint start")
    ap.add_argument("--probe-grounding", type=int, default=0,
                    help="decode the gap subset real-vs-zeroed and report "
                         "viseme-line sensitivity, then exit (use with "
                         "--resume)")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="pause training at this step (0 = off) WITHOUT "
                         "touching --max-steps (the LR schedule's horizon); "
                         "fixed chain boundaries keep multi-segment runs "
                         "reproducible instead of timeout-dependent")
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise-reproducible training on fixed hardware: "
                         "torch deterministic algorithms + cudnn "
                         "deterministic, eager attention (GEMMA_ATTN), CTC "
                         "on CPU. Pair with CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                         "exported before launch")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=1.5e-5)
    ap.add_argument("--lr-head", type=float, default=5e-6)
    ap.add_argument("--ctc-weight", type=float, default=0.3)
    ap.add_argument("--ml-lexicons", type=Path,
                    default=None,
                    help="dir of lexicon_<lang>.json word->viseme lexicons; only read for non-English rows "
                         "(the released English-only training data never reads it)")
    ap.add_argument("--ctx-word-drop", type=float, default=0.0)
    ap.add_argument("--ctx-full-drop", type=float, default=0.0)
    ap.add_argument("--cfd-schedule", default="",
                    help="context full-drop curriculum 'step:p,step:p,...' "
                         "(global optimizer steps, ascending; last p is the "
                         "floor and holds). Overrides --ctx-full-drop.")
    ap.add_argument("--cfd-cosine", action="store_true",
                    help="cosine context full-drop: hold at --cfd-start "
                         "through warmup, then cosine-anneal start->floor "
                         "over --cfd-anneal steps of the joint stage, then "
                         "hold the floor. Overrides --cfd-schedule.")
    ap.add_argument("--cfd-start", type=float, default=1.0)
    ap.add_argument("--cfd-floor", type=float, default=0.15)
    ap.add_argument("--cfd-anneal", type=int, default=2500)
    ap.add_argument("--time-mask-p", type=float, default=0.0)
    ap.add_argument("--time-mask-len", type=int, default=4)
    ap.add_argument("--feat-noise", type=float, default=0.0)
    ap.add_argument("--soft-markers", default="", choices=["", "audio"],
                    help="bracket soft tokens with the model's native "
                         "modality markers (<|audio> ... <audio|>) — "
                         "Gemma-4's learned slot for continuous speech "
                         "features at 25 tok/s")
    ap.add_argument("--chat-soft-label", default="",
                    help="text label placed IMMEDIATELY before the soft "
                         "visual tokens in the user turn (Gemma-4 treats "
                         "unlabeled soft vectors as spam/CJK garbage and "
                         "derails; a label like 'Lip reading data:' frames "
                         "them as sensor data)")
    ap.add_argument("--chat-wrap", action="store_true",
                    help="wrap prompt/target in the tokenizer chat template "
                         "(REQUIRED for Gemma-4: raw continuations "
                         "degenerate; see g4_probe)")
    ap.add_argument("--one-cut-per-window", action="store_true",
                    help="serve one random k-cut per base window per epoch "
                         "(anti duplication/leakage; B4 arm)")
    ap.add_argument("--one-cut-fixed-k", action="store_true",
                    help="STRICT partition: one cut per window with a fixed "
                         "k for the whole run (~4x less data, zero leakage)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--init-adapter", type=Path, default=None,
                    help="warm-start the adapter (and its owned head) from "
                         "another run's checkpoint (e.g. arm W's prompt-"
                         "tuned adapter); --resume takes precedence")
    ap.add_argument("--inject-mode", choices=["prefix", "xattn"],
                    default="prefix",
                    help="prefix: visual soft tokens in the input sequence "
                         "(default, all arms so far). xattn: NO prefix — "
                         "Flamingo-style gated cross-attention blocks every "
                         "--xattn-every Gemma layers read the adapter's "
                         "pre-projection states (true encoder/decoder "
                         "seq2seq; gates init 0 = base model at step 0)")
    ap.add_argument("--xattn-every", type=int, default=8)
    ap.add_argument("--xattn-inner", type=int, default=512)
    ap.add_argument("--xattn-heads", type=int, default=8)
    ap.add_argument("--lr-xattn", type=float, default=1e-4)
    ap.add_argument("--gate-lr-mult", type=float, default=10.0,
                    help="LR multiplier for the 6 gate scalars: the visual "
                         "pathway's gradient is scaled by tanh(gate)~0 at "
                         "init, so the gates bootstrap slowly (probe: froze "
                         "at ~0.01-0.02); a hot gate LR opens the channel "
                         "without touching block weights")
    ap.add_argument("--lang-balance-alpha", type=float, default=1.0,
                    help="<1.0 = temperature-smoothed language balancing: "
                         "sampling share ~ n_lang^alpha (0.5 = sqrt "
                         "smoothing, standard multilingual practice). "
                         "1.0 = natural frequencies (off)")
    ap.add_argument("--lr-schedule", choices=["none", "cosine"],
                    default="none",
                    help="cosine: 250-step linear LR ramp then cosine decay "
                         "to 10%% of base over max-steps (all param groups; "
                         "survives the stage-switch optimizer rebuild)")
    _fix_paper_options(ap)
    args = ap.parse_args()
    global _READER_K, _READER_RNG, _PLAN_WORDS, _KEEP_DECISION, _MCL_SLOTS, _SELECT_WRITE, _SELECT_EDIT, _NO_VIS
    if args.reader_pool:
        import random as _random
        for spec in args.reader_pool:
            _READERS.append(load_reader_nbest(spec))
        _READER_RNG = _random.Random(args.seed + 104729)
        print("[readers] " + ", ".join(f"{n} w={w} windows={len(d)}" for n, w, d in _READERS), flush=True)
    _READER_K = args.reader_nbest_k
    _PLAN_WORDS, _KEEP_DECISION, _MCL_SLOTS = args.plan_words, args.keep_decision, args.mcl_slots
    for mp in args.keep_meta or []:
        for c in json.loads(Path(mp).read_text())["clips"]:
            _KEEP_META.setdefault(c["id"], c)
    if args.keep_decision:
        print(f"[keep] truth-based labels for {len(_KEEP_META)} cuts", flush=True)
    _SELECT_WRITE = bool(args.select_write)
    _SELECT_EDIT = bool(args.select_edit)
    _NO_VIS = args.evidence == "none"
    if _NO_VIS:
        # viseme-head ablation: no soft tokens (so no modality-marker block), no viseme line, no CTC anchor
        assert args.ctc_weight == 0, "--evidence none has no viseme head: pass --ctc-weight 0"
        assert args.chat_wrap and args.arm == "G" and (args.select_edit or args.select_write), \
            "--evidence none is implemented for the chat-wrapped arm-G select recipes only"
        args.soft_markers, args.emb_init = "", False
        print("[evidence none] no lip tokens, no viseme line, no CTC (viseme-head ablation)", flush=True)
    assert not (_SELECT_WRITE and _SELECT_EDIT), "--select-write and --select-edit are exclusive"
    if _SELECT_EDIT:
        assert _READERS and 1 <= _READER_K <= len(SW_LETTERS) and _KEEP_META, \
            "--select-edit needs --reader-pool, --reader-nbest-k 1..10 and --keep-meta"
        assert not args.keep_decision and not args.plan_words, "--select-edit replaces the plan lines"
        print(f"[select-edit] readings shown: 1..{_READER_K} per training row; closest-reading + corrected-window "
              f"targets for {len(_KEEP_META)} cuts", flush=True)
    if _SELECT_WRITE:
        assert _READERS and 1 <= _READER_K <= len(SW_LETTERS) and _KEEP_META, \
            "--select-write needs --reader-pool, --reader-nbest-k 1..10 and --keep-meta"
        assert not args.keep_decision, "--select-write replaces --keep-decision"
        print(f"[select-write] readings shown: 1..{_READER_K} per training row, {_READER_K} at val; "
              f"truth-based pick labels for {len(_KEEP_META)} cuts", flush=True)
    if args.avsr_json:
        _AVSR.update(load_avsr_json([str(p) for p in args.avsr_json]))
        print(f"[avsr] {len(_AVSR)} transcript keys loaded from "
              f"{[str(p) for p in args.avsr_json]}", flush=True)
    if args.avsr_noise:
        import random as _random
        global _AVSR_NOISE
        _p, _mr = (float(x) for x in args.avsr_noise.split(","))
        _vocab = sorted({w for t in _AVSR.values() for w in (t or "").split()})
        _AVSR_NOISE = {"p": _p, "max_rate": _mr, "rng": _random.Random(args.seed + 7919), "vocab": _vocab}
        print(f"[avsr-noise] p={_p} max_rate={_mr} vocab={len(_vocab)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str,
                                                   indent=1))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        # bitwise repro on fixed hardware+versions: kill every hidden
        # reseed (nondet kernels). CUBLAS env must predate cublas init —
        # set here AND exported in the sbatch. CTC moves to CPU (the CUDA
        # kernel has no deterministic variant); attention goes eager.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("GEMMA_ATTN", "eager")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("DETERMINISTIC mode: eager attn, CPU CTC, "
              f"cublas ws {os.environ['CUBLAS_WORKSPACE_CONFIG']}",
              flush=True)
    device = "cuda"

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(str(args.out / "tensorboard"))

    PT, PV = None, None
    if args.aux_line == "phones":
        assert args.evidence == "draft_phones", \
            "--aux-line phones takes its IPA vocab from the draft_phones " \
            "evidence dir"
        assert args.phone_targets, "--phone-targets required"
        PT = {}
        for p in args.phone_targets:
            PT.update(json.loads(Path(p).read_text())["targets"])
        PV, _ = load_phone_vocab(args.avsr_evidence_dir[0])
        set_aux_label("Phones:")
        print(f"[aux-line phones] {len(PT)} gold phone targets from "
              f"{[str(p) for p in args.phone_targets]} | label {AUX_LABEL!r}"
              f" | decode budget {args.aux_max_new}", flush=True)
    args.aux_label = AUX_LABEL              # saved with the ckpt for eval
    train_ds = GapDataset(args.data / "train.jsonl",
                          viseme_targets=(args.arm in ("C", "G", "H")),
                          ml_lexicons=args.ml_lexicons,
                          ctx_word_drop=args.ctx_word_drop,
                          ctx_full_drop=args.ctx_full_drop,
                          time_mask_p=args.time_mask_p,
                          time_mask_len=args.time_mask_len,
                          feat_noise=args.feat_noise,
                          full_window=args.full_window,
                          slice_pad=round(args.slice_pad_s * 25),
                          evidence=args.evidence,
                          evidence_dirs=args.avsr_evidence_dir,
                          evidence_topk=args.avsr_topk,
                          route_langs=args.route_langs,
                          drop_langs=args.drop_langs,
                          aux_line=args.aux_line, phone_targets=PT,
                          phone_vocab=PV)
    val_ds = GapDataset(args.data / "val.jsonl",
                        full_window=args.full_window,   # always clean
                        slice_pad=round(args.slice_pad_s * 25),
                        evidence=args.evidence,
                        evidence_dirs=args.avsr_evidence_dir,
                        evidence_topk=args.avsr_topk,
                        route_langs=args.route_langs,
                        drop_langs=args.drop_langs,
                        aux_line=args.aux_line, phone_targets=PT,
                        phone_vocab=PV)
    print(f"train {len(train_ds)} val {len(val_ds)} arm {args.arm} "
          f"aug(cwd={args.ctx_word_drop},cfd={args.ctx_full_drop},"
          f"tm={args.time_mask_p},fn={args.feat_noise})", flush=True)

    def _winit(wid: int) -> None:      # distinct python/numpy RNG per worker
        s = (torch.initial_seed() + wid) % 2**31
        random.seed(s)
        np.random.seed(s)

    if args.one_cut_fixed_k:
        loader_ds = OneCutFixedK(train_ds)
    elif args.one_cut_per_window:
        loader_ds = OneCutPerWindow(train_ds)
        print(f"one-cut-per-window: {len(loader_ds)} windows/epoch "
              f"(from {len(train_ds)} cuts)", flush=True)
    else:
        loader_ds = train_ds

    def _lang_of(i: int) -> str:
        if isinstance(loader_ds, OneCutPerWindow):
            r = train_ds.rows[loader_ds.groups[i][0]]
        elif isinstance(loader_ds, OneCutFixedK):
            r = train_ds.rows[loader_ds.picks[i]]
        else:
            r = train_ds.rows[i]
        return r.get("language", "en")

    sampler_weights = None
    if args.lang_balance_alpha < 1.0:
        from collections import Counter
        langs = [_lang_of(i) for i in range(len(loader_ds))]
        cnt = Counter(langs)
        tot = sum(cnt.values())
        wl = {lg: (c / tot) ** (args.lang_balance_alpha - 1.0)
              for lg, c in cnt.items()}
        sampler_weights = [wl[lg] for lg in langs]
        eff = {lg: round(c * wl[lg] / sum(c2 * wl[lg2] for lg2, c2
                                          in cnt.items()), 3)
               for lg, c in cnt.items()}
        print(f"lang balance a={args.lang_balance_alpha}: natural "
              f"{dict(cnt)} -> sampling shares {eff}", flush=True)

    def make_loader():
        if sampler_weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                sampler_weights, num_samples=len(loader_ds),
                replacement=True)
            return torch.utils.data.DataLoader(
                loader_ds, batch_size=args.batch_size, sampler=sampler,
                num_workers=2, collate_fn=collate, drop_last=True,
                persistent_workers=True, worker_init_fn=_winit)
        return torch.utils.data.DataLoader(
            loader_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
            collate_fn=collate, drop_last=True, persistent_workers=True,
            worker_init_fn=_winit)

    # context full-drop curriculum: list of (from_step, p), ascending
    cfd_stages = ([(int(s.split(":")[0]), float(s.split(":")[1]))
                   for s in args.cfd_schedule.split(",")]
                  if args.cfd_schedule else [])

    def cfd_at(s: int) -> float:
        if args.cfd_cosine:
            frac = min(1.0, max(0.0, (s - args.warmup_steps)
                                / max(1, args.cfd_anneal)))
            return args.cfd_floor + (args.cfd_start - args.cfd_floor) \
                * 0.5 * (1.0 + np.cos(np.pi * frac))
        p = args.ctx_full_drop
        for t, v in cfd_stages:
            if s >= t:
                p = v
        return p

    loader = make_loader()

    # ── viseme head (warm start) ─────────────────────────────────────────────
    head = None
    # (--evidence none: no viseme head)
    if args.evidence in ("avsr", "route", "draft_phones"):
        # avsr/route/draft_phones evidence: arm G per-frame + emb-init mixture
        # is the only supported configuration; avsr and draft_phones alone
        # have no head -> no CTC anchor
        assert args.arm == "G" and args.emb_init and \
            args.inject_mode == "prefix", \
            "--evidence avsr/route/draft_phones needs --arm G --emb-init (prefix)"
        if args.evidence in ("avsr", "draft_phones"):
            assert args.ctc_weight == 0, \
                f"--evidence {args.evidence} has no viseme head: " \
                f"pass --ctc-weight 0"
        for d in args.avsr_evidence_dir:
            if args.evidence == "draft_phones":
                ph, blank_id = load_phone_vocab(d)
                # packing convention: a zeroed (time-masked) frame reads id 0
                # with p 0 -> id 0 MUST be the blank so unit_ok[0] == 0
                assert blank_id == 0 and ph[0] == "<pad>", (blank_id, ph[:2])
                continue
            mp = Path(d) / "_meta.json"
            assert mp.exists(), f"{mp} missing (run extract_avsr_evidence)"
            n_units_meta = json.loads(mp.read_text())["n_units"]
            assert n_units_meta == len(load_avsr_units(args.avsr_units_file)), \
                (n_units_meta, str(mp))

    # ── Gemma + LoRA ─────────────────────────────────────────────────────────
    tok, gemma = load_gemma()
    emb = gemma.get_input_embeddings()
    gemma_dim = emb.weight.shape[1]
    emb_rms = emb.weight.float().pow(2).mean(-1).sqrt().mean().item()
    from peft import LoraConfig, get_peft_model
    if hasattr(gemma, "language_model") \
            or "ConditionalGeneration" in type(gemma).__name__:
        target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
    else:
        target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    gemma = get_peft_model(gemma, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=target, bias="none"))
    gemma.config.use_cache = False
    gemma.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    emb = gemma.get_input_embeddings()
    lora_params = [p for n, p in gemma.named_parameters() if "lora_" in n]
    print(f"gemma_dim {gemma_dim} emb_rms {emb_rms:.4f} "
          f"lora tensors {len(lora_params)}", flush=True)
    chat = chat_affixes(tok, args.chat_soft_label, args.soft_markers) if args.chat_wrap else None
    if chat is not None:
        print(f"chat-wrap ON: pre={len(chat[0])} post={len(chat[1])} "
              f"end_id={chat[2]} soft_label={args.chat_soft_label!r}",
              flush=True)

    if args.evidence in ("avsr", "route", "draft_phones"):
        phones = args.evidence == "draft_phones"
        if phones:
            units, _ = load_phone_vocab(args.avsr_evidence_dir[0])
            for d in args.avsr_evidence_dir[1:]:      # dirs must agree
                assert load_phone_vocab(d)[0] == units, str(d)
            unit_text, special = phone_text, PHONE_SPECIAL
        else:
            units = load_avsr_units(args.avsr_units_file)
            unit_text, special = avsr_unit_text, AVSR_SPECIAL
        args.avsr_n_units = len(units)          # saved with the ckpt args
        adapter = AvsrAdapter(gemma_dim, n_units=len(units), K=args.avsr_topk,
                              max_pf=args.max_pf, min_stride=args.min_stride,
                              cls_dim=args.avsr_cls_dim,
                              tokens=args.avsr_tokens,
                              spike_thresh=args.avsr_spike_thresh,
                              enc_dim=0 if phones else AVSR_ENC_DIM).to(device)
        # unit embeddings: Gemma's own embedding of each CTC unit's text
        # (emb(ids), NOT emb.weight, so the scaling matches the text tokens
        # around the soft span; multi-token pieces -> mean). emb-init
        # with 5,046 units instead of 13 viseme letters.
        with torch.no_grad():
            n_ok, ntok = 0, []
            for uid, u in enumerate(units):
                txt = unit_text(u)
                if txt is None:
                    continue
                ids = tok(txt, add_special_tokens=False).input_ids
                if not ids:
                    continue
                adapter.unit_emb[uid] = emb(torch.tensor(ids, device=device)) \
                    .float().mean(0)
                adapter.unit_ok[uid] = 1.0
                n_ok += 1
                ntok.append(len(ids))
        assert n_ok >= len(units) - len(special) - 8, n_ok
        print(f"{args.evidence} evidence ON: {len(units)} CTC units, "
              f"{n_ok} mapped to "
              f"Gemma text ({np.mean(ntok):.2f} tok/unit) | unit_emb rms "
              f"{adapter.unit_emb[adapter.unit_ok > 0].pow(2).mean().sqrt():.4f}"
              f" | top-{args.avsr_topk} | tokens={args.avsr_tokens}"
              f"(thresh {args.avsr_spike_thresh}) | evidence dirs "
              f"{[str(d) for d in args.avsr_evidence_dir]}", flush=True)
        if args.evidence == "route":
            avh_adapter = VisAdapter(head, gemma_dim, K=args.K, per_frame=True,
                                     max_pf=args.max_pf,
                                     min_stride=args.min_stride,
                                     gap_channel=False, emb_init=True)
            adapter = RouteAdapter(avh_adapter, adapter).to(device)
            print(f"lang_routed_tokens: avsr channel for {args.route_langs}, "
                  f"viseme-head channel otherwise; packed width "
                  f"{route_width(args.avsr_topk)}", flush=True)
    elif args.evidence == "none":
        adapter = NullAdapter(gemma_dim).to(device)
    else:
        adapter = VisAdapter(head, gemma_dim, K=args.K,
                             per_frame=(args.arm in ("G", "H")),
                             max_pf=args.max_pf, min_stride=args.min_stride,
                             gap_channel=(args.full_window
                                          and not args.no_gap_channel),
                             emb_init=args.emb_init).to(device)
    if args.emb_init and args.evidence in ("avh", "route"):
        vis_target = adapter.avh if args.evidence == "route" else adapter
        # viseme letters as they appear in the target line (" K", " TH", ...);
        # emb(ids) (NOT emb.weight) so Gemma's embedding scaling matches the
        # text tokens around the soft span; multi-token letters -> mean
        with torch.no_grad():
            ntok = {}
            for c in range(N_CLS):
                if c == BLANK:
                    continue
                ids = tok(f" {ID2CLS[c]}", add_special_tokens=False).input_ids
                ntok[ID2CLS[c]] = len(ids)
                vis_target.vis_emb[c] = emb(torch.tensor(ids, device=device)) \
                    .float().mean(0)
        print(f"emb-init ON: viseme-letter token counts {ntok} | "
              f"vis_emb rms {vis_target.vis_emb[1:].pow(2).mean().sqrt():.4f}",
              flush=True)

    # arm XA: install gated cross-attention blocks (AFTER get_peft_model so
    # LoRA key paths in checkpoints match eval-side reconstruction)
    xa_ctx: dict = {"enc": None, "keep": None}
    xa_list, xa_params, xa_gates = None, [], []
    if args.inject_mode == "xattn":
        assert adapter.per_frame, "xattn needs per-frame states (arm G/H)"
        adapter.stash_states = True
        xa_list = install_xattn(gemma, d_lm=gemma_dim, d_enc=adapter.d,
                                every=args.xattn_every,
                                inner=args.xattn_inner,
                                heads=args.xattn_heads,
                                ctx=xa_ctx, device=device)
        xa_params = [p for n, p in xa_list.named_parameters()
                     if not n.endswith("gate")]
        xa_gates = [p for n, p in xa_list.named_parameters()
                    if n.endswith("gate")]

    def gap_mask(rows_b, T, dev):
        if not (args.full_window and not args.no_gap_channel):
            return None
        m = torch.zeros(len(rows_b), T, device=dev)
        for i, rr in enumerate(rows_b):
            m[i, rr["f0"]: rr["f1"]] = 1.0
        return m
    adapter.out_gain.data.fill_(emb_rms)
    if args.init_adapter and args.init_adapter.exists():
        st0 = torch.load(args.init_adapter, map_location="cpu",
                         weights_only=False)
        adapter.load_state_dict(st0["adapter"])
        xa_note = ""
        if xa_list is not None and "xattn" in st0:
            xa_list.load_state_dict(st0["xattn"])   # XW-style donor: carry
            xa_note = " + xattn blocks"             # the opened gates too
        print(f"adapter{xa_note} warm-started from {args.init_adapter} "
              f"(donor step {st0.get('step')}, val {st0.get('best_pos')})",
              flush=True)
    head_params = list(adapter.head.parameters()) \
        if getattr(adapter, "head", None) is not None else []
    head_ids = {id(p) for p in head_params}
    core_params = [p for p in adapter.parameters() if id(p) not in head_ids]

    step, best_pos, bad_evals = 0, -1.0, 0
    if args.resume and args.resume.exists():
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        adapter.load_state_dict(st["adapter"])
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(gemma, st["lora"])
        if xa_list is not None and "xattn" in st:
            xa_list.load_state_dict(st["xattn"])
        step, best_pos = st["step"], st.get("best_pos", -1.0)
        print(f"resumed at step {step} (best_pos {best_pos})", flush=True)
    elif args.init_from and args.init_from.exists():
        st_i = torch.load(args.init_from, map_location="cpu", weights_only=False)
        adapter.load_state_dict(st_i["adapter"])
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(gemma, st_i["lora"])
        if xa_list is not None and "xattn" in st_i:
            xa_list.load_state_dict(st_i["xattn"])
        print(f"warm start from {args.init_from} (donor step {st_i.get('step')}): adapter + head + LoRA, "
              f"step 0, fresh optimizer", flush=True)

    def set_stage(joint: bool):
        for p in core_params:
            p.requires_grad_(True)
        for p in xa_params + xa_gates:
            p.requires_grad_(True)      # xattn blocks train in both stages
        for p in lora_params:
            p.requires_grad_(joint)
        for p in head_params:
            p.requires_grad_(joint and args.arm in ("C", "G", "H"))
        groups = [{"params": core_params, "lr": args.lr_adapter}]
        if xa_params:
            groups.append({"params": xa_params, "lr": args.lr_xattn})
            groups.append({"params": xa_gates,
                           "lr": args.lr_xattn * args.gate_lr_mult})
        if joint:
            groups.append({"params": lora_params, "lr": args.lr_lora,
                           "is_lora": True})
            if args.arm in ("C", "G", "H") and head_params:
                groups.append({"params": head_params, "lr": args.lr_head})
        o = torch.optim.AdamW(groups, weight_decay=0.01)
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
        return o

    def lr_factor(s: int) -> float:
        if args.lr_schedule != "cosine":
            return 1.0
        ramp = 250
        if s < ramp:
            return max(s, 1) / ramp
        frac = min(1.0, (s - ramp) / max(1, args.max_steps - ramp))
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * frac))

    if args.warmup_only:
        args.warmup_steps = args.max_steps + 1   # joint never starts
    joint = step >= args.warmup_steps
    opt = set_stage(joint)
    if args.resume and args.resume.exists() and "opt" in st:
        # chunked training (45-min debug slots): carry AdamW moments across chunks when the
        # stage (param-group layout) is the same as at save time
        try:
            if [len(g["params"]) for g in st["opt"]["param_groups"]] == [len(g["params"]) for g in opt.param_groups]:
                opt.load_state_dict(st["opt"])
                for g in opt.param_groups:
                    g.setdefault("base_lr", g["lr"])
                print("resumed optimizer state", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"optimizer state not restored ({e})", flush=True)
    print(f"stage: {'joint' if joint else 'warmup'}"
          f"{' (warmup-only run)' if args.warmup_only else ''}", flush=True)

    def save(path, extra=None, with_opt=False):
        from peft import get_peft_model_state_dict
        tmp = Path(str(path) + ".tmp")
        torch.save({"adapter": adapter.state_dict(),
                    "lora": get_peft_model_state_dict(gemma),
                    **({"xattn": xa_list.state_dict()} if xa_list else {}),
                    **({"opt": opt.state_dict()} if with_opt else {}),
                    "step": step, "best_pos": best_pos,
                    "args": vars(args), **(extra or {})}, tmp)
        os.replace(tmp, path)          # a kill mid-save never leaves a truncated last.pt

    pool = list(range(len(val_ds)))
    if args.val_k_max:
        pool = [i for i in pool if val_ds.rows[i]["k"] <= args.val_k_max]
    if args.val_one_cut:
        byw: dict[str, list[int]] = {}
        for i in pool:
            byw.setdefault(val_ds.rows[i]["id"].rsplit("__k", 1)[0],
                           []).append(i)
        val_idx = []
        for j, w in enumerate(sorted(byw)):
            byk = {val_ds.rows[i]["k"]: i for i in byw[w]}
            want = 1 + j % 4
            k = want if want in byk else \
                sorted(byk, key=lambda x: (abs(x - want), x))[0]
            val_idx.append(byk[k])
        val_idx.sort()
        from collections import Counter as _C
        print(f"val one-cut: {len(val_idx)} windows | lang "
              f"{dict(_C(val_ds.rows[i].get('language', 'en') for i in val_idx))}"
              f" | k {dict(sorted(_C(val_ds.rows[i]['k'] for i in val_idx).items()))}",
              flush=True)
    elif args.val_en_all:
        en_i = [i for i in pool
                if val_ds.rows[i].get("language", "en") == "en"]
        rest = [i for i in pool
                if val_ds.rows[i].get("language", "en") != "en"]
        random.Random(0).shuffle(rest)
        val_idx = sorted((en_i + rest)[: args.val_decode_n])
    else:
        random.Random(0).shuffle(pool)
        val_idx = sorted(pool[: args.val_decode_n])
    # grounding-gap subset: language-stratified round-robin over the FULL
    # val set (all k, independent of the k<=val_k_max selection pool) —
    # long spans carry the partial-credit signal that makes real-vs-zeroed
    # features visible; a short-span-only subset pins both sides to the
    # floor and reads gap~0 regardless of grounding
    by_lang: dict[str, list[int]] = {}
    for i in range(len(val_ds)):
        by_lang.setdefault(val_ds.rows[i].get("language", "en"), []).append(i)
    for lg in by_lang:
        random.Random(0).shuffle(by_lang[lg])
    n_gap = min(50, len(val_ds))
    gap_idx: list[int] = []
    while len(gap_idx) < n_gap:
        grew = False
        for lg in sorted(by_lang):
            if by_lang[lg] and len(gap_idx) < n_gap:
                gap_idx.append(by_lang[lg].pop(0))
                grew = True
        if not grew:
            break

    def run_decode(idxs_, **kw):
        kw.setdefault("gap_dur", args.gap_dur_prompt)
        with _eval_attn(gemma, args.eval_attn):
            if args.decode_batch > 1 and kw.get("xa_ctx") is None:
                return decode_rows_batched(
                    val_ds, idxs_, adapter, gemma, tok, emb, device,
                    batch_size=args.decode_batch, **kw)
            return decode_rows(val_ds, idxs_, adapter, gemma, tok, emb,
                               device, **kw)

    if args.probe_grounding:
        # decisive grounding diagnostic: decode the gap rows with real vs
        # zeroed features and compare the VISEME LINES. Feature-driven
        # lines change when lips vanish; context-derived lines (the LM
        # back-transliterating its word guess) do not.
        vc, il = args.arm == "G", args.arm == "H"
        kwd = dict(max_new=60 if il else (args.aux_max_new if vc else 12),
                   viseme_cot=vc, interleaved=il, chat=chat)
        rg = run_decode(gap_idx, **kwd)
        rz = run_decode(gap_idx, zero_visual=True, **kwd)

        def vline(t: str) -> str:
            return t.split("Missing words:")[0].replace(
                AUX_LABEL, "").strip()

        diff = sum(vline(a["pred"]) != vline(b["pred"])
                   for a, b in zip(rg, rz)) / max(1, len(rg))
        pr = sum(r["pos"] for r in rg) / max(1, len(rg))
        pz = sum(r["pos"] for r in rz) / max(1, len(rz))
        print(f"PROBE_GROUNDING n={len(rg)}: viseme-line real!=zero rate "
              f"{diff:.2f} | word pos real {pr:.3f} zero {pz:.3f} "
              f"gap {pr - pz:+.3f}", flush=True)
        for a, b in list(zip(rg, rz))[:12]:
            print(f"  gt='{a['gt'][:44]}'\n   real='{vline(a['pred'])[:84]}'"
                  f"\n   zero='{vline(b['pred'])[:84]}'", flush=True)
        print("PROBE_GROUNDING_DONE", flush=True)
        return

    if args.validate_batched:
        # sequential vs batched decoder validation on N val rows: report
        # text agreement, batched-rerun repeatability, metric deltas, speed
        n = min(args.validate_batched, len(val_idx))
        sub = val_idx[:n]
        vc, il = args.arm == "G", args.arm == "H"
        kwd = dict(max_new=60 if il else (args.aux_max_new if vc else 12),
                   viseme_cot=vc, interleaved=il, chat=chat)
        for nb in (1, max(1, args.val_nbest)):
            t0 = time.time()
            seq = decode_rows(val_ds, sub, adapter, gemma, tok, emb,
                              device, nbest=nb, **kwd)
            t1 = time.time()
            b1 = run_decode(sub, nbest=nb, **kwd)
            t2 = time.time()
            b2 = run_decode(sub, nbest=nb, **kwd)
            agree = sum(s["pred"] == b["pred"]
                        for s, b in zip(seq, b1)) / n
            rep = all(x["pred"] == y["pred"] for x, y in zip(b1, b2))
            dpos = sum(b["pos"] for b in b1) / n \
                - sum(s["pos"] for s in seq) / n
            dbp = sum(b["bestpos"] for b in b1) / n \
                - sum(s["bestpos"] for s in seq) / n
            print(f"VALIDATE nbest={nb} n={n}: top1-text agree "
                  f"{agree:.3f} | batched rerun identical {rep} | "
                  f"d_pos {dpos:+.4f} d_bestpos {dbp:+.4f} | "
                  f"seq {t1 - t0:.0f}s batched {t2 - t1:.0f}s "
                  f"({(t1 - t0) / max(1e-9, t2 - t1):.1f}x)", flush=True)
            for s, b in zip(seq, b1):
                if s["pred"] != b["pred"]:
                    print(f"  DIFF id={s['id']}\n   seq='{s['pred'][:90]}'"
                          f"\n   bat='{b['pred'][:90]}'", flush=True)
        print("VALIDATE_BATCHED_DONE", flush=True)
        return

    adapter.train()
    gemma.train()
    # warmup-viseme-only: lips are the ONLY loss-reducing path — context
    # fully dropped (via the existing ___ mechanism, format preserved) and
    # supervision truncated to the viseme line (see build_batch vis_only)
    def cfd_now(s: int) -> float:
        if args.warmup_viseme_only and s < args.warmup_steps:
            return 1.0
        return cfd_at(s)

    train_ds.ctx_full_drop = cfd_now(step)     # resume lands mid-schedule
    print(f"ctx_full_drop starts at {train_ds.ctx_full_drop}", flush=True)
    t0, run_loss, run_ctc, n_acc = time.time(), 0.0, 0.0, 0
    it = iter(loader)
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps and \
            not (args.stop_at_step and step >= args.stop_at_step):
        if not joint and step >= args.warmup_steps:
            joint = True
            opt = set_stage(True)
            if _NO_VIS:      # the skipped warmup batches must not dilute the first logged joint loss
                t0, run_loss, run_ctc, n_acc = time.time(), 0.0, 0.0, 0
            print(f"[{step}] switched to joint stage", flush=True)
        p_now = cfd_now(step)
        # persistent workers snapshot dataset attrs -> loader rebuild needed
        # on change; for the smooth cosine path only rebuild on >=0.05 moves
        # (or on reaching the floor) to avoid thrashing workers every step
        if abs(p_now - train_ds.ctx_full_drop) >= 0.05 or \
                (args.cfd_cosine and p_now == args.cfd_floor
                 and train_ds.ctx_full_drop != args.cfd_floor) or \
                (not args.cfd_cosine and p_now != train_ds.ctx_full_drop):
            train_ds.ctx_full_drop = p_now
            del it, loader
            loader = make_loader()
            it = iter(loader)
            print(f"[{step}] ctx_full_drop -> {p_now:.3f}", flush=True)
        try:
            feats, lens, rows, vis, visw = next(it)
        except StopIteration:
            it = iter(loader)
            feats, lens, rows, vis, visw = next(it)
        if _NO_VIS and not joint:
            # no lip tokens => nothing trains before the joint stage (the warmup only fits the lip-token
            # adapter with Gemma frozen): consume the batch so the data stream stays aligned, advance the step
            n_acc += 1
            if n_acc % args.grad_accum == 0:
                step += 1
            continue
        feats = feats.to(device)
        soft, slens, logits = adapter(feats, lens,
                                      gapm=gap_mask(rows, feats.shape[1],
                                                    device))
        vis_strs, interleaved = None, False
        if args.arm == "G" and not _NO_VIS:
            # aux_str (--aux-line phones) overrides the gold viseme string
            vis_strs = [r.get("aux_str") if "aux_str" in r
                        else (" ".join(ID2CLS[j] for j in v) if v else "?")
                        for r, v in zip(rows, vis)]
        elif args.arm == "H":
            interleaved = True
            vis_strs = []
            for vw, r in zip(visw, rows):
                ws = r["target"].split()
                if vw and len(vw) == len(ws):
                    vis_strs.append(" ".join(
                        f"[{' '.join(ID2CLS[j] for j in wid)}] {w}"
                        for wid, w in zip(vw, ws)))
                else:
                    vis_strs.append(" ".join(f"[?] {w}" for w in ws))
        if xa_list is not None:
            xa_ctx["enc"] = adapter.last_states
            xa_ctx["keep"] = (torch.arange(adapter.last_states.shape[1],
                                           device=device)[None]
                              < slens.to(device)[:, None])
            slens = torch.zeros_like(slens)     # no prefix: xattn only
        global _AVSR_NOISE_ON
        _AVSR_NOISE_ON = True           # training batch: noise (if configured) applies
        K_ = args.mcl_slots
        if K_ > 1:
            # idea 4: every row once per slot; the slot marker sits in the user turn
            B_ = len(rows)
            b_rows = [dict(r, _slot=j) for j in range(K_) for r in rows]
            b_soft, b_slens = soft.repeat(K_, 1, 1), slens.repeat(K_)
            b_vis = None if vis_strs is None else vis_strs * K_
        else:
            b_rows, b_soft, b_slens, b_vis = rows, soft, slens, vis_strs
        ie, am, lb = build_batch(tok, emb, b_soft, b_slens, b_rows, device,
                                 vis_strs=b_vis, interleaved=interleaved,
                                 chat=chat,
                                 vis_only=(args.warmup_viseme_only
                                           and not joint),
                                 gap_dur=args.gap_dur_prompt)
        _AVSR_NOISE_ON = False          # val / decode prompts stay clean
        if K_ > 1:
            out = gemma(inputs_embeds=ie, attention_mask=am)
            lg_ = out.logits[:, :-1].float()
            lb_ = lb[:, 1:]
            tok_ce = F.cross_entropy(lg_.reshape(-1, lg_.shape[-1]), lb_.reshape(-1),
                                     ignore_index=-100, reduction="none").view(lb_.shape)
            n_tok = (lb_ != -100).sum(1).clamp(min=1)
            seq_ce = (tok_ce.sum(1) / n_tok).view(K_, B_)            # [K, B]
            # relaxed winner-take-all: the slot closest to the target takes the gradient
            loss = (seq_ce.min(0).values + args.mcl_eps * seq_ce.mean(0)).mean()
            win_ = seq_ce.argmin(0)
            mcl_win_counts = torch.bincount(win_, minlength=K_).tolist()
        else:
            out = gemma(inputs_embeds=ie, attention_mask=am, labels=lb)
            loss = out.loss.float()
            mcl_win_counts = None
        ctc = torch.tensor(0.0, device=device)
        if joint and args.arm in ("C", "G", "H") and args.ctc_weight > 0 \
                and logits is not None:
            chan = getattr(adapter, "last_channel", None)   # route: avh only
            tgt = [(i, v) for i, v in enumerate(vis)
                   if v and not (chan and chan[i])]
            if tgt:
                lp = F.log_softmax(logits.float(), -1)
                ids = torch.cat([torch.tensor(v) for _, v in tgt]).to(device)
                tlen = torch.tensor([len(v) for _, v in tgt])
                if args.full_window or args.slice_pad_s > 0:
                    # CTC labels are GAP visemes -> score the gap slice only
                    # (clamped: some windows' feats are shorter than f1).
                    # Wide mode: sf0/sf1 = the span's position in the slice,
                    # so the anchor is IDENTICAL to the exact-span recipe.
                    segs_g = [lp[i, rows[i].get("sf0", rows[i]["f0"]):
                              min(rows[i].get("sf1", rows[i]["f1"]),
                                  int(lens[i]))]
                              for i, _ in tgt]
                    glens = [s.shape[0] for s in segs_g]
                    gmax = max(glens)
                    lpg = torch.full((len(tgt), gmax, lp.shape[-1]),
                                     0.0, device=device)
                    for j, s in enumerate(segs_g):
                        lpg[j, : s.shape[0]] = s
                    if args.deterministic:   # CUDA ctc has no det. kernel
                        ctc = F.ctc_loss(lpg.cpu().transpose(0, 1),
                                         ids.cpu(), torch.tensor(glens),
                                         tlen, blank=BLANK,
                                         zero_infinity=True).to(device)
                    else:
                        ctc = F.ctc_loss(lpg.transpose(0, 1), ids,
                                         torch.tensor(glens), tlen,
                                         blank=BLANK, zero_infinity=True)
                else:
                    ilen = torch.tensor([int(lens[i]) for i, _ in tgt])
                    sel = torch.tensor([i for i, _ in tgt])
                    if args.deterministic:
                        ctc = F.ctc_loss(lp[sel].cpu().transpose(0, 1),
                                         ids.cpu(), ilen, tlen, blank=BLANK,
                                         zero_infinity=True).to(device)
                    else:
                        ctc = F.ctc_loss(lp[sel].transpose(0, 1), ids, ilen,
                                         tlen, blank=BLANK,
                                         zero_infinity=True)
        total = loss + (args.ctc_weight * ctc
                        if args.arm in ("C", "G", "H") else 0.0)
        (total / args.grad_accum).backward()
        run_loss += loss.item()
        run_ctc += float(ctc.detach())
        n_acc += 1
        if n_acc % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad]
                + xa_params + xa_gates
                + [p for p in lora_params if p.requires_grad], 1.0)
            f = lr_factor(step)
            # LoRA LR ramp-in (anti-bistability): the joint switch used to
            # unfreeze the LoRA at FULL LR — a shock that races the weak
            # soft-token pathway against the context shortcut and lets
            # float noise pick the basin. Ramp 0->base over N joint steps.
            lr_ramp = 1.0
            if args.lora_ramp_steps and step >= args.warmup_steps:
                lr_ramp = min(1.0, (step - args.warmup_steps)
                              / max(1, args.lora_ramp_steps))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f \
                    * (lr_ramp if g.get("is_lora") else 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                dl = run_loss / n_acc
                dc = run_ctc / n_acc
                tb.add_scalar("train/loss/CE", dl, step)
                tb.add_scalar("train/loss/ctc", dc, step)
                tb.add_scalar("train/loss/total",
                              dl + args.ctc_weight * dc, step)
                tb.add_scalar("train/step_time_s",
                              (time.time() - t0) / max(1, n_acc /
                                                       args.grad_accum), step)
                tb.add_scalar("train/ctx_full_drop",
                              train_ds.ctx_full_drop, step)
                if xa_list is not None:
                    gv = [float(torch.tanh(b.gate.detach().float()))
                          for b in xa_list]
                    tb.add_scalar("train/xattn_gate_meanabs",
                                  sum(abs(g) for g in gv) / len(gv), step)
                for gi, g in enumerate(opt.param_groups):
                    tb.add_scalar(f"train/lr_g{gi}", g["lr"], step)
                if mcl_win_counts is not None:
                    for j_, c_ in enumerate(mcl_win_counts):
                        tb.add_scalar(f"train/mcl_win_slot{j_ + 1}", c_, step)
                if args.save_every and step % args.save_every == 0 and step % args.eval_every:
                    save(args.out / "last.pt", with_opt=True)
                print(f"[{step}] loss {dl:.4f} ctc {dc:.3f} "
                      f"({'joint' if joint else 'warmup'})"
                      + (f" mcl_wins {mcl_win_counts}" if mcl_win_counts is not None else ""), flush=True)
                t0, run_loss, run_ctc, n_acc = time.time(), 0.0, 0.0, 0

            if step % args.eval_every == 0 or step == args.max_steps:
                vc = args.arm == "G"
                il = args.arm == "H"
                mn = 60 if il else (args.aux_max_new if vc else 12)
                kwd = dict(max_new=mn, viseme_cot=vc, interleaved=il,
                           xa_ctx=xa_ctx if xa_list else None, chat=chat)
                # FAST components first: the greedy gap
                # subset (monitoring-only) logs and flushes to TB before
                # the heavy beam selection decode. Back to EVERY eval:
                # batched decoding made its 100
                # greedy rows ~35s, no longer worth skipping.
                do_gap = True
                gap_s = "gap: skipped "
                if do_gap:
                    resg = run_decode(gap_idx, **kwd)
                    resz = run_decode(gap_idx, zero_visual=True, **kwd)
                    posz = sum(r["pos"] for r in resz) / max(1, len(resz))
                    pos_r = sum(r["pos"] for r in resg) / max(1, len(resg))
                    tb.add_scalar("val/pos_zero", posz, step)
                    tb.add_scalar("val/grounding_gap", pos_r - posz, step)
                    # viseme-line sensitivity: fraction of rows whose viseme
                    # line CHANGES when lips are zeroed — the direct
                    # is-it-reading-lips meter (context-derived lines don't
                    # change; the wuvis failure read 0.24)
                    vd = sum(a_["pred"].split("Missing words:")[0]
                             != b_["pred"].split("Missing words:")[0]
                             for a_, b_ in zip(resg, resz)) \
                        / max(1, len(resg))
                    tb.add_scalar("val/gap_visline_diff", vd, step)
                    tb.flush()
                    gap_s = (f"gap subset: real {pos_r:.3f} zero {posz:.3f} "
                             f"gap {pos_r - posz:+.3f} visdiff {vd:.2f} ")
                    print(f"[{step}] VAL(gap-first) {gap_s}", flush=True)
                res = run_decode(val_idx, nbest=args.val_nbest, **kwd)
                pos = sum(r["pos"] for r in res) / len(res)
                exact = sum(r["exact"] for r in res) / len(res)
                tb.add_scalar("val/pos", pos, step)
                tb.add_scalar("val/exact", exact, step)
                sel = pos
                if args.val_nbest > 1:
                    bp = sum(r["bestpos"] for r in res) / len(res)
                    hit = sum(r["hit"] for r in res) / len(res)
                    tb.add_scalar(f"val/bestpos{args.val_nbest}", bp, step)
                    tb.add_scalar(f"val/hit{args.val_nbest}", hit, step)
                    sel = bp
                if args.val_lang_weights:
                    wts = {kv.split(":")[0]: float(kv.split(":")[1])
                           for kv in args.val_lang_weights.split(",")}
                    key = "bestpos" if args.val_nbest > 1 else "pos"
                    acc: dict[str, list[float]] = {}
                    for i, r in zip(val_idx, res):
                        acc.setdefault(val_ds.rows[i].get("language", "en"),
                                       []).append(r[key])
                    num = sum(wts[lg] * sum(v) / len(v)
                              for lg, v in acc.items() if lg in wts)
                    den = sum(wts[lg] for lg in acc if lg in wts)
                    sel = num / max(den, 1e-9)
                    tb.add_scalar("val/sel_weighted", sel, step)
                en_sub = [r["pos"] for i, r in zip(val_idx, res)
                          if val_ds.rows[i].get("language", "en") == "en"]
                if en_sub:
                    tb.add_scalar("val/pos_en",
                                  sum(en_sub) / len(en_sub), step)
                # per-language val pos (the val set is es/fr/ru-heavy; an
                # English-only evidence channel is invisible in the overall
                # number)
                by_lg: dict[str, list[float]] = {}
                for i, r in zip(val_idx, res):
                    by_lg.setdefault(val_ds.rows[i].get("language", "en"),
                                     []).append(r["pos"])
                for lg, v in sorted(by_lg.items()):
                    tb.add_scalar(f"val/pos_{lg}", sum(v) / len(v), step)
                print(f"[{step}] VAL per-language pos " + " ".join(
                    f"{lg} {sum(v) / len(v):.3f}(n={len(v)})"
                    for lg, v in sorted(by_lg.items())), flush=True)
                for kk in sorted({r["k"] for r in res}):
                    sub = [r["pos"] for r in res if r["k"] == kk]
                    tb.add_scalar(f"val/pos_k{kk}",
                                  sum(sub) / len(sub), step)
                tb.flush()
                tb.add_text("val/samples", "  \n".join(
                    f"[{r['k']}] gt='{r['gt']}' pred='{r['pred']}'"
                    for r in res[:8]), step)
                gemma.train()
                gate_s = ""
                if xa_list is not None:
                    gate_s = " | gates " + ",".join(
                        f"{float(torch.tanh(b.gate.detach().float())):+.3f}"
                        for b in xa_list)
                bp_s = (f" bestpos{args.val_nbest} {sel:.3f}"
                        if args.val_nbest > 1 else "")
                print(f"[{step}] VAL pos {pos:.3f} exact {exact:.3f}{bp_s} | "
                      f"{gap_s}(best {best_pos:.3f})"
                      f"{gate_s}", flush=True)
                save(args.out / "last.pt", with_opt=True)
                if args.keep_ckpts and step in {int(x) for x in args.keep_ckpts.split(",") if x}:
                    import shutil as _shutil
                    _shutil.copyfile(args.out / "last.pt", args.out / f"ckpt_s{step:05d}.pt")
                    print(f"[{step}] kept ckpt_s{step:05d}.pt", flush=True)
                # during a cfd anneal the early evals are expectedly depressed
                # (context still mostly withheld) — freeze the patience clock
                # until the schedule reaches its floor
                annealing = (args.cfd_cosine
                             and step < args.warmup_steps + args.cfd_anneal)
                # >= not >: on the long val plateau, ties go to the LATER
                # (more joint-CTC-mature) checkpoint
                if sel >= best_pos:
                    best_pos, bad_evals = sel, 0
                    save(args.out / "best.pt", {"val": res})
                elif not annealing:
                    bad_evals += 1
                    if (joint or args.warmup_only) \
                            and bad_evals >= args.patience:
                        print(f"early stop at {step} (best {best_pos:.3f})",
                              flush=True)
                        break
    save(args.out / "last.pt")
    tb.close()
    print(f"DONE step {step} best val pos {best_pos:.3f}", flush=True)


if __name__ == "__main__":
    main()
