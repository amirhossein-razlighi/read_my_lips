"""Support code of the L trainer (train_vis2gemma.py).

  load_gemma            Gemma-4 from the GEMMA_ROOT environment variable
  viseme classes,       the 12-class viseme map built by GapDataset at start-up (not used by the loss or the prompt
  Phonemizer, visemes,  with --evidence none and CTC weight 0). Needs nltk's cmudict and g2p_en.
  collapse
"""
from __future__ import annotations

import re

import torch

VISEME_CLASSES = ["P", "F", "TH", "T", "SH", "K", "W", "Y", "E", "A", "O", "U", "SIL"]
BLANK = 0
CLS2ID = {c: i + 1 for i, c in enumerate(VISEME_CLASSES)}  # 1..13, 0=blank

# ── Viseme classes (ARPAbet → class), Jeffers & Barley-derived ──────────────
# 12 classes; stress digits stripped before lookup.
VISEME_MAP: dict[str, str] = {}
for cls, phones in {
    "P":  ["P", "B", "M"],                       # bilabial closure
    "F":  ["F", "V"],                            # labiodental
    "TH": ["TH", "DH"],                          # tongue-between-teeth
    "T":  ["T", "D", "S", "Z", "N", "L"],        # alveolar
    "SH": ["SH", "ZH", "CH", "JH"],              # postalveolar / rounded
    "K":  ["K", "G", "NG", "HH"],                # velar / open-back (near-invisible)
    "W":  ["W", "R"],                            # rounded approximant
    "Y":  ["Y", "IY", "IH"],                     # spread lips
    "E":  ["EH", "EY", "AE"],                    # mid-front open
    "A":  ["AA", "AH", "AY", "AW"],              # open jaw
    "O":  ["AO", "OY", "OW"],                    # rounded open
    "U":  ["UH", "UW", "ER"],                    # rounded close
}.items():
    for p in phones:
        VISEME_MAP[p] = cls

_STRESS = re.compile(r"\d")


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


class Phonemizer:
    """CMUdict lookup first; g2p_en (neural grapheme-to-phoneme) fallback so
    EVERY word gets phonemes — OOV only meant 'not in the lookup dictionary',
    never 'has no pronunciation'. self.last_source tags cmudict vs g2p."""

    def __init__(self) -> None:
        from nltk.corpus import cmudict
        self.d = cmudict.dict()
        self._g2p = None
        self.last_source = None

    def phonemes(self, word: str) -> list[str] | None:
        raw = word.lower().strip(".,!?;:\"'()")
        w = norm_word(word)
        if any(c.isdigit() for c in raw):
            # numbers/alphanumerics: g2p_en expands digits to words internally
            # ("2012" -> "twenty twelve")
            if self._g2p is None:
                from g2p_en import G2p
                self._g2p = G2p()
            phones = [_STRESS.sub("", p) for p in self._g2p(raw)
                      if p.strip() and _STRESS.sub("", p) in VISEME_MAP]
            self.last_source = "g2p" if phones else None
            return phones or None
        if not w:
            self.last_source = None
            return None
        for cand in (w, w.replace("'", "")):
            if cand in self.d:
                self.last_source = "cmudict"
                return [_STRESS.sub("", p) for p in self.d[cand][0]]
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        phones = [_STRESS.sub("", p) for p in self._g2p(w)
                  if p.strip() and _STRESS.sub("", p) in VISEME_MAP]
        self.last_source = "g2p" if phones else None
        return phones or None


def visemes(phones: list[str]) -> list[str]:
    return [VISEME_MAP[p] for p in phones]  # KeyError = mapping bug, fail loud


def collapse(seq: list[str]) -> list[str]:
    """Merge adjacent repeats (P P -> P): what the lips actually show."""
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def load_gemma():
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gd = os.environ.get("GEMMA_ROOT")
    if not gd:
        raise SystemExit("set GEMMA_ROOT to the local Gemma-4 12B (it-qat-q4_0-unquantized) model directory")
    tok = AutoTokenizer.from_pretrained(gd)
    kw = {"torch_dtype": torch.bfloat16}
    if os.environ.get("GEMMA_ATTN"):     # e.g. "eager" for --deterministic
        kw["attn_implementation"] = os.environ["GEMMA_ATTN"]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            gd, **kw).to("cuda").eval()
    except Exception:
        # multimodal wrappers (Gemma3/Gemma4Unified...ForConditionalGeneration)
        # aren't AutoModelForCausalLM — resolve the class from the config
        import transformers as _tf
        from transformers import AutoConfig
        arch = (AutoConfig.from_pretrained(gd).architectures
                or ["Gemma3ForConditionalGeneration"])[0]
        model = getattr(_tf, arch).from_pretrained(
            gd, **kw).to("cuda").eval()
    return tok, model
