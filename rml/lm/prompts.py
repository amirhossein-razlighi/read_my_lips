"""Prompt of the language model L (Sec. 4.2, Fig. 5): the audible context with the gap marked, the lip reader's
distinct readings under letters A..J (best first) and the gap duration, inside the Gemma-4 chat template. The answer
is 'Pick: <letter>' and then 'Corrected: <window transcript>'."""
from __future__ import annotations

import json
import re

SW_LETTERS = "ABCDEFGHIJ"
READER_K = 10   # readings shown at test time


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower(), flags=re.UNICODE)


def load_readers(path) -> dict:
    """reader n-best json ({rows: [{id, nbest}]}) -> {window or sample id: [readings]}; ids may carry __k<N>"""
    d = {}
    j = json.loads(open(path).read())
    for r in (j["rows"] if isinstance(j, dict) and "rows" in j else j):
        nb = r.get("nbest") or []
        nb = json.loads(nb) if isinstance(nb, str) else nb
        d.setdefault(r["id"].rsplit("__k", 1)[0], list(nb))
    return d


def readings_for(r: dict, readers: dict) -> list:
    """the readings of a row: keyed by its own id (the pipeline), else by its window (ids may carry __k<N>)"""
    if r["id"] in readers:
        return readers[r["id"]]
    w = r["id"].rsplit("__k", 1)[0]
    return readers.get(w) or readers.get(str(r.get("feats", "")).rsplit("/", 1)[-1][:-4]) or []


def shown_readings(r: dict, readers: dict, k: int = READER_K) -> list:
    """the reader's distinct readings shown in the prompt (best first)"""
    out, seen = [], set()
    for h in readings_for(r, readers):
        key = " ".join(h.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(h.strip())
        if len(out) >= k:
            break
    return out


def reader_lines(r: dict, readers: dict) -> str:
    out = shown_readings(r, readers)
    if not out:
        return ""
    return "Lip reading candidates (best first):\n" + "".join(f"{SW_LETTERS[i]}. {h}\n" for i, h in enumerate(out))


def user_text(r: dict, readers: dict) -> str:
    """the user turn: context, candidates, gap duration (frames f0..f1 at 25 fps)"""
    s = f"Transcript: {r['context']}\n"
    s += reader_lines(r, readers)
    s += f"Gap duration: {(r['f1'] - r['f0']) / 25.0:.1f} s\n"
    return s


def chat_affixes(tok):
    """(pre_ids, post_ids, end_id): user content inside the model's chat template. Gemma-4 is template-locked: raw
    continuations degenerate, inside the template it answers cleanly. The generation prompt ends with an empty
    pre-closed thought channel, so the answer follows it directly."""
    m = "\x00C\x00"
    s = tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)
    pre, post = s.split(m)
    end = tok.convert_tokens_to_ids("<turn|>")
    if end is None or end == getattr(tok, "unk_token_id", None):
        end = tok.eos_token_id
    return tok(pre, add_special_tokens=False).input_ids, tok(post, add_special_tokens=False).input_ids, end
