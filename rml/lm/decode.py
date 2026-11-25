"""L: Gemma-4 12B + LoRA (r 32), select-then-edit decoding.

Per row: the answer up to 'Corrected:' is decoded greedily ('Pick: <letter>', whose next-token distribution over the
shown letters and 'none' is recorded), then a beam of 10 decodes the corrected window transcript. final_list = the
corrected transcripts (beam order), then the shown readings by pick probability, distinct, at most 10; its first
entry is L's transcript y_L.

Rows (jsonl): {"id", "context", "f0", "f1", ...}; readers: {id: [readings]} (prompts.load_readers).
  python -m rml.lm.decode --rows rows.jsonl --readers readers.json --out decode.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from rml.lm.prompts import READER_K, SW_LETTERS, chat_affixes, load_readers, shown_readings, user_text

# the deployed checkpoint's training configuration (checked on load)
EXPECTED_ARGS = {"evidence": "none", "select_edit": True, "chat_wrap": True, "gap_dur_prompt": True,
                 "reader_nbest_k": READER_K, "mcl_slots": 1, "full_window": False, "inject_mode": "prefix"}


def load_gemma(gemma_dir, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(gemma_dir)
    kw = {"torch_dtype": torch.bfloat16}
    if os.environ.get("GEMMA_ATTN"):     # "eager" for bitwise-reproducible decoding
        kw["attn_implementation"] = os.environ["GEMMA_ATTN"]
    try:
        model = AutoModelForCausalLM.from_pretrained(gemma_dir, **kw).to(device).eval()
    except Exception:
        import transformers as _tf
        from transformers import AutoConfig
        arch = (AutoConfig.from_pretrained(gemma_dir).architectures or ["Gemma3ForConditionalGeneration"])[0]
        model = getattr(_tf, arch).from_pretrained(gemma_dir, **kw).to(device).eval()
    return tok, model


class LanguageModel:
    def __init__(self, gemma_dir, ckpt, device="cuda"):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        ta = st.get("args", {})
        for k, v in EXPECTED_ARGS.items():
            got = ta.get(k, {"full_window": False, "inject_mode": "prefix", "mcl_slots": 1}.get(k))
            assert got == v, f"checkpoint {ckpt} has {k}={got!r}, the released decoder expects {v!r}"
        self.max_new = max(128, int(ta.get("aux_max_new", 44)))
        self.tok, gemma = load_gemma(gemma_dir, device)
        self.chat = chat_affixes(self.tok)
        if hasattr(gemma, "language_model") or "ConditionalGeneration" in type(gemma).__name__:
            target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
        else:
            target = ["q_proj", "k_proj", "v_proj", "o_proj"]
        r = int(ta.get("lora_r", 16))
        gemma = get_peft_model(gemma, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules=target,
                                                 bias="none"))
        set_peft_model_state_dict(gemma, st["lora"])
        self.gemma = gemma.eval()
        self.emb = gemma.get_input_embeddings()
        self.device = device
        tok = self.tok
        base_ = tok("Pick:", add_special_tokens=False).input_ids
        self.sw_opts = {}
        for o in [*list(SW_LETTERS), "none"]:
            ids_ = tok(f"Pick: {o}", add_special_tokens=False).input_ids
            assert ids_[:len(base_)] == base_ and len(ids_) == len(base_) + 1, \
                f"'Pick: {o}' is not exactly one token after 'Pick:' ({ids_})"
            self.sw_opts[o] = ids_[-1]

    @torch.no_grad()
    def decode_row(self, r, readers, nbest=10):
        tok, gemma, emb, device, chat = self.tok, self.gemma, self.emb, self.device, self.chat
        nb = max(nbest, 10)
        cue_txt = "Corrected:"
        a = chat[0] + tok(user_text(r, readers), add_special_tokens=False).input_ids
        b = list(chat[1])
        ie = torch.cat([emb(torch.tensor(a, device=device)), emb(torch.tensor(b, device=device))])[None]
        eos_ids = [chat[2], tok.eos_token_id]
        # 1) greedy until 'Corrected:'  2) beam over the corrected transcript only
        g1 = gemma.generate(inputs_embeds=ie, attention_mask=torch.ones(1, ie.shape[1], dtype=torch.long, device=device),
                            max_new_tokens=self.max_new, do_sample=False, num_beams=1, eos_token_id=eos_ids,
                            pad_token_id=tok.eos_token_id, use_cache=True, return_dict_in_generate=True,
                            output_scores=True)
        ids1 = g1.sequences[0].tolist()
        prefix_ids = []
        for cut in range(1, len(ids1) + 1):
            if tok.decode(ids1[:cut], skip_special_tokens=True).rstrip().endswith(cue_txt):
                prefix_ids = ids1[:cut]
                break
        sw_shown = shown_readings(r, readers)
        sw_probs = None
        opts = [*list(SW_LETTERS[:len(sw_shown)]), "none"]
        for cut in range(1, len(ids1)):
            if tok.decode(ids1[:cut], skip_special_tokens=True).rstrip().endswith("Pick:"):
                if cut < len(g1.scores):
                    lg_ = g1.scores[cut][0].float()
                    v_ = torch.stack([lg_[self.sw_opts[o]] for o in opts]).softmax(0).tolist()
                    sw_probs = {o: round(x, 4) for o, x in zip(opts, v_)}
                break
        prefix_txt = ""
        if prefix_ids:
            prefix_txt = tok.decode(prefix_ids, skip_special_tokens=True)
            ie = torch.cat([ie[0], emb(torch.tensor(prefix_ids, device=device))])[None]
        am = torch.ones(1, ie.shape[1], dtype=torch.long, device=device)
        gen = gemma.generate(inputs_embeds=ie, attention_mask=am, max_new_tokens=(64 if prefix_ids else self.max_new),
                             do_sample=False, num_beams=nb, num_return_sequences=nbest, length_penalty=1.0,
                             eos_token_id=eos_ids, pad_token_id=tok.eos_token_id, use_cache=True,
                             return_dict_in_generate=True, output_scores=True)
        ss = getattr(gen, "sequences_scores", None)
        sc = ss.tolist() if ss is not None else [0.0] * len(gen.sequences)
        lst = [((prefix_txt + " " + tok.decode(s, skip_special_tokens=True)).strip()
                if prefix_ids else tok.decode(s, skip_special_tokens=True).strip(), q) for s, q in zip(gen.sequences, sc)]
        texts, scores, seen = [], [], set()
        for t, q in lst:                                   # dedup on the words after the cue
            key = " ".join(t.split("Missing words:")[-1].lower().split())
            if key not in seen:
                seen.add(key)
                texts.append(t)
                scores.append(round(q, 3))
        row = {"id": r["id"], "nbest": texts, "nbest_scores": scores, "shown_readings": sw_shown, "pick_probs": sw_probs}
        head_ = texts[0].split("Missing words:")[0] if texts else ""
        pick_text = "none"
        for ln in head_.splitlines():
            if ln.startswith("Pick:"):
                pick_text = row["pick_text"] = ln.split(":", 1)[1].strip()
        order = sorted(sw_probs, key=lambda o: -sw_probs[o]) if sw_probs else \
            [pick_text] + (["none"] if pick_text != "none" else [])
        row["pick"] = order[0]

        def full(o):
            return {"type": "full", "text": sw_shown[SW_LETTERS.index(o)]} if o in SW_LETTERS[:len(sw_shown)] else None
        row["mode"] = "select_edit"
        row["picked_reading"] = full(order[0])["text"] if full(order[0]) else ""
        seen_, fl = set(), []
        for c in [{"type": "full", "text": t.split("Corrected:")[-1].strip()} for t in texts] + \
                [full(o) for o in order if full(o)]:
            k_ = " ".join(c["text"].lower().split())
            if k_ and k_ not in seen_:
                seen_.add(k_)
                fl.append(c)
        row["final_list"] = fl[:10]
        return row


def decode_rows(rows, readers, gemma_dir, ckpt, out=None, nbest=10, device="cuda"):
    """decode every row (resumable through <out>.partial.jsonl)"""
    lm = LanguageModel(gemma_dir, ckpt, device)
    done, res = set(), []
    partial = Path(str(out) + ".partial.jsonl") if out else None
    if partial and partial.exists():
        for line in partial.read_text().splitlines():
            try:
                rr = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rr["id"] not in done:
                done.add(rr["id"])
                res.append(rr)
    fh = open(partial, "a") if partial else None
    for r in rows:
        if r["id"] in done:
            continue
        row = lm.decode_row(r, readers, nbest)
        res.append(row)
        if fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    if out:
        Path(out).write_text(json.dumps({"rows": res}, indent=1))
    return res


def main():
    from rml.paths import load_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--readers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    P = load_paths(a.paths)
    rows = [json.loads(line) for line in a.rows.read_text().splitlines() if line.strip()]
    decode_rows(rows, load_readers(a.readers), P.gemma_dir, P.lm_checkpoint, a.out)


if __name__ == "__main__":
    main()
