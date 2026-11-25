"""Label-free evidence for every candidate transcript of a sample.

Per distinct transcript y:
  hear, hear0   Whisper large-v3 log p(y | input audio) and log p(y | silence of the same length), teacher-forced
  vis, vis0     the lip reader's log p(y | lips) and log p(y | frozen mouth) (Auto-AVSR VSR, forced decoding,
                0.9 attention + 0.1 CTC totals, the reader's own decoding weights)
  lm            Gemma-4 log p(y) (a completeness flag for the selection; it does not enter the rule)
Each recogniser's language prior cancels in its PMI (hear - hear0, vis - vis0); a sense whose input carries no word
information gives PMI ~ 0 to every hypothesis and drops out by itself.

A spec is {sample_id: {"wav": input audio, "roi": mouth-ROI video (.avi) or null, "hyp": {key: transcript}}};
`score_spec` returns {sample_id: {"hyp": {key: canonical transcript}, "s": {transcript: scores}}}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def canon(t):
    """canonical transcript: task tag and 'Transcript:' prefix dropped, lower case, [a-z0-9'] only"""
    t = re.sub(r"<(denoise|inpaint)>\s*", "", str(t or ""))
    t = t.split("Transcript:", 1)[-1]
    t = re.sub(r"[^a-z0-9' ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def read_gray_frames(avi):
    import cv2
    cap, fr = cv2.VideoCapture(str(avi)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return fr


class Vision:
    """Auto-AVSR VSR as a scorer of transcripts given the lips"""

    def __init__(self, dev, auto_avsr_dir, vsr_ckpt):
        import torch
        aa = Path(auto_avsr_dir)
        sys.path.insert(0, str(aa))
        from datamodule.transforms import VideoTransform
        from lightning import ModelModule
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"data": {"modality": "video"},
                                "model": {"visual_backbone": OmegaConf.load(aa / "configs/model/visual_backbone/resnet_conformer.yaml")},
                                "pretrained_model_path": str(vsr_ckpt),
                                "optimizer": OmegaConf.load(aa / "configs/optimizer/adamw.yaml"),
                                "trainer": OmegaConf.load(aa / "configs/trainer/train.yaml"),
                                "transfer_frontend": False, "transfer_encoder": False})
        self.mm = ModelModule(cfg).to(dev).eval()
        self.vt, self.torch, self.dev = VideoTransform(subset="test"), torch, dev

    def encode_frames(self, frames):
        v = self.torch.from_numpy(np.stack(frames))[:, None].repeat(1, 3, 1, 1)
        with self.torch.no_grad():
            h, _ = self.mm.model.encoder(self.vt(v).to(self.dev).unsqueeze(0), None)
        return h

    def score_parts(self, hs, text):
        """totals (not per token): attention log P(text | lips) and CTC log P(text | lips) over every video frame -
        CTC must place blanks on lip frames the text leaves unexplained, so it penalises missing words"""
        torch, m = self.torch, self.mm.model
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
        ids = self.mm.text_transform.tokenize(text.upper()).to(self.dev)
        if len(ids) == 0:
            return None
        with torch.no_grad():
            ctc = -torch.nn.functional.ctc_loss(m.ctc.log_softmax(hs).transpose(0, 1), ids[None], [hs.shape[1]],
                                                [len(ids)], blank=0, reduction="sum", zero_infinity=True)
            mem = m.proj_decoder(hs) if getattr(m, "proj_decoder", None) is not None else hs
            ys_in = torch.cat([torch.tensor([m.sos], device=self.dev), ids])
            ys_out = torch.cat([ids, torch.tensor([m.eos], device=self.dev)])
            pred, _ = m.decoder(ys_in[None], subsequent_mask(len(ys_in), device=self.dev)[None], mem, None)
            att = torch.log_softmax(pred[0], -1).gather(1, ys_out[:, None]).sum()
        return {"att": float(att), "ctc": float(ctc), "n": len(ids) + 1, "frames": int(hs.shape[1])}


class WhisperHearing:
    """log P(text | audio) by Whisper large-v3, teacher-forced; scored against the input and against an all-silent
    input of the same length, so the difference keeps only acoustic evidence"""

    def __init__(self, dev, whisper_dir):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(whisper_dir)
        self.m = WhisperForConditionalGeneration.from_pretrained(whisper_dir, torch_dtype=torch.float16).to(dev).eval()
        t = self.proc.tokenizer
        self.prefix = t.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eot = t.convert_tokens_to_ids("<|endoftext|>")
        self.torch, self.dev = torch, dev

    def encode(self, y):
        x = self.proc.feature_extractor(y, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev, self.torch.float16)
        with self.torch.no_grad():
            return self.m.model.encoder(x).last_hidden_state

    def score(self, enc, texts):
        torch = self.torch
        seqs = []
        for t in texts:
            t = re.sub(r"\bi\b", "I", t)
            seqs.append(self.proc.tokenizer(" " + t[:1].upper() + t[1:], add_special_tokens=False).input_ids + [self.eot])
        n = max(len(s) for s in seqs)
        lab = torch.full((len(seqs), n), -1, dtype=torch.long)
        for i, s in enumerate(seqs):
            lab[i, :len(s)] = torch.tensor(s)
        dec_in = torch.cat([torch.tensor(self.prefix).expand(len(seqs), -1), lab[:, :-1].clamp(min=0)], 1).to(self.dev)
        lab = lab.to(self.dev)
        with torch.no_grad():
            lg = self.m(encoder_outputs=(enc.expand(len(seqs), -1, -1),), decoder_input_ids=dec_in).logits[:, len(self.prefix) - 1:]
            lp = torch.log_softmax(lg.float(), -1).gather(2, lab.clamp(min=0)[..., None])[..., 0]
            tot = (lp * (lab >= 0)).sum(1)
        return {t: {"ll": float(v), "n": len(s)} for t, v, s in zip(texts, tot, seqs)}


class Language:
    """Gemma-4 (base, no LoRA) log p(text)"""

    def __init__(self, dev, gemma_dir):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(gemma_dir)
        self.lm = AutoModelForCausalLM.from_pretrained(gemma_dir, torch_dtype=torch.bfloat16).to(dev).eval()
        self.torch, self.dev = torch, dev

    def total(self, t):
        ids = [self.tok.bos_token_id] + self.tok(t, add_special_tokens=False).input_ids   # Gemma-4 adds no BOS itself
        x = self.torch.tensor([ids], device=self.dev)
        with self.torch.no_grad():
            lp = self.torch.log_softmax(self.lm(x).logits[0, :-1].float(), -1)
        return float(lp.gather(1, x[0, 1:, None]).sum())


def score_spec(spec, models, out_path=None, device="cuda", with_lm=True):
    """Evidence for every transcript of every sample of `spec`. models: {"auto_avsr_dir", "vsr_ckpt", "whisper_dir",
    "gemma_dir"}. Resumable through out_path (a sample counts as done when every transcript has 'lm').
    with_lm=False writes lm = None instead of running Gemma-4: the rule only checks that the key exists."""
    import librosa
    import torch
    out_p = Path(out_path) if out_path else None
    res = json.load(open(out_p)) if out_p and out_p.exists() else {}
    todo = []
    for sid, sp in sorted(spec.items()):
        ts = {canon(t) for t in sp["hyp"].values()} - {""}
        r = res.get(sid)
        if r and set(r["hyp"]) == set(sp["hyp"]) and all("lm" in r["s"].get(t, {}) for t in ts):
            continue
        res[sid] = {"hyp": {k: canon(v) for k, v in sp["hyp"].items()}, "s": {t: {} for t in ts}}
        todo.append(sid)

    def save():
        if out_p:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out_p, "w"))
    print(f"evidence: {len(todo)} samples to score", flush=True)
    vis = Vision(device, models["auto_avsr_dir"], models["vsr_ckpt"])
    venc = {}
    for sid in todo:
        roi = spec[sid].get("roi")
        if roi in venc:
            continue
        if not roi or not Path(roi).exists():
            venc[roi] = None
            continue
        fr = read_gray_frames(roi)
        if len(fr) < 5:
            venc[roi] = None
            continue
        venc[roi] = [vis.encode_frames(frames) for frames in (fr, [fr[0]] * len(fr))]   # real lips, frozen mouth
    for sid in todo:
        e = venc.get(spec[sid].get("roi"))
        for t, d in res[sid]["s"].items():
            if e is None:
                d["vis"] = d["vis0"] = None
                continue
            p1, p0 = vis.score_parts(e[0], t), vis.score_parts(e[1], t)
            d["vis"] = 0.9 * p1["att"] + 0.1 * p1["ctc"] if p1 else None
            d["vis0"] = 0.9 * p0["att"] + 0.1 * p0["ctc"] if p0 else None
    del vis, venc
    torch.cuda.empty_cache()
    save()
    wh = WhisperHearing(device, models["whisper_dir"])
    for sid in todo:
        ts = list(res[sid]["s"])
        if not ts or not Path(spec[sid]["wav"]).exists():
            continue
        y, _ = librosa.load(spec[sid]["wav"], sr=16000, mono=True)
        sc, nu = wh.score(wh.encode(y), ts), wh.score(wh.encode(np.zeros_like(y)), ts)
        for t in ts:
            res[sid]["s"][t].update({"hear": sc[t]["ll"], "hear0": nu[t]["ll"]})
    del wh
    torch.cuda.empty_cache()
    save()
    if with_lm:
        lm = Language(device, models["gemma_dir"])
        for sid in todo:
            for t in res[sid]["s"]:
                res[sid]["s"][t]["lm"] = lm.total(t)
        del lm
        torch.cuda.empty_cache()
    else:
        for sid in todo:
            for t in res[sid]["s"]:
                res[sid]["s"][t]["lm"] = None
    save()
    return res
