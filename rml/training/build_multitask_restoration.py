#!/usr/bin/env python
"""Generator (G) training data, stages 2 and 3: the multi-task audio-visual speech restoration root.

ONE LTX-2.5 LoRA is trained on both tasks:
  <inpaint>  silenced spans (k = 1..10 words), text = '<inpaint>\n' + the lip reader's transcript line
  <denoise>  noise over the whole 4.8 s window, text = '<denoise>' (stage 2) or a transcript variant (stage 3)
en / es / fr, varied noise TYPES from the noise bank (build_noise_bank.py) plus a competing speaker (another
window's clean speech).

Stages (resumable), in order:
  train          <out>/train: for each window of the stage-1 training cuts (<src>/conditions_avsr_vsrfw_k10): one
                 INPAINT sample (one cut of the window, k balanced 1..10 by round-robin over sorted windows; latents /
                 clean audio / muted audio / reference hard-linked from <src>) and one DENOISE sample (noise type
                 uniform over babble / sfx_seen / colored / speech, SNR U[-15,+5] dB, 20% get a second source at
                 +5 dB relative; noisy latents encoded here with the LTX-2.5 audio VAE).
                 Writes metadata.json + _texts_inpaint.json (per-id tagged text).
  native-texts   es/fr inpaint texts from the native Spanish/French lip readers (docs/TRAINING.md).
  (encode)       rml/generator/conditions.py -> <out>/train/conditions_mt (incl. the '<denoise>' template)
  link-denoise   hard-link every denoise id's conditions to the ONE encoded '<denoise>' template.
  text-conds     stage 3: conditions_mt_text texts (denoise ids get the reader line / the reference sentence / no
                 text by crc32(id) % 4); then encode them with rml/generator/conditions.py.
  train-wavs     the 16 kHz NOISY waveforms of the train root's denoise samples (build_train keeps only their latents),
                 rebuilt bit-for-bit from the metadata (noise, noise2, snr_db) and the same per-sample RNG; --verify N
                 re-encodes N of them and compares with the stored noisy_audio_latents. Needed only to build the
                 language model's stage-3 (denoise) rows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))   # audio_latents.py

# Paths: set from the command line in main() (defaults from the RML_* environment variables, see docs/TRAINING.md).
SRC: Path       # the stage-1 train root (talkvid_mt_span/train)
BANK: Path      # the noise bank (build_noise_bank.py --out)
OUT: Path       # the multi-task root's parent dir; the train root is OUT / "train"
CASCADE_TEXTS: Path
CKPT: Path      # LTX-2.5 audio VAE safetensors
NATIVE_GLOB: str
TRAIN_TYPES = ("babble", "sfx_seen", "colored", "speech")
LANGS = ("en", "es", "fr")


class Bank:
    def __init__(self, split: str):
        idx = json.loads((BANK / "index.json").read_text())
        self.sr = idx["sr"]
        self.arr, self.segs = {}, {}
        for typ, splits in idx["types"].items():
            if split in splits:
                self.arr[typ] = np.load(BANK / f"{typ}_{split}.npy", mmap_mode="r")
                self.segs[typ] = splits[split]

    def draw(self, typ: str, n: int, rng: random.Random) -> np.ndarray:
        segs = self.segs[typ]
        s = rng.choices(segs, weights=[x["len"] for x in segs])[0]
        y = np.asarray(self.arr[typ][s["start"]:s["start"] + s["len"]], dtype=np.float32)
        if len(y) < n:
            y = np.tile(y, n // len(y) + 1)
        o = rng.randint(0, len(y) - n)
        return y[o:o + n].copy()


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)) + 1e-9)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    return clean + noise * (rms(clean) / (10 ** (snr_db / 20.0)) / rms(noise))


def finish(y: np.ndarray) -> np.ndarray:
    p = float(np.abs(y).max())
    return (y / p * 0.95 if p > 0.99 else y).astype(np.float32)


def resample(y: np.ndarray, a: int, b: int) -> np.ndarray:
    if a == b:
        return y
    import librosa
    return librosa.resample(y, orig_sr=a, target_sr=b)


def noise_for(typ, n, sr, bank, rng, speech_pool, own_vid, load):
    """n samples at the VAE rate sr of noise type typ; speech = another video's clean window."""
    if typ == "speech":
        while True:
            clip, vid = rng.choice(speech_pool)
            if vid != own_vid:
                break
        y = load(clip, sr)
        return np.tile(y, n // len(y) + 1)[:n] if len(y) < n else y[:n]
    m = int(np.ceil(n * bank.sr / sr)) + 1
    return resample(bank.draw(typ, m, rng), bank.sr, sr)[:n]


def vae_and_loader(device):
    from ltx_core.model.audio_vae import AudioProcessor
    from ltx_trainer.model_loader import load_audio_vae_encoder
    from audio_latents import encode_waveform, load_audio_window
    vae = load_audio_vae_encoder(CKPT, device=device, dtype=torch.bfloat16)
    proc = AudioProcessor(target_sample_rate=vae.sample_rate, mel_bins=vae.mel_bins,
                          mel_hop_length=vae.mel_hop_length, n_fft=vae.n_fft).to(device)
    return vae, proc, encode_waveform, (lambda clip, sr: load_audio_window(clip, 0.0, sr))


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.link(src, dst)


def build_train(seed: int) -> None:
    out = OUT / "train"
    texts = json.loads(CASCADE_TEXTS.read_text())
    meta = {c["id"]: c for c in json.loads((SRC / "metadata.json").read_text())["clips"]}
    wins = defaultdict(list)
    for cid in texts:
        if cid in meta and meta[cid].get("language", "en") in LANGS:
            wins[cid.rsplit("__k", 1)[0]].append(cid)
    inp, rows, jobs = {}, [], []
    for gi, w in enumerate(sorted(wins)):                     # one inpaint cut per window, k balanced 1..10
        byk = {int(c.rsplit("__k", 1)[1]): c for c in wins[w]}
        want = 1 + gi % 10
        k = min(byk, key=lambda x: (abs(x - want), x))
        cut = byk[k]
        sid = f"{cut}__inp"
        inp[sid] = "<inpaint>\n" + texts[cut]
        rows.append({**meta[cut], "id": sid, "task": "inpaint", "source_cut": cut, "clip": str(SRC / meta[cut]["clip"])})
        jobs.append((f"{w}__dn", cut))
    for d in ("latents", "audio_latents", "noisy_audio_latents", "reference_latents", "conditions_mt"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for r in rows:
        c = r["source_cut"]
        for d in ("latents", "audio_latents", "noisy_audio_latents", "reference_latents"):
            link(SRC / d / f"{c}.pt", out / d / f"{r['id']}.pt")
    rng = random.Random(seed)
    speech_pool = [(str(SRC / meta[c]["clip"]), c.split("_NA_")[0]) for _, c in jobs]
    plan = []
    for sid, cut in jobs:
        typ = rng.choice(TRAIN_TYPES)
        second = rng.choice([t for t in TRAIN_TYPES if t != typ]) if rng.random() < 0.2 else None
        plan.append((sid, cut, typ, round(rng.uniform(-15, 5), 2), second))
        for d in ("latents", "audio_latents", "reference_latents"):
            link(SRC / d / f"{cut}.pt", out / d / f"{sid}.pt")
        rows.append({**meta[cut], "id": sid, "task": "denoise", "source_cut": cut, "clip": str(SRC / meta[cut]["clip"]), "noise": typ,
                     "noise2": second, "snr_db": plan[-1][3], "mask_segments_s": []})
    (out / "metadata.json").write_text(json.dumps({"clips": rows}, indent=1))
    (out / "_texts_inpaint.json").write_text(json.dumps({**inp, "_denoise_template": "<denoise>"}, indent=1))
    langs = defaultdict(lambda: defaultdict(int))
    for r in rows:
        langs[r["task"]][r.get("language", "en")] += 1
    print("train rows:", len(rows), {t: dict(v) for t, v in langs.items()}, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, proc, encode_waveform, load = vae_and_loader(device)
    sr = vae.sample_rate
    bank = Bank("train")
    for sid, cut, typ, snr, second in tqdm(plan, desc="denoise train"):
        dst = out / "noisy_audio_latents" / f"{sid}.pt"
        if dst.exists():
            continue
        r = random.Random(f"{seed}:{sid}")
        clean = load(str(SRC / meta[cut]["clip"]), sr)
        own = cut.split("_NA_")[0]
        noisy = mix(clean, noise_for(typ, len(clean), sr, bank, r, speech_pool, own, load), snr)
        if second:
            noisy = mix(noisy, noise_for(second, len(clean), sr, bank, r, speech_pool, own, load), snr + 5)
        torch.save(encode_waveform(vae, proc, finish(noisy), sr), dst)
    print("TRAIN_DONE", out)


def train_wavs(root: Path, seed: int, verify: int) -> None:
    rows = json.loads((root / "metadata.json").read_text())["clips"]
    dn = [r for r in rows if r.get("task") == "denoise"]
    speech_pool = [(r["clip"], r["source_cut"].split("_NA_")[0]) for r in dn]   # = build_train's jobs order
    out = root / "noisy_wavs_16k"
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, proc, encode_waveform, load = vae_and_loader(device)
    sr = vae.sample_rate
    bank = Bank("train")
    checked = []
    for r in tqdm(dn, desc="noisy train wavs"):
        dst = out / f"{r['id']}.wav"
        if dst.exists() and len(checked) >= verify:
            continue
        rr = random.Random(f"{seed}:{r['id']}")
        clean = load(r["clip"], sr)
        own = r["source_cut"].split("_NA_")[0]
        noisy = mix(clean, noise_for(r["noise"], len(clean), sr, bank, rr, speech_pool, own, load), r["snr_db"])
        if r.get("noise2"):
            noisy = mix(noisy, noise_for(r["noise2"], len(clean), sr, bank, rr, speech_pool, own, load), r["snr_db"] + 5)
        noisy = finish(noisy)
        if len(checked) < verify:
            ref = root / "noisy_audio_latents" / f"{r['id']}.pt"
            if ref.exists():
                a = encode_waveform(vae, proc, noisy, sr)
                b = torch.load(ref, map_location="cpu")
                ta, tb = (a["latents"] if isinstance(a, dict) else a).float().cpu(), (b["latents"] if isinstance(b, dict) else b).float()
                checked.append(float((ta - tb).abs().max() / tb.abs().max()))
                print(f"verify {r['id']}: max rel diff {checked[-1]:.2e}", flush=True)
        sf.write(dst, resample(noisy, sr, 16000), 16000)
    print(f"TRAIN_WAVS_DONE {len(dn)} -> {out}; verify max rel diffs {checked}")


def link_denoise(root: Path) -> None:
    cd = root / "conditions_mt"
    tpl = cd / "_denoise_template.pt"
    assert tpl.exists(), f"encode the template first: {tpl}"
    n = 0
    for r in json.loads((root / "metadata.json").read_text())["clips"]:
        if r.get("task") == "denoise":
            dst = cd / f"{r['id']}.pt"
            if not dst.exists():
                os.link(tpl, dst)
                n += 1
    print(f"linked {n} denoise conditions -> {cd}")


def native_texts(out_json: Path) -> None:
    """es/fr inpaint training texts from the native readers (Ma et al. CMU-MOSEAS VSR, full-window top-1):
    '<inpaint>\nTranscript: <native top-1>' for every es/fr inpaint sample (replaces the English Auto-AVSR line)."""
    import glob
    root = OUT / "train"
    rd = {}
    for f in glob.glob(NATIVE_GLOB):
        for r in json.loads(Path(f).read_text())["rows"]:
            rd[r["id"]] = (r.get("nbest") or [""])[0].strip()
    texts, miss = {}, 0
    for r in json.loads((root / "metadata.json").read_text())["clips"]:
        if r.get("task") != "inpaint" or r.get("language") not in ("es", "fr"):
            continue
        w = r.get("base_id") or r["source_cut"].rsplit("__k", 1)[0]
        if w not in rd:
            miss += 1
            continue
        texts[r["id"]] = "<inpaint>\nTranscript: " + rd[w]
    out_json.write_text(json.dumps(texts, indent=1, ensure_ascii=False))
    print(f"native es/fr inpaint texts: {len(texts)} (missing native reading: {miss}) -> {out_json}")


def text_conds() -> None:
    """Stage 3: conditions_mt_text for avse25_multitask_text = transcript-guided denoising.
    inpaint ids: hard-linked to conditions_mt (unchanged). denoise ids, deterministic by crc32(id) % 4:
      0, 1 -> '<denoise>\n' + the cascade reader line of the source cut ('Transcript: <Auto-AVSR VSR line>')
      2    -> '<denoise>\nTranscript: ' + the reference sentence, lower case, punctuation stripped
      3    -> '<denoise>' (hard link to conditions_mt/_denoise_template.pt)
    Writes train/_texts_conditions_mt_text.json (buckets 0-2) for rml/generator/conditions.py."""
    import re
    import zlib
    root = OUT / "train"
    src, dst = root / "conditions_mt", root / "conditions_mt_text"
    dst.mkdir(exist_ok=True)
    reader = json.loads(CASCADE_TEXTS.read_text())
    texts, buckets, linked = {}, defaultdict(int), 0
    for r in json.loads((root / "metadata.json").read_text())["clips"]:
        sid = r["id"]
        if r["task"] == "inpaint":
            link(src / f"{sid}.pt", dst / f"{sid}.pt")
            linked += 1
            continue
        b = zlib.crc32(sid.encode()) % 4
        buckets[b] += 1
        if b in (0, 1):
            texts[sid] = "<denoise>\n" + reader[r["source_cut"]]
        elif b == 2:
            texts[sid] = "<denoise>\nTranscript: " + re.sub(r"[^a-z0-9' ]+", " ", r["sentence"].lower()).strip()
        else:
            link(src / "_denoise_template.pt", dst / f"{sid}.pt")
            linked += 1
    (root / "_texts_conditions_mt_text.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))
    print(f"conditions_mt_text: {linked} linked (inpaint + text-free denoise), {len(texts)} to encode; "
          f"denoise buckets {dict(sorted(buckets.items()))}")


def main() -> None:
    global SRC, BANK, OUT, CASCADE_TEXTS, CKPT, NATIVE_GLOB
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "native-texts", "link-denoise", "text-conds", "train-wavs"])
    ap.add_argument("--src-root", type=Path, default=env("RML_G_STAGE1_ROOT"),
                    help="stage-1 train root (talkvid_mt_span/train); default $RML_G_STAGE1_ROOT")
    ap.add_argument("--out-root", type=Path, default=env("RML_G_MULTITASK_DIR"),
                    help="parent of the multi-task train root; default $RML_G_MULTITASK_DIR")
    ap.add_argument("--noise-bank", type=Path, default=env("RML_NOISE_BANK"), help="default $RML_NOISE_BANK")
    ap.add_argument("--audio-vae", type=Path, default=env("RML_LTX_AUDIO_VAE"),
                    help="LTX-2.5 audio VAE safetensors; default $RML_LTX_AUDIO_VAE")
    ap.add_argument("--native-readings", default=env("RML_NATIVE_TRAIN_READINGS", ""),
                    help="native-texts: glob of the es/fr native lip-reader train decodes ({rows:[{id, nbest}]})")
    ap.add_argument("--root", type=Path, default=None, help="link-denoise / train-wavs: the train root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", type=int, default=3, help="train-wavs: re-encode this many and compare latents")
    a = ap.parse_args()
    SRC, BANK, OUT, CKPT, NATIVE_GLOB = a.src_root, a.noise_bank, a.out_root, a.audio_vae, a.native_readings
    CASCADE_TEXTS = SRC / "conditions_avsr_vsrfw_k10/_final_conditions.json" if SRC else None
    if a.stage == "train":
        build_train(a.seed)
    elif a.stage == "native-texts":
        native_texts(OUT / "train" / "_texts_native_esfr.json")
    elif a.stage == "train-wavs":
        train_wavs(a.root or OUT / "train", a.seed, a.verify)
    elif a.stage == "text-conds":
        text_conds()
    else:
        link_denoise(a.root or OUT / "train")


if __name__ == "__main__":
    main()
