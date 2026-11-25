"""The RML pipeline on a root (Sec. 4): R -> L -> G under six configurations, then the evidence-weighted selection.

A root is a directory with metadata.json: {"clips": [row, ...]}, one row per SAMPLE:
  id, clip (4.8 s face window, relative to the root), base_id (the window), language,
  task  "inpaint" (a silenced span; mask_segments_s = [[start, end]]) or "denoise" (a noisy track),
  input "muted" for the window's muted twin <base_id>__full_silence (added by add_muted_twins), which every
        "audio muted" configuration renders;
  masked_words (optional, benchmarks only): the reference words of the span, used by the audible-context leak guard.

The six configurations (CONFIGS) and what renders them (Sec. 4.3, Appendix B):
                        silenced input                                   noisy input
  R -> L -> G, kept     main LoRA, '<inpaint>\nTranscript: y_L', cfg 5    main LoRA, '<denoise>\nTranscript: y_L', cfg 1
  R -> G,      kept     stage-1 LoRA, 'Transcript: h1', cfg 5             main LoRA, '<denoise>\nTranscript: h1', cfg 1
  G alone,     kept     main LoRA, '<inpaint>', cfg 5                     stage-2 LoRA, '<denoise>', cfg 1
  R -> L -> G, muted    main LoRA on the muted twin, '<inpaint>\nTranscript: y_L(muted)', cfg 5
  R -> G,      muted    main LoRA on the muted twin, '<inpaint>\nTranscript: h1', cfg 5
  G alone,     muted    main LoRA on the muted twin, '<inpaint>', cfg 5
main = avse25_multitask_text (stage 3), stage-1 = avse25_vsr_cascade_k1to10, stage-2 = avse25_multitask_restoration.
30 steps, seed 42 everywhere. y_L(muted) is L in denoise mode on the muted twin.

Recipes (which numbers each one produced):
  tables  the table above: Tab. 2 and Tab. 3 (TalkVid, VoxMM) and the default for a new video
  hard    Tab. 4: the main LoRA for every configuration (R -> G kept: '<inpaint>\nTranscript: h1'; G alone kept on
          noisy input: the main LoRA)
  grid    Fig. 4: as hard; the evidence hears the 16 kHz input track instead of its audio-VAE reconstruction

Stages (each resumable, each writes under the root): latents, roi, reader, whisper, lm, texts, encode, render,
transcribe, evidence, select.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rml.lm.rows import clean_txt, denoise_answer, denoise_row, inpaint_answer, inpaint_row, merged

CONFIGS = {
    # name: (selection key, sample kind, LoRA key, text-CFG)
    "inp_gemma": ("gemma|keep", "inpaint", "g_main_lora", 5.0),
    "inp_lip": ("lip_text|keep", "inpaint", "g_cascade_lora", 5.0),
    "inp_av": ("av_only|keep", "inpaint", "g_main_lora", 5.0),
    "den_gemma": ("gemma|keep", "denoise", "g_main_lora", 1.0),
    "den_lip": ("lip_text|keep", "denoise", "g_main_lora", 1.0),
    "den_av": ("av_only|keep", "denoise", "g_restoration_lora", 1.0),
    "mute_gemma": ("gemma|mute", "muted", "g_main_lora", 5.0),
    "mute_lip": ("lip_text|mute", "muted", "g_main_lora", 5.0),
    "mute_av": ("av_only|mute", "muted", "g_main_lora", 5.0),
}
SINGLE_LORA = {"inp_lip": "g_main_lora", "den_av": "g_main_lora"}   # recipes hard and grid
RECIPES = ("tables", "hard", "grid")
STAGES = ("latents", "roi", "reader", "whisper", "lm", "texts", "encode", "render", "transcribe", "evidence", "select")


def kind(row):
    return "muted" if row.get("input") == "muted" else row["task"]


def twin_id(base_id):
    return f"{base_id}__full_silence"


def words_of(text: str) -> str:
    def norm(w):
        return re.sub(r"[^\w']", "", w.lower(), flags=re.UNICODE)
    t = text.split("Missing words:")[-1] if "Missing words:" in text else text
    return " ".join(w for w in t.split() if norm(w))


def add_muted_twins(rows):
    """append one muted twin per window (whole window silent, task inpaint)"""
    out, seen = list(rows), {r["id"] for r in rows}
    for r in rows:
        t = twin_id(r["base_id"])
        if t not in seen:
            seen.add(t)
            out.append({"id": t, "clip": r["clip"], "base_id": r["base_id"], "language": r.get("language", "en"),
                        "task": "inpaint", "input": "muted", "mask_segments_s": [[0.0, 4.8]], "masked_words": []})
    return out


class Pipeline:
    def __init__(self, root: Path, paths, device="cuda", recipe="tables"):
        assert recipe in RECIPES, recipe
        self.root, self.P, self.device, self.recipe = Path(root), paths, device, recipe
        self.meta = json.loads((self.root / "metadata.json").read_text())["clips"]
        self.by_id = {r["id"]: r for r in self.meta}
        self.work = self.root / "rml"
        self.work.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def rows(self, *kinds):
        return [r for r in self.meta if kind(r) in kinds]

    def windows(self):
        w = {}
        for r in self.meta:
            w.setdefault(r["base_id"], r)
        return w

    def evaluated(self):
        """the samples the selection runs on: every task sample, and a muted twin that is itself a test sample
        (the whole-clip silence cells of the grid, flagged evaluate: true), whose kept and muted outputs coincide"""
        return [r for r in self.meta if kind(r) != "muted" or r.get("evaluate")]

    def jload(self, name, default=None):
        p = self.work / name
        return json.loads(p.read_text()) if p.exists() else default

    def jsave(self, name, obj):
        (self.work / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False))

    def input_wav(self, sid):
        return self.root / "inputs_16k" / f"{sid}.wav"

    def roi(self, base_id):
        return self.work / "roi" / "roi" / f"{base_id}.avi"

    # ------------------------------------------------------------------ stages
    def latents(self):
        from rml.data.latents import encode_root
        encode_root(self.root, self.P.ltx_video_vae, self.P.ltx_audio_vae, self.device)

    def roi_stage(self, procs=8):
        from rml.reader.roi import prep_rows
        wins = [{"id": w, "clip": r["clip"]} for w, r in sorted(self.windows().items())]
        prep_rows(wins, self.root, self.work / "roi", self.P.roi_config(), procs)

    def reader(self):
        """R: VSR 10-best per window; Auto-AVSR AV n-best per noisy sample and per muted twin"""
        from rml.reader.auto_avsr import read_rows
        if self.jload("reader_vsr.json") is None:
            rows = [{"id": w, "roi": self.roi(w)} for w in sorted(self.windows())]
            self.jsave("reader_vsr.json", {"rows": read_rows(rows, self.P.auto_avsr_dir, self.P.auto_avsr_vsr, "vsr",
                                                             device=self.device)})
        if self.jload("reader_av.json") is None:
            rows = [{"id": r["id"], "roi": self.roi(r["base_id"]), "wav": self.input_wav(r["id"])}
                    for r in self.rows("denoise", "muted")]
            self.jsave("reader_av.json", {"rows": read_rows(rows, self.P.auto_avsr_dir, self.P.auto_avsr_avsr, "avsr",
                                                            device=self.device)})

    def whisper(self):
        """audible context of every silenced sample; Whisper's transcript of every noisy sample and muted twin"""
        import soundfile as sf

        from rml.data.context import context_or_placeholder, norm
        from rml.eval.asr import transcribe_16k
        res = self.jload("whisper.json", {})
        for r in self.meta:
            if r["id"] in res:
                continue
            y, _ = sf.read(self.input_wav(r["id"]), dtype="float32")
            if kind(r) == "inpaint":
                g0, g1 = r["mask_segments_s"][0]
                forbid = {norm(w) for w in r.get("masked_words") or []}
                res[r["id"]] = context_or_placeholder(y, g0, g1, forbid, r["id"], r.get("language", "en"))
            else:
                res[r["id"]] = transcribe_16k(y, language=r.get("language", "en"))
            self.jsave("whisper.json", res)

    def vsr(self):
        return {r["id"]: r.get("nbest") or [] for r in self.jload("reader_vsr.json")["rows"]}

    def lm(self):
        """L on every sample: inpaint rows (reader 10-best) and denoise rows (merged candidates)"""
        from rml.lm.decode import decode_rows
        wh, vsr = self.jload("whisper.json"), self.vsr()
        av = {r["id"]: r.get("nbest") or [] for r in self.jload("reader_av.json")["rows"]}
        rows, readers = [], {}
        for r in self.meta:
            sid = r["id"]
            if kind(r) == "inpaint":
                rows.append(inpaint_row(sid, wh[sid], r["mask_segments_s"][0], language=r.get("language", "en")))
                readers[sid] = vsr.get(r["base_id"], [])
            else:
                rows.append(denoise_row(sid, wh[sid], r.get("language", "en")))
                readers[sid] = merged(vsr.get(r["base_id"], []), av.get(sid, []), clean_txt(wh[sid]))
        self.jsave("lm_rows.json", rows)
        self.jsave("lm_readers.json", readers)
        decode_rows(rows, readers, self.P.gemma_dir, self.P.lm_checkpoint, out=self.work / "lm_decode.json",
                    device=self.device)

    def answers(self):
        """y_L per sample (the denoise-mode answer with the deployment guard)"""
        dec = {r["id"]: r for r in self.jload("lm_decode.json")["rows"]}
        rows = {r["id"]: r for r in self.jload("lm_rows.json")}
        readers = self.jload("lm_readers.json")
        out = {}
        for r in self.meta:
            sid = r["id"]
            if kind(r) == "inpaint":
                out[sid] = inpaint_answer(dec[sid], rows[sid]["context"])
            else:
                out[sid] = denoise_answer(dec[sid], readers[sid])[0]
        return out

    def texts(self):
        """the conditioning text of every configuration, per rendered sample"""
        ans, vsr = self.answers(), self.vsr()
        T = {name: {} for name in CONFIGS}
        for r in self.meta:
            sid, h = r["id"], vsr.get(r["base_id"], [])
            top = (h or [""])[0]
            k = kind(r)
            if k == "inpaint":
                g = clean_txt(ans[sid]) if self.recipe == "grid" else ans[sid]
                T["inp_gemma"][sid] = "<inpaint>\nTranscript: " + g
                if self.recipe == "tables":
                    first = next((words_of(t) for t in h if words_of(t)), "")
                    T["inp_lip"][sid] = f"Transcript: {first}"
                else:
                    T["inp_lip"][sid] = f"<inpaint>\nTranscript: {clean_txt(top)}"
                T["inp_av"][sid] = "<inpaint>"
            elif k == "denoise":
                T["den_gemma"][sid] = f"<denoise>\nTranscript: {ans[sid]}"
                if self.recipe == "grid":
                    T["den_lip"][sid] = f"<denoise>\nTranscript: {clean_txt(top)}"
                else:
                    line = re.sub(r"[^a-z0-9' ]+", " ", top.lower()).strip()
                    T["den_lip"][sid] = "<denoise>\nTranscript: " + line if line else "<denoise>"
                T["den_av"][sid] = "<denoise>"
            else:
                T["mute_gemma"][sid] = f"<inpaint>\nTranscript: {clean_txt(ans[sid])}"
                T["mute_lip"][sid] = f"<inpaint>\nTranscript: {clean_txt(top)}"
                T["mute_av"][sid] = "<inpaint>"
        self.jsave("texts.json", T)
        self.jsave("text_voters.json", {r["id"]: {"text|lip_text": (vsr.get(r["base_id"]) or [""])[0],
                                                  "text|gemma": ans[r["id"]]} for r in self.meta})

    def encode(self):
        from rml.generator.conditions import ConditionEncoder
        T = self.jload("texts.json")
        enc = ConditionEncoder(self.P.ltx_transformer, self.P.ltx_text_encoder)
        for name, texts in T.items():
            if texts:
                enc.encode_texts(texts, self.root / f"conditions_{name}")
        enc.encode_empty([r["id"] for r in self.meta], self.root / "conditions_empty")

    def render(self):
        from rml.generator.render import render
        T = self.jload("texts.json")
        for name, (_key, _k, lora, cfg) in CONFIGS.items():
            if self.recipe != "tables":
                lora = SINGLE_LORA.get(name, lora)
            ids = sorted(T[name])
            if ids:
                render(self.root, ids, getattr(self.P, lora), f"conditions_{name}", self.work / "renders" / name,
                       transformer_path=self.P.ltx_transformer, text_encoder_path=self.P.ltx_text_encoder,
                       audio_vae_path=self.P.ltx_audio_vae, text_cfg=cfg, steps=30, seed=42)

    def transcribe(self):
        from rml.eval.asr import transcribe_file
        res = self.jload("transcripts.json", {})
        for name in CONFIGS:
            d = self.work / "renders" / name
            for w in sorted(d.glob("*_avse.wav")) if d.exists() else []:
                k = f"{name}|{w.name[:-len('_avse.wav')]}"
                if k not in res:
                    res[k] = transcribe_file(w)
        self.jsave("transcripts.json", res)

    def hypotheses(self):
        """{task sample: {selection key: transcript}} (six configurations + the two text voters)"""
        tx, voters = self.jload("transcripts.json"), self.jload("text_voters.json")
        out = {}
        for r in self.evaluated():
            sid = r["id"]
            pre = "mute" if kind(r) == "muted" else "inp" if r["task"] == "inpaint" else "den"
            h = {}
            for arm, cfg in (("gemma", "gemma"), ("lip_text", "lip"), ("av_only", "av")):
                if f"{pre}_{cfg}|{sid}" in tx:
                    h[f"{arm}|keep"] = tx[f"{pre}_{cfg}|{sid}"]
                if f"mute_{cfg}|{twin_id(r['base_id'])}" in tx:
                    h[f"{arm}|mute"] = tx[f"mute_{cfg}|{twin_id(r['base_id'])}"]
            h.update(voters[sid])
            out[sid] = h
        return out

    def evidence(self, with_lm=True):
        from rml.selection.evidence import score_spec
        spec = {}
        for sid, h in self.hypotheses().items():
            r = self.by_id[sid]
            decoded = self.recipe != "grid" and kind(r) == "inpaint"
            wav = (self.work / "renders/inp_gemma" / f"{sid}_noisy.wav") if decoded else self.input_wav(sid)
            spec[sid] = {"wav": str(wav), "roi": str(self.roi(r["base_id"])), "hyp": h}
        self.jsave("evidence_spec.json", spec)
        score_spec(spec, self.P.evidence_models(), self.work / "output_scores.json", self.device, with_lm=with_lm)

    def select(self):
        from rml.selection.rule import NAME, select
        sc = self.jload("output_scores.json")
        cfg_of = {(v[0], v[1]): k for k, v in CONFIGS.items()}
        res = {}
        for sid, e in sc.items():
            r = self.by_id[sid]
            ch = select(e)
            arm, mode = ch.split("|")
            muted = mode == "mute" or kind(r) == "muted"
            name = cfg_of[(arm + "|mute" if muted else ch, "muted" if muted else r["task"])]
            src = twin_id(r["base_id"]) if muted else sid
            res[sid] = {"choice": ch, "name": NAME[ch], "transcript": e["hyp"][ch],
                        "wav": str(self.work / "renders" / name / f"{src}_avse.wav")}
        self.jsave("selection.json", res)
        return res

    def run(self, stages=STAGES, with_lm=True):
        for st in stages:
            print(f"=== stage {st}", flush=True)
            {"latents": self.latents, "roi": self.roi_stage, "reader": self.reader, "whisper": self.whisper,
             "lm": self.lm, "texts": self.texts, "encode": self.encode, "render": self.render,
             "transcribe": self.transcribe, "evidence": lambda: self.evidence(with_lm), "select": self.select}[st]()


def mux(video: Path, wav: Path, out: Path) -> None:
    """the restored track on the input window (video copied, audio AAC)"""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v:0",
                    "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)

