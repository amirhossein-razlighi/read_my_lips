"""R: Auto-AVSR on the mouth ROI of the whole window, beam search 40, the 10 best distinct transcripts.

Two modes:
  vsr   the video-only checkpoint (vsr_trlrwlrs2lrs3vox2avsp_base.pth): the lip reader R
  avsr  the audio-visual checkpoint (avsr_trlrwlrs2lrs3vox2avsp_base.pth) on the window's input audio: its n-best
        joins the candidate list of L for noisy inputs (Sec. 4.2)
Rows: [{"id", "roi": <window ROI .avi>, "wav": <16 kHz input wav, avsr only>}] -> [{"id", "nbest": [...]}].
`degrade` weakens the reader for the training pools of L only (downN: resolution 96/N and back; dropP: P% of the
frames replaced by the previous one, fixed seed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def degrade_frames(frames: list, spec: str) -> list:
    if not spec:
        return frames
    if spec.startswith("down"):
        f = int(spec[4:])
        return [cv2.resize(cv2.resize(x, (96 // f, 96 // f), interpolation=cv2.INTER_AREA),
                           (96, 96), interpolation=cv2.INTER_LINEAR) for x in frames]
    if spec.startswith("drop"):
        p = int(spec[4:]) / 100.0
        rng = np.random.default_rng(0)
        out = [frames[0]]
        for x in frames[1:]:
            out.append(out[-1] if rng.random() < p else x)
        return out
    raise ValueError(spec)


def read_roi(avi: Path, degrade: str = "") -> torch.Tensor | None:
    """the whole ROI video as [T, 3, 96, 96] uint8"""
    cap = cv2.VideoCapture(str(avi))
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        return None
    frames = degrade_frames(frames, degrade)
    v = torch.from_numpy(np.stack(frames))
    return v[:, None].repeat(1, 3, 1, 1)


class AutoAVSR:
    def __init__(self, auto_avsr_dir, ckpt, mode="vsr", beam_size=40, device="cuda"):
        aa = Path(auto_avsr_dir)
        sys.path.insert(0, str(aa))
        from datamodule.transforms import VideoTransform
        from omegaconf import OmegaConf
        self.video_transform = VideoTransform(subset="test")
        common = {"optimizer": OmegaConf.load(aa / "configs/optimizer/adamw.yaml"),
                  "trainer": OmegaConf.load(aa / "configs/trainer/train.yaml"),
                  "transfer_frontend": False, "transfer_encoder": False}
        if mode == "vsr":
            from lightning import ModelModule, get_beam_search_decoder
            cfg = OmegaConf.create({
                "data": {"modality": "video"},
                "model": {"visual_backbone": OmegaConf.load(aa / "configs/model/visual_backbone/resnet_conformer.yaml")},
                "pretrained_model_path": str(ckpt), **common})
        else:
            from lightning_av import ModelModule, get_beam_search_decoder
            cfg = OmegaConf.create({
                "data": {"modality": "audiovisual"},
                "model": {"audiovisual_backbone": OmegaConf.load(
                    aa / "configs/model/audiovisual_backbone/resnet_conformer.yaml")},
                "pretrained_model_path": str(ckpt), **common})
        self.mode, self.device = mode, device
        self.mm = ModelModule(cfg).to(device).eval()
        self.beam = get_beam_search_decoder(self.mm.model, self.mm.token_list, beam_size=beam_size)

    @torch.inference_mode()
    def nbest(self, roi_avi, wav=None, nbest=10, degrade=""):
        """distinct lower-case transcripts, best first; None when the ROI is missing or too short"""
        v = read_roi(Path(roi_avi), degrade) if Path(roi_avi).exists() else None
        if v is None or v.shape[0] < 2:
            return None
        x = self.video_transform(v).to(self.device)        # [T,1,88,88]
        feat, _ = self.mm.model.encoder(x.unsqueeze(0), None)
        if self.mode == "avsr":
            import soundfile as sf
            y, _sr = sf.read(str(wav), dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            need = x.shape[0] * 640
            y = np.pad(y, (0, max(0, need - len(y))))[:need]
            aud = torch.from_numpy(y)[:, None].to(self.device)
            afeat, _ = self.mm.model.aux_encoder(aud.unsqueeze(0), None)
            feat = self.mm.model.fusion(torch.cat((feat, afeat), dim=-1))
        hyps = self.beam(feat.squeeze(0))
        texts, seen = [], set()
        for h in hyps[: nbest * 2]:
            ids = torch.tensor(list(map(int, h.asdict()["yseq"][1:])))
            t = self.mm.text_transform.post_process(ids).replace("<eos>", "").strip().lower()
            if t not in seen:
                seen.add(t)
                texts.append(t)
            if len(texts) >= nbest:
                break
        return texts


def read_rows(rows, auto_avsr_dir, ckpt, mode="vsr", nbest=10, beam_size=40, degrade="", device="cuda"):
    """n-best for every row; rows whose ROI is missing are skipped"""
    r = AutoAVSR(auto_avsr_dir, ckpt, mode, beam_size, device)
    out = []
    for row in rows:
        t = r.nbest(row["roi"], row.get("wav"), nbest, degrade)
        if t is not None:
            out.append({"id": row["id"], "nbest": t})
    return out
