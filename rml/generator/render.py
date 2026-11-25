"""G: LTX-2.5 22B + an AVSE LoRA, rendered through the official ltx-pipelines building blocks.

Token layout per sample (all built by official conditioning items):
  video state:  [target_video (sigma 0, frozen) | ref_video (sigma 0, appended)]
  audio state:  [target_audio (flow-matching noise, denoised) | ref_audio (the input track, sigma 0)]
Training used [ref | target] order; the official items append references after the target. With identical RoPE
positions (references get the same coordinates as the target) and full attention, the two orders are equivalent.
Video Modality.sigma is pinned to 0 (it feeds prompt_adaln and the video-to-audio attention gate), as in training.
Text CFG: the null context is conditions_empty/ (Gemma-4-encoded ""); x0 blending cond + (w-1)(cond - uncond).
The LoRA is fused into the bf16 base and the transformer quantised with int8-quanto (the training precision).
Per-sample noise: a stable hash of (seed, sample id), so outputs do not depend on the order or the subset.

A root holds metadata.json ({"clips": [{"id", ...}]}), latents/, reference_latents/, audio_latents/,
noisy_audio_latents/ (the input track), <conditions_dir>/ and conditions_empty/ (data/latents.py,
generator/conditions.py). Output: <out>/<id>_{avse,noisy,clean}.wav (generated, decoded input, decoded reference).
  python -m rml.generator.render --lora L.safetensors --root R --conditions-dir C --text-cfg 5 --out O [--ids-file F]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from safetensors import safe_open

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import AudioConditionByReferenceLatent
from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.blocks import DiffusionStage, _build_state
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import DenoisedLatentResult, ModalitySpec
from ltx_trainer.model_loader import load_audio_vae_decoder, load_embeddings_processor, load_vocoder
from ltx_trainer.quantization import quantize_model

VIDEO_TIME_SF, VIDEO_H_SF, VIDEO_W_SF = 8, 32, 32  # ltx-core default scale factors


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{sample_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def check_lora_checkpoint(lora_path: Path) -> int:
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    n_lora_a = sum(1 for k in keys if k.endswith(".lora_A.weight"))
    if n_lora_a == 0:
        raise RuntimeError(f"{lora_path} has no '.lora_A.weight' tensors: not a LoRA checkpoint in ComfyUI format")
    return n_lora_a


def load_condition_context(embeddings_processor, cond_path: Path, device):
    """pre-connector video/audio features + binary mask -> additive mask -> connectors -> (v_ctx, a_ctx)"""
    if not cond_path.exists():
        raise RuntimeError(f"Missing condition file {cond_path}: encode the texts first (rml.generator.conditions)")
    connector_dtype = next(embeddings_processor.parameters()).dtype
    cond = torch.load(cond_path, weights_only=False)
    video_features = cond["video_prompt_embeds"].to(device, dtype=connector_dtype).unsqueeze(0)
    audio_features = cond.get("audio_prompt_embeds")
    if audio_features is not None:
        audio_features = audio_features.to(device, dtype=connector_dtype).unsqueeze(0)
    attn_mask = cond["prompt_attention_mask"].clone().to(device).unsqueeze(0)
    additive = convert_to_additive_mask(attn_mask, video_features.dtype)
    with torch.inference_mode():
        v_ctx, a_ctx, _binary_mask = embeddings_processor.create_embeddings(video_features, audio_features, additive)
    return v_ctx.detach(), a_ctx.detach()


def audio_ref_conditioning(noisy_audio_latent: torch.Tensor, device) -> AudioConditionByReferenceLatent:
    """the input track with the SAME (overlapping) RoPE positions as the target audio"""
    patchifier = AudioPatchifier(patch_size=1)
    patchified = patchifier.patchify(noisy_audio_latent)
    b, c, t, mel = noisy_audio_latent.shape
    positions = patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=b, channels=c, frames=t, mel_bins=mel), device=device)
    return AudioConditionByReferenceLatent(patchified=patchified, positions=positions, strength=1.0)


def build_modality_specs(target_video_latent, ref_video_latent, noisy_audio_latent, v_ctx, a_ctx, device):
    video_spec = ModalitySpec(
        context=v_ctx,
        conditionings=[VideoConditionByReferenceLatent(latent=ref_video_latent, downscale_factor=1,
                                                       temporal_scale_factor=1, strength=1.0)],
        noise_scale=0.0, frozen=True, initial_latent=target_video_latent)   # clean, frozen video
    audio_spec = ModalitySpec(
        context=a_ctx, conditionings=[audio_ref_conditioning(noisy_audio_latent, device)],
        noise_scale=1.0, frozen=False, initial_latent=None)                 # target audio from pure noise
    return video_spec, audio_spec


class AVSEDenoiser:
    """single conditional pass, video Modality.sigma = 0"""

    def __init__(self, v_context, a_context):
        self.v_context, self.a_context = v_context, a_context

    @staticmethod
    def _forward(transformer, video_state: LatentState, audio_state: LatentState, sigma, v_ctx, a_ctx):
        video = modality_from_latent_state(video_state, v_ctx, torch.zeros_like(sigma))
        audio = modality_from_latent_state(audio_state, a_ctx, sigma)
        return transformer(video=video, audio=audio, perturbations=None)

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        dv, da = self._forward(transformer, video_state, audio_state, sigmas[step_index], self.v_context, self.a_context)
        return DenoisedLatentResult.result_or_none(denoised=dv), DenoisedLatentResult.result_or_none(denoised=da)


class AVSETextCFGDenoiser(AVSEDenoiser):
    """text CFG on the audio: cond + null-text pass blended by MultiModalGuider (video stays positive-only); the two
    passes run sequentially to keep the pinned video sigma"""

    def __init__(self, v_context, a_context, neg_v_context, neg_a_context, text_cfg_scale: float):
        super().__init__(v_context, a_context)
        self.neg_v_context, self.neg_a_context = neg_v_context, neg_a_context
        self.audio_guider = MultiModalGuider(
            params=MultiModalGuiderParams(cfg_scale=text_cfg_scale, stg_scale=0.0, modality_scale=1.0),
            negative_context=neg_a_context)

    def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
        sigma = sigmas[step_index]
        dv_c, da_c = self._forward(transformer, video_state, audio_state, sigma, self.v_context, self.a_context)
        _dv_u, da_u = self._forward(transformer, video_state, audio_state, sigma, self.neg_v_context, self.neg_a_context)
        da = self.audio_guider.calculate(da_c, da_u, 0.0, 0.0)
        return (DenoisedLatentResult.result_or_none(denoised=dv_c),
                DenoisedLatentResult.result_or_none(denoised=da, uncond=da_u, cond=da_c))


def stage_run(stage, transformer, denoiser, sigmas, noiser, width, height, frames, fps, video, audio, stepper, device):
    """DiffusionStage's loop body (states -> Euler loop -> clear conditioning -> unpatchify) on a held transformer"""
    pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
    v_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=stage.video_scale_factors)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps,
                                   scale_factors=stage.video_scale_factors)
    video_state = _build_state(video, video_tools, noiser, torch.bfloat16, device)
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    audio_state = _build_state(audio, audio_tools, noiser, torch.bfloat16, device)
    wrapped = BatchSplitAdapter(transformer, max_batch_size=1)
    video_state, audio_state = euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state,
                                                    stepper=stepper, transformer=wrapped, denoiser=denoiser)
    video_state = video_tools.unpatchify(video_tools.clear_conditioning(video_state))
    audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state))
    return video_state, audio_state


def derive_pixel_shape(video_meta: dict) -> VideoPixelShape:
    lat_f, lat_h, lat_w = int(video_meta["num_frames"]), int(video_meta["height"]), int(video_meta["width"])
    return VideoPixelShape(batch=1, frames=(lat_f - 1) * VIDEO_TIME_SF + 1, height=lat_h * VIDEO_H_SF,
                           width=lat_w * VIDEO_W_SF, fps=float(video_meta.get("fps", 25.0)))


def decode_to_wav(audio_vae, vocoder, latent: torch.Tensor, device) -> torch.Tensor:
    with torch.inference_mode():
        mel = audio_vae(latent.to(device, dtype=torch.bfloat16))
        return vocoder(mel)[0].cpu().float()


def render(root: Path, ids, lora: Path, conditions_dir: str, out: Path, *, transformer_path: Path,
           text_encoder_path: Path, audio_vae_path: Path, text_cfg: float, steps: int = 30, seed: int = 42,
           quantization: str = "int8-quanto", skip_existing: bool = True) -> None:
    """render every id of the root with one LoRA, one conditions dir and one text-CFG scale"""
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    check_lora_checkpoint(lora)
    meta = {c["id"]: c for c in json.loads((root / "metadata.json").read_text())["clips"]}
    chosen = [i for i in ids if not (skip_existing and (out / f"{i}_avse.wav").exists())]
    missing = [i for i in chosen if i not in meta]
    assert not missing, f"{len(missing)} ids not in {root}/metadata.json"
    if not chosen:
        return
    embeddings_processor = load_embeddings_processor((transformer_path, text_encoder_path), text_encoder_path,
                                                     device="cuda", dtype=torch.bfloat16)
    ctx = {i: load_condition_context(embeddings_processor, root / conditions_dir / f"{i}.pt", device) for i in chosen}
    neg = None
    if text_cfg != 1.0:
        empty = sorted((root / "conditions_empty").glob("*.pt"))
        if not empty:
            raise RuntimeError(f"text CFG {text_cfg} needs the null-text context {root}/conditions_empty")
        neg = load_condition_context(embeddings_processor, empty[0], device)
    del embeddings_processor
    gc.collect()
    torch.cuda.empty_cache()
    stage = DiffusionStage.from_checkpoint(
        checkpoint_path=str(transformer_path), dtype=torch.bfloat16, device=device,
        loras=(LoraPathStrengthAndSDOps(str(lora), 1.0, LTXV_LORA_COMFY_RENAMING_MAP),), quantization=None)
    scheduler, stepper = LTX2Scheduler(), EulerDiffusionStep()
    with stage._transformer_ctx() as transformer:
        if quantization == "int8-quanto":
            quantize_model(transformer.velocity_model, "int8-quanto")
            gc.collect()
            torch.cuda.empty_cache()
        audio_vae = load_audio_vae_decoder(audio_vae_path, device="cuda", dtype=torch.bfloat16)
        vocoder = load_vocoder(audio_vae_path, device="cuda")
        sr = vocoder.output_sampling_rate
        for sid in chosen:
            v_data = torch.load(root / "latents" / f"{sid}.pt", weights_only=False)
            ref_p = root / "reference_latents" / f"{sid}.pt"
            ref_v_data = torch.load(ref_p, weights_only=False) if ref_p.exists() else v_data
            a_clean = torch.load(root / "audio_latents" / f"{sid}.pt", weights_only=False)["latents"].unsqueeze(0)
            a_noisy = torch.load(root / "noisy_audio_latents" / f"{sid}.pt", weights_only=False)["latents"].unsqueeze(0)
            v_lat = v_data["latents"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            ref_v_lat = ref_v_data["latents"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            a_noisy_dev = a_noisy.to(device, dtype=torch.bfloat16)
            pixel_shape = derive_pixel_shape(v_data)
            expected = AudioLatentShape.from_video_pixel_shape(pixel_shape).frames
            if expected != a_noisy.shape[2]:
                raise RuntimeError(f"[{sid}] audio latent has T={a_noisy.shape[2]}, the video shape implies {expected}")
            v_ctx, a_ctx = ctx[sid]
            video_spec, audio_spec = build_modality_specs(v_lat, ref_v_lat, a_noisy_dev, v_ctx, a_ctx, device)
            denoiser = AVSETextCFGDenoiser(v_ctx, a_ctx, *neg, text_cfg) if text_cfg != 1.0 else AVSEDenoiser(v_ctx, a_ctx)
            noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(stable_sample_seed(seed, sid)))
            sigmas = scheduler.execute(steps=steps).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _v, audio_state = stage_run(stage, transformer, denoiser=denoiser, sigmas=sigmas, noiser=noiser,
                                            width=pixel_shape.width, height=pixel_shape.height,
                                            frames=pixel_shape.frames, fps=pixel_shape.fps, video=video_spec,
                                            audio=audio_spec, stepper=stepper, device=device)
            denoised = audio_state.latent.float()
            for tag, lat in (("clean", a_clean), ("noisy", a_noisy), ("avse", denoised)):
                arr = decode_to_wav(audio_vae, vocoder, lat, device).squeeze().cpu().numpy()
                if arr.ndim == 2:
                    arr = arr[0]
                arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                scipy.io.wavfile.write(str(out / f"{sid}_{tag}.wav"), int(sr), arr)
            print(f"[render] {sid} -> {out}", flush=True)
            del denoised, audio_state, _v, v_lat, ref_v_lat, a_noisy_dev
            torch.cuda.empty_cache()


def main():
    from rml.paths import load_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--conditions-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids-file", type=Path, default=None, help="json list of ids (default: every metadata row)")
    ap.add_argument("--text-cfg", type=float, required=True, help="5 with <inpaint>, 1 with <denoise>")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    P = load_paths(a.paths)
    P.export_env()
    ids = json.loads(a.ids_file.read_text()) if a.ids_file else \
        sorted(c["id"] for c in json.loads((a.root / "metadata.json").read_text())["clips"])
    render(a.root, ids, a.lora, a.conditions_dir, a.out, transformer_path=P.ltx_transformer,
           text_encoder_path=P.ltx_text_encoder, audio_vae_path=P.ltx_audio_vae, text_cfg=a.text_cfg,
           steps=a.steps, seed=a.seed)


if __name__ == "__main__":
    main()
