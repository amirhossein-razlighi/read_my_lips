# NOTICE: this file is NOT part of the upstream LTX-2 repository (Lightricks, commit 400fd31). It was added
# for "Read My Lips" and is distributed under the LTX-2.x Community License Agreement together with the rest
# of this vendored package (see packages/LICENSE and packages/NOTICE.md).
"""AVSE (audio-visual speech enhancement / restoration) training strategy.

The AUDIO stream is generated given the clean video and the corrupted audio
as conditioning:

  - Clean speech audio latent  -> loaded from disk (the flow-matching target)
  - Corrupted audio latent     -> the same window with the speech silenced over the
                                  target span (inpainting) or mixed with noise
                                  (denoising); encoded once, used as ref_audio
                                  (conditioning, sigma=0)
  - Clean video latent         -> conditioning, sigma=0 (no FM noise)
  - Text embeddings            -> conditioning (dropout for CFG)

Token layout (reference-style CONCATENATION, not inline masking):

  video: [ref_video (s=0) | target_video (s=0)]
  audio: [ref_audio (s=0) | target_audio (FM)]

Flow-matching objective on the audio stream only:
  noisy_target = (1 - sigma) * clean + sigma * gaussian
  target_velocity = gaussian - clean
  loss = MSE(audio_pred - target_velocity) over target tokens (per-sample [B,]),
         with tokens overlapping the silenced span up-weighted by gap_audio_loss_weight.

Notes on the trainer API:
  * ``Modality`` carries a per-batch ``sigma``; conditioning modalities get
    zeros, the generated audio gets the sampled sigma.
  * ``get_data_sources`` lives on the config.
  * ``compute_loss`` returns a per-element ``[B,]`` loss (the trainer reduces).
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


@dataclass
class AVSEModelInputs(ModelInputs):
    """ModelInputs extended with the AVSE-specific loss plumbing."""

    audio_loss_weights: Tensor | None = None
    ref_video_seq_len: int = 0
    ref_audio_seq_len: int = 0


class AVSEConfig(TrainingStrategyConfigBase):
    """Configuration for AVSE (audio-visual speech enhancement / restoration) training."""

    name: Literal["avse"] = "avse"  # type: ignore[assignment]

    with_audio: bool = Field(
        default=True,
        description="AVSE is inherently joint AV. Must be true.",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for clean target speech audio latents.",
    )

    noisy_audio_latents_dir: str = Field(
        default="noisy_audio_latents",
        description=(
            "Directory name for the corrupted (silenced span or noisy) audio "
            "latents. Used as the `ref_audio_latents` conditioning stream."
        ),
    )

    reference_latents_dir: str = Field(
        default="reference_latents",
        description=(
            "Directory of clean reference video latents. For AVSE this can be "
            "a copy of the target clean video latents — the trainer plumbing "
            "still expects a non-empty ref_video stream."
        ),
    )

    conditions_dir: str = Field(
        default="conditions",
        description="Directory name for text condition embeddings.",
    )

    text_dropout_p: float = Field(
        default=0.5,
        description=(
            "Probability of zeroing out the text embedding during training so "
            "the model handles the missing-text inference case via CFG."
        ),
        ge=0.0,
        le=1.0,
    )

    gap_audio_loss_weight: float = Field(
        default=1.0,
        description=(
            "Focal weight applied to audio target tokens overlapping the "
            "inpaint gap. 1.0 recovers the plain whole-clip FM loss."
        ),
        ge=1.0,
    )

    gap_audio_loss_margin_s: float = Field(
        default=0.04,
        description="Temporal margin around each gap when building the latent-frame loss mask.",
        ge=0.0,
    )

    def get_data_sources(self) -> dict[str, str]:
        return {
            "latents": "latents",
            self.conditions_dir: "conditions",
            self.audio_latents_dir: "audio_latents",
            self.noisy_audio_latents_dir: "ref_audio_latents",
            self.reference_latents_dir: "ref_latents",
        }


class AVSEStrategy(TrainingStrategy):
    """AVSE training strategy on the LTX-2 trainer stack."""

    config: AVSEConfig

    def __init__(self, config: AVSEConfig):
        super().__init__(config)
        self._last_loss_breakdown: dict[str, float] = {}

    @property
    def requires_audio(self) -> bool:
        return True

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> AVSEModelInputs:
        latents = batch["latents"]
        target_video_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        ref_latents_info = batch["ref_latents"]
        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()
        ref_video_latents = self._video_patchifier.patchify(ref_latents_info["latents"])

        target_video_latents = self._video_patchifier.patchify(target_video_latents)

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS in batch: {fps.tolist()}, using {fps[0].item()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_video_latents.shape[0]
        device = target_video_latents.device
        dtype = target_video_latents.dtype

        if self.config.text_dropout_p > 0.0:
            drop = torch.rand(batch_size, device=device) < self.config.text_dropout_p
            if drop.any():
                video_prompt_embeds = torch.where(
                    drop.view(-1, 1, 1), torch.zeros_like(video_prompt_embeds), video_prompt_embeds
                )
                audio_prompt_embeds = torch.where(
                    drop.view(-1, 1, 1), torch.zeros_like(audio_prompt_embeds), audio_prompt_embeds
                )

        ref_video_seq_len = ref_video_latents.shape[1]

        # Single sigma per sample for the generated audio stream.
        sigmas = timestep_sampler.sample_for(target_video_latents)
        sigmas_flat = sigmas.view(-1).to(dtype)

        target_video_seq_len = target_video_latents.shape[1]

        # ALL video tokens are conditioning (sigma=0). No FM noise on video.
        video_latents = torch.cat([ref_video_latents, target_video_latents], dim=1)
        total_video_seq = ref_video_seq_len + target_video_seq_len
        video_timesteps = torch.zeros(batch_size, total_video_seq, device=device, dtype=dtype)

        # Pure conditioning video: no targets/loss (also signals the trainer's
        # sigma-bucket tracking to use the audio sigma).
        video_targets = None
        video_loss_mask = None

        ref_positions = self._get_video_positions(
            num_frames=ref_frames, height=ref_height, width=ref_width,
            batch_size=batch_size, fps=fps, device=device,
        )
        target_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device,
        )
        video_positions = torch.cat([ref_positions, target_positions], dim=2)

        video_modality = Modality(
            enabled=True,
            latent=video_latents,
            sigma=torch.zeros(batch_size, device=device, dtype=dtype),
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        audio_modality, audio_targets, audio_loss_mask, ref_audio_seq_len = self._prepare_audio_inputs(
            batch=batch,
            sigmas=sigmas,
            sigmas_flat=sigmas_flat,
            audio_prompt_embeds=audio_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio_loss_weights = self._build_audio_gap_loss_weights(
            batch=batch,
            target_audio_seq_len=audio_modality.latent.shape[1] - ref_audio_seq_len,
            ref_audio_seq_len=ref_audio_seq_len,
            device=device,
            dtype=dtype,
        )

        return AVSEModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            audio_loss_weights=audio_loss_weights,
            ref_video_seq_len=ref_video_seq_len,
            ref_audio_seq_len=ref_audio_seq_len,
        )

    def _build_audio_gap_loss_weights(
        self,
        batch: dict[str, Any],
        target_audio_seq_len: int,
        ref_audio_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.config.gap_audio_loss_weight <= 1.0:
            return None
        required = {"has_gap", "gap_start_s", "gap_end_s"}
        if not required.issubset(batch.keys()):
            return None

        batch_size = batch["has_gap"].shape[0]
        has_gap = batch["has_gap"].to(device=device).float() > 0.5
        gap_start = (
            batch["gap_start_s"].to(device=device, dtype=torch.float32) - self.config.gap_audio_loss_margin_s
        )
        gap_end = batch["gap_end_s"].to(device=device, dtype=torch.float32) + self.config.gap_audio_loss_margin_s

        token_start = self._audio_patchifier._get_audio_latent_time_in_sec(
            0, target_audio_seq_len, torch.float32, device
        )
        token_end = self._audio_patchifier._get_audio_latent_time_in_sec(
            1, target_audio_seq_len + 1, torch.float32, device
        )
        overlaps = (token_start.unsqueeze(0) < gap_end.unsqueeze(1)) & (
            token_end.unsqueeze(0) > gap_start.unsqueeze(1)
        )
        gap_mask = overlaps & has_gap.unsqueeze(1)

        target_weights = torch.ones(batch_size, target_audio_seq_len, device=device, dtype=dtype)
        target_weights = torch.where(
            gap_mask,
            torch.full_like(target_weights, float(self.config.gap_audio_loss_weight)),
            target_weights,
        )

        prefix = torch.ones(batch_size, ref_audio_seq_len, device=device, dtype=dtype)
        return torch.cat([prefix, target_weights], dim=1)

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        sigmas_flat: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor, int]:
        """Audio is the denoising target; ref_audio carries the corrupted speech."""
        target_audio_data = batch["audio_latents"]
        target_audio_latents = target_audio_data["latents"]  # clean
        ref_audio_latents = batch["ref_audio_latents"]["latents"]  # corrupted

        target_audio_latents = self._audio_patchifier.patchify(target_audio_latents)
        ref_audio_latents = self._audio_patchifier.patchify(ref_audio_latents)

        target_audio_seq_len = target_audio_latents.shape[1]
        ref_audio_seq_len = ref_audio_latents.shape[1]

        sigmas_exp = sigmas.view(-1, 1, 1)
        audio_noise = torch.randn_like(target_audio_latents)
        noisy_target = (1.0 - sigmas_exp) * target_audio_latents + sigmas_exp * audio_noise
        audio_targets = audio_noise - target_audio_latents
        target_audio_timesteps = sigmas.view(-1, 1).expand(batch_size, target_audio_seq_len).to(dtype)

        audio_latents = torch.cat([ref_audio_latents, noisy_target], dim=1)

        ref_audio_cond_mask = torch.ones(batch_size, ref_audio_seq_len, dtype=torch.bool, device=device)
        target_audio_cond_mask = torch.zeros(batch_size, target_audio_seq_len, dtype=torch.bool, device=device)
        audio_conditioning_mask = torch.cat([ref_audio_cond_mask, target_audio_cond_mask], dim=1)

        ref_audio_timesteps = torch.zeros(batch_size, ref_audio_seq_len, device=device, dtype=dtype)
        audio_timesteps = torch.cat([ref_audio_timesteps, target_audio_timesteps], dim=1)

        ref_audio_positions = self._get_audio_positions(
            num_time_steps=ref_audio_seq_len, batch_size=batch_size, device=device
        )
        target_audio_positions = self._get_audio_positions(
            num_time_steps=target_audio_seq_len, batch_size=batch_size, device=device
        )
        audio_positions = torch.cat([ref_audio_positions, target_audio_positions], dim=2)

        audio_modality = Modality(
            enabled=True,
            latent=audio_latents,
            sigma=sigmas_flat,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        audio_loss_mask = ~audio_conditioning_mask

        return audio_modality, audio_targets, audio_loss_mask, ref_audio_seq_len

    def compute_loss(
        self,
        video_pred: Tensor | None,
        audio_pred: Tensor | None,
        inputs: AVSEModelInputs,
    ) -> Tensor:
        """Masked, gap-weighted velocity MSE. Returns per-element loss [B,]."""
        assert audio_pred is not None, "AVSE requires audio prediction"
        ref_audio_seq_len = inputs.ref_audio_seq_len
        target_audio_pred = audio_pred[:, ref_audio_seq_len:, :]
        target_audio_loss_mask = inputs.audio_loss_mask[:, ref_audio_seq_len:]

        audio_sq_err = (target_audio_pred - inputs.audio_targets).pow(2)
        mask = target_audio_loss_mask.unsqueeze(-1).float()
        if inputs.audio_loss_weights is not None:
            target_weights = (
                inputs.audio_loss_weights[:, ref_audio_seq_len:].unsqueeze(-1).to(audio_sq_err.dtype)
            )
            mask = mask * target_weights
        # Per-sample weighted mean over (tokens, channels) -> [B,]
        audio_loss = (audio_sq_err * mask).mean(dim=[-2, -1]) / mask.mean(dim=[-2, -1]).clamp(min=1e-8)

        self._last_loss_breakdown = {
            "audio_loss": float(audio_loss.mean().item()),
            "gap_audio_loss_weight": float(self.config.gap_audio_loss_weight),
        }
        return audio_loss
