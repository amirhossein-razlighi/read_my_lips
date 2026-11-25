# Vendored LTX-2 packages — notice of modifications

`ltx-core`, `ltx-pipelines` and `ltx-trainer` in this directory are a copy of the official
[LTX-2 repository](https://github.com/Lightricks/LTX-2) by Lightricks Ltd. at commit
`400fd31054597515f47125691032c04b1c3ee24e` (2026-08-16). They are distributed under the
**LTX-2.x Community License Agreement** (`LICENSE` / `LICENSE.md` in this directory and in each package),
not under the MIT licence of the rest of this repository. Your use of these packages, and of any model
weights derived from LTX-2 (including the Read My Lips generator LoRAs), is subject to that agreement,
including Section 4 and Attachment A (use-based restrictions).

`ltx-kernels` (optional compiled CUDA kernels) is not vendored: nothing in our training or inference path
imports it (the only `ltx_kernels` imports in `ltx-core` are lazy and belong to the DiffVAE video decoder).

## Files changed relative to upstream 400fd31 (Section 3.3 of the licence)

Added:

- `ltx-trainer/src/ltx_trainer/training_strategies/avse.py` — the AVSE (audio-visual speech
  restoration) training strategy used to train the Read My Lips generator: clean video + corrupted
  audio as sigma=0 conditioning, flow-matching loss on the generated audio only, focal up-weighting
  of the silenced span.
- `LICENSE`, `LICENSE.md` copied into each package directory; this `NOTICE.md`.

Modified (each file carries a `NOTICE` header comment):

- `ltx-trainer/src/ltx_trainer/training_strategies/__init__.py` — registers `AVSEConfig` / `AVSEStrategy`.
- `ltx-trainer/src/ltx_trainer/config.py` — `avse` in the strategy discriminated union;
  `optimization.learning_rate_groups` (regex -> learning rate); a guard that requires
  `validation.interval: null` for the `avse` strategy (the generic validation sampler does not implement
  the AVSE token layout).
- `ltx-trainer/src/ltx_trainer/datasets.py` — `PrecomputedDataset` attaches `has_gap`,
  `gap_start_s`, `gap_end_s` per sample from `<data root>/metadata.json` (`clips[].mask_segments_s`).
- `ltx-trainer/src/ltx_trainer/trainer.py` — per-step strategy loss breakdown in the logs; TensorBoard
  scalars under `<output_dir>/tensorboard`; fp32 precomputed text features cast to the connector dtype; a
  hard error when legacy single-stream conditions meet a two-stream checkpoint; a hard error when the
  conditions directory's `_encoder.json` stamp does not match the model generation; sigma tracking uses
  the audio sigma when the video stream is pure conditioning; per-pattern learning-rate groups.
- `ltx-core/pyproject.toml`, `ltx-trainer/pyproject.toml` — removed the `natten` extra and the
  CUDA 13.2 index routing (and the trainer's dev dependency group); the top-level project pins torch.
- `ltx-trainer/README.md`, `ltx-trainer/docs/{quick-start,training-guide,training-modes}.md` — the
  links to the upstream agent skill directory (not shipped here) were replaced by plain text.

Removed: the upstream coding-agent instruction files (`ltx-trainer/AGENTS.md` and two assistant-specific
instruction markdown files in `ltx-trainer/` and `ltx-pipelines/`).

Everything else is byte-identical to upstream 400fd31.
