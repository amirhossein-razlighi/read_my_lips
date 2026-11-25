# Training Read My Lips

This page gives the recipe of the two trained components:

- **L**, the language model: Gemma-4 12B with a LoRA adapter, trained in 3 SFT stages.
- **G**, the generator: LTX-2.5 22B with an audio LoRA, trained in 3 stages.

The lip reader R (Auto-AVSR), Whisper and the selection rule are used as released and are not trained here.

Every hyperparameter below is a value of the released runs.

Contents: [Requirements](#requirements) · [Paths](#paths) · [Data](#data) · [L: language model](#l-the-language-model-3-stages) · [G: generator](#g-the-generator-3-stages) · [Compute](#compute) · [Reproducibility](#reproducibility)

## Requirements

- **GPU:** one 80 GB GPU per job (an H100).
- **Software:**
  - the repository environment, which includes the vendored `packages/ltx-*`;
  - `torch` 2.9.1 (CUDA 12.8 wheels);
  - `peft`, `transformers` 5.x (`<5.15`), `tensorboard`;
  - `nltk` with the `cmudict` corpus, and `g2p_en`;
  - `soundfile`, `librosa` and `ffmpeg` for the data builders;
  - `yt-dlp` to fetch the noise clips.
- **Models:**
  - the LTX-2.5 split pack from Lightricks: `diffusion_models/`, `text_encoders/` and `vae/` under one directory;
  - Gemma-4 12B `it-qat-q4_0-unquantized`, as a local directory.
- **Licences:** `packages/` and every LoRA trained from LTX-2.5 fall under the LTX-2.x Community License (`packages/LICENSE`). TalkVid, DEMAND and VGGSound have their own terms.

## Paths

Configs contain `${RML_*}` placeholders. The launchers fill them from the environment and refuse to start if one is unset.

| Variable | Meaning |
|---|---|
| `RML_LTX_DIR` | LTX-2.5 split-pack directory |
| `RML_OUT_DIR` | where every run writes `<RML_OUT_DIR>/<run name>/` |
| `RML_G_STAGE1_ROOT` | G stage-1 data root (`talkvid_mt_span/train`) |
| `RML_G_MULTITASK_DIR` | parent of the multi-task root. `RML_G_MULTITASK_ROOT` = `$RML_G_MULTITASK_DIR/train` |
| `RML_NOISE_BANK`, `RML_LTX_AUDIO_VAE` | noise bank directory; LTX-2.5 audio VAE file |
| `RML_LM_STAGE12_DATA`, `RML_LM_STAGE3_DATA` | L rows for stages 1–2 and for stage 3 |
| `RML_LM_READERS` | the reader n-best files of the L training pool (names below) |
| `RML_AVH_FEATS` | AV-HuBERT feature directory (see [the feature-length dependency](#the-av-hubert-feature-length-dependency-of-l)) |
| `GEMMA_ROOT` | the Gemma-4 12B directory |

## Data

Everything comes from TalkVid (English, Spanish and French) in 4.8 s windows of 121 frames at 25 fps.

- A **cut** is a window with one silenced span of k = 1..10 words.
- Cut ids read `<youtube id>_NA_<clip start>_<clip end>_t<offset>__k<k>`.
- Test and validation videos are excluded by YouTube id. The list itself is not part of this release.

### What this release ships and what you must build

The raw data cannot be redistributed: the videos, latents and features are not included. What is shipped:

| Item | Status |
|---|---|
| G stage-1 cut list with silenced spans and the **exact condition text** of each cut | shipped: `rml/training/manifests/generator_stage1_cuts.json.gz` (15,816 cuts) |
| G multi-task samples: inpaint/denoise, source cut, noise type, SNR, second source, and the candidate texts | shipped: `rml/training/manifests/generator_multitask_samples.json.gz` (13,018 samples) |
| Noise-bank sources: DEMAND channels, VGGSound clip ids per split | shipped: `rml/training/manifests/noise_bank_sources.json` |
| Scripts that run the scripted steps | `rml/training/*.py` (table below) |
| L training rows (6,792 English windows, 66,186 cuts) | built with `build_vis2gemma_data.py`, then filtered to English, deduplicated and cleared of test videos |
| G stage-1 root: window mining, merge of two mining generations, purge of bad windows, dedup, test-video removal | the cut manifest above lists exactly the training cuts: build the latents of those cuts |

### Data steps

**G stage 1: `RML_G_STAGE1_ROOT`**

1. **Latents.** For every cut in `generator_stage1_cuts.json.gz`, encode latents with the repository's latent encoder (`python -m rml.data.latents --root <root>`, rml/data/latents.py):
   - `latents/` (video);
   - `reference_latents/`, a copy of `latents/`;
   - `audio_latents/` (clean);
   - `noisy_audio_latents/`, the audio with the span silenced.

   Also write a `metadata.json` with one `clips[]` entry per cut carrying `id`, `language`, `masked_words`, `mask_segments_s`, `window_words` and `sentence`.

   The trainer up-weights the loss on `mask_segments_s`. **The trainer trains on exactly the ids that have a file in every source directory**, so the condition directory decides the training set.

   Video latents come from the LTX-2.5 video VAE encoder.
2. **Condition text.** Each cut's `condition_text` (`Transcript: <Auto-AVSR VSR 1-best of the whole window>`) goes to `conditions_avsr_vsrfw_k10/`, encoded with `python -m rml.generator.conditions` (rml/generator/conditions.py) (Gemma-4, which writes the `_encoder.json` stamp the trainer checks).
   - **Spanish and French cuts in stage 1 carry the English Auto-AVSR transcript.**
   - `build_k10_train_subset.py` draws these cuts; their ids are in `rml/training/manifests/generator_stage1_cuts.json.gz`.

**G noise bank: `RML_NOISE_BANK`**

3. Fetch the sound-effect clips: `fetch_vggsound_sfx.py --csv vggsound.csv --out <sfx>`. YouTube availability changes, so check the result against `noise_bank_sources.json`.
4. `build_noise_bank.py --out $RML_NOISE_BANK --pcafeter <DEMAND>/PCAFETER --sfx-manifest <sfx>/manifest.json` builds:
   - babble: PCAFETER channels 1–12 for train, 13–16 for test;
   - 9 seen VGGSound classes (train and test clips);
   - 3 held-out classes (church bell, sneezing, train horn), test only;
   - white, pink and brown noise.

**G stages 2–3: `RML_G_MULTITASK_ROOT`**

5. `build_multitask_restoration.py train --src-root $RML_G_STAGE1_ROOT --out-root $RML_G_MULTITASK_DIR --noise-bank $RML_NOISE_BANK --audio-vae $RML_LTX_AUDIO_VAE`. Per window of the stage-1 set this writes:
   - one inpaint sample: one cut, with k balanced 1..10;
   - one denoise sample: noise type uniform over {babble, seen sfx, colored, competing speaker}, SNR U[−15, 5] dB, and a second source 5 dB weaker for 20% of samples. The noisy latents are encoded here.

   This gives 13,018 samples: en 4,901 / es 1,523 / fr 85 per task.
6. `python -m rml.generator.conditions --root $RML_G_MULTITASK_ROOT --json conditions_mt=$RML_G_MULTITASK_ROOT/_texts_inpaint.json`, then `build_multitask_restoration.py link-denoise`. Denoise samples share the single `<denoise>` template.
7. **Spanish and French inpaint texts** come from the native Spanish/French lip readers (Ma et al.): `build_multitask_restoration.py native-texts --native-readings '<glob>'` writes `_texts_native_esfr.json`.
   - Encode them into `conditions_mt` with `--json conditions_mt=_texts_native_esfr.json --overwrite`.
   - Both candidate texts are listed per sample in the manifest.
8. Stage 3: `build_multitask_restoration.py text-conds`, then `python -m rml.generator.conditions --root $RML_G_MULTITASK_ROOT --json conditions_mt_text=$RML_G_MULTITASK_ROOT/_texts_conditions_mt_text.json`.
   - Inpaint samples are linked unchanged.
   - Denoise samples are bucketed by `crc32(id) % 4`: two buckets get the reader line, one the reference sentence, and one no text.

**L stages 1–2: `RML_LM_STAGE12_DATA`**

9. Reader n-best files of the training windows go in `RML_LM_READERS`. The pool:

   | Reader | Share of rows | File |
   |---|---|---|
   | Auto-AVSR VSR on the original video, beam 40, 10-best | 50% | `autoavsr_vsr_clean_train_shard*.json` |
   | the same reader on mouth crops downscaled 96→24 px and back | 15% | `autoavsr_vsr_down4_train_shard*.json` |
   | the same reader with half of the frames replaced by the previous frame | 15% | `autoavsr_vsr_drop50_train_shard*.json` |
   | AV-HuBERT large (video only), beam 20, 10-best | 20% | `avhubert_vsr_train.json` |

   Each file has the format `{rows: [{id, nbest}]}`, keyed by window or cut id.
10. Whisper context on the **muted** audio, with `___` at the detected silence: `python -m rml.data.context --root <root>` (rml/data/context.py, the inference code). This gives `conditions_span/_provenance.json` in the stage-1 root.
11. `build_vis2gemma_data.py --root $RML_G_STAGE1_ROOT --feats $RML_AVH_FEATS --out $RML_LM_STAGE12_DATA --split train --language en --val-videos 8` writes `train.jsonl` / `val.jsonl`.

**L stage 3: `RML_LM_STAGE3_DATA`**

12. For the English denoise samples of step 5:
    - their 16 kHz noisy waves: `build_multitask_restoration.py train-wavs --verify 3`;
    - Whisper large-v3 on them: `whisper_noisy.json` (`{id: text}`);
    - Auto-AVSR audio-visual 10-best: `autoavsr_av_noisy.json`.
13. `build_gemma_denoise_rows.py train --out $RML_LM_STAGE3_DATA --inpaint-rows $RML_LM_STAGE12_DATA --train-meta $RML_G_STAGE1_ROOT/metadata.json --denoise-meta $RML_G_MULTITASK_ROOT/metadata.json --recognizers <dir> --feats $RML_AVH_FEATS --reader clean='…' --reader down4='…' --reader drop50='…' --reader avhubert='…'`. It writes:
    - the stage-1/2 rows plus 4,901 whole-window denoise rows;
    - `keep_meta.json`;
    - `readers_denoise/<reader>.json`, in which each reader's readings are interleaved with the AV n-best and the Whisper line.

### The AV-HuBERT feature-length dependency of L

L is trained with `--evidence none`, so no lip features reach Gemma. **The trainer loads each row's AV-HuBERT feature file** (`row["feats"]`, an `.npz` with `feats [T, 1024]` at 25 fps over the window) and slices `[f0:f1]`. Only the slice's **length** matters:

- the time-mask augmentation (`time-mask-p 0.5`) draws `random.randrange` only when the slice is longer than 6 frames;
- the feature-noise draw consumes numpy random numbers in proportion to the slice size.

The slice length therefore shifts the random stream that also drives context word-drop and the cut chosen per window. To reproduce the data order, a row's file must exist and have at least `f1` frames: `f1 ≤ 120` for silenced rows, `f1 = 120` for denoise rows.

The features are AV-HuBERT features of the `python -m rml.reader.roi` mouth crops.

## L: the language model (3 stages)

```
GEMMA_ROOT=… RML_OUT_DIR=… RML_LM_STAGE12_DATA=… RML_LM_STAGE3_DATA=… RML_LM_READERS=… RML_G_STAGE1_ROOT=… \
python scripts/train/train_lm.py configs/training/lm_stage1_select_or_write.yaml
python scripts/train/train_lm.py configs/training/lm_stage2_select_then_edit.yaml
python scripts/train/train_lm.py configs/training/lm_stage3_multitask_denoise.yaml   # L = ckpt_s01000.pt
```

`--dry-run` prints the command line. Re-running a stage resumes from `<out>/last.pt`.

**Model:** Gemma-4 12B with LoRA r 32, α 64, dropout 0.1 on q/k/v/o of every attention layer.

**Shared recipe:**
- batch 4 × 4 accumulation steps = 16; cosine LR to 10% over `max-steps`;
- `--deterministic`: eager attention, deterministic kernels and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; seed 0;
- one cut per window per epoch;
- 1..10 lettered hypotheses per row, drawn from the 4-reader pool;
- context word-drop 0.15; context full-drop 0.5 → 0.3 (step 4000) → 0.15 (step 7000);
- `Gap duration` line in the prompt; chat template.

Labels (`Pick:` and the closest reading) are computed on the fly against the reference with `rml/training/label_gap_score.py`.

| Stage | Run | Init | Target | LR (LoRA) | Steps | Data |
|---|---|---|---|---|---|---|
| 1 | `gemma_select_or_write_novis` | base Gemma | `Pick: <letter\|none>` + missing words | 1e-4 | 3000; horizon 20000 | stage-1/2 rows |
| 2 | `gemma_select_then_edit_novis` | stage 1 `ckpt_s03000` | `Pick: <closest>` + `Corrected: <window transcript>` | 5e-5 | 3000; horizon 4000 | stage-1/2 rows |
| 3 | `gemma_multitask_denoise_novis` | stage 2 `ckpt_s03000` | same, plus 4,901 denoise rows | 5e-5 | 1000 (`ckpt_s01000.pt`); horizon 2500 | stage-3 rows |

**Expected training loss**, per 25-step window:

| Stage | Loss |
|---|---|
| 1 | about 1.59 at step 1525, then 0.73 (2000) → 0.66 (2925) |
| 2 | 0.51 (1425) → 0.35 (2850) |
| 3 | 0.48 (500) → 0.39 (1000) |

The in-training `VAL` monitor decodes `Missing words:`. It is meaningful in stage 1 (pos about 0.70 at step 3000) and reads about 0.05 in stages 2–3, whose answer format it does not parse. Evaluate with the repository's L decoder (`python -m rml.lm.decode`, the deployed select-then-edit decoding).

## G: the generator (3 stages)

```
RML_LTX_DIR=… RML_OUT_DIR=… RML_G_STAGE1_ROOT=… RML_G_MULTITASK_ROOT=… \
bash scripts/train/train_generator.sh configs/training/generator_stage1_avse25_vsr_cascade_k1to10.yaml
bash scripts/train/train_generator.sh configs/training/generator_stage2_avse25_multitask_restoration.yaml
bash scripts/train/train_generator.sh configs/training/generator_stage3_avse25_multitask_text.yaml   # main G
```

**Model:** the LTX-2.5 22B dev transformer, frozen in bf16 with int8-quanto. It carries a LoRA of rank 64 (α 64) on:
- audio self-attention, audio text cross-attention and the audio feed-forward;
- video→audio attention.

The video pathway is untouched.

**Strategy `avse`:**
- token layout `video [reference | target]` (σ = 0) and `audio [corrupted input (σ = 0) | target (flow matching)]`;
- the loss is computed on the target audio only, with weight 12 on tokens within the silenced span ±60 ms;
- text dropout 0.1, which trains the CFG null branch.

**Optimisation:** AdamW at LR 1e-4 for all LoRA parameters (via `learning_rate_groups`), linear schedule, batch 2, gradient clipping 1.0, seed 42.

| Stage | Run | Init | Data / conditions | Steps |
|---|---|---|---|---|
| 1 | `avse25_vsr_cascade_k1to10` | base LTX-2.5 | 15,816 silenced cuts, `Transcript: <reader line>` | 5000 |
| 2 | `avse25_multitask_restoration` | stage 1 step 5000, fresh optimizer | 13,018 inpaint/denoise samples; denoise = `<denoise>` | 3000 |
| 3 | `avse25_multitask_text` (main) | stage 2 step 3000, fresh optimizer | same samples; denoise text varied (`conditions_mt_text`) | 2000 |

Inference also uses two intermediate stages:
- stage 1 (step 5000) at cfg 5 for the silenced R→G candidate;
- stage 2 (step 3000) at cfg 1 for the noisy "G alone, audio kept" candidate.

Checkpoints are saved every 125 steps with the full optimizer state; `train_generator.sh` resumes from the newest one. Losses are in `<out>/tensorboard` (`train/loss`, `train/audio_loss`).

## Compute

Compute (Appendix B.2):
- **G:** about 2 s per step at batch 2. The three stages take about 6 GPU-hours; stage 2 alone was 121 min with a 39.4 GB peak.
- **L:** about 2.5 s per step, about 5 GPU-hours over the three stages.
- **Data:** encoding the latents and conditions is a one-off GPU job. The noise bank and the row builders run on CPU.

## Reproducibility

- Every stage resumes from its newest checkpoint (weights, optimizer and step).
- **Deterministic mode** makes L bitwise reproducible across runs on the same hardware and library versions. G is not run in deterministic mode.
