# Training launchers

The launchers are thin wrappers around `python`, with no cluster or scheduler commands. Run each one inside whatever GPU job your system provides; every stage needs one 80 GB GPU. Set the `RML_*` variables first, as described in `docs/TRAINING.md`, which also covers data preparation.

| File | What it does |
|---|---|
| `render_config.py` | fills the `${RML_*}` placeholders of a config from the environment and fails if any is unset |
| `train_generator.sh <cfg>` | runs one generator stage with `packages/ltx-trainer/scripts/train.py`; resumes from the newest checkpoint |
| `train_lm.py <cfg> [--dry-run]` | runs one language-model stage with `rml/training/train_vis2gemma.py`; resumes from `last.pt` |

## The chain

Stage 1 of each model starts from its base model. Stages 2 and 3 warm-start from the final checkpoint of the previous stage, with a fresh optimizer.

```
# generator G
bash scripts/train/train_generator.sh configs/training/generator_stage1_avse25_vsr_cascade_k1to10.yaml    # 5,000 steps
bash scripts/train/train_generator.sh configs/training/generator_stage2_avse25_multitask_restoration.yaml # 3,000 steps
bash scripts/train/train_generator.sh configs/training/generator_stage3_avse25_multitask_text.yaml        # 2,000 steps -> main G

# language model L
python scripts/train/train_lm.py configs/training/lm_stage1_select_or_write.yaml      # to step 3,000
python scripts/train/train_lm.py configs/training/lm_stage2_select_then_edit.yaml     # 3,000 steps
python scripts/train/train_lm.py configs/training/lm_stage3_multitask_denoise.yaml    # to step 1,000 (ckpt_s01000.pt)
```

L and G do not depend on each other's checkpoints, so the two chains can run in parallel.

The released system uses these checkpoints:

| Checkpoint | Role | Released as (Hugging Face `AmirHossein-razlighi/read-my-lips`) |
|---|---|---|
| `avse25_multitask_text` step 2000 | main generator | `generator/main/lora.safetensors` |
| `avse25_vsr_cascade_k1to10` step 5000 | silenced R→G candidate | `generator/silenced_reader/lora.safetensors` |
| `avse25_multitask_restoration` step 3000 | noisy candidate in which G keeps the audio | `generator/noisy_keep/lora.safetensors` |
| `gemma_multitask_denoise_novis/ckpt_s01000.pt` | L | `lm/adapter_model.safetensors` + `lm/config.json` (its LoRA tensors and decoding configuration) |
