#!/bin/bash
# Download every checkpoint into checkpoints/ (the layout of configs/paths.example.yaml). Needs the Hugging Face CLI
# (pip install -U huggingface_hub; `hf auth login`) and, for Gemma-4, an accepted licence on its model page.
set -euo pipefail
cd "$(dirname "$0")/.."
C=checkpoints; mkdir -p $C
RML_HF=${RML_HF:-AmirHossein-razlighi/read-my-lips}
if command -v hf >/dev/null; then dl() { hf download "$@"; }; else dl() { huggingface-cli download "$@"; }; fi
# Read My Lips: L (Gemma-4 LoRA: lm/) and the three LoRAs of G (generator/{main,silenced_reader,noisy_keep}/)
dl "$RML_HF" --local-dir $C/read_my_lips
(cd $C/read_my_lips && sha256sum -c --quiet SHA256SUMS)
# LTX-2.5 split pack (Lightricks)
dl Lightricks/LTX-2.5 --local-dir $C/ltx-2.5 \
  --include "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
            "vae/ltx-2.5-audio-vae-bf16.safetensors" "vae/ltx-2.5-video-vae-bf16.safetensors"
# Gemma-4 12B (the base of L and the language score)
dl google/gemma-4-12B-it-qat-q4_0-unquantized --local-dir $C/gemma-4-12B-it-qat-q4_0-unquantized
# Whisper large-v3: HF weights (evidence) and the faster-whisper conversion (transcripts)
dl openai/whisper-large-v3 --local-dir $C/whisper-large-v3
dl Systran/faster-whisper-large-v3 --local-dir $C/whisperx/faster-whisper-large-v3
echo "Auto-AVSR checkpoints, dlib models, the mean face and the MediaPipe face landmarker: docs/CHECKPOINTS.md"
