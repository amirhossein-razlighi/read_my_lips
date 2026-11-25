#!/bin/bash
# Download every checkpoint into checkpoints/ (the layout of configs/paths.example.yaml). Needs huggingface-cli
# (pip install -U huggingface_hub) and, for Gemma-4, an accepted licence on the Hugging Face model page.
# The Read My Lips repositories are placeholders until the release: see docs/CHECKPOINTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
C=checkpoints; mkdir -p $C
RML_HF=${RML_HF:-read-my-lips/read-my-lips}            # placeholder repo id
hf() { huggingface-cli download "$@"; }
# Read My Lips: L (Gemma-4 LoRA) and the three LoRAs of G
hf "$RML_HF" --local-dir $C/read_my_lips
# LTX-2.5 split pack (Lightricks)
hf Lightricks/LTX-2.5 --local-dir $C/ltx-2.5 \
  --include "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" "vae/*"
# Gemma-4 12B (the base of L and the language score)
hf google/gemma-4-12B-it-qat-q4_0-unquantized --local-dir $C/gemma-4-12B-it-qat-q4_0-unquantized
# Whisper large-v3: HF weights (evidence) and the faster-whisper conversion (transcripts)
hf openai/whisper-large-v3 --local-dir $C/whisper-large-v3
hf Systran/faster-whisper-large-v3 --local-dir $C/whisperx/faster-whisper-large-v3
echo "Auto-AVSR checkpoints, dlib models, the mean face and the MediaPipe face landmarker: docs/CHECKPOINTS.md"
