#!/bin/bash
# Download the Read My Lips data (Hugging Face dataset AmirHossein-razlighi/read-my-lips-data, gated: request access on
# its page, then `hf auth login`) into data/read-my-lips-data (the rml_data key of configs/paths.example.yaml).
#
#   bash scripts/download_data.sh                 # test sets and our outputs (test_sets/, rml_outputs/)
#   bash scripts/download_data.sh test_sets       # only the test sets (inputs and references)
#   bash scripts/download_data.sh rml_outputs     # only our selected outputs and per-sample results
#   bash scripts/download_data.sh train           # the training data (train/, tar shards; unpacked here)
#   bash scripts/download_data.sh all
set -euo pipefail
cd "$(dirname "$0")/.."
D=${RML_DATA:-data/read-my-lips-data}; mkdir -p "$D"
RML_HF_DATA=${RML_HF_DATA:-AmirHossein-razlighi/read-my-lips-data}
if command -v hf >/dev/null; then dl() { hf download "$@"; }; else dl() { huggingface-cli download "$@"; }; fi
parts=("$@"); [ ${#parts[@]} -eq 0 ] && parts=(test_sets rml_outputs)
[ "${parts[0]}" = all ] && parts=(test_sets rml_outputs train)
for p in "${parts[@]}"; do
  case $p in
    test_sets|rml_outputs) dl "$RML_HF_DATA" --repo-type dataset --local-dir "$D" --include "README.md" "$p/*" ;;
    train)
      dl "$RML_HF_DATA" --repo-type dataset --local-dir "$D" --include "train/*"
      (cd "$D/train/generator" && sha256sum -c --quiet SHA256SUMS && for t in windows-*.tar; do tar -xf "$t"; done)
      (cd "$D/train/noise_bank" && sha256sum -c --quiet SHA256SUMS) ;;
    *) echo "unknown part '$p' (test_sets, rml_outputs, train, all)" >&2; exit 1 ;;
  esac
done
echo "data in $D (docs/EVALUATION.md, docs/TRAINING.md)"
