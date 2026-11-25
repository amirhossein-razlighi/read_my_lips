#!/usr/bin/env bash
# Train one generator (G) stage with the vendored LTX-2 trainer.
#   bash scripts/train/train_generator.sh configs/training/generator_stage1_avse25_vsr_cascade_k1to10.yaml
# Needs: RML_LTX_DIR, RML_OUT_DIR and the stage's data root (RML_G_STAGE1_ROOT or RML_G_MULTITASK_ROOT); one 80 GB GPU.
# Stages 2 and 3 warm-start from the previous stage's final LoRA (model.load_checkpoint in their configs).
# Re-running the command after an interruption resumes from the newest checkpoint of the run (lora weights + full
# training state, written every checkpoints.interval steps).
set -euo pipefail
CFG=${1:?config yaml}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PYTHON:-python}
NAME=$(basename "$CFG" .yaml); NAME=${NAME#generator_stage[123]_}
OUT=${RML_OUT_DIR:?set RML_OUT_DIR}/$NAME
CK=$OUT/checkpoints
STEPS=$(grep -E '^\s+steps:' "$CFG" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
FINAL=$(printf "%s/lora_weights_step_%05d.safetensors" "$CK" "$STEPS")
[ -f "$FINAL" ] && { echo "run complete: $FINAL"; exit 0; }
mkdir -p "$OUT"
RUN=$OUT/resolved_$(basename "$CFG")
"$PY" "$REPO/scripts/train/render_config.py" "$CFG" "$RUN" >/dev/null
# resume: the newest checkpoint step that has both lora weights and a training state
RES=""
for L in $(ls -r "$CK"/lora_weights_step_*.safetensors 2>/dev/null); do
  st=$(basename "$L" .safetensors | sed 's/lora_weights_step_//')
  [ -f "$CK/training_state_step_$st.pt" ] && { RES=$L; break; }
done
if [ -n "$RES" ]; then
  echo "resuming from $RES"
  if grep -qE '^\s+load_checkpoint:' "$RUN"; then
    sed -i -E -e "s|^(\s*)load_checkpoint:.*|\1load_checkpoint: \"$RES\"|" -e "s|^(\s*)no_resume:.*|\1no_resume: false|" "$RUN"
  else
    sed -i -E "s|^(\s*)model_path:|\1load_checkpoint: \"$RES\"\n\1model_path:|" "$RUN"
  fi
fi
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
cd "$REPO"
"$PY" packages/ltx-trainer/scripts/train.py "$RUN" --disable-progress-bars
[ -f "$FINAL" ] || { echo "trainer exited without the final checkpoint $FINAL"; exit 1; }
echo "done: $FINAL"
