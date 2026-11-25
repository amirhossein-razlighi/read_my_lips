#!/bin/bash
# Fetch the third-party code the pipeline imports, at the commits the paper's results were produced with.
#   Auto-AVSR (Apache-2.0)  -> third_party/auto_avsr      the lip reader R and the visual evidence
#   AV-HuBERT preparation   -> third_party/av_hubert      the mouth-ROI recipe (align_mouth.py); own licence
#   USR 2.0 (MIT)           -> third_party/usr2           R = USR 2.0 (Tab. 2, 3: "USR 2.0 as R")
#   Ma et al. multilingual VSR (own licence) -> third_party/vsr_multilang   R for Spanish and French (Tab. 5b)
set -euo pipefail
cd "$(dirname "$0")/.."
fetch() {  # url dir commit
  [ -d "$2/.git" ] || git clone --quiet "$1" "$2"
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "$2" rev-parse --short HEAD)"
}
fetch https://github.com/mpc001/auto_avsr.git third_party/auto_avsr 30951b51f4608e0b135ee5eae54358b324356415
fetch https://github.com/facebookresearch/av_hubert.git third_party/av_hubert 258fb50e155134eec2c4b49c2ae8de267075fd18
fetch https://github.com/ahaliassos/usr2.git third_party/usr2 e9e9c94e8d827efbb28e9a4b9d5d2d97a583b129
fetch https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.git third_party/vsr_multilang \
  5e1405db0ae816509fb312f9a578724c2e0de0c7
