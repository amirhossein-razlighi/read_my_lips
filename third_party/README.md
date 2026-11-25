# Third-party code

Nothing here is vendored: `scripts/fetch_third_party.sh` clones the two repositories below at the commits the paper's
results were produced with. Each keeps its own licence.

| Directory | Repository | Commit | Licence | Used for |
|---|---|---|---|---|
| `third_party/auto_avsr` | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | `30951b51f4608e0b135ee5eae54358b324356415` | Apache-2.0 | the lip reader R (visual Auto-AVSR), the audio-visual n-best for noisy inputs, the visual evidence of the selection |
| `third_party/av_hubert` | [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) | `258fb50e155134eec2c4b49c2ae8de267075fd18` | AV-HuBERT licence (see its LICENSE) | `avhubert/preparation/align_mouth.py`: the mean-face mouth crop that Auto-AVSR expects |

The model files these need (Auto-AVSR checkpoints, the dlib detectors and the 20-word mean face) are listed in
[docs/CHECKPOINTS.md](../docs/CHECKPOINTS.md).
