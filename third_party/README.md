# Third-party code

Nothing here is vendored: `scripts/fetch_third_party.sh` clones the repositories below at the commits the paper's
results were produced with. Each keeps its own licence.

| Directory | Repository | Commit | Licence | Used for |
|---|---|---|---|---|
| `third_party/auto_avsr` | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | `30951b51f4608e0b135ee5eae54358b324356415` | Apache-2.0 | the lip reader R (visual Auto-AVSR), the audio-visual n-best for noisy inputs, the visual evidence of the selection |
| `third_party/av_hubert` | [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) | `258fb50e155134eec2c4b49c2ae8de267075fd18` | AV-HuBERT licence (see its LICENSE) | `avhubert/preparation/align_mouth.py`: the mean-face mouth crop that Auto-AVSR expects |
| `third_party/usr2` | [ahaliassos/usr2](https://github.com/ahaliassos/usr2) | `e9e9c94e8d827efbb28e9a4b9d5d2d97a583b129` | MIT | USR 2.0 as the lip reader R (`--reader usr2`; Tab. 2, 3) |
| `third_party/vsr_multilang` | [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) | `5e1405db0ae816509fb312f9a578724c2e0de0c7` | its own licence (research and benchmarking use; see its LICENSE) | the Spanish and French lip readers of Ma et al. (`--reader vsr_multilang`; Tab. 5b) |

The model files these need (Auto-AVSR, USR 2.0 and the CMU-MOSEAS Spanish/French checkpoints, the dlib detectors
and the 20-word mean face) are listed in
[docs/CHECKPOINTS.md](../docs/CHECKPOINTS.md).
