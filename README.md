# Read My Lips

Code for **Read My Lips: Context-aware Lip Reading Using Language and Audio-Visual Models** (arXiv
[2611.33117](https://arxiv.org/abs/2611.33117)).

**[Project page](https://amirhossein-razlighi.github.io/read_my_lips/)** · **[Paper](https://arxiv.org/abs/2611.33117)**

Read My Lips (RML) recovers speech that was silenced or buried in noise. Three models communicate through plain text:
a lip reader **R** (Auto-AVSR) proposes the 10 best transcripts of the whole clip, a language model **L** (Gemma-4 12B
with a LoRA) chooses one and corrects it with the audible context, and an audio-visual generator **G** (LTX-2.5 22B with
LoRA adapters) speaks the result in the speaker's voice, synchronised with the lips. At test time the pipeline runs six
configurations (text from L, from R or none; the input audio kept or replaced by silence) and selects the output best
supported by the audio and lip evidence, with no knowledge of the noise level or of where the damage is.

| | Silenced speech, exact span recovery ↑ | Noisy speech, WER ↓ | Hard-to-read lips (silenced ↑ / noisy ↓) |
|---|---|---|---|
| Auto-AVSR (text output) | 30.7 | 49.4 | 8.6 / 72.5 |
| **RML (Auto-AVSR as R)** | **39.3** | **27.2** | **25.9 / 29.6** |

Pooled TalkVid + VoxMM test sets (Tab. 2 and 3) and the hard set (Tab. 4). Auto-AVSR reads the lips only, except
on the noisy hard set, where it also hears the noisy audio.

## Install

```bash
git clone <this repository> read_my_lips && cd read_my_lips
uv sync --extra roi                      # Python 3.12, torch 2.9.1 (CUDA 12.8), packages/, dlib
bash scripts/fetch_third_party.sh        # Auto-AVSR, the mouth-ROI recipe and the other readers at pinned commits
bash scripts/download_checkpoints.sh     # checkpoints -> checkpoints/ (docs/CHECKPOINTS.md)
cp configs/paths.example.yaml configs/paths.yaml   # edit if your checkpoints live elsewhere
```
Details, including the dlib build for the mouth crops: [docs/INSTALL.md](docs/INSTALL.md).

## Restore your own video

```bash
uv run python scripts/rml_infer.py --video talk.mp4 --start 12.0 --out outputs/talk
```
This cuts a 4.8 s face window at 12 s, detects whether it has a silenced span or noisy speech, runs R, L and the six
configurations of G, and writes `outputs/talk/talk_restored.mp4` (the window with the selected speech) and
`talk_result.json` (transcript, chosen configuration, every candidate). Give a known span with `--span 1.2,2.05` or
force the task with `--task denoise`. One 80 GB GPU. See [docs/INFERENCE.md](docs/INFERENCE.md).

## Evaluate

`benchmarks/` holds the test sets of the paper as text manifests (ids, source videos and window starts, silenced
spans, reviewed references; no media). `scripts/run_benchmark.py` runs the pipeline on one of them and
`scripts/make_tables.py` regenerates Tab. 2, 3, 4, 5a, 5b and Fig. 4 from the per-sample results, including the rows with
USR 2.0 as the lip reader (`--reader usr2`) and the Spanish and French set. See
[docs/EVALUATION.md](docs/EVALUATION.md).

## Training

The three training stages of G and the fine-tuning of L are described in [docs/TRAINING.md](docs/TRAINING.md).

## Checkpoints

[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) lists every checkpoint: ours (L and the LoRAs of G, on Hugging Face) and
the third-party models (LTX-2.5, Gemma-4, Auto-AVSR, Whisper large-v3, dlib, MediaPipe).

## Citation

```bibtex
@article{razlighi2026readmylips,
  title   = {Read My Lips: Context-aware Lip Reading Using Language and Audio-Visual Models},
  author  = {Razlighi, Amir and Ning, Zhuo and Ken Korem, Naomi and Patel, Nisarg and Chrysanthou, Yiorgos and
             Cohen-Or, Daniel and Mahdavi Amiri, Ali},
  journal = {arXiv preprint arXiv:2611.33117},
  year    = {2026}
}
```

## Licence

Our code is MIT-licensed ([LICENSE](LICENSE)). `packages/` is the LTX-2 monorepo of Lightricks under its own licence;
Auto-AVSR (Apache-2.0), the AV-HuBERT preparation code, USR 2.0 (MIT) and the multilingual lip readers of Ma et al.
are fetched into `third_party/` under theirs
([third_party/README.md](third_party/README.md)). The model weights follow the licences of their bases (LTX-2.5,
Gemma-4). Please read the limitations and responsible-use section of the paper (Appendix F) before use.
