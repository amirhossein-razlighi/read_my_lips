# Evaluation

## Test sets (benchmarks/)

| Manifest | Rows | Paper |
|---|---|---|
| `silenced_talkvid_en.jsonl` | 958 cuts (875 ASR-verified) of 114 windows, k = 1..10 | Tab. 1, 2, 5a |
| `silenced_voxmm.jsonl` | 1,424 cuts (1,217 verified) of 159 windows | Tab. 1, 2, 5a |
| `silenced_hard_en.jsonl` | 119 cuts (116 verified) of 44 windows, k = 1, 3, 5 | Tab. 1, 4 |
| `silenced_val_en.jsonl` | 373 validation cuts (99 videos), for tuning only | Sec. 5, Data |
| `noisy_{talkvid_en,voxmm,hard_en}.jsonl` | 570 / 795 / 220 windows x conditions | Tab. 1, 3, 4 |
| `grid_{talkvid_en,voxmm}.jsonl` | 1,421 / 1,999 condition-grid samples | Fig. 4, App. C |
| `silenced_talkvid_esfr.jsonl` | 431 Spanish and French cuts (401 verified; 334 / 97 windows of 69 / 7 TalkVid videos), k = 1..4, one per window | Tab. 1, 5b |

Each row carries the sample id, the window (`base_id`, its source video `video`, `source_clip` and
`window_start_s`), the silenced span (`mask_segments_s`, `masked_words`, `span_start_index`) or the noise condition,
the reviewed reference (`sentence`, `window_words` with WhisperX timings), `gt_review` and, for silenced cuts,
`asr_verified` (Whisper on the clean window recovers the span exactly: the cuts the paper scores).

## Data

The media of every test set and our outputs are in the gated Hugging Face dataset
[AmirHossein-razlighi/read-my-lips-data](https://huggingface.co/datasets/AmirHossein-razlighi/read-my-lips-data)
(request access on its page, then `hf auth login`):

```bash
bash scripts/download_data.sh             # test_sets/ and rml_outputs/ -> data/read-my-lips-data (rml_data in configs/paths.yaml)
```

| Path | Content |
|---|---|
| `test_sets/<benchmark>/manifest.jsonl` | the manifest of `benchmarks/<benchmark>.jsonl` |
| `test_sets/<benchmark>/video/<base_id>.mp4` | the 4.8 s face window (768 x 512, 25 fps, with its original audio): the model's video input |
| `test_sets/<benchmark>/input/<id>.wav` | the corrupted audio the pipeline receives (16 kHz mono): the window with the span silenced, or the noisy mix |
| `test_sets/<benchmark>/reference/<base_id>.wav` | the clean window audio (16 kHz mono) |
| `rml_outputs/<benchmark>/audio/<id>.wav` | RML's selected restored audio |
| `rml_outputs/<benchmark>/selection.jsonl` | per sample: the configuration the selection chose, whether it kept the input audio, its transcript |
| `rml_outputs/<benchmark>/results.json` | our per-sample results in the format of `scripts/make_tables.py` |
| `rml_outputs/<benchmark>/results.usr2.json` | the same with USR 2.0 as the lip reader (silenced and noisy TalkVid and VoxMM) |

`<benchmark>` is a name of `benchmarks/index.json` (the validation set is not included). The Spanish and French set has its results (R, R -> L, R -> L -> G) but no audio in `rml_outputs/`. Without the data, rebuild each
face window from its source clip (TalkVid, VoxMM; the hard set's scenes) with
`rml.data.windows.make_window(source, window_start_s, out)` into a directory of `<base_id>.mp4` and pass it as
`--windows`; noisy inputs and the grid's babble are then mixed deterministically from the test split of the noise bank
(`noise_bank` in configs/paths.yaml; built with the training data tools, docs/TRAINING.md).

## Running

```bash
uv run python scripts/run_benchmark.py --benchmark silenced_talkvid_en --work data/runs/silenced_talkvid_en \
    --results outputs/results
uv run python scripts/run_benchmark.py --benchmark silenced_talkvid_en --reader usr2 \
    --work data/runs/silenced_talkvid_en_usr2 --results outputs/results     # USR 2.0 as R
uv run python scripts/run_benchmark.py --benchmark silenced_talkvid_esfr \
    --work data/runs/silenced_talkvid_esfr --results outputs/results        # Spanish / French, R = Ma et al.
uv run python scripts/make_tables.py --results outputs/results --out outputs/tables.txt
```
`run_benchmark.py` reads the windows (and, for the noisy and grid sets, the noisy inputs) from
`<rml_data>/test_sets/<benchmark>/` (override with `--data` or `--windows`). `make_tables.py` prints Tab. 2 (silenced,
exact span recovery and word accuracy, per set and pooled), Tab. 3 (noisy, WER per condition, pooled), the hard
columns of Tab. 4, the component rows of Tab. 5a and Fig. 4 (WER per grid condition and the configurations chosen),
each against the selection with a 95% paired bootstrap interval over source videos. With `<benchmark>.usr2.json`
results (`--reader usr2`) it prints the same silenced and noisy tables for USR 2.0 as R (the "Ours (USR 2.0 as R)"
rows of Tab. 2 and 3), and with `silenced_talkvid_esfr.json` Tab. 5b (R, R -> L and R -> L -> G per language and
pooled; no selection). Baselines enter as transcripts: `<results>/<benchmark>.baselines.json` =
`{method: {id: text}}`.

Our rows of the paper's tables, from our stored per-sample results, without a GPU:

```bash
bash scripts/download_data.sh rml_outputs
uv run python scripts/make_tables.py --results data/read-my-lips-data/rml_outputs
```

## Metrics

- Silenced speech: exact span recovery and word accuracy in the span (`rml/eval/gap_score.py`): the whole-window
  transcript is aligned to the reference window (Levenshtein on Whisper-normalised tokens); the span region runs
  from the last left-context reference token to the first right-context one, so insertions inside it count.
  Stutters are optional and compounds may be joined, on both sides.
- Spanish and French (Tab. 5b, and the `asr_verified` flags of `silenced_talkvid_esfr`): the same alignment, with the
  reference normalised word by word and without the stutter, compound and spelling tolerances
  (`rml/training/label_gap_score.py`); Whisper in the cut's language.
- Noisy speech and the grid: WER of the whole window against the reviewed reference.
- Normalisation: Whisper's English normaliser with its British-to-American spelling map (`rml/eval/scoring.py`).
