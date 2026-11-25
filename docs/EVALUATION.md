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

Each row carries the sample id, the window (`base_id`, its source video `video`, `source_clip` and
`window_start_s`), the silenced span (`mask_segments_s`, `masked_words`, `span_start_index`) or the noise condition,
the reviewed reference (`sentence`, `window_words` with WhisperX timings), `gt_review` and, for silenced cuts,
`asr_verified` (Whisper on the clean window recovers the span exactly: the cuts the paper scores). No media is
distributed: rebuild each face window from its source clip (TalkVid, VoxMM; the hard set's scenes) with
`rml.data.windows.make_window(source, window_start_s, out)` into a directory of `<base_id>.mp4`.

Noisy inputs and the grid's babble are mixed deterministically from the test split of the noise bank
(`noise_bank` in configs/paths.yaml; built with the training data tools, docs/TRAINING.md).

## Running

```bash
uv run python scripts/run_benchmark.py --benchmark silenced_talkvid_en --windows data/windows/talkvid_en \
    --work data/runs/silenced_talkvid_en --results outputs/results
uv run python scripts/make_tables.py --results outputs/results --out outputs/tables.txt
```
`make_tables.py` prints Tab. 2 (silenced, exact span recovery and word accuracy, per set and pooled), Tab. 3 (noisy,
WER per condition, pooled), the hard columns of Tab. 4, the component rows of Tab. 5a and Fig. 4 (WER per grid
condition and the configurations chosen), each against the selection with a 95% paired bootstrap interval over
source videos. Baselines enter as transcripts: `<results>/<benchmark>.baselines.json` = `{method: {id: text}}`.

## Metrics

- Silenced speech: exact span recovery and word accuracy in the span (`rml/eval/gap_score.py`): the whole-window
  transcript is aligned to the reference window (Levenshtein on Whisper-normalised tokens); the span region runs
  from the last left-context reference token to the first right-context one, so insertions inside it count.
  Stutters are optional and compounds may be joined, on both sides.
- Noisy speech and the grid: WER of the whole window against the reviewed reference.
- Normalisation: Whisper's English normaliser with its British-to-American spelling map (`rml/eval/scoring.py`).
