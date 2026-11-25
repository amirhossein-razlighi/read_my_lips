# Inference

`scripts/rml_infer.py` restores one 4.8 s window of a video; `rml/pipeline.py` implements the stages and is shared
with the benchmark runner.

## Stages

| Stage | What it does | Paper |
|---|---|---|
| window | MediaPipe face box over five frames, crop with margin to 3:2, 768x512 at 25 fps, 4.8 s (`rml/data/windows.py`) | Sec. 5, Data |
| task | silence detector on the window's audio: 5 ms frames, 70 dB below the peak, >= 50 ms, the longest region with audio on both sides (`rml/data/silence.py`); found -> `<inpaint>` with that span, else `<denoise>` | Sec. 4.2, App. B.1 |
| latents | video latents of the window; the input track G hears (muted span, noisy track, or silence for the muted twin) | Sec. 4.2 |
| roi | dlib 68 landmarks, mean-face warp, 96x96 mouth crop of the whole window (`rml/reader/roi.py`) | App. B.1 |
| reader | R: visual Auto-AVSR, beam 40, the 10 best distinct transcripts; for noisy inputs also the audio-visual Auto-AVSR n-best (`rml/reader/auto_avsr.py`) | Sec. 4.2 |
| whisper | silenced input: Whisper large-v3 on the input with the span filled with low noise, words inside the span dropped, written `left ___ right`; noisy input: Whisper's transcript (`rml/data/context.py`) | Sec. 4.2 |
| lm | L picks a candidate and corrects it (`Pick: <letter>`, `Corrected: ...`; Fig. 5); silenced inputs see the reader's 10-best and the gap duration, noisy inputs the reader, Auto-AVSR AV and Whisper interleaved (`rml/lm/`) | Sec. 4.2, Fig. 5 |
| texts, encode | the conditioning text of each configuration, encoded by the LTX-2.5 Gemma-4 text encoder (`rml/generator/conditions.py`) | Sec. 4.2 |
| render | G renders each configuration: 30 steps, seed 42, text guidance 5 with `<inpaint>` and 1 with `<denoise>` (`rml/generator/render.py`) | App. B.1 |
| transcribe | Whisper large-v3 on every output | Sec. 4.3 |
| evidence | e_A = log p(y given the input) - log p(y given silence) by Whisper large-v3; e_R = log p(y given the lips) - log p(y given a frozen first mouth frame) by visual Auto-AVSR, 0.9 attention + 0.1 CTC (`rml/selection/evidence.py`) | Eq. (2), App. C |
| select | evidence-weighted MBR, T = 100, nested voters (`rml/selection/rule.py`) | Eq. (3) |

## Candidate set

Every window gets a muted twin `<window>__full_silence` (its whole input track replaced by silence); the three
"audio muted" configurations render it.

| Configuration | Silenced input | Noisy input |
|---|---|---|
| R -> L -> G, audio kept | main LoRA, `<inpaint>\nTranscript: y_L`, cfg 5 | main LoRA, `<denoise>\nTranscript: y_L`, cfg 1 |
| R -> G, audio kept | stage-1 LoRA, `Transcript: h1`, cfg 5 | main LoRA, `<denoise>\nTranscript: h1`, cfg 1 |
| G alone, audio kept | main LoRA, `<inpaint>`, cfg 5 | stage-2 LoRA, `<denoise>`, cfg 1 |
| R -> L -> G, audio muted | main LoRA on the twin, `<inpaint>\nTranscript: y_L(twin)`, cfg 5 | same |
| R -> G, audio muted | main LoRA on the twin, `<inpaint>\nTranscript: h1`, cfg 5 | same |
| G alone, audio muted | main LoRA on the twin, `<inpaint>`, cfg 5 | same |

main LoRA = the final training stage of G (`g_main_lora`), stage-1 = `g_cascade_lora`, stage-2 =
`g_restoration_lora` (docs/CHECKPOINTS.md). h1 = the reader's top reading; y_L = L's transcript (for noisy input
and for the twin, L runs in denoise mode, and an empty or runaway answer falls back to the candidate L picked).
This is the set of the TalkVid and VoxMM results (Tab. 2, 3; recipe `tables`, the default). The hard set (Tab. 4,
recipe `hard`) and the condition grid (Fig. 4, recipe `grid`) were rendered with the main LoRA for every
configuration (R -> G kept on silenced input: `<inpaint>\nTranscript: h1`).

Voters: the six transcripts, the reader's top reading and y_L; when y_L votes, the reader's reading and R -> G with
the audio muted do not (L already read the whole N-best list). y_L votes but cannot be selected. An output without
speech has no evidence and neither votes nor can be selected. Ties go to the earlier configuration of the table
(more components, audio kept).

## Options

- `--span s,e` / `--task`: skip the detector.
- `--no-crop`: the input already is a 768x512, 4.8 s face window.
- `--stages`: rerun from a stage (every stage resumes from what it wrote under `<out>/root/rml/`).
- `--no-lm-score`: skip the Gemma-4 language score of the candidates, which is computed for completeness and does
  not enter Eq. (3).

The reader is replaceable without retraining (Sec. 5.5): any recogniser that writes the 10 best transcripts per window
into `<out>/root/rml/reader_vsr.json` (`{"rows": [{"id": <window>, "nbest": [...]}]}`) can stand in for Auto-AVSR;
run the remaining stages with `--stages whisper,lm,texts,encode,render,transcribe,evidence,select` after the
`latents,roi,reader` stages (the audio-visual n-best and the visual evidence stay Auto-AVSR, as in the paper).
