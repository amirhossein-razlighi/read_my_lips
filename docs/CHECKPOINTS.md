# Checkpoints

Default locations are those of `configs/paths.example.yaml` (under `checkpoints/`).

## Read My Lips

The Hugging Face repository id is a placeholder until the release (`RML_HF` in scripts/download_checkpoints.sh).

| Key | File | What it is |
|---|---|---|
| `lm_checkpoint` | `read_my_lips/lm/ckpt_s01000.pt` | L: LoRA (r 32) of Gemma-4 12B, select-then-edit, trained on silenced and noisy windows (Sec. 4.2, Appendix B.1) |
| `g_main_lora` | `read_my_lips/g_multitask_text/lora_weights_step_02000.safetensors` | G, final stage (2,000 steps): every configuration except the two below |
| `g_cascade_lora` | `read_my_lips/g_vsr_cascade/lora_weights_step_05000.safetensors` | G, first stage (5,000 steps): R -> G with the input audio kept, on silenced inputs |
| `g_restoration_lora` | `read_my_lips/g_multitask_restoration/lora_weights_step_03000.safetensors` | G, second stage (3,000 steps): G alone with the noisy audio kept |

The three G checkpoints are the three training stages of Appendix B.1 (docs/TRAINING.md); docs/INFERENCE.md says
which configuration renders with which.

## Third-party models

| Key | Model | Source |
|---|---|---|
| `ltx_transformer`, `ltx_text_encoder`, `ltx_audio_vae`, `ltx_video_vae` | LTX-2.5 22B dev, split pack (transformer, Gemma-4 text encoder with projection, audio and video VAEs) | Lightricks (Hugging Face) |
| `gemma_dir` | Gemma-4 12B it, QAT q4_0 unquantized | Google (Hugging Face; licence to accept) |
| `auto_avsr_vsr`, `auto_avsr_avsr` | `vsr_trlrwlrs2lrs3vox2avsp_base.pth`, `avsr_trlrwlrs2lrs3vox2avsp_base.pth` | Auto-AVSR model zoo (links in its README) |
| `usr2_ckpt` | USR 2.0 Huge, high-resource, fine-tuned (`usr2_huge_high_resource.pth`, sha256 `42aef587d507aa390873b2744ffb3a306b37267fa88c803e6cb950112794d318`): R in "USR 2.0 as R" (Tab. 2, 3) | the "Huge (high-resource)" link of the USR 2.0 README (Google Drive) |
| `vsr_multilang_es`, `vsr_multilang_lm_es` | `CMUMOSEAS_V_ES_WER44.5/` (model.pth, model.json) and its language model `lm_es/`: R for Spanish (Tab. 5b) | Ma et al. model zoo, CMU-MOSEAS rows (links in the Visual_Speech_Recognition_for_Multiple_Languages README) |
| `vsr_multilang_fr`, `vsr_multilang_lm_fr` | `CMUMOSEAS_V_FR_WER58.6/` and `lm_fr/`: R for French (Tab. 5b) | same |
| `whisper_hf_dir` | Whisper large-v3 (transformers format) | `openai/whisper-large-v3` |
| `whisperx_dir` | faster-whisper large-v3 + the WhisperX English alignment model (wav2vec2) | `Systran/faster-whisper-large-v3`; the alignment model is fetched by WhisperX on first use |
| `dlib_face_predictor`, `dlib_cnn_detector` | `shape_predictor_68_face_landmarks.dat`, `mmod_human_face_detector.dat` | dlib model files (dlib.net) |
| `mean_face` | `20words_mean_face.npy` | the Auto-AVSR / AV-HuBERT preparation recipe |
| `face_landmarker` | `face_landmarker.task` | MediaPipe face landmarker |
