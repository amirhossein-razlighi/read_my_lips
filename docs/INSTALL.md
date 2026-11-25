# Installation

Tested on Linux x86-64 with one NVIDIA H100 80 GB, CUDA 12.8 drivers, Python 3.12, ffmpeg on the PATH.

```bash
uv sync                                   # creates .venv; torch 2.9.1 (cu128) and the workspace packages
bash scripts/fetch_third_party.sh         # third_party/auto_avsr, third_party/av_hubert (pinned commits)
uv sync --extra roi                       # dlib, for the mouth crops (needs cmake and a C++ compiler)
bash scripts/download_checkpoints.sh      # see docs/CHECKPOINTS.md for the files it cannot fetch
cp configs/paths.example.yaml configs/paths.yaml
```

- `packages/` holds the LTX-2 monorepo (ltx-core, ltx-pipelines, ltx-trainer) with the audio-visual speech
  strategy used to train G; `pyproject.toml` declares them as a uv workspace.
- Keep torch at 2.9.1: torchaudio and torchvision must match its minor version, and newer default wheels need
  newer drivers.
- Everything runs offline once the checkpoints are local (`HF_HUB_OFFLINE=1` is set by the scripts). WhisperX's
  Silero VAD is read from the torch hub cache: put `snakers4_silero-vad_master` under `<torch_home>/hub/`
  (`torch_home` in configs/paths.yaml; one `torch.hub.load("snakers4/silero-vad", "silero_vad")` with network fills it).
- `uv run pytest` runs the CPU unit tests; `uv run ruff check` lints.
