# Installation

Tested on Linux x86-64 with one NVIDIA H100 80 GB, CUDA 12.8 drivers, Python 3.12, ffmpeg on the PATH (4.4 and
7.1 tested; versions decode AAC audio slightly differently, which can change G's outputs slightly).

```bash
uv sync --extra roi                       # creates .venv: torch 2.9.1 (cu128), the workspace packages and dlib
bash scripts/fetch_third_party.sh         # third_party/: auto_avsr, av_hubert, usr2, vsr_multilang (pinned)
bash scripts/download_checkpoints.sh      # see docs/CHECKPOINTS.md for the files it cannot fetch
cp configs/paths.example.yaml configs/paths.yaml
```

- dlib (the `roi` extra, for the mouth crops) is built from source and needs cmake and a C++ compiler. Keep
  `--extra roi` on every later `uv sync`: a plain `uv sync` uninstalls it.
- `packages/` holds the LTX-2 monorepo (ltx-core, ltx-pipelines, ltx-trainer) with the audio-visual speech
  strategy used to train G; `pyproject.toml` declares them as a uv workspace.
- Keep torch at 2.9.1: torchaudio and torchvision must match its minor version, and newer default wheels need
  newer drivers.
- Everything runs offline once the checkpoints are local (`HF_HUB_OFFLINE=1` is set by the scripts). WhisperX's
  Silero VAD is read from the torch hub cache: put `snakers4_silero-vad_master` under `<torch_home>/hub/`
  (`torch_home` in configs/paths.yaml; one `torch.hub.load("snakers4/silero-vad", "silero_vad")` with network fills it).
- The Spanish and French readers of Ma et al. (`--reader vsr_multilang`) run in this environment. USR 2.0
  (`--reader usr2`) runs in its own interpreter with the requirements of `third_party/usr2/requirements.txt`
  (hydra-core, omegaconf, pytorch-lightning, torchmetrics, mediapipe, av, opencv; tested with Python 3.11 and
  torch 2.5.1 + CUDA 12.1); give its path as `usr2_python` in configs/paths.yaml.
- `uv run pytest` runs the CPU unit tests; `uv run ruff check` lints.
- HPC clusters whose compilers and libraries come from a software stack separate from the host system (for example
  a CVMFS stack): dlib then links against that stack, and the standalone Python that uv downloads cannot load it
  (`ImportError: libgif.so.7 ...` or `GLIBC_2.35 not found`). Build the environment on the cluster's own Python:
  load its Python 3.12 module, then `uv sync --python "$(which python3.12)" --extra roi`.
