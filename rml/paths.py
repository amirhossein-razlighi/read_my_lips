"""Paths of checkpoints and data: configs/paths.yaml (copy configs/paths.example.yaml), overridable per key by the
environment variable RML_<KEY>. Relative paths resolve against the repository root."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class Paths:
    def __init__(self, values: dict):
        self._v = values

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        env = os.environ.get(f"RML_{key.upper()}")
        v = env if env is not None else self._v.get(key)
        if v is None:
            raise KeyError(f"path '{key}' is not set: add it to configs/paths.yaml or set RML_{key.upper()}")
        p = Path(os.path.expanduser(str(v)))
        return p if p.is_absolute() else REPO / p

    def roi_config(self) -> dict:
        return {"prep_dir": self.roi_prep_dir, "face_predictor": self.dlib_face_predictor,
                "cnn_detector": self.dlib_cnn_detector, "mean_face": self.mean_face}

    def evidence_models(self) -> dict:
        return {"auto_avsr_dir": self.auto_avsr_dir, "vsr_ckpt": self.auto_avsr_vsr,
                "whisper_dir": self.whisper_hf_dir, "gemma_dir": self.gemma_dir}

    def export_env(self) -> None:
        """environment read by the ASR and generator modules"""
        os.environ.setdefault("RML_WHISPERX_DIR", str(self.whisperx_dir))
        os.environ["TORCH_HOME"] = str(self.torch_home)   # the configured cache wins over a global TORCH_HOME
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_paths(config: str | Path | None = None) -> Paths:
    import yaml
    cfg = Path(config) if config else REPO / "configs/paths.yaml"
    if not cfg.exists():
        cfg = REPO / "configs/paths.example.yaml"
    return Paths(yaml.safe_load(cfg.read_text()) or {})
