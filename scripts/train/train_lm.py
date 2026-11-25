#!/usr/bin/env python
"""Train one language-model (L) stage: resolve a configs/training/lm_stage*.yaml and run rml/training/train_vis2gemma.py.

    python scripts/train/train_lm.py configs/training/lm_stage1_select_or_write.yaml [--dry-run]

Needs GEMMA_ROOT (local Gemma-4 12B it-qat-q4_0-unquantized), RML_OUT_DIR and the data variables the config uses;
one 80 GB GPU. Output: $RML_OUT_DIR/<run_name>/ (ckpt_sNNNNN.pt at keep-ckpts, last.pt every save-every steps,
args.json, tensorboard/). Re-running resumes from <out>/last.pt (then --init-from is dropped).
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/train"))
from render_config import render  # noqa: E402


def argv_of(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg["args"].items():
        if v is True:
            out.append(f"--{k}")
        elif v is False or v is None:
            continue
        elif isinstance(v, list):
            out += [f"--{k}", *map(str, v)]
        else:
            out += [f"--{k}", str(v)]
    return out


def main() -> None:
    cfg_path = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv[2:]
    cfg = yaml.safe_load(render(cfg_path.read_text()))
    out = Path(os.environ["RML_OUT_DIR"]) / cfg["run_name"]
    argv = argv_of(cfg) + ["--out", str(out)]
    if (out / "last.pt").exists():
        argv += ["--resume", str(out / "last.pt")]
    elif cfg.get("init_from"):
        if not dry and not Path(cfg["init_from"]).exists():
            raise SystemExit(f"previous stage not finished: {cfg['init_from']}")
        argv += ["--init-from", cfg["init_from"]]
    cmd = [sys.executable, str(REPO / "rml/training/train_vis2gemma.py"), *argv]
    if dry:
        print(" ".join(cmd))
        return
    if not os.environ.get("GEMMA_ROOT"):
        raise SystemExit("set GEMMA_ROOT to the local Gemma-4 12B (it-qat-q4_0-unquantized) directory")
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "GEMMA_ATTN": "eager",
           "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")}
    out.mkdir(parents=True, exist_ok=True)
    with open(out.parent / f"{cfg['run_name']}.log", "a") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        done = False
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            done |= line.startswith(f"DONE step {cfg['args']['stop-at-step']}")
        rc = p.wait()
    if rc or not done:
        raise SystemExit(f"{cfg['run_name']} not finished (exit {rc}); re-run to resume")


if __name__ == "__main__":
    main()
