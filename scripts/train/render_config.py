#!/usr/bin/env python
"""Fill the ${RML_*} placeholders of a training config from the environment and write the resolved copy.

    python scripts/train/render_config.py <config.yaml> <resolved.yaml>

Fails on any placeholder whose variable is unset, so a run never starts with a half-resolved path.
"""
import os
import re
import sys
from pathlib import Path


def render(text: str) -> str:
    missing = sorted({m for m in re.findall(r"\$\{([A-Z0-9_]+)\}", text) if m not in os.environ})
    if missing:
        raise SystemExit(f"unset environment variables: {', '.join(missing)} (see docs/TRAINING.md)")
    return re.sub(r"\$\{([A-Z0-9_]+)\}", lambda m: os.environ[m.group(1)], text)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render(src.read_text()))
    print(dst)
