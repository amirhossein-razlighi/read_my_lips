"""Text conditions of G: each text is encoded by the LTX-2.5 Gemma-4 text encoder and its feature extractor into
the pre-connector format {video_prompt_embeds, audio_prompt_embeds, prompt_attention_mask (left-padded),
transcript}. Every output dir carries an _encoder.json stamp. conditions_empty/ holds the encoded empty prompt, the
null branch of text CFG.
  python -m rml.generator.conditions --root R --json NAME=texts.json [--empty-out conditions_empty]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ENCODER_STAMP = {"text_encoder": "gemma4-ltx2.5"}


class ConditionEncoder:
    def __init__(self, transformer_path, text_encoder_path, device=None):
        from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = load_text_encoder(text_encoder_path, device=device, dtype=torch.bfloat16)
        self.embeddings_processor = load_embeddings_processor((transformer_path, text_encoder_path), text_encoder_path,
                                                              device=device, dtype=torch.bfloat16)

    def encode(self, prompt: str) -> dict:
        with torch.inference_mode():
            hidden_states, attn = self.text_encoder.encode([prompt])[0]       # pads left
            video_features, audio_features = self.embeddings_processor.feature_extractor(hidden_states, attn, "left")
        assert video_features.ndim == 3 and video_features.shape[0] == 1 and attn.shape[-1] == video_features.shape[1]
        out = {"video_prompt_embeds": video_features[0].cpu().contiguous(),
               "prompt_attention_mask": attn[0].cpu().contiguous(), "transcript": prompt}
        if audio_features is not None:
            out["audio_prompt_embeds"] = audio_features[0].cpu().contiguous()
        return out

    def encode_texts(self, texts: dict, cond_dir: Path, overwrite: bool = False) -> int:
        cond_dir.mkdir(parents=True, exist_ok=True)
        (cond_dir / "_encoder.json").write_text(json.dumps(ENCODER_STAMP))
        (cond_dir / "_final_conditions.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))
        n = 0
        for sid, text in texts.items():
            p = cond_dir / f"{sid}.pt"
            if p.exists() and not overwrite:
                continue
            torch.save(self.encode(text), p)
            n += 1
        return n

    def encode_empty(self, ids, cond_dir: Path, overwrite: bool = False) -> None:
        empty = self.encode("")
        cond_dir.mkdir(parents=True, exist_ok=True)
        (cond_dir / "_encoder.json").write_text(json.dumps(ENCODER_STAMP))
        for sid in ids:
            p = cond_dir / f"{sid}.pt"
            if not p.exists() or overwrite:
                torch.save(empty, p)


def main() -> None:
    from rml.paths import load_paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--json", action="append", default=[], help="NAME=texts.json ({id: text}) -> <root>/NAME/")
    ap.add_argument("--empty-out", default=None, help="dir name for the empty-prompt conditions (one per metadata row)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    P = load_paths(a.paths)
    enc = ConditionEncoder(P.ltx_transformer, P.ltx_text_encoder)
    for spec in a.json:
        name, sep, jpath = spec.partition("=")
        if not sep:
            raise SystemExit(f"--json expects NAME=JSON, got {spec!r}")
        texts = {k: (v.get("transcript") if isinstance(v, dict) else str(v)) for k, v in json.loads(Path(jpath).read_text()).items()}
        print(f"[{name}] {enc.encode_texts(texts, a.root / name, a.overwrite)} encoded", flush=True)
    if a.empty_out:
        ids = [r["id"] for r in json.loads((a.root / "metadata.json").read_text())["clips"]]
        enc.encode_empty(ids, a.root / a.empty_out, a.overwrite)


if __name__ == "__main__":
    main()
