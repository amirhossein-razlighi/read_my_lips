"""The Hugging Face layouts: L's released checkpoint (safetensors + config.json) and the data's rml_outputs/."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_lm_checkpoint_safetensors_with_config(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from rml.lm.decode import EXPECTED_ARGS, load_lm_checkpoint
    lora = {"base_model.model.x.q_proj.lora_A.weight": torch.randn(4, 8),
            "base_model.model.x.q_proj.lora_B.weight": torch.randn(8, 4)}
    save_file(lora, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({**EXPECTED_ARGS, "lora_r": 32, "aux_max_new": 128}))
    st = load_lm_checkpoint(tmp_path / "adapter_model.safetensors")
    assert set(st["lora"]) == set(lora) and all(torch.equal(st["lora"][k], lora[k]) for k in lora)
    assert all(st["args"][k] == v for k, v in EXPECTED_ARGS.items())


def test_make_tables_reads_the_rml_outputs_layout(tmp_path):
    spec = importlib.util.spec_from_file_location("make_tables", REPO / "scripts/make_tables.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    name = "silenced_hard_en"
    (tmp_path / name).mkdir()
    (tmp_path / name / "results.json").write_text(json.dumps({"x": {"hyp": {}}}))
    (tmp_path / name / "results.usr2.json").write_text(json.dumps({"y": {"hyp": {}}}))
    res, man, _ = mt.load(tmp_path, REPO / "benchmarks", name)
    assert set(res) == {"x"} and len(man) == 119
    res, _, _ = mt.load(tmp_path, REPO / "benchmarks", name, ".usr2")
    assert set(res) == {"y"}


def test_prepare_lm_data_lays_out_the_released_rows(tmp_path):
    np = pytest.importorskip("numpy")
    lm, out = tmp_path / "lm", tmp_path / "out"
    row = {"id": "v_NA_1_2_t0.0__k1", "window": "v_NA_1_2_t0.0", "f0": 3, "f1": 9, "context": "a ___ c", "target": "b",
           "k": 1, "language": "en"}
    for st in ("stage1_2", "stage3"):
        (lm / st).mkdir(parents=True)
        for split in ("train", "val"):
            (lm / st / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        (lm / st / "keep_meta.json").write_text(json.dumps({"clips": [{"id": row["id"]}]}))
    (lm / "readers").mkdir()
    for r in ("clean", "down4", "drop50", "avhubert"):
        for t in ("inpaint", "denoise"):
            (lm / "readers" / f"{r}_{t}.json").write_text(json.dumps({"reader": r, "rows": [{"id": row["id"], "nbest": ["a b c"]}]}))
    (lm / "feat_lengths.tsv").write_text("window\tn_frames\tdim\nv_NA_1_2_t0.0\t120\t1024\n")
    import subprocess
    import sys
    subprocess.run([sys.executable, str(REPO / "scripts/train/prepare_lm_data.py"), "--lm", str(lm), "--out", str(out)],
                   check=True, capture_output=True)
    got = json.loads((out / "stage1_2/train.jsonl").read_text())
    assert {k: v for k, v in got.items() if k != "feats"} == row
    assert np.load(got["feats"])["feats"].shape == (120, 1024)
    assert (out / "readers/autoavsr_vsr_clean_train_shard0.json").exists()
    assert (out / "stage3/readers_denoise/avhubert.json").exists() and (out / "stage3/keep_meta.json").exists()
