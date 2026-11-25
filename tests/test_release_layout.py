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
