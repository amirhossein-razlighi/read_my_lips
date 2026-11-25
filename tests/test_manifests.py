import json
from pathlib import Path

B = Path(__file__).resolve().parents[1] / "benchmarks"


def rows(name):
    return [json.loads(line) for line in (B / f"{name}.jsonl").read_text().splitlines() if line]


def test_test_set_sizes_match_table_1():
    for name, n in (("silenced_talkvid_en", 875), ("silenced_voxmm", 1217), ("silenced_hard_en", 116)):
        assert sum(1 for r in rows(name) if r["asr_verified"] and r["gt_review"] != "removed") == n
    for name, n in (("noisy_talkvid_en", 570), ("noisy_voxmm", 795), ("noisy_hard_en", 220)):
        assert len(rows(name)) == n
    assert len(rows("grid_talkvid_en")) == 1421


def test_cut_rows_are_consistent():
    for r in rows("silenced_talkvid_en")[:200]:
        k, s = r["n_masked_words"], r["span_start_index"]
        assert [w[0] for w in r["window_words"]][s:s + k] == r["masked_words"]
        assert r["id"].startswith(r["base_id"])
