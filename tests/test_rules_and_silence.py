import numpy as np

from rml.data.silence import find_silenced_regions, pick_gap
from rml.eval.benchmarks.silenced import pick
from rml.eval.span_rules import edge_ok


def test_edge_ok():
    assert edge_ok(["market"], "en")
    assert not edge_ok(["the", "market"], "en")
    assert not edge_ok(["market", "yeah"], "en")
    assert not edge_ok(["a"], "en")
    assert not edge_ok([], "en")


def test_cut_position_obeys_the_rules():
    words = ["so", "i", "went", "to", "the", "market", "yesterday", "morning", "with", "my", "brother"]
    ws = [{"word": w, "start_s": 0.3 * i, "end_s": 0.3 * i + 0.25} for i, w in enumerate(words)]
    r = {"window_words": ws, "sentence": " ".join(words), "language": "en", "target_index": 5}
    st, s, e = pick(r, 1)
    assert words[st] == "market" and s < ws[st]["start_s"] and e > ws[st]["end_s"]
    st, s, e = pick(r, 3)
    span = words[st:st + 3]
    assert edge_ok(span, "en") and "market" in span


def test_silence_detection():
    sr = 16000
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(int(4.8 * sr))).astype(np.float32)
    y[int(1.2 * sr):int(2.0 * sr)] = 0.0
    y[-int(0.1 * sr):] = 0.0                     # digital padding at the end: not the gap
    gap = pick_gap(find_silenced_regions(y, sr), len(y) / sr)
    assert abs(gap[0] - 1.2) < 0.002 and abs(gap[1] - 2.0) < 0.002
    assert pick_gap(find_silenced_regions(y[: int(1.0 * sr)], sr), 1.0) is None
