"""L's inference prompt (rml/lm/prompts.py) must be the one the trainer (rml/training/train_vis2gemma.py) builds
for the deployed checkpoint: select-then-edit, 10 readings, gap duration."""
import pytest

pytest.importorskip("torch")


def test_user_text_matches_the_trainer():
    import rml.training.train_vis2gemma as T
    from rml.lm.prompts import shown_readings, user_text
    readers = {"w": ["the first thing", "The first  thing", "a first thing"] + [f"reading {i}" for i in range(12)]}
    T._READERS[:] = [("eval", 1.0, readers)]
    T._READER_K, T._SELECT_EDIT, T._SELECT_WRITE, T._MCL_SLOTS = 10, True, False, 1
    for r in ({"id": "w__k2", "context": "first thing ___ candle holder", "f0": 10, "f1": 32, "feats": "x/w.npz"},
              {"id": "w__k7", "context": "<denoise> a b c", "f0": 0, "f1": 120, "feats": "x/w.npz"}):
        assert T._ctx_text(r, True) == user_text(r, readers)
        assert T.shown_readings(r) == shown_readings(r, readers)
