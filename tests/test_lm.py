from rml.lm.prompts import shown_readings, user_text
from rml.lm.rows import denoise_answer, inpaint_answer, inpaint_row, merged


def test_inpaint_row_frames_and_context():
    r = inpaint_row("w__k2", "so i went to ___ yesterday", (1.0, 1.9))
    assert (r["f0"], r["f1"]) == (23, 50)
    assert r["context"] == "so i went to ___ yesterday"


def test_prompt_matches_the_paper_format():
    r = {"id": "w__k2", "context": "first thing ___ candle holder", "f0": 10, "f1": 32}
    readers = {"w": ["the first thing", "The first  thing", "a first thing"]}
    assert shown_readings(r, readers) == ["the first thing", "a first thing"]
    assert user_text(r, readers) == ("Transcript: first thing ___ candle holder\n"
                                     "Lip reading candidates (best first):\nA. the first thing\nB. a first thing\n"
                                     "Gap duration: 0.9 s\n")


def test_merged_candidates_interleave():
    assert merged(["r1", "r2", "r3"], ["a1", "A1", "a2"], "w") == ["r1", "a1", "w", "r2", "r3", "a2"]


def test_answers():
    dec = {"final_list": [{"type": "full", "text": "the  cat sat"}], "pick": "A"}
    assert inpaint_answer(dec, "") == "the cat sat"
    assert inpaint_answer({"final_list": [{"type": "gap", "text": "cat"}]}, "the ___ sat") == "the cat sat"
    assert denoise_answer(dec, ["a cat sat"]) == ("the cat sat", False)
    runaway = {"final_list": [{"type": "full", "text": "word " * 20}], "pick": "A"}
    assert denoise_answer(runaway, ["a cat sat"]) == ("a cat sat", True)
