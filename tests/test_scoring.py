from rml.eval.gap_score import score_full, score_gap_only
from rml.eval.scoring import contains_norm, eq_norm, norm_tokens, wer


def test_normaliser_numbers_contractions_spelling():
    assert norm_tokens("Twenty four seven") == norm_tokens("24 seven")
    assert norm_tokens("we're here") == norm_tokens("we are here")
    assert norm_tokens("the organisation") == norm_tokens("the organization")
    assert norm_tokens("well...") == ["well"]


def test_wer_and_equality():
    assert wer(["a", "b", "c"], ["a", "x", "c"]) == 1 / 3
    assert wer(["a"], []) == 1.0
    assert eq_norm(["story", "teller"], ["storyteller"])
    assert contains_norm(["i", "said", "hello", "there"], ["hello", "there"])


def cut(words, start, k):
    return {"language": "en", "window_words": [{"word": w} for w in words], "n_masked_words": k,
            "span_start_index": start, "masked_words": words[start:start + k]}


def test_exact_span_recovery():
    c = cut(["so", "i", "went", "to", "the", "market", "yesterday", "morning"], 4, 2)    # gap: "the market"
    assert score_full(c, "so i went to the market yesterday morning")["g_exact"] == 1.0
    v = score_full(c, "so i went to a market yesterday morning")
    assert v["g_exact"] == 0.0 and v["g_acc"] == 0.5
    # errors outside the gap do not count against it
    assert score_full(c, "so we went to the market yesterday")["g_exact"] == 1.0
    # an insertion inside the gap does
    assert score_full(c, "so i went to the big market yesterday morning")["g_exact"] == 0.0
    assert score_gap_only(c, "the market")["g_exact"] == 1.0


def test_stutter_is_optional():
    c = cut(["i", "i", "think", "that", "is", "right"], 3, 2)                         # gap: "that is"
    assert score_full(c, "i think that is right")["g_exact"] == 1.0
