import math

from rml.selection.rule import SIX, TEXTS, dist, evidence, mbr, select


def entry(hyp, ev):
    """hyp: {key: transcript}; ev: {transcript: (audio PMI, lip PMI)}"""
    return {"hyp": hyp, "s": {t: {"hear": a, "hear0": 0.0, "vis": v, "vis0": 0.0, "lm": -1.0} for t, (a, v) in ev.items()}}


def test_dist_is_symmetric_mean_of_wers():
    assert dist("a b c", "a b c") == 0.0
    assert dist("", "") == 0.0
    assert dist("a b", "") == 1.0
    assert math.isclose(dist("a b c d", "a b"), 0.5 * 0.5 + 0.5 * 1.0)


def test_evidence_sums_audio_and_lip_pmi():
    e = entry({"x|keep": "hello there"}, {"hello there": (3.0, 2.0)})
    assert evidence(e, "hello there") == 5.0
    e["s"]["hello there"]["vis"] = None
    assert evidence(e, "hello there") == 3.0
    assert evidence(e, "") is None


def test_consensus_wins():
    good, bad = "the cat sat on the mat", "a dog ran in the park"
    hyp = {"gemma|keep": good, "lip_text|keep": good, "av_only|keep": bad, "gemma|mute": good,
           "lip_text|mute": bad, "av_only|mute": "", "text|lip_text": bad, "text|gemma": good}
    e = entry(hyp, {good: (10.0, 5.0), bad: (1.0, 1.0)})
    assert select(e) == "gemma|keep"


def test_empty_output_never_selected_and_tie_break_order():
    hyp = dict.fromkeys(SIX, "same words")
    hyp["av_only|mute"] = ""
    e = entry(hyp, {"same words": (1.0, 0.0)})
    assert select(e) == SIX[0]      # every risk is 0: the first configuration in tie-break order


def test_nested_voters_drop_reader_lines_when_l_votes():
    a, b = "alpha beta gamma", "delta epsilon zeta"
    # the reader's text and R -> G muted both say b; without nesting they would outvote a
    hyp = {"gemma|keep": a, "lip_text|keep": b, "text|gemma": a, "text|lip_text": b, "lip_text|mute": b}
    e = entry(hyp, {a: (1.0, 0.0), b: (1.0, 0.0)})
    sel = ["gemma|keep", "lip_text|keep"]
    voters = [*sel, "lip_text|mute", *TEXTS]
    assert mbr(e, sel, voters) == "gemma|keep"
    # without L's line the reader's lines keep their votes
    del e["hyp"]["text|gemma"]
    assert mbr(e, sel, voters) == "lip_text|keep"


def test_temperature_weights_by_evidence():
    a, b = "one two three", "four five six"
    hyp = {"gemma|keep": a, "lip_text|keep": b, "av_only|keep": b}
    e = entry(hyp, {a: (500.0, 0.0), b: (0.0, 0.0)})
    sel = list(hyp)
    assert mbr(e, sel, sel, T=100.0) == "gemma|keep"          # a's weight exp(5) beats two votes for b
    assert mbr(e, sel, sel, T=float("inf")) == "lip_text|keep"  # uniform weights: majority
