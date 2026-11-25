import json

from rml.pipeline import Pipeline, add_muted_twins


class _P:
    pass


def test_configuration_texts(tmp_path):
    rows = add_muted_twins([
        {"id": "w1__k2", "clip": "clips/w1.mp4", "base_id": "w1", "task": "inpaint", "mask_segments_s": [[1.0, 1.5]]},
        {"id": "w2__babble-10", "clip": "clips/w2.mp4", "base_id": "w2", "task": "denoise", "mask_segments_s": []}])
    (tmp_path / "metadata.json").write_text(json.dumps({"clips": rows}))
    p = Pipeline(tmp_path, _P())
    p.jsave("reader_vsr.json", {"rows": [{"id": "w1", "nbest": ["Hello, world!", "x"]}, {"id": "w2", "nbest": ["good day"]}]})
    p.jsave("lm_rows.json", [{"id": r["id"], "context": "hello ___"} for r in rows])
    p.jsave("lm_readers.json", {r["id"]: ["good day"] for r in rows})
    p.jsave("lm_decode.json", {"rows": [{"id": r["id"], "pick": "A", "final_list": [{"type": "full", "text": "Hello  World"}]}
                                        for r in rows]})
    p.texts()
    T = p.jload("texts.json")
    assert T["inp_gemma"]["w1__k2"] == "<inpaint>\nTranscript: Hello World"
    assert T["inp_lip"]["w1__k2"] == "Transcript: Hello, world!"
    assert T["inp_av"]["w1__k2"] == "<inpaint>"
    assert T["den_gemma"]["w2__babble-10"] == "<denoise>\nTranscript: hello world"
    assert T["den_lip"]["w2__babble-10"] == "<denoise>\nTranscript: good day"
    assert T["mute_lip"]["w1__full_silence"] == "<inpaint>\nTranscript: hello world"
    assert set(T["mute_av"]) == {"w1__full_silence", "w2__full_silence"}
    assert [r["id"] for r in p.evaluated()] == ["w1__k2", "w2__babble-10"]
