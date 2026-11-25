import json

import rml.reader.vsr_multilang as V
from rml.eval import tables as T
from rml.pipeline import Pipeline


class _P:
    pass


def _root(tmp_path, rows, **kw):
    (tmp_path / "metadata.json").write_text(json.dumps({"clips": rows}))
    return Pipeline(tmp_path, _P(), **kw)


def _decodes(p, rows):
    p.jsave("lm_rows.json", [{"id": r["id"], "context": "hola ___"} for r in rows])
    p.jsave("lm_readers.json", {r["id"]: ["buenos dias"] for r in rows})
    p.jsave("lm_decode.json", {"rows": [{"id": r["id"], "pick": "A", "final_list": [{"type": "full", "text": "Hola  Mundo"}]}
                                        for r in rows]})


def test_usr2_reader_renders_r_to_g_with_the_main_lora(tmp_path):
    rows = [{"id": "w1__k2", "clip": "clips/w1.mp4", "base_id": "w1", "task": "inpaint", "mask_segments_s": [[1.0, 1.5]]}]
    p = _root(tmp_path, rows, reader="usr2")
    p.jsave("reader_vsr.json", {"rows": [{"id": "w1", "nbest": ["hello world", "x"]}]})
    _decodes(p, rows)
    p.texts()
    assert p.jload("texts.json")["inp_lip"]["w1__k2"] == "<inpaint>\nTranscript: hello world"
    assert p.lora("inp_lip") == "g_main_lora" and p.lora("den_av") == "g_restoration_lora"
    assert _root(tmp_path, rows).lora("inp_lip") == "g_cascade_lora"


def test_multilingual_recipe_renders_r_to_l_to_g_only(tmp_path):
    rows = [{"id": "w1__k1", "clip": "clips/w1.mp4", "base_id": "w1", "language": "es", "task": "inpaint",
             "mask_segments_s": [[1.0, 1.5]]}]
    p = _root(tmp_path, rows, recipe="multilingual", reader="vsr_multilang")
    p.jsave("reader_vsr.json", {"rows": [{"id": "w1", "nbest": ["hola mundo"]}]})
    _decodes(p, rows)
    p.texts()
    T_ = p.jload("texts.json")
    assert T_["inp_gemma"] == {"w1__k1": "<inpaint>\nTranscript: Hola Mundo"}
    assert not any(v for k, v in T_.items() if k != "inp_gemma")
    assert p.jload("text_voters.json")["w1__k1"] == {"text|lip_text": "hola mundo", "text|gemma": "Hola Mundo"}


def test_multilingual_reader_per_language(monkeypatch):
    seen = []

    class Fake:
        def __init__(self, vsr_dir, lang, model_dir, lm_dir, device="cuda"):
            seen.append((lang, model_dir))
            self.lang = lang

        def nbest(self, roi, nbest=10):
            return None if roi == "missing" else [f"{self.lang}:{roi}"]
    monkeypatch.setattr(V, "MultilingualVSR", Fake)
    rows = [{"id": "a", "roi": "r1", "language": "fr"}, {"id": "b", "roi": "r2", "language": "es"},
            {"id": "c", "roi": "missing", "language": "es"}]
    out = V.read_rows(rows, "vsr", {"es": ("m_es", "lm_es"), "fr": ("m_fr", "lm_fr")})
    assert seen == [("es", "m_es"), ("fr", "m_fr")]
    assert out == [{"id": "b", "nbest": ["es:r2"]}, {"id": "a", "nbest": ["fr:r1"]}]


def test_spanish_french_table():
    ww = [["uno", 0.0, 0.3], ["dos", 0.4, 0.7], ["tres", 0.8, 1.1]]
    man = {f"c{i}": {"id": f"c{i}", "video": f"v{i % 2}", "language": "es" if i < 3 else "fr", "asr_verified": i != 4,
                     "window_words": ww, "masked_words": ["dos"], "n_masked_words": 1, "span_start_index": 1,
                     "sentence": "uno dos tres", "gt_review": None} for i in range(5)}
    res = {f"c{i}": {"hyp": {}, "raw": {"text|lip_text": "uno tos tres", "text|gemma": "uno dos tres",
                                        "gemma|keep": "uno dos tres" if i % 2 else "uno tres"}} for i in range(5)}
    for sc in (T.score_full, T.word_by_word_scorer()):
        per = T.fixed_scores(res, man, "g_exact", sc)
        assert len(per) == 4
        txt = T.table_languages(per)
        assert txt.splitlines()[1].split()[-3:] == ["0.0", "0.0", "0.0"]
        assert txt.splitlines()[2].split()[-3:] == ["100.0", "100.0", "100.0"]
        assert txt.splitlines()[3].split()[5:8] == ["33.3", "100.0", "50.0"]
