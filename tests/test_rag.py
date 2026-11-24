import pytest

from desk.rag import evaluate
from desk.rag.index import FLOOR, Index, Section, stem, words


def test_every_expected_section_exists(index):
    ids = {s.id for s in index.sections}
    rows = evaluate.questions("dev") + evaluate.questions("test")
    assert {r["section"] for r in rows if r["section"]} <= ids
    assert len(ids) == len(index.sections) == 41


def test_halves_share_no_section():
    a, b = ({r["section"] for r in evaluate.questions(h) if r["section"]} for h in ("dev", "test"))
    assert not a & b


def test_the_floor_in_use_is_the_one_tuned_on_the_development_half(index):
    assert evaluate.tune(index) == pytest.approx(FLOOR)


def test_bm25_prefers_rare_words_and_shorter_sections():
    s = [
        Section("a#1", "A", "x", "kettle kettle water"),
        Section("b#1", "B", "x", "water water"),
        Section("c#1", "C", "x", "kettle " + "filler " * 60),
    ]
    ix = Index(s)
    scores = ix.bm25("kettle")
    assert scores[0] > scores[2] > scores[1] == 0


def test_stemming_and_stop_words():
    assert words("How are the deliveries delayed?") == [stem("deliveries"), stem("delayed")]
    assert stem("batteries") == "battery" and stem("is") == "is"


def test_quote_is_word_for_word(index):
    h = index.search("How do I descale the kettle")[0]
    q = index.quote(h, "How do I descale the kettle", 1)
    assert q in h.section.text


def test_quality_does_not_slip(index):
    r = evaluate.run()["test"]
    assert r["hybrid"]["recall@3"] >= 0.8
    assert r["refuses_unanswerable"] >= 0.75
    assert r["refuses_answerable"] <= 0.3


@pytest.mark.parametrize("q", ["", "???", "zzz qqq", "a" * 300])
def test_odd_questions_do_not_crash(index, q):
    assert len(index.search(q)) == 3
