import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import softmax

from desk.ml import data, drift, metrics
from desk.ml.baseline import Baseline
from desk.ml.registry import ROOT, Gate, Registry

Y = metrics.truth(data.test_set())


def test_training_validation_and_test_do_not_overlap():
    tr, va, te = (
        ({e.text.lower() for e in s})
        for s in (data.train_set(), data.validation_set(), data.test_set())
    )
    assert not (tr & va or tr & te or va & te)
    assert len(tr) > 1400 and len(te) == 120


def test_data_is_reproducible_and_fingerprint_notices_a_change():
    a, b = data.train_set(), data.train_set()
    assert a == b and data.fingerprint(a) == data.fingerprint(b)
    changed = [data.Example(a[0].text + "!", a[0].intent), *a[1:]]
    assert data.fingerprint(changed) != data.fingerprint(a)


def test_classes_are_balanced():
    for s, n in ((data.train_set(), 150), (data.validation_set(), 30), (data.test_set(), 12)):
        assert set(np.bincount(metrics.truth(s))) == {n}


def test_baseline_learns():
    s = Baseline().train(data.train_set()).scores([e.text for e in data.validation_set()])
    assert (s.argmax(1) == metrics.truth(data.validation_set())).mean() > 0.35


def test_perfect_scores_give_perfect_metrics():
    m = metrics.evaluate(np.eye(10)[Y] * 50, Y)
    assert m["accuracy"] == m["macro_f1"] == 1 and m["ece"] < 1e-6
    assert all(p["accuracy"] == 1 for p in m["selective"])


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 4000)
    true = rng.normal(size=(4000, 10)) + 1.5 * np.eye(10)[y]
    loud = true * 6
    t = metrics.fit_temperature(loud, y)
    assert t > 2
    assert metrics.ece(softmax(loud / t, 1), y) < metrics.ece(softmax(loud, 1), y) / 3


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 5000), st.floats(0.1, 5))
def test_temperature_never_changes_the_answer(seed, t):
    s = np.random.default_rng(seed).normal(size=(30, 10))
    assert (softmax(s, 1).argmax(1) == softmax(s / t, 1).argmax(1)).all()


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    p = np.random.default_rng(1).integers(0, 10, 120)
    assert metrics.macro_f1(Y, p) == pytest.approx(f1_score(Y, p, average="macro"))


# --- registry and gate ---------------------------------------------------------------------------


def scores_with(right: int, seed: int = 0, strength: float = 4.0):
    rng = np.random.default_rng(seed)
    s = rng.normal(0, 0.1, (120, 10))
    ok = np.zeros(120, bool)
    ok[rng.permutation(120)[:right]] = True
    s[np.arange(120), np.where(ok, Y, (Y + 1) % 10)] += strength
    return s


def submit(reg, name, s, test="t1"):
    m = metrics.evaluate(s, Y, metrics.fit_temperature(s, Y))
    reg.register(name, {"test_data": test, "metrics": m}, s)
    return reg.promote(name, Y, Gate(max_intent_drop=1.0))


def test_better_model_is_promoted_and_can_be_rolled_back(tmp_path):
    reg = Registry(tmp_path)
    assert submit(reg, "v1", scores_with(60))["passed"]
    assert submit(reg, "v2", scores_with(100, 1))["passed"]
    assert reg.stages()["champion"] == "v2"
    assert reg.rollback() == "v1"
    with pytest.raises(ValueError):
        reg.rollback()


def test_a_lucky_difference_is_not_promoted(tmp_path):
    reg = Registry(tmp_path)
    submit(reg, "v1", scores_with(90))
    r = submit(reg, "v2", scores_with(93, 1))
    assert not r["passed"] and reg.stages()["champion"] == "v1"


def test_a_model_that_ruins_one_intent_is_not_promoted():
    old = {"per_intent": {k: 0.6 for k in data.LABELS}}
    new = {"per_intent": {k: 1.0 for k in data.LABELS} | {"late": 0.2}, "ece": 0.02}
    r = Gate().check(Y, old, new, np.where(np.arange(120) % 2, Y, 0), Y)
    assert not r["passed"] and r["worst_intent"]["intent"] == "late"


def test_dishonest_confidence_is_not_promoted():
    old = {"per_intent": {k: 0.2 for k in data.LABELS}}
    new = {"per_intent": {k: 1.0 for k in data.LABELS}, "ece": 0.3}
    assert not Gate().check(Y, old, new, (Y + 1) % 10, Y)["passed"]


def test_versions_are_immutable_and_tests_must_match(tmp_path):
    reg = Registry(tmp_path)
    submit(reg, "v1", scores_with(60))
    with pytest.raises(FileExistsError):
        submit(reg, "v1", scores_with(70))
    with pytest.raises(ValueError):
        submit(reg, "v2", scores_with(100), test="another test set")


# --- drift ----------------------------------------------------------------------------------------


def test_psi_is_zero_for_the_same_mix_and_grows():
    a = np.array([10.0, 20, 70])
    assert drift.psi(a, a * 3) == pytest.approx(0)
    assert drift.psi(a, np.array([30.0, 30, 40])) < drift.psi(a, np.array([80.0, 15, 5]))


def test_monitor_is_quiet_on_the_same_mail_and_loud_on_new_mail():
    val = data.validation_set()
    b = Baseline().train(data.train_set())
    texts = [e.text for e in val]
    s = b.scores(texts)
    mon = drift.Monitor(s[::2], texts[::2])
    assert mon.check(s[1::2], texts[1::2])["state"] != "alarm"
    flood = [e.text for e in val if e.intent == "late"] * 3
    assert mon.check(b.scores(flood), flood)["state"] == "alarm"


# --- what is committed ----------------------------------------------------------------------------


def test_committed_registry_is_consistent():
    reg = Registry(ROOT)
    assert reg.stages()["champion"] in reg.versions()
    for v in reg.versions():
        m = reg.manifest(v)
        assert m["test_data"] == data.fingerprint(data.test_set())
        assert m["train_data"] == data.fingerprint(data.train_set())
        s = np.load(ROOT / v / "test_scores.npy")
        assert (s.argmax(1) == Y).mean() == pytest.approx(m["metrics"]["accuracy"])


def test_baseline_in_registry_can_be_rebuilt():
    s = Baseline().train(data.train_set()).scores([e.text for e in data.test_set()])
    want = Registry(ROOT).manifest("v1-tfidf")["metrics"]["accuracy"]
    assert (s.argmax(1) == Y).mean() == pytest.approx(want, abs=0.02)


def test_gate_reports_are_recorded():
    for v in Registry(ROOT).versions():
        assert "checks" in json.loads((ROOT / v / "gate.json").read_text())
