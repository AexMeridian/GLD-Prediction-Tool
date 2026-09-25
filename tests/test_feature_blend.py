import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.learning.feature_blend import (
    BASE,
    feature_value,
    fit_and_score,
    fit_feature_blend,
    paired_diff,
    split_dev_holdout,
    walk_forward,
)
from gold_edge.learning.market_study import BookQuote, StudyRow

D0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def row(i: int, day: int, fair: float, mid: float, outcome: float, extra=0.0) -> StudyRow:
    t = D0 + timedelta(days=day, minutes=5)
    bid, ask = Decimal(str(round(mid - 0.01, 2))), Decimal(str(round(mid + 0.01, 2)))
    return StudyRow(
        f"W{day}-{i}", 5, fair, BookQuote(bid, ask, 1 - ask, 1 - bid, t), outcome, {"mom1": extra}
    )


def synth(n_days=10, per_day=60, extra_informative=False, seed=0) -> list[StudyRow]:
    rng = random.Random(seed)
    out = []
    for d in range(n_days):
        for i in range(per_day):
            truth = rng.random()
            outcome = 1.0 if rng.random() < truth else 0.0
            mid = min(max(truth + rng.gauss(0, 0.15), 0.03), 0.97)
            hint = (truth - 0.5) * 2 + rng.gauss(0, 0.2)
            fair = rng.random() * 0.9 + 0.05
            out.append(row(i, d, fair, mid, outcome, hint if extra_informative else 0.0))
    return out


def test_base_features_reproduce_the_two_input_blend_and_trust_the_market_on_noise():
    rows = synth(extra_informative=False)
    model = fit_feature_blend(rows, BASE)
    assert model.names == BASE
    assert model.weights[1] > 0.5  # weight on logit(market)
    assert abs(model.weights[2]) < 0.3  # model column is noise here


def test_informative_extra_gets_weight_and_improves_out_of_sample():
    rows = synth(n_days=12, extra_informative=True)
    base = walk_forward(rows, BASE, min_train_windows=100, n_boot=200)
    plus = walk_forward(rows, (*BASE, "mom1"), min_train_windows=100, n_boot=200)
    assert base is not None and plus is not None
    assert plus.brier < base.brier
    diff, lo, _ = paired_diff(base, plus, n_boot=200)
    assert diff > 0 and lo > 0


def test_useless_extra_does_not_beat_base_reliably():
    rows = synth(n_days=12, extra_informative=False)
    base = walk_forward(rows, BASE, min_train_windows=100, n_boot=200)
    plus = walk_forward(rows, (*BASE, "mom1"), min_train_windows=100, n_boot=200)
    _, lo, _ = paired_diff(base, plus, n_boot=200)
    assert lo <= 0.001


def test_walk_forward_never_scores_days_before_enough_training():
    rows = synth(n_days=6, per_day=50)
    res = walk_forward(rows, BASE, min_train_windows=100, n_boot=100)
    assert res is not None
    days = {r.quote.receive_time.date() for r in res.rows}
    assert (D0 + timedelta(days=0)).date() not in days
    assert (D0 + timedelta(days=1)).date() not in days


def test_split_holds_out_the_most_recent_days_only():
    rows = synth(n_days=6, per_day=10)
    dev, hold = split_dev_holdout(rows, 2)
    assert max(r.quote.receive_time for r in dev) < min(r.quote.receive_time for r in hold)
    assert len({r.quote.receive_time.date() for r in hold}) == 2
    assert len(dev) + len(hold) == len(rows)


def test_fit_and_score_uses_only_dev_for_fitting():
    rows = synth(n_days=8, per_day=40)
    dev, hold = split_dev_holdout(rows, 2)
    res = fit_and_score(dev, hold, BASE, n_boot=100)
    assert res is not None and res.n_rows == len(hold)


def test_feature_value_builtin_and_extra_lookup():
    r = row(0, 0, 0.6, 0.5, 1.0, extra=0.25)
    assert feature_value(r, "mom1") == 0.25
    assert feature_value(r, "missing") == 0.0
    assert abs(feature_value(r, "spread") - 0.02) < 1e-9
    assert abs(feature_value(r, "logit_mid")) < 1e-6
