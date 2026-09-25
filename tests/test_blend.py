import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gold_edge.learning.blend import fit_blend, format_blend, walk_forward_blend
from gold_edge.learning.market_study import BookQuote, StudyRow

D0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def row(
    ticker: str, day: int, fair: float, mid: float, outcome: float, minute: int = 5
) -> StudyRow:
    t = D0 + timedelta(days=day, minutes=minute)
    bid, ask = Decimal(str(round(mid - 0.01, 2))), Decimal(str(round(mid + 0.01, 2)))
    return StudyRow(ticker, minute, fair, BookQuote(bid, ask, 1 - ask, 1 - bid, t), outcome)


def synthetic(n_days: int, per_day: int, informative_model: bool, seed: int = 0) -> list[StudyRow]:
    rng = random.Random(seed)
    rows = []
    for d in range(n_days):
        for i in range(per_day):
            truth = rng.random()
            outcome = 1.0 if rng.random() < truth else 0.0
            noisy = min(max(truth + rng.gauss(0, 0.05), 0.02), 0.98)
            mid = 0.5 if informative_model else noisy
            fair = noisy if informative_model else rng.random() * 0.9 + 0.05
            rows.append(row(f"W{d}-{i}", d, fair, mid, outcome))
    return rows


def test_blend_puts_weight_on_the_model_when_only_the_model_is_informative():
    model = fit_blend(synthetic(10, 60, informative_model=True), l2=0.1)
    assert model.weights[2] > 0.5


def test_blend_defaults_to_trusting_the_market_when_the_model_is_noise():
    model = fit_blend(synthetic(10, 60, informative_model=False), l2=1.0)
    assert model.weights[1] > 0.5
    assert abs(model.weights[2]) < 0.5


def test_walk_forward_scores_only_days_after_enough_training_and_never_the_first():
    rows = synthetic(8, 50, informative_model=True)
    wf = walk_forward_blend(rows, min_train_windows=100, n_boot=100)
    assert wf is not None
    assert wf.n_days_scored == 6  # days 0,1 supply 100 windows; days 2..7 are scored
    scored_days = {r.quote.receive_time.date() for r in wf.scored_rows}
    assert (D0 + timedelta(days=0)).date() not in scored_days
    assert (D0 + timedelta(days=1)).date() not in scored_days


def test_informative_model_beats_a_flat_market_out_of_sample():
    wf = walk_forward_blend(synthetic(12, 80, informative_model=True), 200, n_boot=300)
    assert wf is not None and wf.diff_brier > 0 and wf.ci_low > 0


def test_pure_noise_model_does_not_beat_the_market():
    wf = walk_forward_blend(synthetic(12, 80, informative_model=False), 200, n_boot=300)
    assert wf is not None and wf.ci_low <= 0.005


def test_not_enough_history_returns_none_and_formats_cleanly():
    wf = walk_forward_blend(synthetic(2, 5, informative_model=True), 200)
    assert wf is None
    assert "not enough history" in format_blend(wf)
