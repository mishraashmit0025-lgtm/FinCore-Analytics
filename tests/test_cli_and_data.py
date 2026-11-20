from pathlib import Path

import pytest

from fincore import cli
from fincore.data import market

ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_loads_cleanly():
    md = market.load()
    assert md.prices.shape[1] == 25
    assert md.prices.index.is_monotonic_increasing
    assert md.prices.index.equals(md.benchmark.index)
    r = md.returns
    assert r.isna().sum().sum() == 0
    # Unadjusted splits show up as -50%/-80% days; adjusted data has none.
    assert r.min().min() > -0.30


def test_adv_is_lagged():
    md = market.load().window("2024-01-01", "2024-03-31")
    traded = md.volume * md.prices
    day = md.prices.index[30]
    expected = traded.iloc[10:30].mean()
    assert md.adv_value.loc[day].to_numpy() == pytest.approx(expected.to_numpy())


def test_cli_recon_demo(capsys):
    cli.main(["recon-demo", "--n", "800", "--seed", "3"])
    out = capsys.readouterr().out
    assert "fincore matcher" in out and "naive" in out


def test_cli_value(tmp_path, capsys):
    cli.main(["value", str(ROOT / "examples/valuation/example_industrials.toml"), "--out", str(tmp_path)])
    text = (tmp_path / "valuation_exind.md").read_text(encoding="utf-8")
    assert "per share" in text and "Reverse DCF" in text
