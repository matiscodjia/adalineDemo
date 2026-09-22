"""The chip projection model: exact at the one verified point, linear and structurally justified
elsewhere (see `adalineDemo.chip`'s module docstring)."""

import pytest

from adalineDemo import chip


def test_verified_point_reproduces_the_documented_margin():
    # docs/BENCHMARKS.md: 575 cycles per forward+backward+update, real-time budget at 8 kHz is
    # 125 µs, margin 36.5x — the one number this whole project is built to reproduce exactly.
    p = chip.project(16)
    assert p.cycles == 575 and p.verified
    assert p.seconds_per_sample * chip.CHIP_CLOCK_HZ == pytest.approx(575, rel=1e-6)
    assert chip.margin(16, 8000) == pytest.approx(36.5, rel=0.01)


def test_projection_is_linear_in_taps():
    p8, p16, p32 = chip.project(8), chip.project(16), chip.project(32)
    assert p16.cycles == pytest.approx(2 * p8.cycles) == pytest.approx(p32.cycles / 2)
    assert p8.verified is False and p32.verified is False


def test_margin_shrinks_with_more_taps_or_a_higher_sample_rate():
    assert chip.margin(64, 8000) < chip.margin(16, 8000)
    assert chip.margin(16, 96000) < chip.margin(16, 8000)


def test_the_wall_is_reachable_within_the_ladders():
    # At the extreme of both ladders the chip should fall behind real time: the demo's whole point
    # (a margin that can flip) would be missing if every combination always fit comfortably.
    worst = chip.margin(max(chip.TAPS_LADDER), max(r["hz"] for r in chip.SAMPLE_RATES))
    best = chip.margin(min(chip.TAPS_LADDER), min(r["hz"] for r in chip.SAMPLE_RATES))
    assert worst < 1 < best


def test_host_margin():
    assert chip.host_margin(seconds_per_sample=1e-6, fs=8000) == pytest.approx(125.0)
