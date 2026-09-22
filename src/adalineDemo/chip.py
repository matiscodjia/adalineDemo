"""What the STM32F446RE would do, and how sure we are.

Two kinds of numbers, kept apart everywhere they're shown:

- **Measured on the chip**: one data point. `docs/BENCHMARKS.md` in `frugal_ml` reports
  575 cycles (≈ 3.42 µs at 168 MHz) for one full forward + backward + `Sgd` update of this exact
  `Linear<16, 1>` ADALINE — cycles read from the DWT counter, over an ST-Link, on real hardware.
- **Projected**: every other tap count. Unlike `directConv`'s conv kernel (a sweep of real
  measurements across shapes), there is only this one point for the ADALINE. The projection is a
  straight line through it, but it isn't a curve-fit convenience: because the output is a single
  scalar (`Linear<TAPS, 1>`), the forward pass, the backward pass and the SGD update are each one
  O(TAPS) pass over the window, with no O(TAPS²) term anywhere — the cost is structurally linear
  in `TAPS`, and 16 taps is the one point pinning that line down. Nothing above or below 16 taps
  has been run on the board.
"""

from dataclasses import dataclass

CHIP_NAME = "STM32F446RE (Cortex-M4F, 168 MHz)"
CHIP_CLOCK_HZ = 168_000_000.0
VERIFIED_TAPS = 16
VERIFIED_CYCLES = 575  # docs/BENCHMARKS.md: "cost per forward+backward+update"

# Filter lengths `frugal_ml`'s `Linear<TAPS, 1>` is compiled for (rust/src/anc.rs's register_taps!).
TAPS_LADDER = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

# Sample rates worth trying, low to high: from a slow sensor to studio audio. Frequencies in the
# signal/noise are fixed (below 1 kHz), so all of these are comfortably above Nyquist.
SAMPLE_RATES = [
    {"hz": 4000, "label": "4 kHz", "note": "a slow sensor channel"},
    {"hz": 8000, "label": "8 kHz", "note": "phone quality — the benchmark's own rate"},
    {"hz": 16000, "label": "16 kHz", "note": "wideband voice (VoIP)"},
    {"hz": 44100, "label": "44.1 kHz", "note": "CD quality"},
    {"hz": 48000, "label": "48 kHz", "note": "video / broadcast audio"},
    {"hz": 96000, "label": "96 kHz", "note": "studio / pro audio"},
]

DURATIONS_S = [2, 4, 8, 16, 30]
DEFAULT_DURATION_S = 6  # not in the ladder on purpose: see runner.snap_duration


@dataclass(frozen=True)
class Projection:
    cycles: float
    seconds_per_sample: float
    verified: bool  # False beyond the one point actually run on the chip


def project(taps: int) -> Projection:
    cycles = VERIFIED_CYCLES * taps / VERIFIED_TAPS
    return Projection(cycles=cycles, seconds_per_sample=cycles / CHIP_CLOCK_HZ, verified=taps == VERIFIED_TAPS)


def margin(taps: int, fs: float) -> float:
    """Real-time budget (1/fs) divided by the chip's projected time per sample. >1 means it keeps up."""
    return (1.0 / fs) / project(taps).seconds_per_sample


def host_margin(seconds_per_sample: float, fs: float) -> float:
    return (1.0 / fs) / seconds_per_sample
