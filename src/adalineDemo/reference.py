"""An independent reimplementation of the same LMS update rule, in plain Python.

Used only by the test suite, as the trust device: `frugal_ml`'s `Linear` + `mse` + `Sgd` is a
generic autodiff path, this is the closed-form LMS update (`w += lr * error * x`) it is mathematically
equivalent to (Widrow & Hoff, 1960). Agreement between the two, to float32 rounding, is evidence the
Rust binding (window shift, coupling path, loss gradient, `Sgd.step`) matches the algorithm, not just
that it matches itself. It is a plain Python loop (LMS is sequential, state carries from one sample to
the next), so it is only run on short clips in tests, never from the live page.
"""

import math

from adalineDemo.chip import SAMPLE_RATES  # noqa: F401  (re-exported for convenience in tests)

F_S = (60.0, 140.0, 310.0)
A_S = (0.30, 0.15, 0.10)
F_N = (200.0, 400.0, 600.0)
A_N = (0.50, 0.25, 0.15)
NOISE_JITTER = 0.03
COUPLING_DIRECT = 0.7
COUPLING_DELAYED = 0.3


def xorshift(state: int) -> tuple[int, float]:
    """The exact xorshift64 both this reference and the Rust side use, for the same noise samples
    given the same seed: bit-for-bit reproducible pseudo-randomness, not just "a" random generator."""
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x, (x / 0xFFFFFFFFFFFFFFFF) * 2.0 - 1.0


def run(taps: int, fs: float, lr: float, n: int, seed: int = 424242) -> dict[str, list[float]]:
    weights = [0.0] * taps
    bias = 0.0
    window = [0.0] * taps
    rng = seed or 1
    n0_prev = 0.0
    out = {"s_true": [], "n0": [], "d": [], "y": [], "e": []}

    for i in range(n):
        t = i / fs
        s_true = sum(a * math.sin(2 * math.pi * f * t) for f, a in zip(F_S, A_S))
        noise = sum(a * math.sin(2 * math.pi * f * t) for f, a in zip(F_N, A_N))
        rng, jitter = xorshift(rng)
        noise += NOISE_JITTER * jitter
        reference = COUPLING_DIRECT * noise + COUPLING_DELAYED * n0_prev
        n0_prev = noise
        primary = s_true + noise

        window = [reference] + window[:-1]
        y = sum(w * x for w, x in zip(weights, window)) + bias

        # d/dy of mse(y, primary) for a length-1 output is 2*(y - primary): the LMS instantaneous
        # error gradient, same rule frugal_ml's `mse` + `Sgd.step` compute generically.
        grad = 2.0 * (y - primary)
        weights = [w - lr * grad * x for w, x in zip(weights, window)]
        bias -= lr * grad

        out["s_true"].append(s_true)
        out["n0"].append(noise)
        out["d"].append(primary)
        out["y"].append(y)
        out["e"].append(primary - y)
    return out
