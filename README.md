# adalineDemo

Hear an ADALINE cancel noise, live: `frugal_ml`'s Rust core (the same `Linear` + `mse` + `Sgd`
that runs on an STM32F446RE) trained one audio sample at a time, with the exact real-time margin
the chip's own benchmark measured.

```
uv sync            # builds the Rust extension (needs a Rust toolchain)
uv run adaline-demo # opens http://localhost:8010
uv run pytest       # 31 tests
```

## What it is

One neuron, no hidden layer (`Linear<TAPS, 1>`), trained online against `mse` with plain `Sgd`.
That configuration alone reproduces the Widrow-Hoff LMS adaptive filter exactly (Widrow & Hoff,
1960) — the classic adaptive-noise-cancellation setup:

- `s[n]`: a 3-tone chord to **keep** (60, 140, 310 Hz)
- `n0[n]`: a fundamental + 2 harmonics to **cancel** (200, 400, 600 Hz, engine-like), plus a
  little jitter
- `x[n]`: a **reference** input — `n0[n]` through an unknown 2-tap coupling path, correlated with
  the noise, never with the signal
- `d[n] = s[n] + n0[n]`: the **primary** input, what a microphone would pick up
- `y[n]`: the ADALINE's noise estimate, learned from `x[n]` alone
- `e[n] = d[n] - y[n]`: the **cancelled** output

Move the sliders (filter length, sample rate, learning rate, clip length), press Run, and listen
to the noisy input next to the cancelled output and the clean signal (only knowable because this
is a simulation). A chart shows correlation with the noise and residual error converging over the
clip.

## The real-time margin

Two cards show the same thing on two devices: the fraction of one sample's time budget (`1/fs`)
the filter actually uses.

- **This computer**: measured live, timed in 256-sample batches (median shown, 5th–95th
  percentile available).
- **STM32 chip**: a projection. `frugal_ml`'s `docs/BENCHMARKS.md` measured **575 cycles**
  (≈ 3.42 µs at 168 MHz) for one forward + backward + `Sgd` update of exactly this filter at 16
  taps, on real hardware (DWT cycle counter, read over ST-Link) — the one number this project is
  built to reproduce (margin 36.5× at 8 kHz, matching the docs exactly). Other tap counts scale
  that number linearly, which isn't a curve-fit convenience: with a single output, the forward
  pass, the backward pass and the update are each one O(taps) pass over the window, with no
  O(taps²) term anywhere. Only 16 taps has actually run on the board; the page marks every other
  value "projected."

Push the sliders far enough (128 taps, 96 kHz) and the chip's margin drops under 1×: the filter
would no longer keep up with the incoming audio. `tests/test_chip.py::test_the_wall_is_reachable`
checks that this crossing exists within the ladders offered.

## Correctness

The Rust path is cross-checked against an independent, plain-Python reimplementation of the same
update rule (`src/adalineDemo/reference.py`), same PRNG, same coupling path, agreement to float32
rounding (`tests/test_native.py`). The Python reference is a sequential loop (LMS carries state
from one sample to the next, so it can't be vectorised) and is only run in tests, never from the
live page.

## Layout

| | |
|---|---|
| `rust/src/anc.rs` | The LMS loop itself, `Linear`/`mse`/`Sgd` from `frugal_ml`, unchanged from the board. `register_taps!` compiles a fixed ladder of filter lengths. |
| `src/adalineDemo/chip.py` | The projection model, the tap/sample-rate/duration ladders. |
| `src/adalineDemo/reference.py` | The plain-Python LMS reference used only by the tests. |
| `src/adalineDemo/wav.py` | float32 → 16-bit PCM WAV, standard library only. |
| `src/adalineDemo/demo/` | The server (`server.py`), the job (`runner.py`), the page (`static/`). |

## On the network

`uv run adaline-demo --lan` opens the demo to your local network (e.g. a phone on the same
Wi-Fi); the printed URL is the one to use there. `--https` adds a self-signed certificate for
browsers (Safari's "HTTPS-Only") that refuse plain `http://` links — the browser will warn once
about the certificate being untrusted; that's expected, choose to visit the site anyway. Without
`--lan`, the server only accepts `localhost`.

## Limits

- Filter length is compiled ahead into a ladder (`rust/src/anc.rs`'s `register_taps!`); anything
  else raises an error naming the line to add.
- Frequencies and amplitudes of the signal and noise are fixed, matching the on-chip benchmark
  exactly; only filter length, sample rate, learning rate and clip length are adjustable.
- The chip projection is a straight line through one measured point, not a sweep — see "The
  real-time margin" above for why that's still a structurally grounded claim, not a guess.
