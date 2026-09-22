"""The demo's one job: run the ADALINE, measure it, describe it, render it as audio.

What is measured live is measured on this machine. What is said about the STM32F446RE is a
projection from the one point `frugal_ml`'s docs measured on the real board (see
`adalineDemo.chip`); the two are never mixed in one figure.
"""

import uuid
from dataclasses import asdict

import numpy as np

from adalineDemo import _native, chip, wav
from adalineDemo.chip import DURATIONS_S, SAMPLE_RATES, TAPS_LADDER

LR_MIN, LR_MAX = 0.0005, 0.05
DEFAULT_LR = 0.004
CONVERGENCE_POINTS = 48  # samples of the convergence curve shown on the page
# Fixed headroom (not a per-run normalisation) above the signals' theoretical peak (sum of the
# tone amplitudes + jitter, about 1.48): a louder or noisier run should sound louder, not be
# silently rescaled back to the same volume every time.
PLAYBACK_SCALE = 1.0 / 1.6


def _fs_values() -> list[int]:
    return [r["hz"] for r in SAMPLE_RATES]


def validate(taps: int, fs: int, lr: float, duration_s: float) -> None:
    if taps not in TAPS_LADDER:
        raise ValueError(f"taps must be one of {TAPS_LADDER}")
    if fs not in _fs_values():
        raise ValueError(f"fs must be one of {_fs_values()}")
    if not LR_MIN <= lr <= LR_MAX:
        raise ValueError(f"lr must be between {LR_MIN} and {LR_MAX}")
    if not 0.5 <= duration_s <= max(DURATIONS_S):
        raise ValueError(f"duration_s must be between 0.5 and {max(DURATIONS_S)}")


def _stats(seconds_per_sample: np.ndarray) -> dict:
    ms = 1e6 * seconds_per_sample  # microseconds/sample: cycle-scale numbers read better in us than ms
    return {
        "us_per_sample": float(np.median(ms)),
        "p5_us": float(np.percentile(ms, 5)),
        "p95_us": float(np.percentile(ms, 95)),
        "batches": len(seconds_per_sample),
    }


def _convergence(n0: np.ndarray, y: np.ndarray, s: np.ndarray, e: np.ndarray, fs: float, taps: int) -> dict:
    """Correlation(y, n0) and RMS(e - s) over sliding windows, the same two curves the original
    on-chip benchmark's plot showed as two static snapshots (`early` vs `converged`), animated here."""
    n = len(n0)
    window = max(taps * 8, n // CONVERGENCE_POINTS, 8)
    starts = np.linspace(0, max(0, n - window), min(CONVERGENCE_POINTS, max(2, n // window)), dtype=int)
    t, corr, rms = [], [], []
    for start in starts:
        sl = slice(start, start + window)
        yw, n0w = y[sl], n0[sl]
        c = float(np.corrcoef(yw, n0w)[0, 1]) if np.std(yw) > 1e-9 and np.std(n0w) > 1e-9 else 0.0
        t.append((start + window / 2) / fs)
        corr.append(c)
        rms.append(float(np.sqrt(np.mean((e[sl] - s[sl]) ** 2))))
    return {"t_s": t, "corr": corr, "rms": rms}


def run(taps: int, fs: int, lr: float, duration_s: float, out_dir: str, progress=lambda stage: None) -> dict:
    validate(taps, fs, lr, duration_s)
    n = int(round(duration_s * fs))

    progress(f"Training the ADALINE on {n:,} samples ({taps} taps, {fs:,} Hz)")
    s_true, n0, d, y, e, timings = _native.run_anc(taps=taps, fs=float(fs), lr=lr, n=n)
    host_stats = _stats(np.asarray(timings["seconds_per_sample"]))

    progress("Measuring convergence")
    conv = _convergence(n0, y, s_true, e, fs, taps)
    early, late = slice(0, min(n, taps * 8)), slice(max(0, n - taps * 8), n)
    early_corr = float(np.corrcoef(y[early], n0[early])[0, 1]) if n > 16 else 0.0
    late_corr = float(np.corrcoef(y[late], n0[late])[0, 1]) if n > 16 else 0.0
    early_rms = float(np.sqrt(np.mean((e[early] - s_true[early]) ** 2)))
    late_rms = float(np.sqrt(np.mean((e[late] - s_true[late]) ** 2)))

    projection = asdict(chip.project(taps))
    chip_margin = chip.margin(taps, fs)
    host_margin = chip.host_margin(host_stats["us_per_sample"] * 1e-6, fs)

    progress("Encoding audio")
    files = {}
    for name, arr in [("noisy", d), ("cancelled", e), ("clean", s_true)]:
        files[name] = f"{uuid.uuid4().hex}.wav"
        with open(f"{out_dir}/{files[name]}", "wb") as f:
            f.write(wav.encode(arr * PLAYBACK_SCALE, fs))

    return {
        "taps": taps,
        "fs": fs,
        "lr": lr,
        "duration_s": duration_s,
        "samples": n,
        "convergence": conv,
        "early": {"corr": early_corr, "rms": early_rms},
        "late": {"corr": late_corr, "rms": late_rms},
        "host": host_stats,
        "chip_projection": projection,
        "chip_margin": chip_margin,
        "host_margin": host_margin,
        "budget_us": 1e6 / fs,
        "audio": {name: f"/media/{f}" for name, f in files.items()},
        "waveform": {  # a light-weight preview curve for the page, not the full-resolution audio
            "d": _downsample(d, 800).tolist(),
            "e": _downsample(e, 800).tolist(),
            "s": _downsample(s_true, 800).tolist(),
        },
    }


def _downsample(arr: np.ndarray, points: int) -> np.ndarray:
    if len(arr) <= points:
        return arr
    # min/max per bucket, not a mean: a mean would smear out the noise peaks that make d and e
    # visibly different, which is the whole point of drawing them.
    bins = np.array_split(arr, points // 2)
    return np.concatenate([[np.min(b), np.max(b)] for b in bins])
