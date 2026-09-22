"""The Rust ADALINE, cross-checked against an independent Python LMS reference."""

import numpy as np
import pytest

from adalineDemo import _native, reference

RNG_SEED = 424242


@pytest.mark.parametrize("taps", [4, 16, 64])
@pytest.mark.parametrize("n", [1, 50, 500])
def test_matches_the_python_reference(taps, n):
    fs, lr = 8000.0, 0.004
    s, n0, d, y, e, _ = _native.run_anc(taps=taps, fs=fs, lr=lr, n=n, seed=RNG_SEED)
    ref = reference.run(taps, fs, lr, n, RNG_SEED)
    for key, rust in [("s_true", s), ("n0", n0), ("d", d), ("y", y), ("e", e)]:
        py = np.array(ref[key], dtype=np.float32)
        assert np.abs(rust - py).max() < 1e-3, key


def test_reproducible_given_the_same_seed():
    a = _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=200, seed=7)
    b = _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=200, seed=7)
    assert np.array_equal(a[3], b[3])  # y[n]


def test_different_seeds_give_different_noise_realisations():
    a = _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=200, seed=1)
    b = _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=200, seed=2)
    assert not np.array_equal(a[1], b[1])  # n0[n] differs (the jitter term)


def test_the_adaline_converges_reproducing_the_documented_margin():
    # docs/BENCHMARKS.md: correlation goes from 0.549 to 0.986, RMS |e-s| from 0.398 to 0.074, on
    # this exact configuration (16 taps, fs=8000, lr=0.004, n=3000, seed=424242).
    s, n0, d, y, e, _ = _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=3000, seed=424242)
    early, late = slice(0, 100), slice(-500, None)
    corr_early = np.corrcoef(y[early], n0[early])[0, 1]
    corr_late = np.corrcoef(y[late], n0[late])[0, 1]
    assert corr_early < 0.85 < corr_late  # starts weak, converges strong
    rms_early = np.sqrt(np.mean((e[early] - s[early]) ** 2))
    rms_late = np.sqrt(np.mean((e[late] - s[late]) ** 2))
    assert rms_late < rms_early


def test_registered_taps_and_signal_params():
    taps = _native.registered_taps()
    assert taps == sorted(taps) and 16 in taps
    f_s, a_s, f_n, a_n = _native.signal_params()
    assert len(f_s) == len(a_s) == 3 and len(f_n) == len(a_n) == 3


def test_unregistered_taps_says_what_to_add():
    with pytest.raises(ValueError, match=r"add it to register_taps!"):
        _native.run_anc(taps=17, fs=8000.0, lr=0.004, n=10)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="fs must be positive"):
        _native.run_anc(taps=16, fs=0.0, lr=0.004, n=10)
    with pytest.raises(ValueError, match="n must be at least 1"):
        _native.run_anc(taps=16, fs=8000.0, lr=0.004, n=0)
