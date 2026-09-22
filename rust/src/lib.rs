//! Python bindings for `frugal_ml`'s ADALINE/LMS adaptive noise canceller, run on the host.
//!
//! Mirrors `directConv`'s bindings in spirit: the Rust core is unchanged from what runs on the
//! STM32F446RE (same `Linear` + `mse` + `Sgd`, same signal), this crate is just the desktop harness
//! that runs it, times it in chunks, and hands the arrays to numpy. See [`anc`] for the loop itself.

mod anc;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Runs the ADALINE noise canceller for `n` samples at `fs` Hz, `taps`-long filter, learning rate
/// `lr`, and returns (s_true, n0, d, y, e) arrays plus a dict of `{"seconds_per_sample": [...]}`,
/// one entry per timing chunk (see `anc::TIMING_CHUNK`), for a median/percentile distribution.
#[pyfunction]
#[pyo3(signature = (taps, fs, lr, n, seed=424242))]
fn run_anc<'py>(
    py: Python<'py>,
    taps: usize,
    fs: f32,
    lr: f32,
    n: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyDict>,
)> {
    if fs <= 0.0 {
        return Err(PyValueError::new_err("fs must be positive"));
    }
    if n == 0 {
        return Err(PyValueError::new_err("n must be at least 1"));
    }
    let trace = py.detach(|| anc::run(taps, fs, lr, n, seed)).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{taps} taps is not compiled into this build, add it to register_taps! in rust/src/anc.rs \
             and rebuild. Available: {:?}",
            anc::TAPS
        ))
    })?;

    let timings = PyDict::new(py);
    timings.set_item("seconds_per_sample", PyArray1::from_vec(py, trace.seconds_per_sample))?;
    Ok((
        PyArray1::from_vec(py, trace.s_true),
        PyArray1::from_vec(py, trace.n0),
        PyArray1::from_vec(py, trace.d),
        PyArray1::from_vec(py, trace.y),
        PyArray1::from_vec(py, trace.e),
        timings,
    ))
}

/// The filter lengths (taps) this build is compiled for.
#[pyfunction]
fn registered_taps() -> Vec<usize> {
    anc::TAPS.to_vec()
}

/// (frequencies, amplitudes) of the 3-tone signal to keep and the 3-harmonic noise to cancel, as
/// used by `run_anc`: (f_signal, a_signal, f_noise, a_noise).
#[pyfunction]
fn signal_params() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (anc::F_S.to_vec(), anc::A_S.to_vec(), anc::F_N.to_vec(), anc::A_N.to_vec())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_anc, m)?)?;
    m.add_function(wrap_pyfunction!(registered_taps, m)?)?;
    m.add_function(wrap_pyfunction!(signal_params, m)?)?;
    Ok(())
}
