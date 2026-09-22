//! The adaptive noise canceller (ADALINE / LMS) exactly as run on the STM32F446RE in
//! `ferrite-embedded/src/bin/adaline_anc.rs`: `Linear<TAPS, 1>` (no hidden layer, no activation)
//! trained online, one sample at a time, with `mse` + `Sgd`. That configuration alone reproduces
//! the Widrow-Hoff LMS update rule exactly (Widrow & Hoff, 1960).
//!
//! `TAPS` (the filter length) is a const generic in `frugal_ml`, fixed when the program is built,
//! the same constraint `directConv`'s conv strategies work under. `register_taps!` at the bottom
//! turns a runtime tap count into one of a fixed ladder of compiled instantiations.
//!
//! The problem (classic adaptive-noise-cancellation setup):
//!   s\[n\]  = the signal to keep: a 3-tone chord, frequencies unrelated to the noise
//!   n0\[n\] = the noise to cancel: a fundamental + 2 harmonics (engine-like) + a little jitter
//!   x\[n\]  = a reference input: n0\[n\] through an unknown 2-tap coupling path, correlated with
//!            the noise, not with the signal
//!   d\[n\]  = s\[n\] + n0\[n\], the primary input (what a microphone would pick up)
//!   y\[n\]  = the ADALINE's noise estimate, learned from x\[n\] alone
//!   e\[n\]  = d\[n\] - y\[n\], the cancelled output

use frugal_ml::autodiff::core::module::Module;
use frugal_ml::autodiff::layers::linear::Linear;
use frugal_ml::autodiff::loss::mse;
use frugal_ml::autodiff::optim::optimizer::Optimizer;
use frugal_ml::autodiff::optim::sgd::Sgd;
use frugal_ml::linalg::Vector;
use frugal_ml::Scalar;
use std::time::Instant;

pub const F_S: [f32; 3] = [60.0, 140.0, 310.0];
pub const A_S: [f32; 3] = [0.30, 0.15, 0.10];
pub const F_N: [f32; 3] = [200.0, 400.0, 600.0];
pub const A_N: [f32; 3] = [0.50, 0.25, 0.15];
const NOISE_JITTER: f32 = 0.03;
const COUPLING_DIRECT: f32 = 0.7;
const COUPLING_DELAYED: f32 = 0.3;
/// How many samples share one `Instant` pair: timing every single sample would make the timer
/// itself the dominant cost at these speeds (a few hundred cycles per step).
const TIMING_CHUNK: usize = 256;

/// One run: every sample of every signal, plus the per-chunk timings that give a distribution.
pub struct Trace {
    pub s_true: Vec<f32>,
    pub n0: Vec<f32>,
    pub d: Vec<f32>,
    pub y: Vec<f32>,
    pub e: Vec<f32>,
    /// Seconds per sample, one entry per `TIMING_CHUNK`-sample chunk.
    pub seconds_per_sample: Vec<f64>,
}

fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f32) / (u64::MAX as f32) * 2.0 - 1.0
}

fn run_typed<const TAPS: usize>(fs: f32, lr: Scalar, n: usize, seed: u64) -> Trace {
    let mut adaline = Linear::<TAPS, 1>::zeros();
    let mut opt = Sgd::new(lr);
    let mut window = [0.0 as Scalar; TAPS];
    let mut rng = if seed == 0 { 1 } else { seed };
    let mut n0_prev: f32 = 0.0;

    let mut t = Trace {
        s_true: Vec::with_capacity(n),
        n0: Vec::with_capacity(n),
        d: Vec::with_capacity(n),
        y: Vec::with_capacity(n),
        e: Vec::with_capacity(n),
        seconds_per_sample: Vec::with_capacity(n.div_ceil(TIMING_CHUNK)),
    };

    let mut chunk_start = Instant::now();
    for i in 0..n {
        let time_s = i as f32 / fs;
        let s_true: f32 = F_S.iter().zip(A_S).map(|(f, a)| a * (2.0 * std::f32::consts::PI * f * time_s).sin()).sum();
        let mut noise: f32 = F_N.iter().zip(A_N).map(|(f, a)| a * (2.0 * std::f32::consts::PI * f * time_s).sin()).sum();
        noise += NOISE_JITTER * xorshift(&mut rng);
        let reference = COUPLING_DIRECT * noise + COUPLING_DELAYED * n0_prev;
        n0_prev = noise;
        let primary = s_true + noise;

        window.copy_within(0..TAPS - 1, 1);
        window[0] = reference;

        let (y_vec, ctx) = adaline.forward(Vector::from_data(window));
        let (_loss, grad) = mse(y_vec, Vector::from_data([primary]));
        let (_, grads) = adaline.backward(grad, &ctx);
        opt.step(&mut adaline, &grads);

        let y = y_vec[0];
        t.s_true.push(s_true);
        t.n0.push(noise);
        t.d.push(primary);
        t.y.push(y);
        t.e.push(primary - y);

        if (i + 1) % TIMING_CHUNK == 0 || i + 1 == n {
            let done = i % TIMING_CHUNK + 1;
            t.seconds_per_sample.push(chunk_start.elapsed().as_secs_f64() / done as f64);
            chunk_start = Instant::now();
        }
    }
    t
}

macro_rules! register_taps {
    ($($t:literal),+ $(,)?) => {
        /// Filter lengths this build can run. `frugal_ml` fixes `Linear`'s size at compile time,
        /// so only these are available; add a value here and rebuild to add another.
        pub const TAPS: &[usize] = &[$($t),+];

        pub fn run(taps: usize, fs: f32, lr: Scalar, n: usize, seed: u64) -> Option<Trace> {
            match taps {
                $($t => Some(run_typed::<$t>(fs, lr, n, seed)),)+
                _ => None,
            }
        }
    };
}

register_taps!(4, 8, 12, 16, 24, 32, 48, 64, 96, 128);
