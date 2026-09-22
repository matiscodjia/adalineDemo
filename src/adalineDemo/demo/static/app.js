const $ = (id) => document.getElementById(id);
const num = (n, d = 0) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const paint = (input) => input.style.setProperty("--fill", ((input.value - input.min) / (input.max - input.min)) * 100 + "%");

const S = { cfg: null, tapsIdx: 2, fsIdx: 1, lr: 0.004, duration: 6, run: null, busy: false };

function save() {
  try { localStorage.setItem("adaline.state", JSON.stringify({ tapsIdx: S.tapsIdx, fsIdx: S.fsIdx, lr: S.lr, duration: S.duration })); } catch {}
}
function load() {
  try { return JSON.parse(localStorage.getItem("adaline.state")) || {}; } catch { return {}; }
}

const taps = () => S.cfg.taps_ladder[S.tapsIdx];
const fsEntry = () => S.cfg.sample_rates[S.fsIdx];

// log-scale slider (0..200) <-> learning rate (lr_min..lr_max)
function lrFromSlider(v) {
  const { lr_min, lr_max } = S.cfg;
  return lr_min * Math.pow(lr_max / lr_min, v / 200);
}
function sliderFromLr(lr) {
  const { lr_min, lr_max } = S.cfg;
  return Math.round((200 * Math.log(lr / lr_min)) / Math.log(lr_max / lr_min));
}

async function job(path, body, onStage) {
  const started = performance.now();
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  for (;;) {
    const j = await (await fetch("/api/job/" + data.job)).json();
    if (j.state === "done") return j.result;
    if (j.state === "error") throw new Error(j.error);
    onStage(`${j.stage}… ${((performance.now() - started) / 1000).toFixed(0)} s`);
    await new Promise((r) => setTimeout(r, 200));
  }
}

// ------------------------------------------------------------ margin cards (the same "does it fit"
// grammar as the memory wall in directConv, applied to TIME instead of MEMORY)
function paintMargin(prefix, marginX, usedLabel, budgetLabel) {
  // The bar shows the fraction of the real-time budget actually used: 1/margin. margin < 1 (over
  // budget) still shows a full, red bar rather than overflowing past the dashed budget line.
  const fits = marginX >= 1, usedPct = Math.min(100, (100 / marginX) || 100);
  $(`${prefix}-fill`).style.width = usedPct + "%";
  $(`${prefix}-fill`).style.background = fits ? (usedPct > 70 ? "var(--warn)" : "") : "var(--bad)";
  $(`${prefix}-num`).textContent = num(marginX, marginX < 10 ? 1 : 0) + "×" + (fits ? "" : " ⚠");
  $(`${prefix}-num`).style.color = fits ? "" : "var(--bad)";
  $(`${prefix}-used`).textContent = usedLabel;
  $(`${prefix}-budget`).textContent = budgetLabel;
}

function microseconds(us) {
  return (us >= 1000 ? num(us / 1000, 2) + " ms" : num(us, us < 1 ? 2 : 1) + " µs") + "/sample";
}

// ------------------------------------------------------------ preview (before Run): projected chip
// margin at the current sliders, so moving a slider is felt immediately, not only after Run.
function chipProjection(t) {
  const cyc = S.cfg.chip.verified_cycles * (t / S.cfg.chip.verified_taps);
  return cyc / S.cfg.chip.clock_hz; // seconds/sample
}

function updatePreview() {
  const spsChip = chipProjection(taps()), budget = 1 / fsEntry().hz;
  const marginChip = budget / spsChip;
  paintMargin("chip", marginChip, microseconds(spsChip * 1e6), `budget ${microseconds(budget * 1e6)}`);
  $("chip-pill").textContent = taps() === S.cfg.chip.verified_taps ? "verified on the chip" : "projected";
  $("chip-pill").className = "pill " + (taps() === S.cfg.chip.verified_taps ? "ok" : "warn");
  if (S.run && S.run.taps === taps() && S.run.fs === fsEntry().hz) {
    paintMargin("host", S.run.host_margin, microseconds(S.run.host.us_per_sample), `budget ${microseconds(S.run.budget_us)}`);
  } else {
    $("host-num").textContent = "—"; $("host-fill").style.width = "0%"; $("host-used").textContent = ""; $("host-budget").textContent = "";
  }
}

// ------------------------------------------------------------ result rendering
function drawChart(conv) {
  const W = 620, H = 180, m = { l: 34, r: 34, t: 10, b: 20 };
  const x = (t) => m.l + (t / Math.max(...conv.t_s, 1e-6)) * (W - m.l - m.r);
  const yC = (c) => m.t + (1 - Math.max(0, Math.min(1, c))) * (H - m.t - m.b);
  const rmsMax = Math.max(...conv.rms, 1e-6);
  const yR = (r) => m.t + (1 - r / rmsMax) * (H - m.t - m.b);
  let g = "";
  for (const f of [0, 0.5, 1]) g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${yC(f)}" y2="${yC(f)}"/><text x="${m.l - 6}" y="${yC(f) + 3}" text-anchor="end">${f}</text>`;
  g += `<text x="${W - m.r + 6}" y="${yR(rmsMax) + 3}">${num(rmsMax, 2)}</text><text x="${W - m.r + 6}" y="${yR(0) + 3}">0</text>`;
  g += `<polyline class="corr" points="${conv.t_s.map((t, i) => `${x(t)},${yC(conv.corr[i])}`).join(" ")}"/>`;
  g += `<polyline class="rms" points="${conv.t_s.map((t, i) => `${x(t)},${yR(conv.rms[i])}`).join(" ")}"/>`;
  g += conv.t_s.filter((_, i) => i === 0 || i === conv.t_s.length - 1).map((t) => `<text x="${x(t)}" y="${H - 4}" text-anchor="${t === 0 ? "start" : "end"}">${num(t, 1)}s</text>`).join("");
  $("chart").innerHTML = g;
}

function drawWave(id, points) {
  const w = 800, h = 46, max = Math.max(1e-6, ...points.map(Math.abs));
  const y = (v) => h / 2 - (v / max) * (h / 2 - 3);
  const pts = points.map((v, i) => `${(i / (points.length - 1)) * w},${y(v)}`).join(" ");
  $(id).innerHTML = `<polyline points="${pts}" fill="none" stroke-width="1.2"/>`;
}

function metric(k, v, s) {
  return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
}

function renderResult(r) {
  $("empty").hidden = true; $("filled").hidden = false;
  for (const [key, id] of [["noisy", "noisy"], ["cancelled", "cancelled"], ["clean", "clean"]]) $("a-" + id).src = r.audio[key];
  drawWave("w-noisy", r.waveform.d); drawWave("w-cancelled", r.waveform.e); drawWave("w-clean", r.waveform.s);
  drawChart(r.convergence);
  const gain = r.late.corr - r.early.corr, drop = r.early.rms - r.late.rms;
  $("metrics").innerHTML =
    metric("Correlation with noise", `${num(r.early.corr, 2)} → ${num(r.late.corr, 2)}`, `+${num(gain, 2)} over the clip`) +
    metric("Residual error (RMS)", `${num(r.early.rms, 2)} → ${num(r.late.rms, 2)}`, `−${num(drop, 2)}, closer to the clean signal`) +
    metric("Speed, this computer", microseconds(r.host.us_per_sample), `5–95%: ${microseconds(r.host.p5_us)}–${microseconds(r.host.p95_us)}`) +
    metric("Speed, chip (projected)", microseconds(r.chip_projection.seconds_per_sample * 1e6), r.chip_projection.verified ? "verified on the board" : "linear projection");
  const chipOk = r.chip_margin >= 1;
  $("hint").textContent = `At ${r.taps} taps and ${num(r.fs)} Hz, the chip's real-time budget is ${microseconds(r.budget_us)}. ` +
    (chipOk
      ? `The projected cost is ${microseconds(r.chip_projection.seconds_per_sample * 1e6)}: a ${num(r.chip_margin, r.chip_margin < 10 ? 1 : 0)}× margin, plenty of room for other work between samples.`
      : `The projected cost is ${microseconds(r.chip_projection.seconds_per_sample * 1e6)}: that is more than the budget (${num(1 / r.chip_margin, 2)}× over), the chip could not keep up at this rate.`);
  updatePreview();
}

// ------------------------------------------------------------ controls
function refresh() {
  $("taps-big").textContent = `${taps()} taps`;
  $("taps-sub").textContent = taps() === S.cfg.chip.verified_taps ? "verified on the chip" : "projected on the chip";
  $("taps-note").textContent = "More taps model a longer, more complex coupling path (better cancellation), at a linear cost in time.";
  $("fs-big").textContent = fsEntry().label;
  $("fs-note").textContent = fsEntry().note + ". Higher rates mean less time per sample.";
  $("lr-big").textContent = num(S.lr, S.lr < 0.01 ? 4 : 3);
  $("lr-sub").textContent = num(S.duration) + " s clip";
  document.querySelectorAll("#duration button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.d === S.duration)));
  ["taps", "fs", "lr"].forEach((id) => paint($(id)));
  updatePreview();
  save();
}

async function doRun() {
  if (S.busy) return;
  S.busy = true; $("run").disabled = true;
  $("runmsg").className = "runmsg"; $("runmsg").textContent = "Starting…";
  try {
    const r = await job("/api/run", { taps: taps(), fs: fsEntry().hz, lr: S.lr, duration_s: S.duration }, (t) => ($("runmsg").textContent = t));
    S.run = r;
    $("runmsg").textContent = `Trained on ${num(r.samples)} samples. ${r.host.batches} timed batches.`;
    renderResult(r);
  } catch (e) {
    $("runmsg").className = "runmsg err"; $("runmsg").textContent = e.message;
  } finally {
    S.busy = false; $("run").disabled = false;
  }
}

async function init() {
  S.cfg = await (await fetch("/api/config")).json();
  const saved = load();
  S.tapsIdx = saved.tapsIdx ?? S.cfg.taps_ladder.indexOf(16);
  S.fsIdx = saved.fsIdx ?? S.cfg.sample_rates.findIndex((r) => r.hz === 8000);
  S.lr = saved.lr ?? S.cfg.default_lr;
  S.duration = saved.duration ?? S.cfg.default_duration_s;

  $("taps").max = S.cfg.taps_ladder.length - 1; $("taps").value = S.tapsIdx;
  $("fs").max = S.cfg.sample_rates.length - 1; $("fs").value = S.fsIdx;
  $("lr").value = sliderFromLr(S.lr);
  $("duration").innerHTML = S.cfg.durations_s.map((d) => `<button type="button" data-d="${d}">${d}s</button>`).join("");
  $("f-signal").textContent = S.cfg.signal.f_signal.map((f) => Math.round(f)).join(", ");
  $("f-noise").textContent = S.cfg.signal.f_noise.map((f) => Math.round(f)).join(", ");

  $("taps").addEventListener("input", () => { S.tapsIdx = +$("taps").value; refresh(); });
  $("fs").addEventListener("input", () => { S.fsIdx = +$("fs").value; refresh(); });
  $("lr").addEventListener("input", () => { S.lr = lrFromSlider(+$("lr").value); refresh(); });
  document.querySelectorAll("#duration button").forEach((b) => b.addEventListener("click", () => { S.duration = +b.dataset.d; refresh(); }));
  $("run").addEventListener("click", doRun);

  refresh();
}
init().catch((e) => { $("runmsg").className = "runmsg err"; $("runmsg").textContent = "Could not reach the demo server: " + e.message; });
