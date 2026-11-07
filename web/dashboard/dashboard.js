"use strict";

const $ = id => document.getElementById(id);
const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];
const LABELS = { shoulder_pan: "Base", shoulder_lift: "Shoulder", elbow_flex: "Elbow",
  wrist_flex: "Wrist flex", wrist_roll: "Wrist roll", gripper: "Gripper" };
const ARM_ENABLED = document.querySelector('meta[name="arm-enabled"]').content === "true";
function status(el, text, warn) { el.textContent = text; el.classList.toggle("warn", !!warn); }
// ---------- arm ----------
// The server reports degrees from the saved home pose (arm pointing straight up).
let arm = null, trail = [], armFrames = 0;
const deg = j => arm.degrees_from_home[j];
const L = [0.8 * 116, 0.8 * 135, 0.8 * 100];   // approximate SO-ARM101 link lengths

const rows = {};
for (const j of JOINTS) {
  const li = document.createElement("li");
  const name = document.createElement("span"), value = document.createElement("span");
  name.textContent = LABELS[j];
  li.append(name, value); $("joints").append(li);
  rows[j] = { li, value };
}

function renderArm() {
  if (!arm) return;
  for (const j of JOINTS) {
    const p = arm.positions[j], lim = arm.joints[j];
    const out = p < lim.minimum || p > lim.maximum;
    rows[j].li.classList.toggle("out", out);
    rows[j].value.textContent = `${deg(j).toFixed(0)}°  ·  ${p}${out ? "  outside limit" : ""}`;
  }
  const r = d => d * Math.PI / 180;
  let a = 0, x = 0, y = -70;
  const pts = [[0, 0], [x, y]];
  ["shoulder_lift", "elbow_flex", "wrist_flex"].forEach((j, i) => {
    a += r(deg(j)); x += L[i] * Math.sin(a); y -= L[i] * Math.cos(a); pts.push([x, y]);
  });
  const fmt = ps => ps.map(p => p.map(v => v.toFixed(1)).join(",")).join(" ");
  $("links").setAttribute("points", fmt(pts));
  $("tip").setAttribute("cx", x.toFixed(1)); $("tip").setAttribute("cy", y.toFixed(1));
  const last = trail[trail.length - 1];
  if (!last || Math.hypot(last[0] - x, last[1] - y) > 1.5) { trail.push([x, y]); if (trail.length > 200) trail.shift(); }
  $("trail").setAttribute("points", fmt(trail));
}

function onArm(d) {
  if (!d.connected) { arm = null; status($("arm-status"), d.error || "not connected", true); return; }
  arm = d; armFrames++;
}


// ---------- EMG ----------
const N = 4096;                                    // 2 s at 2048 Hz
const buf = Array.from({ length: 8 }, () => new Float32Array(N));
const mean = new Float64Array(8), scale = new Float64Array(8).fill(50);
const railHits = new Uint32Array(8);
let head = 0, filled = 0, meanReady = false, batches = 0, samples = 0;

function onEmg(ev) {
  batches++;
  for (const row of ev.payload.values) {
    if (!meanReady) { row.forEach((v, c) => { mean[c] = v; }); meanReady = true; }
    for (let c = 0; c < 8; c++) {
      if (row[c] < 256 || row[c] > 65279) railHits[c]++;   // pinned at a rail: poor skin contact
      mean[c] += (row[c] - mean[c]) * 0.002;
      buf[c][head] = row[c] - mean[c];
    }
    head = (head + 1) % N; samples++; if (filled < N) filled++;
  }
}

const canvas = $("emg"), ctx = canvas.getContext("2d");
function drawEmg() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--trace-bg"); ctx.fillRect(0, 0, w, h);
  const lane = h / 8, per = N / w;
  ctx.strokeStyle = css.getPropertyValue("--trace"); ctx.fillStyle = css.getPropertyValue("--muted");
  ctx.lineWidth = Math.max(1, dpr); ctx.font = `${10 * dpr}px system-ui`;
  for (let c = 0; c < 8; c++) {
    const mid = lane * c + lane / 2;
    ctx.fillText(String(c + 1), 0, mid + 3 * dpr);
    if (!filled) continue;
    let peak = 0;
    for (let i = 0; i < filled; i++) { const v = Math.abs(buf[c][i]); if (v > peak) peak = v; }
    scale[c] = Math.max(30, scale[c] * 0.95 + peak * 0.05);
    const k = lane * 0.45 / scale[c];
    ctx.beginPath();
    for (let x = 14 * dpr; x < w; x++) {
      let lo = Infinity, hi = -Infinity;
      const s0 = Math.floor(x * per), s1 = Math.max(s0 + 1, Math.floor((x + 1) * per));
      for (let s = s0; s < s1; s++) { const v = buf[c][(head + s) % N]; if (v < lo) lo = v; if (v > hi) hi = v; }
      ctx.moveTo(x + 0.5, mid - hi * k); ctx.lineTo(x + 0.5, mid - lo * k + 0.5);
    }
    ctx.stroke();
  }
}

// ---------- motion: orientation (deg) and gyro (deg/s), ~128 Hz each ----------
const M = 256;                                     // 2 s at 128 Hz
const MOTION = ["roll", "pitch", "yaw", "gyro x", "gyro y", "gyro z"];
const mbuf = MOTION.map(() => new Float32Array(M)), mhead = [0, 0], mval = new Array(6).fill(null);
function pushMotion(offset, values) {
  const h = mhead[offset ? 1 : 0];
  values.forEach((v, i) => { mbuf[offset + i][h] = v; mval[offset + i] = v; });
  mhead[offset ? 1 : 0] = (h + 1) % M;
}
function onOrientation(ev) {
  const [w, x, y, z] = ev.payload.values, d = 180 / Math.PI;
  pushMotion(0, [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) * d,
                 Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))) * d,
                 Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * d]);
}
function onGyro(ev) {
  const k = ev.payload.observed_degrees_per_second_per_count || 0.07;
  pushMotion(3, ev.payload.values.map(v => v * k));
}
function onGesture(ev) {
  const p = ev.payload;
  const label = p.label || [p.finger, p.derived_action || p.action].filter(Boolean).join(" ") || "gesture";
  status($("gesture"), `${label}${p.synthetic ? " (synthetic)" : ""} · ${new Date().toLocaleTimeString()}`);
}

const mcanvas = $("motion"), mctx = mcanvas.getContext("2d");
function drawMotion() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(mcanvas.clientWidth * dpr), h = Math.round(mcanvas.clientHeight * dpr);
  if (mcanvas.width !== w || mcanvas.height !== h) { mcanvas.width = w; mcanvas.height = h; }
  const css = getComputedStyle(document.documentElement);
  mctx.fillStyle = css.getPropertyValue("--trace-bg"); mctx.fillRect(0, 0, w, h);
  const lane = h / 6, left = 64 * dpr;
  mctx.font = `${10 * dpr}px system-ui`; mctx.lineWidth = Math.max(1, 1.5 * dpr);
  MOTION.forEach((name, i) => {
    const mid = lane * i + lane / 2, range = i < 3 ? 180 : 360;   // ±180° or ±360°/s
    mctx.fillStyle = css.getPropertyValue("--muted");
    mctx.fillText(name, 0, mid - 2 * dpr);
    if (mval[i] === null) return;
    mctx.fillStyle = css.getPropertyValue("--trace");
    mctx.fillText(mval[i].toFixed(0) + (i < 3 ? "°" : "°/s"), 0, mid + 11 * dpr);
    mctx.strokeStyle = css.getPropertyValue("--line");
    mctx.beginPath(); mctx.moveTo(left, mid); mctx.lineTo(w, mid); mctx.stroke();
    mctx.strokeStyle = css.getPropertyValue("--trace");
    mctx.beginPath();
    const head = mhead[i < 3 ? 0 : 1];
    for (let s = 0; s < M; s++) {
      const v = Math.max(-range, Math.min(range, mbuf[i][(head + s) % M]));
      const x = left + (w - left) * s / (M - 1), y = mid - v / range * lane * 0.45;
      s ? mctx.lineTo(x, y) : mctx.moveTo(x, y);
    }
    mctx.stroke();
  });
}

// ---------- wiring ----------
const form = $("url-form");
form.addEventListener("submit", async e => {
  e.preventDefault();
  const url = $("stream-url").value.trim();
  if (!url) return;
  const r = await fetch("/api/stream-url", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }) }).catch(() => null);
  if (r?.ok) $("stream-url").value = "";
  else status($("band-status"), (await r?.json().catch(() => null))?.error || "could not set URL", true);
});

let link = { state: "waiting" };
const source = new EventSource("/events");
const on = (name, fn) => source.addEventListener(name, e => { try { fn(JSON.parse(e.data)); } catch (err) { console.error(err); } });
on("arm", onArm);
on("emg", onEmg);
on("orientation", onOrientation);
on("gyro", onGyro);
on("gesture", onGesture);
on("band_link", d => { link = d; });
source.onerror = () => { status($("arm-status"), "server offline", true); status($("band-status"), "server offline", true); };
if (!ARM_ENABLED) status($("arm-status"), "not started (no --port)");

setInterval(() => {
  if (arm) status($("arm-status"), `${armFrames} Hz`);
  armFrames = 0;
  const live = batches > 0;
  form.hidden = live;
  if (live) {
    const bad = [...railHits].map((n, c) => n / samples > 0.05 ? c + 1 : 0).filter(Boolean);
    status($("band-status"), bad.length ? `ch ${bad.join(", ")} saturated: tighten band`
                                        : `${batches * 16} Hz`, bad.length);
  } else {
    status($("band-status"), link.state === "error" ? link.detail : "no stream", link.state === "error");
  }
  batches = 0; samples = 0; railHits.fill(0);
}, 1000);

(function frame() { drawEmg(); drawMotion(); renderArm(); requestAnimationFrame(frame); })();
