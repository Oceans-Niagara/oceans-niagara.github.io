import {
  PHI,
  extractInputFeatures,
  simulateOptics,
  wavelengthToRgb,
  makeSampleImage,
  loadImageBlob,
} from "./engine.js?v=3";

// --- PRESETS ---
function makeRandomStatic(len = 180) {
  let seed = 42;
  let out = "";
  for (let i = 0; i < len; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    out += String.fromCharCode(33 + (seed % 93));
  }
  return out;
}

const TEXT_PRESETS = {
  prose:
    "In the midst of chaos, there is also opportunity. " +
    "Patterns hide inside language the same way currents hide beneath still water: " +
    "every word carries a cadence, and every sentence bends the larger rhythm.",
  repetition: "abababababababababababababababababababababababababababababababab",
  code:
    "def fibonacci(n):\n" +
    "    a, b = 0, 1\n" +
    "    out = []\n" +
    "    for _ in range(n):\n" +
    "        out.append(a)\n" +
    "        a, b = b, a + b\n" +
    "    return out\n",
  random: makeRandomStatic(),
};

const PLAY_OPTICS = {
  opticsMode: "phase",
  whiteLight: true,
  wavelengthNm: 532.0,
  distanceMm: 35.0,
  showPhi: false,
};

// --- APP STATE ---
const state = {
  sourceType: "text",
  viewMode: "play",
  visualCmpMode: "baseline",
  uploadedText: null,
  uploadedTextName: "",
  uploadedImageObj: null,
  uploadedImageName: "",
  features: null,
  res: null,
  currentImages: { apertureU8: null, lfU8: null, h: 0, w: 0 },
  plotData: { sig1d: null, fft1d: null },
  snapshots: [],
  snapCounter: 0,
};

// --- DOM REFS ---
const el = (id) => document.getElementById(id);
const appShell = document.querySelector(".app-shell");
const snapshotBar = document.querySelector(".snapshot-bar");
const playTextInput = el("play-text-input");
const playLiveCaption = el("play-live-caption");
const textInput = el("text-input");
const textModeSel = el("text-mode");
const textLayoutSel = el("text-layout");
const hiddenBinaryChk = el("hidden-binary");
const textFileInput = el("text-file-input");
const textFileInfo = el("text-file-info");
const imgSampleSel = el("image-sample-select");
const imgUploadWrap = el("image-upload-wrap");
const imgFileInput = el("image-file-input");
const whiteLightChk = el("white-light");
const wavelengthWrap = el("wavelength-wrap");
const wavelengthSlider = el("wavelength-slider");
const wavelengthVal = el("wavelength-val");
const distanceSlider = el("distance-slider");
const distanceVal = el("distance-val");
const showPhiChk = el("show-phi");
const compareSel = el("compare-select");
const visualCmpTabs = el("visual-cmp-tabs");
const clearSnapsBtn = el("clear-snapshots-btn");
const compareTip = el("compare-tip");

textInput.value = TEXT_PRESETS.prose;
playTextInput.value = TEXT_PRESETS.prose;

// --- CANVAS RENDERING ---
function buildApertureRgbU8(aperture, h, w, showPhi) {
  const buf = new Uint8ClampedArray(h * w * 4);
  for (let i = 0; i < h * w; i++) {
    const v = Math.min(255, Math.max(0, Math.round(aperture[i] * 255)));
    buf[i * 4] = v;
    buf[i * 4 + 1] = v;
    buf[i * 4 + 2] = v;
    buf[i * 4 + 3] = 255;
  }
  if (showPhi) {
    const t = Math.max(1, Math.floor(Math.min(h, w) / 180));
    const fracs = [1.0 / PHI, 1.0 - 1.0 / PHI];
    for (const frac of fracs) {
      const py = Math.floor(h * frac);
      const px = Math.floor(w * frac);
      for (let y = Math.max(0, py - t); y < Math.min(h, py + t); y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          buf[idx] = 255; buf[idx + 1] = 214; buf[idx + 2] = 0;
        }
      }
      for (let y = 0; y < h; y++) {
        for (let x = Math.max(0, px - t); x < Math.min(w, px + t); x++) {
          const idx = (y * w + x) * 4;
          buf[idx] = 255; buf[idx + 1] = 214; buf[idx + 2] = 0;
        }
      }
    }
  }
  return buf;
}

function buildLightRgbU8(res, wavelengthNm) {
  const h = res.height, w = res.width;
  const buf = new Uint8ClampedArray(h * w * 4);
  if (res.is_rgb) {
    const { r, g, b } = res.light_field;
    for (let i = 0; i < h * w; i++) {
      buf[i * 4] = Math.min(255, Math.round((r[i] ** 0.65) * 255));
      buf[i * 4 + 1] = Math.min(255, Math.round((g[i] ** 0.65) * 255));
      buf[i * 4 + 2] = Math.min(255, Math.round((b[i] ** 0.65) * 255));
      buf[i * 4 + 3] = 255;
    }
  } else {
    const [tr, tg, tb] = wavelengthToRgb(wavelengthNm);
    const lf = res.light_field;
    for (let i = 0; i < h * w; i++) {
      const s = Math.min(1, Math.max(0, lf[i] ** 0.65));
      buf[i * 4] = Math.round(s * tr);
      buf[i * 4 + 1] = Math.round(s * tg);
      buf[i * 4 + 2] = Math.round(s * tb);
      buf[i * 4 + 3] = 255;
    }
  }
  return buf;
}

function paintRgbaBuffer(canvas, buf, w, h) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(buf, w, h), 0, 0);
}

function resizeRgbaNearest(srcBuf, sw, sh, dw, dh) {
  const tmp = document.createElement("canvas");
  tmp.width = sw;
  tmp.height = sh;
  tmp.getContext("2d").putImageData(new ImageData(srcBuf, sw, sh), 0, 0);

  const dst = document.createElement("canvas");
  dst.width = dw;
  dst.height = dh;
  const dctx = dst.getContext("2d");
  dctx.drawImage(tmp, 0, 0, dw, dh);
  return dctx.getImageData(0, 0, dw, dh).data;
}

function computeDiffRgba(curBuf, cw, ch, baseBuf, bw, bh) {
  const alignedBase = cw === bw && ch === bh ? baseBuf : resizeRgbaNearest(baseBuf, bw, bh, cw, ch);
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let i = 0; i < cw * ch; i++) {
    const off = i * 4;
    out[off] = Math.min(255, Math.round(Math.abs(curBuf[off] - alignedBase[off]) * 1.5));
    out[off + 1] = Math.min(255, Math.round(Math.abs(curBuf[off + 1] - alignedBase[off + 1]) * 1.5));
    out[off + 2] = Math.min(255, Math.round(Math.abs(curBuf[off + 2] - alignedBase[off + 2]) * 1.5));
    out[off + 3] = 255;
  }
  return out;
}

function drawGrayMatrix(canvas, arr, h, w) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  const denom = max - min + 1e-12;
  const buf = new Uint8ClampedArray(h * w * 4);
  for (let i = 0; i < h * w; i++) {
    const v = Math.min(255, Math.max(0, Math.round(((arr[i] - min) / denom) * 255)));
    buf[i * 4] = v; buf[i * 4 + 1] = v; buf[i * 4 + 2] = v; buf[i * 4 + 3] = 255;
  }
  paintRgbaBuffer(canvas, buf, w, h);
}

function drawInteractiveLinePlot(canvas, plotObj, hoverIdx = null) {
  const w = 560, h = 300;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0b0d13";
  ctx.fillRect(0, 0, w, h);

  if (!plotObj || !plotObj.xs.length) return;
  const { xs, ys, color } = plotObj;
  const padL = 48, padR = 16, padT = 16, padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < xMin) xMin = xs[i];
    if (xs[i] > xMax) xMax = xs[i];
    if (ys[i] < yMin) yMin = ys[i];
    if (ys[i] > yMax) yMax = ys[i];
  }
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;

  // Grid + tick labels
  ctx.strokeStyle = "#1f2430";
  ctx.fillStyle = "#98a2b3";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.lineWidth = 1;

  for (let g = 0; g <= 4; g++) {
    const gy = padT + (plotH * g) / 4;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(w - padR, gy); ctx.stroke();
    const yTick = yMax - (ySpan * g) / 4;
    ctx.textAlign = "right";
    ctx.fillText(yTick.toFixed(1), padL - 6, gy + 4);
  }
  for (let g = 0; g <= 4; g++) {
    const gx = padL + (plotW * g) / 4;
    ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, h - padB); ctx.stroke();
    const xTick = xMin + (xSpan * g) / 4;
    ctx.textAlign = "center";
    ctx.fillText(Math.abs(xSpan) >= 10 ? Math.round(xTick) : xTick.toFixed(2), gx, h - 8);
  }

  // Curve
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const px = padL + ((xs[i] - xMin) / xSpan) * plotW;
    const py = h - padB - ((ys[i] - yMin) / ySpan) * plotH;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Crosshair
  if (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < xs.length) {
    const hx = padL + ((xs[hoverIdx] - xMin) / xSpan) * plotW;
    const hy = h - padB - ((ys[hoverIdx] - yMin) / ySpan) * plotH;
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(hx, padT); ctx.lineTo(hx, h - padB); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, hy); ctx.lineTo(w - padR, hy); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(hx, hy, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function attachPlotHover(canvasId, readoutId, plotKey) {
  const canvas = el(canvasId);
  const readout = el(readoutId);

  const handlePointer = (clientX) => {
    const plotObj = state.plotData[plotKey];
    if (!plotObj || !plotObj.xs.length) return;
    const rect = canvas.getBoundingClientRect();
    const cx = ((clientX - rect.left) / rect.width) * canvas.width;
    const padL = 48, padR = 16;
    const plotW = canvas.width - padL - padR;
    const frac = Math.min(1, Math.max(0, (cx - padL) / plotW));

    const { xs, ys, xLabel, yLabel } = plotObj;
    const targetX = xs[0] + frac * (xs[xs.length - 1] - xs[0]);
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(xs[i] - targetX);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    drawInteractiveLinePlot(canvas, plotObj, best);
    readout.textContent = `${xLabel}: ${xs[best].toFixed(3)}   |   ${yLabel}: ${ys[best].toFixed(3)}`;
  };

  canvas.addEventListener("mousemove", (e) => handlePointer(e.clientX));
  canvas.addEventListener("touchmove", (e) => {
    if (e.touches?.[0]) handlePointer(e.touches[0].clientX);
  }, { passive: true });

  const resetPlot = () => {
    drawInteractiveLinePlot(canvas, state.plotData[plotKey], null);
    readout.textContent = "Hover chart to inspect values";
  };
  canvas.addEventListener("mouseleave", resetPlot);
  canvas.addEventListener("touchend", resetPlot);
}

// --- HELPERS ---
function getActiveSourceLabel() {
  if (state.sourceType === "text") {
    if (state.uploadedText !== null) return state.uploadedTextName;
    const snip = textInput.value.trim().replace(/\s+/g, " ");
    return snip.length > 22 ? `"${snip.slice(0, 22)}..."` : `"${snip}"`;
  }
  const sel = imgSampleSel.value;
  if (sel === "upload") return state.uploadedImageName || "Uploaded image";
  return imgSampleSel.options[imgSampleSel.selectedIndex].text.replace("Sample: ", "");
}

function getOpticsParams() {
  const opticsMode = document.querySelector('input[name="optics-mode"]:checked').value;
  return {
    opticsMode,
    whiteLight: whiteLightChk.checked,
    wavelengthNm: parseFloat(wavelengthSlider.value),
    distanceMm: parseFloat(distanceSlider.value),
    showPhi: showPhiChk.checked,
  };
}

function setDelta(elem, cur, baseVal, invertColor = false) {
  if (baseVal === undefined || baseVal === null) {
    elem.textContent = "";
    elem.className = "metric-delta";
    return;
  }
  const diff = Number((cur - baseVal).toPrecision(3));
  const sign = diff > 0 ? "+" : "";
  elem.textContent = `${sign}${diff}`;
  if (invertColor || diff === 0) {
    elem.className = "metric-delta neutral";
  } else {
    elem.className = diff > 0 ? "metric-delta pos" : "metric-delta neg";
  }
}

function getSelectedBaseline() {
  const idx = compareSel.value;
  if (idx === "") return null;
  return state.snapshots[parseInt(idx, 10)] || null;
}

function flashButtonText(btn, msg) {
  const orig = btn.textContent;
  btn.textContent = msg;
  setTimeout(() => { btn.textContent = orig; }, 1200);
}

function downloadCanvasPng(canvasId, filename) {
  const canvas = el(canvasId);
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = filename;
  a.click();
}

function updateViewVisibility() {
  const isPlay = state.viewMode === "play";
  appShell.classList.toggle("play-mode", isPlay);
  el("play-view").classList.toggle("hidden", state.viewMode !== "play");
  el("simple-view").classList.toggle("hidden", state.viewMode !== "simple");
  el("bench-view").classList.toggle("hidden", state.viewMode !== "bench");
  snapshotBar.classList.toggle("hidden", isPlay);
  el("snapshots-section").classList.toggle("hidden", isPlay || state.snapshots.length === 0);
}

// --- PIPELINE RUNNERS ---
async function recomputeAll() {
  if (state.viewMode === "play") {
    const txt = playTextInput.value;
    if (!txt || !txt.trim()) {
      const c = el("play-light-canvas");
      c.width = 256; c.height = 256;
      c.getContext("2d").fillRect(0, 0, 256, 256);
      playLiveCaption.textContent = "Type anything below to shine white light through your words.";
      return;
    }
    state.features = await extractInputFeatures({ text: txt, textMode: "utf8", textLayout: "grating" });
    runOpticsAndRender();
    return;
  }

  const textMode = hiddenBinaryChk.checked ? "bits" : textModeSel.value;
  const textLayout = textLayoutSel.value;

  if (state.sourceType === "text") {
    const txt = state.uploadedText !== null ? state.uploadedText : textInput.value;
    if (!txt || !txt.trim()) return;
    state.features = await extractInputFeatures({ text: txt, textMode, textLayout });
  } else {
    let imgObj;
    const choice = imgSampleSel.value;
    if (choice === "upload") {
      if (!state.uploadedImageObj) return;
      imgObj = state.uploadedImageObj;
    } else {
      imgObj = makeSampleImage(choice, 256);
    }
    state.features = await extractInputFeatures({ imageObj: imgObj });
  }
  runOpticsAndRender();
}

function runOpticsAndRender() {
  if (!state.features) return;
  const isPlay = state.viewMode === "play";
  const params = isPlay ? PLAY_OPTICS : getOpticsParams();
  const res = simulateOptics(state.features, params);
  state.res = res;

  const h = res.height, w = res.width;
  const lfU8 = buildLightRgbU8(res, params.wavelengthNm);

  if (isPlay) {
    paintRgbaBuffer(el("play-light-canvas"), lfU8, w, h);
    const regime = res.simple.regime;
    if (regime === "Uniform / Blank") {
      playLiveCaption.textContent = "Calm solid beam — every character is identical, so nothing bends the light.";
    } else if (regime === "Ordered / Repetitive") {
      playLiveCaption.textContent = "Locked rhythm — repeating structure splits the beam into crisp bands.";
    } else if (regime === "Chaotic / Static") {
      playLiveCaption.textContent = "Scattered static — unpredictable characters shatter the beam into speckle.";
    } else {
      playLiveCaption.textContent = "Flowing structure — mixed repetition and variation bend light into glowing arcs.";
    }
    return;
  }

  const apertureU8 = buildApertureRgbU8(res.aperture_2d, h, w, params.showPhi);
  state.currentImages = { apertureU8, lfU8, h, w };

  const baseline = getSelectedBaseline();
  const [r, g, b] = res.spectral_rgb;
  const hex = `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
  const s = res.simple;
  const info = res.information;
  const slopeVal = res.input_type === "text" ? res.fourier.spectral_slope : res.fourier.radial_spectral_slope;

  const beamDesc = params.whiteLight ? "White Light" : `${Math.round(params.wavelengthNm)} nm Laser`;
  const leftCap =
    params.opticsMode === "phase"
      ? "Brightness shifts the light wave's timing—bending the beam without blocking it."
      : "Brightness controls transparency—blocking dark areas like a film slide.";
  const rightCap =
    params.opticsMode === "phase" && params.distanceMm === 0
      ? "At z = 0 mm, a pure phase filter is invisible (uniform brightness)—increase z so bent waves collide."
      : "Repeating structure locks into sharp bands; pure noise scatters into speckle.";

  // --- SIMPLE VIEW ---
  const pill = el("regime-pill");
  pill.textContent = `Signal Regime: ${s.regime}`;
  pill.style.background = s.regime_color;
  el("regime-meta").innerHTML = `Information Density: <strong>${s.info_density}%</strong> &nbsp;•&nbsp; ${s.regime_hint}`;

  el("simple-right-title").textContent = `2. Data-Filtered Light (${beamDesc} @ ${Math.round(params.distanceMm)} mm)`;
  el("simple-left-cap").textContent = leftCap;
  el("simple-right-cap").textContent = rightCap;

  paintRgbaBuffer(el("simple-aperture-canvas"), apertureU8, w, h);
  paintRgbaBuffer(el("simple-light-canvas"), lfU8, w, h);

  el("m-roughness").textContent = `${s.texture} / 100`;
  el("p-roughness").style.width = `${Math.min(100, Math.max(0, s.texture))}%`;
  setDelta(el("d-roughness"), s.texture, baseline?.roughness, true);

  el("m-harmony").textContent = `${s.harmony} / 100`;
  el("p-harmony").style.width = `${Math.min(100, Math.max(0, s.harmony))}%`;
  setDelta(el("d-harmony"), s.harmony, baseline?.harmony, false);

  el("m-scatter").textContent = `${s.light_bloom} / 100`;
  el("p-scatter").style.width = `${Math.min(100, Math.max(0, s.light_bloom))}%`;
  setDelta(el("d-scatter"), s.light_bloom, baseline?.light_bloom, true);

  el("simple-swatch").style.background = hex;
  el("simple-hex").textContent = hex;

  // --- OPTICAL BENCH VIEW ---
  el("b-slope-label").textContent = res.input_type === "text" ? "Fourier Power Slope (β)" : "Radial Power Slope (β)";
  el("b-slope").textContent = slopeVal;
  setDelta(el("bd-slope"), slopeVal, baseline?.slope, true);

  el("b-fd").textContent = res.fractal_dimension;
  setDelta(el("bd-fd"), res.fractal_dimension, baseline?.fd, true);

  el("b-lfd").textContent = res.light_fractal_dimension;
  setDelta(el("bd-lfd"), res.light_fractal_dimension, baseline?.light_fd, true);

  el("b-phi-label").textContent = res.input_type === "text" ? "φ Harmonic Match" : "Aspect φ Match";
  el("b-phi").textContent =
    res.input_type === "text"
      ? `${res.fourier.phi_harmonic_score_pct}%`
      : `${res.golden_ratio.aspect_phi_score_pct}%`;

  el("b-spec-ent").textContent = info.spectral_entropy.toFixed(3);
  setDelta(el("bd-spec-ent"), info.spectral_entropy, baseline?.spectral_entropy, true);

  el("bench-swatch").style.background = hex;
  el("bench-hex").textContent = `${hex} (${r}, ${g}, ${b})`;

  el("b-shannon").textContent = info.shannon_bits;
  setDelta(el("bd-shannon"), info.shannon_bits, baseline?.shannon_bits, true);

  el("b-compress").textContent = info.compress_ratio.toFixed(3);
  setDelta(el("bd-compress"), info.compress_ratio, baseline?.compress_ratio, true);

  el("b-scint").textContent = res.scintillation_index;
  setDelta(el("bd-scint"), res.scintillation_index, baseline?.scintillation, true);

  el("b-regime").textContent = s.regime;

  el("bench-left-cap").textContent = `Metadata: ${JSON.stringify(res.meta)}`;
  el("bench-right-title").textContent = `2. Diffraction Field (${beamDesc} @ z = ${Math.round(params.distanceMm)} mm)`;
  el("bench-right-cap").textContent = rightCap;

  paintRgbaBuffer(el("bench-aperture-canvas"), apertureU8, w, h);
  paintRgbaBuffer(el("bench-light-canvas"), lfU8, w, h);

  // --- COMPARISON VISUALS ---
  const simpleCmp = el("simple-compare-row");
  const benchCmp = el("bench-compare-row");
  if (baseline) {
    simpleCmp.classList.remove("hidden");
    benchCmp.classList.remove("hidden");
    if (state.visualCmpMode === "diff") {
      const dAp = computeDiffRgba(apertureU8, w, h, baseline.apertureU8, baseline.w, baseline.h);
      const dLf = computeDiffRgba(lfU8, w, h, baseline.lfU8, baseline.w, baseline.h);
      paintRgbaBuffer(el("simple-base-aperture-canvas"), dAp, w, h);
      paintRgbaBuffer(el("simple-base-light-canvas"), dLf, w, h);
      paintRgbaBuffer(el("bench-base-aperture-canvas"), dAp, w, h);
      paintRgbaBuffer(el("bench-base-light-canvas"), dLf, w, h);
      const c1 = `Filter Difference vs. ${baseline.label} (brighter = bigger shift)`;
      const c2 = `Light Difference vs. ${baseline.label} (brighter = bigger shift)`;
      el("simple-base-left-cap").textContent = c1;
      el("simple-base-right-cap").textContent = c2;
      el("bench-base-left-cap").textContent = c1;
      el("bench-base-right-cap").textContent = c2;
    } else {
      paintRgbaBuffer(el("simple-base-aperture-canvas"), baseline.apertureU8, baseline.w, baseline.h);
      paintRgbaBuffer(el("simple-base-light-canvas"), baseline.lfU8, baseline.w, baseline.h);
      paintRgbaBuffer(el("bench-base-aperture-canvas"), baseline.apertureU8, baseline.w, baseline.h);
      paintRgbaBuffer(el("bench-base-light-canvas"), baseline.lfU8, baseline.w, baseline.h);
      const c1 = `Baseline Filter — ${baseline.label}`;
      const c2 = `Baseline Light — ${baseline.label}`;
      el("simple-base-left-cap").textContent = c1;
      el("simple-base-right-cap").textContent = c2;
      el("bench-base-left-cap").textContent = c1;
      el("bench-base-right-cap").textContent = c2;
    }
  } else {
    simpleCmp.classList.add("hidden");
    benchCmp.classList.add("hidden");
  }

  // --- BENCH PLOTS ---
  if (res.input_type === "text") {
    el("bench-text-plots").classList.remove("hidden");
    el("bench-image-plots").classList.add("hidden");

    const sig = res.signal_1d;
    const step = Math.max(1, Math.floor(sig.length / 1000));
    const sx = [], sy = [];
    for (let i = 0; i < sig.length; i += step) {
      sx.push(i);
      sy.push(sig[i]);
    }
    state.plotData.sig1d = { xs: sx, ys: sy, xLabel: "Index", yLabel: "Amplitude", color: "#3b82f6" };
    drawInteractiveLinePlot(el("plot-1d-canvas"), state.plotData.sig1d);

    const fx = [], fy = [];
    for (let i = 0; i < res.fourier.freqs.length; i++) {
      fx.push(Math.log10(res.fourier.freqs[i]));
      fy.push(Math.log10(res.fourier.power[i] + 1e-12));
    }
    state.plotData.fft1d = { xs: fx, ys: fy, xLabel: "log10(freq)", yLabel: "log10(power)", color: "#22c55e" };
    drawInteractiveLinePlot(el("plot-fft-canvas"), state.plotData.fft1d);
  } else {
    el("bench-text-plots").classList.add("hidden");
    el("bench-image-plots").classList.remove("hidden");
    drawGrayMatrix(el("plot-psd2d-canvas"), res.psd2d, res.psdH, res.psdW);
    el("bench-json-details").textContent = JSON.stringify(
      {
        ...res.golden_ratio,
        directional_anisotropy: res.fourier.anisotropy,
        color_chroma: res.meta.color_chroma,
        ...info,
      },
      null,
      2
    );
  }
}

// --- SNAPSHOT TABLE RENDER ---
function refreshSnapshotControls() {
  const has = state.snapshots.length > 0;
  compareSel.classList.toggle("hidden", !has);
  clearSnapsBtn.classList.toggle("hidden", !has);
  compareTip.classList.toggle("hidden", has);
  el("snapshots-section").classList.toggle("hidden", state.viewMode === "play" || !has);

  const prevVal = compareSel.value;
  compareSel.innerHTML = `<option value="">None (no comparison)</option>`;
  state.snapshots.forEach((snap, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = snap.label;
    compareSel.appendChild(opt);
  });
  if (prevVal !== "" && state.snapshots[parseInt(prevVal, 10)]) {
    compareSel.value = prevVal;
  }
  visualCmpTabs.classList.toggle("hidden", compareSel.value === "");

  el("snapshots-title").textContent = `Pinned Snapshots (${state.snapshots.length})`;
  const tbody = el("snapshots-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const snap of state.snapshots) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${snap.label}</td>
      <td>${snap.input_type}</td>
      <td>${snap.regime}</td>
      <td>${snap.roughness}</td>
      <td>${snap.harmony}</td>
      <td>${snap.light_bloom}</td>
      <td>${snap.info_density}</td>
      <td>${snap.slope}</td>
      <td>${snap.fd}</td>
      <td>${snap.light_fd}</td>
      <td>${snap.shannon_bits}</td>
      <td>${snap.spectral_entropy}</td>
      <td>${snap.compress_ratio}</td>
      <td>${snap.scintillation}</td>
      <td class="mono">${snap.wave_color}</td>
    `;
    tbody.appendChild(tr);
  }
}

// --- EVENT WIRING ---
attachPlotHover("plot-1d-canvas", "plot-1d-readout", "sig1d");
attachPlotHover("plot-fft-canvas", "plot-fft-readout", "fft1d");

el("copy-1d-csv").addEventListener("click", async (e) => {
  const p = state.plotData.sig1d;
  if (!p) return;
  const csv = ["index,amplitude", ...p.xs.map((x, i) => `${x},${p.ys[i]}`)].join("\n");
  await navigator.clipboard.writeText(csv);
  flashButtonText(e.target, "Copied!");
});

el("copy-fft-csv").addEventListener("click", async (e) => {
  const p = state.plotData.fft1d;
  if (!p) return;
  const csv = ["log10_freq,log10_power", ...p.xs.map((x, i) => `${x},${p.ys[i]}`)].join("\n");
  await navigator.clipboard.writeText(csv);
  flashButtonText(e.target, "Copied!");
});

el("save-1d-png").addEventListener("click", () => downloadCanvasPng("plot-1d-canvas", "signal_1d.png"));
el("save-fft-png").addEventListener("click", () => downloadCanvasPng("plot-fft-canvas", "fourier_spectrum.png"));
el("save-psd2d-png").addEventListener("click", () => downloadCanvasPng("plot-psd2d-canvas", "psd_2d.png"));

el("copy-json-btn").addEventListener("click", async (e) => {
  await navigator.clipboard.writeText(el("bench-json-details").textContent);
  flashButtonText(e.target, "Copied!");
});

document.querySelectorAll("#source-type-tabs .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#source-type-tabs .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.sourceType = btn.dataset.source;
    el("text-controls").classList.toggle("hidden", state.sourceType !== "text");
    el("image-controls").classList.toggle("hidden", state.sourceType !== "image");
    recomputeAll();
  });
});

document.querySelectorAll("#view-mode-tabs .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#view-mode-tabs .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.viewMode = btn.dataset.view;
    updateViewVisibility();
    recomputeAll();
  });
});

document.querySelectorAll("#visual-cmp-tabs .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#visual-cmp-tabs .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.visualCmpMode = btn.dataset.cmp;
    runOpticsAndRender();
  });
});

document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.uploadedText = null;
    textFileInput.value = "";
    textFileInfo.classList.add("hidden");
    textInput.value = TEXT_PRESETS[btn.dataset.preset];
    playTextInput.value = textInput.value;
    recomputeAll();
  });
});

let textDebounce;
playTextInput.addEventListener("input", () => {
  state.uploadedText = null;
  textFileInput.value = "";
  textFileInfo.classList.add("hidden");
  textInput.value = playTextInput.value;
  clearTimeout(textDebounce);
  textDebounce = setTimeout(recomputeAll, 140);
});

textInput.addEventListener("input", () => {
  state.uploadedText = null;
  textFileInfo.classList.add("hidden");
  playTextInput.value = textInput.value;
  clearTimeout(textDebounce);
  textDebounce = setTimeout(recomputeAll, 140);
});

textFileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  state.uploadedText = await file.text();
  state.uploadedTextName = file.name;
  textFileInfo.textContent = `Loaded ${file.name} (${state.uploadedText.length.toLocaleString()} chars)`;
  textFileInfo.classList.remove("hidden");
  recomputeAll();
});

hiddenBinaryChk.addEventListener("change", () => {
  textModeSel.disabled = hiddenBinaryChk.checked;
  recomputeAll();
});
textModeSel.addEventListener("change", recomputeAll);
textLayoutSel.addEventListener("change", recomputeAll);

imgSampleSel.addEventListener("change", () => {
  imgUploadWrap.classList.toggle("hidden", imgSampleSel.value !== "upload");
  recomputeAll();
});

imgFileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  state.uploadedImageObj = await loadImageBlob(file, 256);
  state.uploadedImageName = file.name;
  recomputeAll();
});

document.querySelectorAll('input[name="optics-mode"]').forEach((r) => {
  r.addEventListener("change", runOpticsAndRender);
});

whiteLightChk.addEventListener("change", () => {
  wavelengthWrap.classList.toggle("hidden", whiteLightChk.checked);
  runOpticsAndRender();
});

let opticsRaf = null;
function scheduleOpticsRender() {
  if (opticsRaf) return;
  opticsRaf = requestAnimationFrame(() => {
    opticsRaf = null;
    runOpticsAndRender();
  });
}

wavelengthSlider.addEventListener("input", () => {
  wavelengthVal.textContent = `${wavelengthSlider.value} nm`;
  scheduleOpticsRender();
});

distanceSlider.addEventListener("input", () => {
  distanceVal.textContent = `${distanceSlider.value} mm`;
  scheduleOpticsRender();
});

showPhiChk.addEventListener("change", runOpticsAndRender);

el("pin-snapshot-btn").addEventListener("click", () => {
  if (!state.res) return;
  const res = state.res;
  const s = res.simple;
  const info = res.information;
  const params = getOpticsParams();
  const slopeVal = res.input_type === "text" ? res.fourier.spectral_slope : res.fourier.radial_spectral_slope;
  const [r, g, b] = res.spectral_rgb;
  const hex = `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
  const beamTag = params.whiteLight ? "RGB" : `${Math.round(params.wavelengthNm)}nm`;

  state.snapCounter += 1;
  state.snapshots.push({
    label: `#${state.snapCounter}: ${getActiveSourceLabel()} (${params.opticsMode}, ${beamTag} @ ${Math.round(params.distanceMm)}mm)`,
    input_type: res.input_type,
    regime: s.regime,
    roughness: s.texture,
    harmony: s.harmony,
    light_bloom: s.light_bloom,
    info_density: s.info_density,
    slope: slopeVal,
    fd: res.fractal_dimension,
    light_fd: res.light_fractal_dimension,
    shannon_bits: info.shannon_bits,
    spectral_entropy: info.spectral_entropy,
    compress_ratio: info.compress_ratio,
    scintillation: res.scintillation_index,
    wave_color: hex,
    apertureU8: new Uint8ClampedArray(state.currentImages.apertureU8),
    lfU8: new Uint8ClampedArray(state.currentImages.lfU8),
    h: state.currentImages.h,
    w: state.currentImages.w,
  });
  refreshSnapshotControls();
});

compareSel.addEventListener("change", () => {
  visualCmpTabs.classList.toggle("hidden", compareSel.value === "");
  runOpticsAndRender();
});

clearSnapsBtn.addEventListener("click", () => {
  state.snapshots = [];
  state.snapCounter = 0;
  refreshSnapshotControls();
  runOpticsAndRender();
});

el("download-csv-btn").addEventListener("click", () => {
  if (!state.snapshots.length) return;
  const cols = [
    "label", "input_type", "regime", "roughness", "harmony", "light_bloom",
    "info_density", "slope", "fd", "light_fd", "shannon_bits",
    "spectral_entropy", "compress_ratio", "scintillation", "wave_color",
  ];
  const lines = [cols.join(",")];
  for (const snap of state.snapshots) {
    lines.push(cols.map((c) => `"${String(snap[c]).replace(/"/g, '""')}"`).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "optics_snapshots.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

// --- BOOT ---
updateViewVisibility();
recomputeAll();
