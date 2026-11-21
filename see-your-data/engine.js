export const PHI = (1.0 + Math.sqrt(5.0)) / 2.0;

// ==========================================
// 1. FAST RADIX-2 1D & 2D FFT
// ==========================================

function nextPow2(v) {
  let p = 1;
  while (p < v) p <<= 1;
  return p;
}

export function fft1d(re, im, invert = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2.0 * Math.PI) / len) * (invert ? 1 : -1);
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let wRe = 1.0;
      let wIm = 0.0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const idx = i + j + half;
        const vRe = re[idx] * wRe - im[idx] * wIm;
        const vIm = re[idx] * wIm + im[idx] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[idx] = uRe - vRe;
        im[idx] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
  if (invert) {
    const invN = 1.0 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}

export function fft2d(re, im, rows, cols, invert = false) {
  const rowRe = new Float64Array(cols);
  const rowIm = new Float64Array(cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    for (let c = 0; c < cols; c++) {
      rowRe[c] = re[off + c];
      rowIm[c] = im[off + c];
    }
    fft1d(rowRe, rowIm, invert);
    for (let c = 0; c < cols; c++) {
      re[off + c] = rowRe[c];
      im[off + c] = rowIm[c];
    }
  }

  const colRe = new Float64Array(rows);
  const colIm = new Float64Array(rows);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const idx = r * cols + c;
      colRe[r] = re[idx];
      colIm[r] = im[idx];
    }
    fft1d(colRe, colIm, invert);
    for (let r = 0; r < rows; r++) {
      const idx = r * cols + c;
      re[idx] = colRe[r];
      im[idx] = colIm[r];
    }
  }
}

// ==========================================
// 2. HELPERS & STATS
// ==========================================

function linregress(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0.0;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    num += dx * (ys[i] - my);
    den += dx * dx;
  }
  return den > 1e-12 ? num / den : 0.0;
}

function minMaxPtp(arr) {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  return { min, max, ptp: max - min };
}

function meanAndVar(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  const mean = sum / Math.max(1, arr.length);
  let v = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - mean;
    v += d * d;
  }
  return { mean, variance: v / Math.max(1, arr.length) };
}

function roundTo(v, digits = 3) {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
}

function reflectIdx(i, n) {
  if (n <= 1) return 0;
  let idx = i;
  while (idx < 0 || idx >= n) {
    if (idx < 0) idx = -idx;
    if (idx >= n) idx = 2 * n - 2 - idx;
  }
  return idx;
}

// ==========================================
// 3. INPUT PARSERS & SAMPLE GENERATORS
// ==========================================

export function makeSampleImage(kind, size = 256) {
  const data = new Float64Array(size * size);
  let seed = 7;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  if (kind === "ripples") {
    for (let y = 0; y < size; y++) {
      const ny = -1 + (2 * y) / (size - 1);
      for (let x = 0; x < size; x++) {
        const nx = -1 + (2 * x) / (size - 1);
        const r = Math.hypot(nx, ny);
        data[y * size + x] = 0.5 + 0.5 * Math.cos(18.0 * Math.PI * r);
      }
    }
  } else if (kind === "grid") {
    for (let y = 0; y < size; y++) {
      const ny = -1 + (2 * y) / (size - 1);
      const gy = Math.sin(10.0 * Math.PI * ny) > 0 ? 1.0 : 0.0;
      for (let x = 0; x < size; x++) {
        const nx = -1 + (2 * x) / (size - 1);
        const gx = Math.sin(10.0 * Math.PI * nx) > 0 ? 1.0 : 0.0;
        data[y * size + x] = 0.5 * gx + 0.5 * gy;
      }
    }
  } else if (kind === "noise") {
    for (let i = 0; i < data.length; i++) {
      data[i] = rand();
    }
  } else {
    // Organic 1/f Cloud (amplitude ~ f^-1.3 => power ~ f^-2.6)
    const re = new Float64Array(size * size);
    const im = new Float64Array(size * size);
    for (let i = 0; i < size * size; i++) {
      const u1 = Math.max(1e-12, rand());
      const u2 = rand();
      re[i] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }
    fft2d(re, im, size, size, false);
    for (let y = 0; y < size; y++) {
      const fy = y < size / 2 ? y / size : (y - size) / size;
      for (let x = 0; x < size; x++) {
        const fx = x < size / 2 ? x / size : (x - size) / size;
        const f = Math.hypot(fx, fy);
        const w = f > 0 ? 1.0 / (f ** 1.3) : 0.0;
        const idx = y * size + x;
        re[idx] *= w;
        im[idx] *= w;
      }
    }
    fft2d(re, im, size, size, true);
    const { min, ptp } = minMaxPtp(re);
    const denom = ptp + 1e-12;
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.min(1, Math.max(0, (re[i] - min) / denom));
    }
  }
  return {
    aperture: data,
    height: size,
    width: size,
    meta: {
      orig_size: [size, size],
      processed_shape: [size, size],
      color_chroma: 0.0,
    },
  };
}

export async function loadImageBlob(fileOrBlob, maxDim = 256) {
  const bitmap = await createImageBitmap(fileOrBlob);
  const origW = bitmap.width;
  const origH = bitmap.height;
  const scale = Math.min(1.0, maxDim / Math.max(origW, origH));
  const w = Math.max(8, Math.round(origW * scale));
  const h = Math.max(8, Math.round(origH * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  const rgba = ctx.getImageData(0, 0, w, h).data;
  const aperture = new Float64Array(w * h);
  let chromaSum = 0.0;

  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4] / 255.0;
    const g = rgba[i * 4 + 1] / 255.0;
    const b = rgba[i * 4 + 2] / 255.0;
    const a = rgba[i * 4 + 3] / 255.0;
    const cMax = Math.max(r, g, b);
    const cMin = Math.min(r, g, b);
    chromaSum += (cMax - cMin) * a;
    aperture[i] = Math.min(1.0, Math.max(0.0, (0.299 * r + 0.587 * g + 0.114 * b) * a));
  }

  return {
    aperture,
    height: h,
    width: w,
    meta: {
      orig_size: [origW, origH],
      processed_shape: [h, w],
      color_chroma: roundTo(chromaSum / (w * h), 3),
    },
  };
}

export function textTo1dSignal(text, mode = "utf8") {
  if (!text) return { signal: new Float64Array(8), mode: "empty", rawBytes: new Uint8Array(0) };
  const rawBytes = new TextEncoder().encode(text);
  let usedMode = mode;
  if (usedMode === "auto") {
    usedMode = rawBytes.length < 64 ? "bits" : "utf8";
  }

  let signal;
  if (usedMode === "bits") {
    signal = new Float64Array(rawBytes.length * 8);
    for (let i = 0; i < rawBytes.length; i++) {
      const byte = rawBytes[i];
      for (let b = 0; b < 8; b++) {
        signal[i * 8 + b] = (byte >> (7 - b)) & 1;
      }
    }
  } else if (usedMode === "word_lengths") {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length > 0) {
      signal = new Float64Array(words.length);
      for (let i = 0; i < words.length; i++) signal[i] = words[i].length;
    } else {
      signal = Float64Array.from(rawBytes);
    }
  } else {
    signal = Float64Array.from(rawBytes);
  }
  return { signal, mode: usedMode, rawBytes };
}

export function textTo2dAperture(text, size = 256, layout = "grating", mode = "utf8") {
  const { signal } = textTo1dSignal(text, mode);
  const { min, ptp } = minMaxPtp(signal);
  const norm = new Float64Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    norm[i] = (signal[i] - min) / (ptp + 1e-12);
  }

  const mask = new Float64Array(size * size);
  if (layout === "matrix" && norm.length >= 64) {
    const side = Math.ceil(Math.sqrt(norm.length));
    for (let y = 0; y < size; y++) {
      const sy0 = Math.floor((y * side) / size);
      const sy1 = Math.max(sy0 + 1, Math.floor(((y + 1) * side) / size));
      for (let x = 0; x < size; x++) {
        const sx0 = Math.floor((x * side) / size);
        const sx1 = Math.max(sx0 + 1, Math.floor(((x + 1) * side) / size));
        let sum = 0, cnt = 0;
        for (let py = sy0; py < sy1; py++) {
          for (let px = sx0; px < sx1; px++) {
            const idx = py * side + px;
            sum += idx < norm.length ? norm[idx] : 0.0;
            cnt++;
          }
        }
        mask[y * size + x] = sum / Math.max(1, cnt);
      }
    }
  } else {
    const row = new Float64Array(size);
    if (norm.length > size) {
      for (let x = 0; x < size; x++) {
        const i0 = Math.floor((x * norm.length) / size);
        const i1 = Math.max(i0 + 1, Math.floor(((x + 1) * norm.length) / size));
        let sum = 0;
        for (let i = i0; i < i1; i++) sum += norm[i];
        row[x] = sum / (i1 - i0);
      }
    } else {
      for (let x = 0; x < size; x++) {
        const idx = Math.floor((x * (norm.length - 1)) / Math.max(1, size - 1));
        row[x] = norm[idx];
      }
    }
    const isCrossed = layout === "crossed";
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        mask[y * size + x] = isCrossed ? 0.5 * (row[x] + row[y]) : row[x];
      }
    }
  }

  const out = new Float64Array(size * size);
  for (let y = 0; y < size; y++) {
    const ny = -1 + (2 * y) / (size - 1);
    for (let x = 0; x < size; x++) {
      const nx = -1 + (2 * x) / (size - 1);
      const beam = Math.exp(-(nx * nx + ny * ny) / (2 * 0.45 * 0.45));
      out[y * size + x] = Math.min(1.0, Math.max(0.0, mask[y * size + x] * beam));
    }
  }
  return out;
}

// ==========================================
// 4. SHANNON, FOURIER, FRACTAL & GOLDEN RATIO
// ==========================================

async function compressRatioDeflate(rawBytes) {
  if (!rawBytes || rawBytes.length === 0) return 0.0;
  if (typeof CompressionStream !== "undefined") {
    try {
      const stream = new Blob([rawBytes]).stream().pipeThrough(new CompressionStream("deflate"));
      const buf = await new Response(stream).arrayBuffer();
      const payload = Math.max(0, buf.byteLength - 8);
      return roundTo(Math.min(1.0, Math.max(0.0, payload / rawBytes.length)), 3);
    } catch {
      // Fall through
    }
  }
  let runs = 1;
  for (let i = 1; i < rawBytes.length; i++) {
    if (rawBytes[i] !== rawBytes[i - 1]) runs++;
  }
  return roundTo(Math.min(1.0, runs / rawBytes.length), 3);
}

async function computeInformationMetrics(rawBytes, discreteValues, powerSpectrum, alphabetBits = 8.0) {
  const counts = new Map();
  for (let i = 0; i < discreteValues.length; i++) {
    const v = discreteValues[i];
    counts.set(v, (counts.get(v) || 0) + 1);
  }
  const n = Math.max(1, discreteValues.length);
  let shannonBits = 0.0;
  for (const c of counts.values()) {
    const p = c / n;
    shannonBits -= p * Math.log2(p + 1e-12);
  }
  const shannonNorm = Math.min(1.0, Math.max(0.0, shannonBits / Math.max(1e-6, alphabetBits)));

  let pSum = 0.0;
  let validCount = 0;
  for (let i = 0; i < powerSpectrum.length; i++) {
    if (powerSpectrum[i] > 0) {
      pSum += powerSpectrum[i];
      validCount++;
    }
  }
  let spectralEntropy = 0.0;
  if (validCount >= 2 && pSum > 0) {
    let hNat = 0.0;
    for (let i = 0; i < powerSpectrum.length; i++) {
      if (powerSpectrum[i] > 0) {
        const p = powerSpectrum[i] / pSum;
        hNat -= p * Math.log(p);
      }
    }
    spectralEntropy = Math.min(1.0, Math.max(0.0, hNat / Math.log(validCount)));
  }

  const compressRatio = await compressRatioDeflate(rawBytes);

  return {
    shannon_bits: roundTo(shannonBits, 3),
    shannon_norm: roundTo(shannonNorm, 3),
    spectral_entropy: roundTo(spectralEntropy, 3),
    compress_ratio: compressRatio,
  };
}

export function analyzeFourier1d(signal) {
  const { ptp } = minMaxPtp(signal);
  if (signal.length < 4 || ptp === 0) {
    return {
      spectral_slope: 0.0,
      phi_harmonic_score_pct: 0.0,
      freqs: new Float64Array([1.0]),
      power: new Float64Array([0.0]),
    };
  }

  const { mean } = meanAndVar(signal);
  const n = nextPow2(signal.length);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < signal.length; i++) re[i] = signal[i] - mean;
  fft1d(re, im, false);

  const half = Math.floor(n / 2);
  const freqs = [];
  const power = [];
  for (let k = 1; k <= half; k++) {
    const p = re[k] * re[k] + im[k] * im[k];
    if (p > 0) {
      freqs.push(k / n);
      power.push(p);
    }
  }

  let slope = 0.0;
  if (freqs.length >= 3) {
    const lx = freqs.map((f) => Math.log10(f));
    const ly = power.map((p) => Math.log10(p));
    slope = -linregress(lx, ly);
  }

  let phiHarm = 0.0;
  if (freqs.length >= 2) {
    let i1 = -1, i2 = -1, p1 = -1, p2 = -1;
    for (let i = 0; i < power.length; i++) {
      if (power[i] > p1) {
        p2 = p1; i2 = i1;
        p1 = power[i]; i1 = i;
      } else if (power[i] > p2) {
        p2 = power[i]; i2 = i;
      }
    }
    const f1 = freqs[i1], f2 = freqs[i2];
    const ratio = Math.max(f1, f2) / (Math.min(f1, f2) + 1e-12);
    phiHarm = Math.max(0.0, 1.0 - Math.abs(ratio - PHI) / PHI) * 100.0;
  }

  return {
    spectral_slope: roundTo(slope, 3),
    phi_harmonic_score_pct: roundTo(phiHarm, 2),
    freqs: Float64Array.from(freqs),
    power: Float64Array.from(power),
  };
}

export function analyzeFourier2d(matrix, h, w) {
  const { ptp } = minMaxPtp(matrix);
  if (ptp < 1e-6) {
    return {
      radial_spectral_slope: 0.0,
      anisotropy: 0.0,
      psd2d: new Float64Array(h * w),
      psdH: h,
      psdW: w,
      freqs: new Float64Array([1.0]),
      power: new Float64Array([0.0]),
    };
  }

  const { mean } = meanAndVar(matrix);
  const ph = nextPow2(h);
  const pw = nextPow2(w);
  const re = new Float64Array(ph * pw);
  const im = new Float64Array(ph * pw);
  const winRe = new Float64Array(ph * pw);
  const winIm = new Float64Array(ph * pw);

  for (let y = 0; y < h; y++) {
    const wy = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * y) / Math.max(1, h - 1)));
    for (let x = 0; x < w; x++) {
      const wx = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * x) / Math.max(1, w - 1)));
      const val = matrix[y * w + x] - mean;
      re[y * pw + x] = val;
      winRe[y * pw + x] = val * wy * wx;
    }
  }

  fft2d(re, im, ph, pw, false);
  fft2d(winRe, winIm, ph, pw, false);

  const psd2d = new Float64Array(ph * pw);
  const cy = Math.floor(ph / 2);
  const cx = Math.floor(pw / 2);
  const maxR = Math.max(2, Math.min(cx, cy));
  const radSum = new Float64Array(maxR);
  const radCnt = new Uint32Array(maxR);
  const angSum = new Float64Array(36);
  const angCnt = new Uint32Array(36);

  for (let sy = 0; sy < ph; sy++) {
    const y = (sy + cy) % ph;
    const dy = sy - cy;
    for (let sx = 0; sx < pw; sx++) {
      const x = (sx + cx) % pw;
      const dx = sx - cx;
      const idx = y * pw + x;
      const p = re[idx] * re[idx] + im[idx] * im[idx];
      psd2d[sy * pw + sx] = Math.log1p(p);

      const r = Math.floor(Math.hypot(dx, dy));
      if (r >= 1 && r < maxR) {
        radSum[r] += p;
        radCnt[r]++;
      }
      if (r >= 3 && r < maxR) {
        let theta = Math.atan2(dy, dx) % Math.PI;
        if (theta < 0) theta += Math.PI;
        const bin = Math.min(35, Math.max(0, Math.floor((theta / Math.PI) * 36)));
        angSum[bin] += Math.hypot(winRe[idx], winIm[idx]);
        angCnt[bin]++;
      }
    }
  }

  const freqs = [];
  const power = [];
  for (let r = 1; r < maxR; r++) {
    const val = radCnt[r] > 0 ? radSum[r] / radCnt[r] : 0;
    if (val > 0) {
      freqs.push(r);
      power.push(val);
    }
  }

  let slope = 0.0;
  if (freqs.length >= 3) {
    slope = -linregress(freqs.map((f) => Math.log10(f)), power.map((p) => Math.log10(p)));
  }

  const angMeans = new Float64Array(36);
  for (let b = 0; b < 36; b++) {
    angMeans[b] = angCnt[b] > 0 ? angSum[b] / angCnt[b] : 0;
  }
  const { mean: am, variance: av } = meanAndVar(angMeans);
  const anisotropy = Math.sqrt(av) / (am + 1e-12);

  return {
    radial_spectral_slope: roundTo(slope, 3),
    anisotropy: roundTo(anisotropy, 3),
    psd2d,
    psdH: ph,
    psdW: pw,
    freqs: Float64Array.from(freqs),
    power: Float64Array.from(power),
  };
}

export function higuchiFd1d(signal, kmax = 8) {
  let sig = signal;
  if (sig.length > 20000) {
    const step = Math.floor(sig.length / 20000);
    const down = [];
    for (let i = 0; i < sig.length; i += step) down.push(sig[i]);
    sig = Float64Array.from(down);
  }
  const N = sig.length;
  const { ptp } = minMaxPtp(sig);
  if (N < 6 || ptp === 0) return 1.0;

  const km = Math.min(kmax, Math.max(2, Math.floor(N / 3)));
  const xs = [];
  const ys = [];
  for (let k = 1; k <= km; k++) {
    let lkSum = 0;
    let lkCnt = 0;
    for (let m = 0; m < k; m++) {
      let lenIdx = 0;
      let diffSum = 0;
      let prev = sig[m];
      for (let i = m + k; i < N; i += k) {
        diffSum += Math.abs(sig[i] - prev);
        prev = sig[i];
        lenIdx++;
      }
      if (lenIdx >= 1) {
        lkSum += (diffSum * (N - 1)) / (lenIdx * k * k);
        lkCnt++;
      }
    }
    const L = lkCnt > 0 ? lkSum / lkCnt : 0;
    // Skip zero-diff harmonic steps (e.g. even k on "abab") to avoid log(1e-12) slope blow-up
    if (L > 1e-12) {
      xs.push(Math.log(1.0 / k));
      ys.push(Math.log(L));
    }
  }
  if (xs.length < 2) return 1.0;
  const slope = linregress(xs, ys);
  return roundTo(Math.min(2.0, Math.max(1.0, slope)), 3);
}

export function boxCount2d(matrix, h, w) {
  const { ptp } = minMaxPtp(matrix);
  if (ptp < 1e-6) return 0.0;

  const p = Math.min(h, w);
  const maxExp = Math.floor(Math.log2(p));
  if (maxExp < 3) return 1.0;

  const xs = [];
  const ys = [];
  for (let exp = maxExp; exp >= 2; exp--) {
    const size = 1 << exp;
    const hBlocks = Math.floor(h / size);
    const wBlocks = Math.floor(w / size);
    const hBox = size / p;
    let totalBoxes = 0;

    for (let by = 0; by < hBlocks; by++) {
      for (let bx = 0; bx < wBlocks; bx++) {
        let bMin = 1.0, bMax = 0.0;
        for (let dy = 0; dy < size; dy++) {
          const rowOff = (by * size + dy) * w + bx * size;
          for (let dx = 0; dx < size; dx++) {
            const v = Math.min(1.0, Math.max(0.0, matrix[rowOff + dx]));
            if (v < bMin) bMin = v;
            if (v > bMax) bMax = v;
          }
        }
        totalBoxes += Math.max(1.0, Math.ceil(bMax / hBox) - Math.floor(bMin / hBox));
      }
    }
    xs.push(Math.log(1.0 / size));
    ys.push(Math.log(Math.max(1.0, totalBoxes)));
  }

  const slope = linregress(xs, ys);
  return roundTo(Math.min(2.0, Math.max(1.0, slope - 1.0)), 3);
}

function goldenRatio1d(signal, tol = 0.05) {
  let valid = 0, matches = 0;
  let prevDiff = signal.length > 1 ? Math.abs(signal[1] - signal[0]) : 0;
  for (let i = 2; i < signal.length; i++) {
    const curDiff = Math.abs(signal[i] - signal[i - 1]);
    if (prevDiff > 0 && curDiff > 0) {
      valid++;
      const ratio = Math.max(prevDiff, curDiff) / Math.min(prevDiff, curDiff);
      if (Math.abs(ratio - PHI) / PHI <= tol) matches++;
    }
    prevDiff = curDiff;
  }
  return valid > 0 ? roundTo((matches / valid) * 100.0, 2) : 0.0;
}

function goldenRatio2d(matrix, h, w, origSize) {
  const oh = origSize ? origSize[1] : h;
  const ow = origSize ? origSize[0] : w;
  const aspect = Math.max(oh, ow) / Math.max(1, Math.min(oh, ow));
  const aspectScore = Math.max(0.0, 1.0 - Math.abs(aspect - PHI) / PHI) * 100.0;

  const band = Math.max(1, Math.floor(0.03 * Math.min(h, w)));
  const y1 = Math.floor(h / PHI), y2 = Math.floor(h * (1 - 1 / PHI));
  const x1 = Math.floor(w / PHI), x2 = Math.floor(w * (1 - 1 / PHI));

  let totalE = 0, phiE = 0, phiCnt = 0;
  for (let y = 0; y < h; y++) {
    const inY = Math.abs(y - y1) < band || Math.abs(y - y2) < band;
    for (let x = 0; x < w; x++) {
      const ym = Math.max(0, y - 1), yp = Math.min(h - 1, y + 1);
      const xm = Math.max(0, x - 1), xp = Math.min(w - 1, x + 1);
      const gy = (matrix[yp * w + x] - matrix[ym * w + x]) / (yp === ym ? 1 : (yp - ym));
      const gx = (matrix[y * w + xp] - matrix[y * w + xm]) / (xp === xm ? 1 : (xp - xm));
      const e = Math.hypot(gx, gy);
      totalE += e;
      const inX = Math.abs(x - x1) < band || Math.abs(x - x2) < band;
      if (inY || inX) {
        phiE += e;
        phiCnt++;
      }
    }
  }
  const meanAll = totalE / Math.max(1, h * w);
  const meanPhi = phiE / Math.max(1, phiCnt);
  return {
    aspect_phi_score_pct: roundTo(aspectScore, 2),
    phi_grid_energy_density: roundTo(meanPhi / (meanAll + 1e-12), 3),
  };
}

// ==========================================
// 5. COLORIMETRY & WAVE OPTICS
// ==========================================

function cieXyzCurves(wl) {
  const g = (mu, t1, t2) => {
    const tau = wl < mu ? t1 : t2;
    const d = (wl - mu) * tau;
    return Math.exp(-0.5 * d * d);
  };
  return [
    0.362 * g(442.0, 0.0624, 0.0374) + 1.056 * g(599.8, 0.0264, 0.0323) - 0.065 * g(501.1, 0.0490, 0.0382),
    0.821 * g(568.8, 0.0213, 0.0247) + 0.286 * g(530.9, 0.0613, 0.0322),
    1.217 * g(437.0, 0.0845, 0.0278) + 0.681 * g(459.0, 0.0385, 0.0725),
  ];
}

function xyzToSrgb(X, Y, Z) {
  const rLin = Math.max(0, 3.2406 * X - 1.5372 * Y - 0.4986 * Z);
  const gLin = Math.max(0, -0.9689 * X + 1.8758 * Y + 0.0415 * Z);
  const bLin = Math.max(0, 0.0557 * X - 0.2040 * Y + 1.0570 * Z);
  const m = Math.max(rLin, gLin, bLin);
  const norm = (c) => {
    const v = m > 0 ? c / m : 0;
    const s = v <= 0.0031308 ? 12.92 * v : 1.055 * (v ** (1.0 / 2.4)) - 0.055;
    return Math.min(255, Math.max(0, Math.round(s * 255)));
  };
  return [norm(rLin), norm(gLin), norm(bLin)];
}

export function wavelengthToRgb(wavelengthNm) {
  let X = 0, Y = 0, Z = 0;
  for (let i = 0; i < 100; i++) {
    const wl = 380.0 + (370.0 * i) / 99;
    const d = (wl - wavelengthNm) / 8.0;
    const spd = Math.exp(-0.5 * d * d);
    const [xb, yb, zb] = cieXyzCurves(wl);
    X += spd * xb; Y += spd * yb; Z += spd * zb;
  }
  return xyzToSrgb(X, Y, Z);
}

export function spectrumToRgb(freqs, power) {
  if (freqs.length < 2) return [0, 0, 0];
  let pSum = 0;
  for (let i = 0; i < power.length; i++) pSum += power[i];
  if (pSum <= 0) return [0, 0, 0];

  const { min: fMin, ptp: fPtp } = minMaxPtp(freqs);
  const denom = fPtp + 1e-12;
  let X = 0, Y = 0, Z = 0;

  let j = 0;
  for (let i = 0; i < 100; i++) {
    const u = i / 99;
    while (j < freqs.length - 2 && (freqs[j + 1] - fMin) / denom < u) j++;
    const u0 = (freqs[j] - fMin) / denom;
    const u1 = (freqs[j + 1] - fMin) / denom;
    const t = u1 > u0 ? Math.min(1, Math.max(0, (u - u0) / (u1 - u0))) : 0;
    const spd = power[j] * (1 - t) + power[j + 1] * t;
    const wl = 720.0 - (330.0 * i) / 99;
    const [xb, yb, zb] = cieXyzCurves(wl);
    X += spd * xb; Y += spd * yb; Z += spd * zb;
  }
  return xyzToSrgb(X, Y, Z);
}

function buildPaddedFieldFft(mask2d, h, w, opticsMode) {
  const padY = Math.max(32, Math.min(128, Math.floor(h / 4)));
  const padX = Math.max(32, Math.min(128, Math.floor(w / 4)));
  const ph = Math.min(512, nextPow2(h + 2 * padY));
  const pw = Math.min(512, nextPow2(w + 2 * padX));
  const py = Math.floor((ph - h) / 2);
  const px = Math.floor((pw - w) / 2);

  const fRe = new Float64Array(ph * pw);
  const fIm = new Float64Array(ph * pw);

  for (let oy = 0; oy < ph; oy++) {
    const sy = oy - py;
    const inY = sy >= 0 && sy < h;
    for (let ox = 0; ox < pw; ox++) {
      const sx = ox - px;
      const inX = sx >= 0 && sx < w;
      if (opticsMode === "phase") {
        const ry = reflectIdx(sy, h);
        const rx = reflectIdx(sx, w);
        const phase = 2.0 * Math.PI * mask2d[ry * w + rx];
        fRe[oy * pw + ox] = Math.cos(phase);
        fIm[oy * pw + ox] = Math.sin(phase);
      } else if (inY && inX) {
        fRe[oy * pw + ox] = mask2d[sy * w + sx];
      }
    }
  }
  fft2d(fRe, fIm, ph, pw, false);
  return { fRe, fIm, ph, pw, py, px, h, w };
}

function propagateFromCachedFft(cached, wavelengthNm, distanceMm, pixelPitchUm = 10.0) {
  const { fRe, fIm, ph, pw, py, px, h, w } = cached;
  const out = new Float64Array(h * w);

  if (distanceMm <= 0.0) {
    const re = new Float64Array(fRe);
    const im = new Float64Array(fIm);
    fft2d(re, im, ph, pw, true);
    let maxI = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (py + y) * pw + (px + x);
        const val = re[idx] * re[idx] + im[idx] * im[idx];
        out[y * w + x] = val;
        if (val > maxI) maxI = val;
      }
    }
    if (maxI > 0) for (let i = 0; i < out.length; i++) out[i] /= maxI;
    return out;
  }

  const wl = wavelengthNm * 1e-9;
  const dx = pixelPitchUm * 1e-6;
  const z = distanceMm * 1e-3;
  const invWlSq = (1.0 / wl) ** 2;

  const pRe = new Float64Array(ph * pw);
  const pIm = new Float64Array(ph * pw);

  for (let y = 0; y < ph; y++) {
    const fyIdx = y <= ph / 2 ? y : y - ph;
    const fy = fyIdx / (ph * dx);
    const fySq = fy * fy;
    for (let x = 0; x < pw; x++) {
      const fxIdx = x <= pw / 2 ? x : x - pw;
      const fx = fxIdx / (pw * dx);
      const kSq = invWlSq - fx * fx - fySq;
      const idx = y * pw + x;
      if (kSq >= 0) {
        const phi = 2.0 * Math.PI * Math.sqrt(kSq) * z;
        const hRe = Math.cos(phi);
        const hIm = Math.sin(phi);
        pRe[idx] = fRe[idx] * hRe - fIm[idx] * hIm;
        pIm[idx] = fRe[idx] * hIm + fIm[idx] * hRe;
      }
    }
  }

  fft2d(pRe, pIm, ph, pw, true);

  let maxI = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (py + y) * pw + (px + x);
      const val = pRe[idx] * pRe[idx] + pIm[idx] * pIm[idx];
      out[y * w + x] = val;
      if (val > maxI) maxI = val;
    }
  }
  if (maxI > 0) {
    for (let i = 0; i < out.length; i++) out[i] /= maxI;
  }
  return out;
}

// ==========================================
// 6. SUMMARY & ORCHESTRATION
// ==========================================

function computeSimpleSummary(res, lightGray) {
  const isText = res.input_type === "text";
  const isFlat = isText
    ? minMaxPtp(res.signal_1d).ptp === 0
    : minMaxPtp(res.aperture_2d).ptp < 1e-6;
  const info = res.information;

  if (isFlat) {
    return {
      texture: 0.0,
      harmony: 0.0,
      light_bloom: 0.0,
      info_density: 0.0,
      regime: "Uniform / Blank",
      regime_color: "#6b7280",
      regime_hint: "Zero variation—light passes straight through unaltered.",
    };
  }

  // 1. DETAIL / EDGE DENSITY (0 = smooth, 50 = balanced detail, 100 = dense sharp edges)
  const slope = isText ? res.fourier.spectral_slope : res.fourier.radial_spectral_slope;
  let texture;
  if (isText) {
    const sig = res.signal_1d;
    const { ptp } = minMaxPtp(sig);
    let diffSum = 0;
    for (let i = 1; i < sig.length; i++) diffSum += Math.abs(sig[i] - sig[i - 1]);
    const meanStep = diffSum / (Math.max(1, sig.length - 1) * (ptp + 1e-12));
    const scaleDiv = res.meta?.mode === "bits" ? 0.65 : 0.42;
    texture = roundTo(Math.min(100.0, Math.max(0.0, (meanStep / scaleDiv) * 100.0)), 1);
  } else {
    const fd = res.fractal_dimension;
    const fdNorm = fd >= 1.0 ? Math.min(1, Math.max(0, fd - 1.0)) : Math.min(1, Math.max(0, fd * 0.5));
    const slopeRoughness = Math.min(1, Math.max(0, 1.0 - slope / 2.5));
    texture = roundTo(100.0 * (0.60 * fdNorm + 0.40 * slopeRoughness), 1);
  }

  // 2. RHYTHM & PROPORTION (unified 1/f center: ~1.0 for 1D, ~2.0 for 2D)
  let harmony = 0.0;
  if (isText) {
    const pinkBalance = Math.exp(-0.5 * ((slope - 1.0) / 0.55) ** 2) * 100.0;
    const phiHarm = res.fourier.phi_harmonic_score_pct;
    const phiTrans = res.golden_ratio.phi_transition_score_pct;
    const phiPart = phiTrans > 0 ? 0.65 * phiHarm + 0.35 * phiTrans : phiHarm;
    const orderPart = (1.0 - info.spectral_entropy) * 100.0;
    harmony = roundTo(Math.min(100, Math.max(0, 0.50 * pinkBalance + 0.30 * phiPart + 0.20 * orderPart)), 1);
  } else {
    const pinkBalance = Math.exp(-0.5 * ((slope - 2.0) / 1.0) ** 2) * 100.0;
    const aspectRaw = res.golden_ratio.aspect_phi_score_pct;
    // Don't give 1:1 square images a free ~62% floor; reward true near-Golden rectangles
    const aspectBonus = Math.max(0, (aspectRaw - 65.0) / 35.0) * 100.0;
    const gridConc = Math.min(100, Math.max(0, (res.golden_ratio.phi_grid_energy_density - 0.8) * 100.0));
    const phiPart = 0.40 * aspectBonus + 0.60 * gridConc;
    const isotropyScore = Math.exp(-res.fourier.anisotropy * 2.0) * 100.0;
    harmony = roundTo(Math.min(100, Math.max(0, 0.45 * pinkBalance + 0.30 * isotropyScore + 0.25 * phiPart)), 1);
  }

  // 3. LIGHT SCATTER
  const lfd = res.light_fractal_dimension;
  const lfdNorm = lfd >= 1.0 ? Math.min(1, Math.max(0, lfd - 1.0)) : Math.min(1, Math.max(0, lfd * 0.5));
  const { variance: lVar } = meanAndVar(lightGray);
  const lightContrast = Math.min(1, Math.max(0, Math.sqrt(lVar) / 0.35));
  const lightBloom = roundTo(100.0 * (0.65 * lfdNorm + 0.35 * lightContrast), 1);

  // 4. SIGNAL REGIME & INFO DENSITY
  const infoDensity = roundTo(100.0 * (0.50 * info.shannon_norm + 0.50 * info.compress_ratio), 1);

  const isOrdered = isText
    ? info.compress_ratio < 0.30 || info.spectral_entropy < 0.35
    : info.compress_ratio < 0.30 || info.shannon_norm < 0.45 || (info.spectral_entropy < 0.22 && (slope < 1.2 || slope > 3.2));

  const isChaotic = info.compress_ratio > 0.72 && info.spectral_entropy > 0.80 && slope < 0.45;

  let regime, regimeColor, regimeHint;
  if (isOrdered) {
    regime = "Ordered / Repetitive";
    regimeColor = "#2563eb";
    regimeHint = "Predictable repeating patterns lock light into sharp bands or symmetries.";
  } else if (isChaotic) {
    regime = "Chaotic / Static";
    regimeColor = "#d97706";
    regimeHint = "High entropy with flat frequencies—light scatters into uniform speckle.";
  } else {
    regime = "Structured / Natural";
    regimeColor = "#16a34a";
    regimeHint = "Lives between repetition and noise—rich multi-scale form.";
  }

  return {
    texture,
    harmony,
    light_bloom: lightBloom,
    info_density: infoDensity,
    regime,
    regime_color: regimeColor,
    regime_hint: regimeHint,
  };
}

export async function extractInputFeatures({ text = null, imageObj = null, textMode = "utf8", textLayout = "grating" }) {
  if (imageObj) {
    const { aperture, height, width, meta } = imageObj;
    const f2d = analyzeFourier2d(aperture, height, width);
    const quant = new Uint8Array(aperture.length);
    for (let i = 0; i < aperture.length; i++) quant[i] = Math.round(aperture[i] * 255);

    const information = await computeInformationMetrics(quant, quant, f2d.power, 8.0);
    return {
      input_type: "image",
      meta,
      fourier: { radial_spectral_slope: f2d.radial_spectral_slope, anisotropy: f2d.anisotropy },
      fractal_dimension: boxCount2d(aperture, height, width),
      golden_ratio: goldenRatio2d(aperture, height, width, meta.orig_size),
      spectral_rgb: spectrumToRgb(f2d.freqs, f2d.power),
      psd2d: f2d.psd2d,
      psdH: f2d.psdH,
      psdW: f2d.psdW,
      information,
      aperture_2d: aperture,
      height,
      width,
      _fftCache: {},
    };
  }

  const { signal, mode: usedMode, rawBytes } = textTo1dSignal(text || "", textMode);
  const size = 256;
  const aperture = textTo2dAperture(text || "", size, textLayout, usedMode);
  const f1d = analyzeFourier1d(signal);

  let alphaBits = 7.0;
  let discVals = rawBytes;
  if (usedMode === "bits") {
    alphaBits = 1.0;
    discVals = Uint8Array.from(signal);
  } else if (usedMode === "word_lengths") {
    const uniq = new Set(signal).size;
    alphaBits = Math.max(2.0, Math.log2(Math.max(2, uniq)));
    discVals = Int32Array.from(signal);
  }

  const information = await computeInformationMetrics(rawBytes, discVals, f1d.power, alphaBits);
  return {
    input_type: "text",
    meta: { chars: (text || "").length, signal_len: signal.length, mode: usedMode },
    signal_1d: signal,
    fourier: f1d,
    fractal_dimension: higuchiFd1d(signal),
    golden_ratio: { phi_transition_score_pct: goldenRatio1d(signal) },
    spectral_rgb: spectrumToRgb(f1d.freqs, f1d.power),
    information,
    aperture_2d: aperture,
    height: size,
    width: size,
    _fftCache: {},
  };
}

export function simulateOptics(features, { opticsMode = "phase", distanceMm = 25.0, wavelengthNm = 532.0, whiteLight = false }) {
  if (!features._fftCache[opticsMode]) {
    features._fftCache[opticsMode] = buildPaddedFieldFft(
      features.aperture_2d,
      features.height,
      features.width,
      opticsMode
    );
  }
  const cached = features._fftCache[opticsMode];

  let lightField, lightGray, isRgb = false;
  if (whiteLight) {
    const r = propagateFromCachedFft(cached, 640.0, distanceMm);
    const g = propagateFromCachedFft(cached, 532.0, distanceMm);
    const b = propagateFromCachedFft(cached, 450.0, distanceMm);
    lightField = { r, g, b };
    lightGray = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) {
      lightGray[i] = ((r[i] ** 0.65) + (g[i] ** 0.65) + (b[i] ** 0.65)) / 3.0;
    }
    isRgb = true;
  } else {
    lightField = propagateFromCachedFft(cached, wavelengthNm, distanceMm);
    lightGray = lightField;
  }

  const { mean, variance } = meanAndVar(lightGray);
  const scintillation = mean > 1e-9 ? variance / (mean * mean + 1e-12) : 0.0;

  const out = {
    ...features,
    light_field: lightField,
    is_rgb: isRgb,
    light_fractal_dimension: boxCount2d(lightGray, features.height, features.width),
    scintillation_index: roundTo(scintillation, 3),
  };
  out.simple = computeSimpleSummary(out, lightGray);
  return out;
}
