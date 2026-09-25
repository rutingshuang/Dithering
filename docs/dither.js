// ============================================================
// dither.js — 调色板图像抖动算法（JS 移植版）
// 与 Python dithers.py 逻辑一致；Delaunay 数据用离线预计算的 DITHER_DATA 查表
// ============================================================

// ---------- 抖动矩阵 ----------
const DITHER_MATRICES = {
  bayer:     [[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]],
  clustered: [[14,11,8,13],[10,6,4,9],[7,2,0,5],[12,3,1,15]],
  diagonal:  [[15,11,5,0],[10,14,9,4],[3,8,13,7],[1,2,9,12]],
  spiral:    [[9,1,6,14],[3,11,15,4],[5,7,0,8],[12,13,10,2]],
};

// ---------- 颜色转换 ----------
function rgbToLab(rgb) {
  let r = rgb[0] / 255, g = rgb[1] / 255, b = rgb[2] / 255;
  const lin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  r = lin(r); g = lin(g); b = lin(b);
  let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
  let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
  let z = r * 0.0193 + g * 0.1192 + b * 0.9505;
  x /= 0.95047; z /= 1.08883;
  const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// ---------- 最近调色板色 ----------
function nearestRGBIdx(p, palette) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < palette.length; i++) {
    const q = palette[i];
    const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function nearestLabIdx(p, labPalette) {
  const lp = rgbToLab([p[0], p[1], p[2]]);
  let best = 0, bestD = Infinity;
  for (let i = 0; i < labPalette.length; i++) {
    const q = labPalette[i];
    const d = (lp[0] - q[0]) ** 2 + (lp[1] - q[1]) ** 2 + (lp[2] - q[2]) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// 矩阵 → 归一化偏移量(用于扰动) + 阈值(用于选色)
function matrixVals(mat, strength) {
  const m = mat.length;
  const offs = [], ths = [];
  for (let v of mat.flat()) {
    const norm = (v + 0.5) / (m * m);
    offs.push((norm - 0.5) * strength);
    ths.push(norm);
  }
  return { m, offs, ths };
}

// ============================================================
// 有序抖动
// ============================================================
function orderedDither(img, palette, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const mat = DITHER_MATRICES[opts.matrix || 'bayer'];
  const { m, offs } = matrixVals(mat, opts.strength ?? 64);
  const colorSpace = opts.colorSpace || 'RGB';
  const labPal = colorSpace === 'Lab' ? palette.map(rgbToLab) : null;
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      if (data[i + 3] === 0) { out[i + 3] = 0; continue; }
      const off = offs[(y % m) * m + (x % m)];
      if (colorSpace === 'Lab') {
        // 与 Python 一致：Lab 路径先裁剪到 [0,255]
        const pc = [
          Math.min(255, Math.max(0, data[i] + off)),
          Math.min(255, Math.max(0, data[i + 1] + off)),
          Math.min(255, Math.max(0, data[i + 2] + off)),
        ];
        const idx = nearestLabIdx(pc, labPal);
        out[i] = palette[idx][0]; out[i + 1] = palette[idx][1]; out[i + 2] = palette[idx][2];
      } else {
        const idx = nearestRGBIdx([data[i] + off, data[i + 1] + off, data[i + 2] + off], palette);
        out[i] = palette[idx][0]; out[i + 1] = palette[idx][1]; out[i + 2] = palette[idx][2];
      }
      out[i + 3] = data[i + 3] > 0 ? 255 : 0;
    }
  }
  return { width, height, data: out };
}

// ============================================================
// 误差扩散（FS / Jarvis / Stucki）
// ============================================================
const ED_FILTERS = {
  FS:     [[0,1,7/16],[1,-1,3/16],[1,0,5/16],[1,1,1/16]],
  Jarvis: [[0,1,7/48],[0,2,5/48],[1,-2,3/48],[1,-1,5/48],[1,0,7/48],[1,1,5/48],[1,2,3/48],
           [2,-2,1/48],[2,-1,3/48],[2,0,5/48],[2,1,3/48],[2,2,1/48]],
  Stucki: [[0,1,8/42],[0,2,4/42],[1,-2,2/42],[1,-1,4/42],[1,0,8/42],[1,1,4/42],[1,2,2/42],
           [2,-2,1/42],[2,-1,2/42],[2,0,4/42],[2,1,2/42],[2,2,1/42]],
};

function errorDiffusion(img, palette, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const weights = ED_FILTERS[opts.filter || 'FS'];
  const alpha = opts.alpha ?? 1.0;
  // 工作数组：RGB float + 透明度
  const arr = new Float32Array(width * height * 3);
  const alphaCh = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    arr[i * 3] = data[i * 4]; arr[i * 3 + 1] = data[i * 4 + 1]; arr[i * 3 + 2] = data[i * 4 + 2];
    alphaCh[i] = data[i * 4 + 3];
  }
  const out = new Uint8ClampedArray(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (alphaCh[i] === 0) { out[i * 4 + 3] = 0; continue; }
      const old = [arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]];
      const idx = nearestRGBIdx(old, palette);
      const nw = palette[idx];
      out[i * 4] = nw[0]; out[i * 4 + 1] = nw[1]; out[i * 4 + 2] = nw[2];
      out[i * 4 + 3] = alphaCh[i] > 0 ? 255 : 0;
      const qe = [(old[0] - nw[0]) * alpha, (old[1] - nw[1]) * alpha, (old[2] - nw[2]) * alpha];
      for (let wgt of weights) {
        const ny = y + wgt[0], nx = x + wgt[1];
        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
          const j = (ny * width + nx) * 3;
          arr[j] += qe[0] * wgt[2]; arr[j + 1] += qe[1] * wgt[2]; arr[j + 2] += qe[2] * wgt[2];
        }
      }
    }
  }
  return { width, height, data: out };
}

// ============================================================
// 重心混合（barycentric）—— 依赖 DITHER_DATA 查表
// ============================================================

// Lab 边缘图（邻近像素 ΔE 之和 / 2）
function labEdgeMap(img) {
  const { width, height, data } = img;
  const n = width * height;
  const lab = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const lp = rgbToLab([data[i * 4], data[i * 4 + 1], data[i * 4 + 2]]);
    lab[i * 3] = lp[0]; lab[i * 3 + 1] = lp[1]; lab[i * 3 + 2] = lp[2];
  }
  const em = new Float32Array(n);
  // gy: 行间 ΔE
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width; x++) {
      const a = y * width + x, b = (y + 1) * width + x;
      const g = Math.hypot(lab[a * 3] - lab[b * 3], lab[a * 3 + 1] - lab[b * 3 + 1], lab[a * 3 + 2] - lab[b * 3 + 2]);
      em[a] += g; em[b] += g;
    }
  }
  // gx: 列间 ΔE
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = y * width + x, b = y * width + x + 1;
      const g = Math.hypot(lab[a * 3] - lab[b * 3], lab[a * 3 + 1] - lab[b * 3 + 1], lab[a * 3 + 2] - lab[b * 3 + 2]);
      em[a] += g; em[b] += g;
    }
  }
  for (let i = 0; i < n; i++) em[i] /= 2;
  return em;
}

// 4×4 矩阵 × 4 向量
function matVec4(A, v) {
  return [
    A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2]+A[0][3]*v[3],
    A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2]+A[1][3]*v[3],
    A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]+A[2][3]*v[3],
    A[3][0]*v[0]+A[3][1]*v[1]+A[3][2]*v[2]+A[3][3]*v[3],
  ];
}

// 完整遍历所有四面体，找包含 p 的（p 不在任何四面体=凸包外 → -1）
// 判定阈值与 scipy find_simplex 一致（权重 ≥ -1e-9 视为包含）
const BARY_EPS = -1e-9;
function findContaining(p, D) {
  const v4 = [p[0], p[1], p[2], 1.0];
  for (let t = 0; t < D.tetras.length; t++) {
    const w = matVec4(D.ainv[t], v4);
    if (w[0] >= BARY_EPS && w[1] >= BARY_EPS && w[2] >= BARY_EPS && w[3] >= BARY_EPS) return t;
  }
  return -1;
}

// ---------- 运行时 3D Delaunay（Bowyer-Watson）用于子集调色板 ----------

// 4×4 矩阵求逆（Gauss-Jordan）
function invMat4(m) {
  const a = m.map(r => r.slice());
  const inv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for (let col = 0; col < 4; col++) {
    let piv = col;
    for (let r = col; r < 4; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) < 1e-12) return null; // 奇异
    [a[col], a[piv]] = [a[piv], a[col]];
    [inv[col], inv[piv]] = [inv[piv], inv[col]];
    const d = a[col][col];
    for (let j = 0; j < 4; j++) { a[col][j] /= d; inv[col][j] /= d; }
    for (let r = 0; r < 4; r++) if (r !== col) {
      const f = a[r][col];
      if (f !== 0) for (let j = 0; j < 4; j++) { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
    }
  }
  return inv;
}

function det3(m) {
  return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
       - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
       + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
}

// 四面体 a,b,c,d 的外接球心与半径平方
function circumsphere(a, b, c, d) {
  const ab2 = a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
  const db2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
  const m = [
    [a[0]-d[0], a[1]-d[1], a[2]-d[2], (ab2-db2)/2],
    [b[0]-d[0], b[1]-d[1], b[2]-d[2], (b[0]*b[0]+b[1]*b[1]+b[2]*b[2]-db2)/2],
    [c[0]-d[0], c[1]-d[1], c[2]-d[2], (c[0]*c[0]+c[1]*c[1]+c[2]*c[2]-db2)/2],
  ];
  const det = det3(m);
  if (Math.abs(det) < 1e-12) return [0,0,0,Infinity];
  const cx = det3([[m[0][3],m[0][1],m[0][2]],[m[1][3],m[1][1],m[1][2]],[m[2][3],m[2][1],m[2][2]]]) / det;
  const cy = det3([[m[0][0],m[0][3],m[0][2]],[m[1][0],m[1][3],m[1][2]],[m[2][0],m[2][3],m[2][2]]]) / det;
  const cz = det3([[m[0][0],m[0][1],m[0][3]],[m[1][0],m[1][1],m[1][3]],[m[2][0],m[2][1],m[2][3]]]) / det;
  const r2 = (a[0]-cx)**2 + (a[1]-cy)**2 + (a[2]-cz)**2;
  return [cx, cy, cz, r2];
}

// 3D Delaunay（Bowyer-Watson），返回四面体（索引元组）
function computeDelaunay(points) {
  const n = points.length;
  if (n < 4) return [];
  let minx=1e9,miny=1e9,minz=1e9,maxx=-1e9,maxy=-1e9,maxz=-1e9;
  for (const p of points) { minx=Math.min(minx,p[0]); maxx=Math.max(maxx,p[0]); miny=Math.min(miny,p[1]); maxy=Math.max(maxy,p[1]); minz=Math.min(minz,p[2]); maxz=Math.max(maxz,p[2]); }
  const dmax = Math.max(maxx-minx, maxy-miny, maxz-minz) || 1;
  const big = 50 * dmax;  // 超四面体要足够大，否则边界四面体会丢失
  const midx=(minx+maxx)/2, midy=(miny+maxy)/2, midz=(minz+maxz)/2;
  const st = [
    [midx-2*big, midy-big, midz-big],
    [midx+2*big, midy-big, midz-big],
    [midx, midy+2*big, midz-big],
    [midx, midy, midz+2*big],
  ];
  const all = points.concat(st);
  const cs0 = circumsphere(st[0], st[1], st[2], st[3]);
  let tetras = [[n, n+1, n+2, n+3, cs0[0], cs0[1], cs0[2], cs0[3]]];
  for (let i = 0; i < n; i++) {
    const p = points[i];
    // 找出外接球包含 p 的四面体（bad）
    const bad = [];
    const badSet = new Set();
    for (let t of tetras) {
      const dx = p[0]-t[4], dy = p[1]-t[5], dz = p[2]-t[6];
      if (dx*dx + dy*dy + dz*dz <= t[7] + 1e-9) { bad.push(t); badSet.add(t); }
    }
    // 边界面（bad 四面体的面中未被两个 bad 共享的）
    const faceCount = new Map();
    for (const t of bad) for (const f of tetraFaces(t)) {
      const k = f.slice().sort().join(',');
      faceCount.set(k, (faceCount.get(k)||0) + 1);
    }
    const boundary = [];
    for (const t of bad) for (const f of tetraFaces(t)) {
      if (faceCount.get(f.slice().sort().join(',')) === 1) boundary.push(f);
    }
    tetras = tetras.filter(t => !badSet.has(t));
    for (const f of boundary) {
      const nt = [f[0], f[1], f[2], i];
      const cs = circumsphere(all[nt[0]], all[nt[1]], all[nt[2]], all[nt[3]]);
      tetras.push([nt[0], nt[1], nt[2], nt[3], cs[0], cs[1], cs[2], cs[3]]);
    }
  }
  // 移除含超四面体顶点的四面体
  return tetras
    .filter(t => t[0] < n && t[1] < n && t[2] < n && t[3] < n)
    .map(t => [t[0], t[1], t[2], t[3]]);
}

function tetraFaces(t) {
  return [[t[0],t[1],t[2]],[t[0],t[1],t[3]],[t[0],t[2],t[3]],[t[1],t[2],t[3]]];
}

// 为选中子集构建重心数据：{palette, tetras, ainv, hull, grid}
function buildBaryData(selectedPalette) {
  const points = selectedPalette.map(c => [c[0], c[1], c[2]]);
  const tetras = computeDelaunay(points);
  // 重心逆矩阵
  const ainv = tetras.map(t => {
    const V = t.map(i => points[i]);
    const A = [
      [V[0][0],V[1][0],V[2][0],V[3][0]],
      [V[0][1],V[1][1],V[2][1],V[3][1]],
      [V[0][2],V[1][2],V[2][2],V[3][2]],
      [1,1,1,1],
    ];
    return invMat4(A);
  });
  // 凸包面（只被一个四面体用到的面）
  const faceCount = new Map();
  for (const t of tetras) for (const f of tetraFaces(t)) {
    const k = f.slice().sort().join(',');
    faceCount.set(k, (faceCount.get(k)||0) + 1);
  }
  const hull = [];
  for (const t of tetras) for (const f of tetraFaces(t)) {
    if (faceCount.get(f.slice().sort().join(',')) === 1) hull.push(f);
  }
  // 查表网格（64³）
  const data = { palette: selectedPalette, tetras, ainv, hull, gridSize: 64, grid: null };
  const gsize = 64, step = 256 / gsize;
  const grid = new Int16Array(gsize*gsize*gsize);
  const d2 = { tetras, ainv };
  for (let ri = 0; ri < gsize; ri++) for (let gi = 0; gi < gsize; gi++) for (let bi = 0; bi < gsize; bi++) {
    const c = [(ri+0.5)*step, (gi+0.5)*step, (bi+0.5)*step];
    grid[(ri*gsize + gi)*gsize + bi] = findContaining(c, d2);
  }
  data.grid = grid;
  return data;
}

// 凸包投影：点到三角形最近点（Ericson 标量版）
function closestOnTri(p, a, b, c) {
  const ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
  const ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
  const ap = [p[0]-a[0], p[1]-a[1], p[2]-a[2]];
  const dot = (u,v)=>u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
  const d1 = dot(ap,ab), d2 = dot(ap,ac);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = [p[0]-b[0], p[1]-b[1], p[2]-b[2]];
  const d3 = dot(bp,ab), d4 = dot(bp,ac);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1*d4 - d3*d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return [a[0]+v*ab[0], a[1]+v*ab[1], a[2]+v*ab[2]];
  }
  const cp = [p[0]-c[0], p[1]-c[1], p[2]-c[2]];
  const d5 = dot(cp,ab), d6 = dot(cp,ac);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5*d2 - d1*d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return [a[0]+w*ac[0], a[1]+w*ac[1], a[2]+w*ac[2]];
  }
  const va = d3*d6 - d5*d4;
  if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
    const w = (d4-d3)/((d4-d3)+(d5-d6));
    const bc = [c[0]-b[0], c[1]-b[1], c[2]-b[2]];
    return [b[0]+w*bc[0], b[1]+w*bc[1], b[2]+w*bc[2]];
  }
  const denom = 1.0/(va+vb+vc);
  const v = vb*denom, w = vc*denom;
  return [a[0]+ab[0]*v+ac[0]*w, a[1]+ab[1]*v+ac[1]*w, a[2]+ab[2]*v+ac[2]*w];
}

// 凸包外像素：投影到凸包表面最近点，返回所在面的 3 个调色板索引 + 三角形重心权重
function projectToHull(p, D) {
  const pal = D.palette;
  let bestQ = null, bestIdx = null, bestD = Infinity;
  for (let f = 0; f < D.hull.length; f++) {
    const hi = D.hull[f];
    const q = closestOnTri(p, pal[hi[0]], pal[hi[1]], pal[hi[2]]);
    const d = (q[0]-p[0])**2+(q[1]-p[1])**2+(q[2]-p[2])**2;
    if (d < bestD) { bestD = d; bestQ = q; bestIdx = hi; }
  }
  // q 在三角形上的重心权重
  const v0 = pal[bestIdx[0]], v1 = pal[bestIdx[1]], v2 = pal[bestIdx[2]];
  const e0 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
  const e1 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
  const e2 = [bestQ[0]-v0[0], bestQ[1]-v0[1], bestQ[2]-v0[2]];
  const d00 = e0[0]*e0[0]+e0[1]*e0[1]+e0[2]*e0[2];
  const d01 = e0[0]*e1[0]+e0[1]*e1[1]+e0[2]*e1[2];
  const d11 = e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2];
  const d20 = e2[0]*e0[0]+e2[1]*e0[1]+e2[2]*e0[2];
  const d21 = e2[0]*e1[0]+e2[1]*e1[1]+e2[2]*e1[2];
  const den = d00*d11 - d01*d01;
  const w1 = (d11*d20 - d01*d21)/den;
  const w2 = (d00*d21 - d01*d20)/den;
  return { idx: bestIdx, w: [1-w1-w2, w1, w2] };
}

// 权重截断后归一化
function pruneWeights(w, minWeight) {
  let s = 0;
  for (let k = 0; k < w.length; k++) { if (w[k] < minWeight) w[k] = 0; s += w[k]; }
  if (s > 0) for (let k = 0; k < w.length; k++) w[k] /= s;
  return w;
}

// 重心混合
function barycentricDither(img, palette, D, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const n = width * height;
  const mat = DITHER_MATRICES[opts.matrix || 'bayer'];
  const { m, offs, ths } = matrixVals(mat, opts.strength ?? 64);
  const minWeight = opts.minWeight ?? 0;
  const edgeGuard = opts.edgeGuard ?? 0;
  const gamut = !!opts.gamut;
  const gsize = D.gridSize, step = 256 / gsize;
  const edge = edgeGuard > 0 ? labEdgeMap(img) : null;
  const out = new Uint8ClampedArray(data.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const pi = y * width + x;
      if (data[i + 3] === 0) { out[i + 3] = 0; continue; }
      const p = [data[i], data[i + 1], data[i + 2]];
      const off = offs[(y % m) * m + (x % m)];
      const tt = ths[(y % m) * m + (x % m)];

      // 边缘保护：边缘像素走有序路径
      if (edge && edge[pi] >= edgeGuard) {
        const q = [p[0]+off, p[1]+off, p[2]+off];
        const idx = nearestRGBIdx(q, palette);
        out[i]=palette[idx][0]; out[i+1]=palette[idx][1]; out[i+2]=palette[idx][2];
        out[i+3]=255; continue;
      }

      // 找四面体：有网格则查表+回退，无网格（子集运行时数据）则直接完整遍历
      let t = -1, w = null;
      if (D.grid) {
        const cell0 = Math.min(gsize-1, Math.max(0, (p[0]/step)|0));
        const cell1 = Math.min(gsize-1, Math.max(0, (p[1]/step)|0));
        const cell2 = Math.min(gsize-1, Math.max(0, (p[2]/step)|0));
        const cidx = (cell0*gsize + cell1)*gsize + cell2;
        t = D.grid[cidx];
        if (t >= 0) {
          w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
          // 查表的四面体不含该像素（任何权重 < BARY_EPS）→ 回退完整遍历（精确，与 scipy 一致）
          if (w[0] < BARY_EPS || w[1] < BARY_EPS || w[2] < BARY_EPS || w[3] < BARY_EPS) {
            t = findContaining(p, D);
            if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
          }
        } else {
          // 网格判定为凸包外，但网格是粗的，像素可能实际在凸包内 → 完整遍历确认
          t = findContaining(p, D);
          if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
        }
      } else {
        // 无网格：直接完整遍历
        t = findContaining(p, D);
        if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
      }

      if (t >= 0) {
        // 凸包内：重心混合
        w = pruneWeights(w, minWeight);
        const maxW = Math.max(w[0], w[1], w[2], w[3]);
        if (maxW >= 0.9) {
          // 近单色 → 有序路径
          const q = [p[0]+off, p[1]+off, p[2]+off];
          const idx = nearestRGBIdx(q, palette);
          out[i]=palette[idx][0]; out[i+1]=palette[idx][1]; out[i+2]=palette[idx][2];
        } else {
          // 阈值选色
          const cum = [w[0], w[0]+w[1], w[0]+w[1]+w[2], 1.0];
          let sel = 3;
          for (let k = 0; k < 4; k++) if (cum[k] >= tt) { sel = k; break; }
          const c = palette[D.tetras[t][sel]];
          out[i]=c[0]; out[i+1]=c[1]; out[i+2]=c[2];
        }
        out[i+3]=255;
      } else {
        // 凸包外
        if (gamut) {
          const pr = projectToHull(p, D);
          let w3 = pruneWeights(pr.w, minWeight);
          const cum = [w3[0], w3[0]+w3[1], 1.0];
          let sel = 2;
          for (let k = 0; k < 3; k++) if (cum[k] >= tt) { sel = k; break; }
          const c = palette[pr.idx[sel]];
          out[i]=c[0]; out[i+1]=c[1]; out[i+2]=c[2];
        } else {
          const q = [p[0]+off, p[1]+off, p[2]+off];
          const idx = nearestRGBIdx(q, palette);
          out[i]=palette[idx][0]; out[i+1]=palette[idx][1]; out[i+2]=palette[idx][2];
        }
        out[i+3]=255;
      }
    }
  }
  return { width, height, data: out };
}

// ============================================================
// 分层投影混色（与 Python layered_dither.py 的 layered_projection_dither 对应）
//
// 两步分离：
//   1. 投影：按明暗分层，每层聚类出 K 个投影色（可不在调色板内），像素投影到最近的
//      那个 —— 每层只有几种颜色，大片区域同色
//   2. 抖动：对每个投影色单独求"用调色板色把它混出来"的权重（四面体重心 / 凸包投影），
//      再按图案阈值分配 —— 输出全部落在调色板内，同一投影色的图案完全一致
// 色差由第 1 步的投影质量决定（层数、每层色数），第 2 步只负责忠实还原。
// ============================================================

// 确定性伪随机（保证同一参数出同一结果）
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// CIELAB → sRGB（0-255），用于把 Lab 空间聚出的投影色转回 RGB
function labToRgb(lab) {
  const fy = (lab[0] + 16) / 116;
  const fx = fy + lab[1] / 500;
  const fz = fy - lab[2] / 200;
  const finv = t => { const t3 = t * t * t; return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787; };
  const x = 0.95047 * finv(fx), y = finv(fy), z = 1.08883 * finv(fz);
  let r = x * 3.2406 + y * -1.5372 + z * -0.4986;
  let g = x * -0.9689 + y * 1.8758 + z * 0.0415;
  let b = x * 0.0557 + y * -0.2040 + z * 1.0570;
  const enc = c => c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
  return [Math.min(255, Math.max(0, enc(r) * 255)),
          Math.min(255, Math.max(0, enc(g) * 255)),
          Math.min(255, Math.max(0, enc(b) * 255))];
}

// k-means++ 初始化 + Lloyd 迭代（在给定空间里），返回 K 个中心
function kmeans(samples, K, seed, iters) {
  const N = samples.length;
  K = Math.max(1, Math.min(K, N));
  const rng = mulberry32(seed);
  const centers = [samples[(rng() * N) | 0].slice()];
  const tmp = new Float64Array(N);
  while (centers.length < K) {
    let sum = 0;
    for (let i = 0; i < N; i++) {
      let d = Infinity;
      for (let j = 0; j < centers.length; j++) {
        const c = centers[j], p = samples[i];
        const dd = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
        if (dd < d) d = dd;
      }
      tmp[i] = d; sum += d;
    }
    let r = rng() * sum, pick = N - 1;
    for (let i = 0; i < N; i++) { r -= tmp[i]; if (r <= 0) { pick = i; break; } }
    centers.push(samples[pick].slice());
  }
  const assign = new Int32Array(N).fill(-1);
  const sums = Array.from({ length: K }, () => [0, 0, 0, 0]);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < N; i++) {
      const p = samples[i];
      let bi = 0, bd = Infinity;
      for (let j = 0; j < K; j++) {
        const c = centers[j];
        const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
        if (d < bd) { bd = d; bi = j; }
      }
      if (assign[i] !== bi) { assign[i] = bi; moved = true; }
    }
    for (let j = 0; j < K; j++) { sums[j][0] = sums[j][1] = sums[j][2] = sums[j][3] = 0; }
    for (let i = 0; i < N; i++) {
      const s = sums[assign[i]], p = samples[i];
      s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; s[3]++;
    }
    for (let j = 0; j < K; j++) {
      if (sums[j][3] > 0) {
        centers[j] = [sums[j][0] / sums[j][3], sums[j][1] / sums[j][3], sums[j][2] / sums[j][3]];
      }
    }
    if (!moved) break;
  }
  return centers;
}

// 求"用调色板色混合出目标色 p"的权重：返回 { idx:[调色板下标...], w:[权重...] }
// 目标色在凸包内 → 四面体重心坐标；在凸包外 → 投影到凸包表面取该面重心坐标
function mixWeights(p, D) {
  const gsize = D.gridSize, step = 256 / gsize;
  let t = -1, w = null;
  if (D.grid) {
    const c0 = Math.min(gsize - 1, Math.max(0, (p[0] / step) | 0));
    const c1 = Math.min(gsize - 1, Math.max(0, (p[1] / step) | 0));
    const c2 = Math.min(gsize - 1, Math.max(0, (p[2] / step) | 0));
    t = D.grid[(c0 * gsize + c1) * gsize + c2];
    if (t >= 0) {
      w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
      if (w[0] < BARY_EPS || w[1] < BARY_EPS || w[2] < BARY_EPS || w[3] < BARY_EPS) {
        t = findContaining(p, D);
        if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
      }
    } else {
      t = findContaining(p, D);
      if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
    }
  } else {
    t = findContaining(p, D);
    if (t >= 0) w = matVec4(D.ainv[t], [p[0], p[1], p[2], 1.0]);
  }
  if (t >= 0) {
    let sum = 0;
    const ww = [];
    for (let k = 0; k < 4; k++) { const x = Math.max(0, w[k]); ww.push(x); sum += x; }
    if (sum <= 0) return { idx: D.tetras[t], w: [0.25, 0.25, 0.25, 0.25] };
    return { idx: D.tetras[t], w: ww.map(v => v / sum) };
  }
  const pr = projectToHull(p, D);
  let sum = 0;
  const ww = [];
  for (let k = 0; k < 3; k++) { const x = Math.max(0, pr.w[k]); ww.push(x); sum += x; }
  if (sum <= 0) return { idx: pr.idx, w: [1 / 3, 1 / 3, 1 / 3] };
  return { idx: pr.idx, w: ww.map(v => v / sum) };
}

function projectionDither(img, palette, D, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const npix = width * height;
  const nBands = Math.max(2, Math.min(64, opts.bands ?? 24));
  const K = Math.max(2, Math.min(32, opts.colors ?? 8));
  const useLab = (opts.clusterSpace || 'Lab') === 'Lab';
  const mat = DITHER_MATRICES[opts.matrix || 'bayer'];
  const { m, ths } = matrixVals(mat, 0);
  const out = new Uint8ClampedArray(data.length);

  // ---- 每个像素的明度 L*，以及（可选）Lab 缓存 ----
  const luma = new Float32Array(npix);
  const labCache = new Float32Array(npix * 3);
  const opaque = new Uint8Array(npix);
  for (let i = 0; i < npix; i++) {
    const o = i * 4;
    out[i * 4 + 3] = data[o + 3];
    if (data[o + 3] === 0) { out[o + 3] = 0; continue; }
    opaque[i] = 1;
    const lab = rgbToLab([data[o], data[o + 1], data[o + 2]]);
    luma[i] = lab[0] / 100;
    labCache[i * 3] = lab[0]; labCache[i * 3 + 1] = lab[1]; labCache[i * 3 + 2] = lab[2];
  }

  // ---- 分层边界：linear 按明度等分（保色彩）/ quantile 按图像分布等频（保细节）----
  const edges = new Float64Array(nBands + 1);
  if (opts.edgeMode === 'quantile') {
    const sorted = Float32Array.from(luma).sort();
    for (let k = 0; k <= nBands; k++) {
      const idx = Math.min(npix - 1, Math.max(0, Math.round(k / nBands * (npix - 1))));
      edges[k] = sorted[idx];
    }
    edges[0] = 0; edges[nBands] = 1;
    for (let k = 1; k <= nBands; k++) if (edges[k] < edges[k - 1]) edges[k] = edges[k - 1];
  } else {
    for (let k = 0; k <= nBands; k++) edges[k] = k / nBands;
  }

  // ---- 按层分桶（一次遍历，避免每层重扫全图）----
  const buckets = Array.from({ length: nBands }, () => []);
  for (let i = 0; i < npix; i++) {
    if (!opaque[i]) continue;
    let b = 0;
    while (b < nBands - 1 && luma[i] >= edges[b + 1]) b++;
    buckets[b].push(i);
  }

  const SAMPLE = 2000;
  for (let b = 0; b < nBands; b++) {
    const members = buckets[b];
    if (!members.length) continue;

    // ---- 第 1 步：聚类投影色 ----
    const step = Math.max(1, Math.floor(members.length / SAMPLE));
    const samples = [];
    for (let i = 0; i < members.length; i += step) {
      const p = members[i];
      samples.push(useLab
        ? [labCache[p * 3], labCache[p * 3 + 1], labCache[p * 3 + 2]]
        : [data[p * 4], data[p * 4 + 1], data[p * 4 + 2]]);
    }
    const centers = kmeans(samples, K, b * 7919 + 13, 20);
    const projRGB = centers.map(c => useLab ? labToRgb(c) : c);

    // ---- 第 2 步：每个投影色单独求混合权重（同一权重负责该颜色的全部像素）----
    const mixes = projRGB.map(c => mixWeights(c, D));

    // ---- 投影分配 + 阈值选色 ----
    for (const p of members) {
      const o = p * 4;
      const r = data[o], g = data[o + 1], bl = data[o + 2];
      let bj = 0, bd = Infinity;
      for (let j = 0; j < projRGB.length; j++) {
        const c = projRGB[j];
        const d = (r - c[0]) ** 2 + (g - c[1]) ** 2 + (bl - c[2]) ** 2;
        if (d < bd) { bd = d; bj = j; }
      }
      const mx = mixes[bj];
      const tt = ths[((p / width | 0) % m) * m + (p % width) % m];
      let acc = 0, sel = mx.w.length - 1;
      for (let k = 0; k < mx.w.length; k++) { acc += mx.w[k]; if (acc >= tt) { sel = k; break; } }
      const c = D.palette[mx.idx[sel]];
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 255;
    }
  }
  // 透明像素
  for (let i = 0; i < npix; i++) if (!opaque[i]) { out[i * 4 + 3] = 0; }
  return { width, height, data: out };
}

// 统一入口
function runDither(img, method, palette, D, opts) {
  if (method === 'ordered') return orderedDither(img, palette, opts);
  if (method === 'error_diffusion') return errorDiffusion(img, palette, opts);
  if (method === 'barycentric') return barycentricDither(img, palette, D, opts);
  if (method === 'projection') return projectionDither(img, palette, D, opts);
  throw new Error('未知方法: ' + method);
}
