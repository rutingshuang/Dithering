// ============================================================
// dither.js — 调色板图像抖动算法（JS 移植版）
// 与 Python dithers.py 逻辑一致；Delaunay 数据用离线预计算的 DITHER_DATA 查表
// ============================================================

// ---------- 抖动矩阵 ----------
// 阈值矩阵：键名带阶数后缀，2×2 与 4×4 各一套。
// matrixVals() 按 mat.length 自动算分阶，新增阶数不用改它。
const DITHER_MATRICES = {
  bayer4:     [[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]],
  clustered4: [[14,11,8,13],[10,6,4,9],[7,2,0,5],[12,3,1,15]],
  diagonal4:  [[15,11,5,0],[10,14,9,4],[3,8,13,7],[1,2,9,12]],
  spiral4:    [[9,1,6,14],[3,11,15,4],[5,7,0,8],[12,13,10,2]],
  // 2×2：只有 4 级阈值，颗粒更粗、色差更大，是「风格取舍」不是「更优」
  bayer2:     [[0,2],[3,1]],
  // 2×2 的点簇：列优先递增，点会从左上角长出来；与 bayer2 结果不同才看得出区别
  clustered2: [[0,1],[2,3]],
  diagonal2:  [[3,1],[0,2]],
  spiral2:    [[1,3],[2,0]],
  // 圆点（欧氏环）：同阶下比 clustered 的方点色差低一档，2×2 退化成一个 L 形
  rounddot4:  [[12,5,6,13],[4,0,1,7],[11,3,2,8],[15,10,9,14]],
  rounddot2:  [[0,1],[3,2]],
};
// 旧键名兼容（外部脚本/工程文件里可能还写着 'bayer'）
DITHER_MATRICES.bayer = DITHER_MATRICES.bayer4;
DITHER_MATRICES.clustered = DITHER_MATRICES.clustered4;
DITHER_MATRICES.diagonal = DITHER_MATRICES.diagonal4;
DITHER_MATRICES.spiral = DITHER_MATRICES.spiral4;
DITHER_MATRICES.rounddot = DITHER_MATRICES.rounddot4;


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
  // Lab 凸包：凸包投影必须在 ΔE76 所在空间算。全局 Delaunay 凸包只保留最外层约 27 个色，
  // 橄榄系全被包在里面，暖黄绿只能靠 LightGreen+Black 压暗 —— 那 30% 的纯黑就是黑点。
  const labPoints = selectedPalette.map(c => rgbToLab(c));
  const labTetras = computeDelaunay(labPoints);
  const labFaceCount = new Map();
  for (const t of labTetras) for (const f of tetraFaces(t)) {
    const key = f.slice().sort().join(',');
    labFaceCount.set(key, (labFaceCount.get(key) || 0) + 1);
  }
  const labHull = [];
  for (const t of labTetras) for (const f of tetraFaces(t)) {
    if (labFaceCount.get(f.slice().sort().join(',')) === 1) labHull.push(f);
  }
  const labL = labPoints.map(c => c[0]);

  // 查表网格（64³）
  const data = { palette: selectedPalette, tetras, ainv, hull, gridSize: 64, grid: null,
                 labPoints, labHull, labL };
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

// Lab 凸包外像素：投影到 Lab 凸包表面最近点（目标口径 ΔE76），返回索引+重心权重
function projectToHullLab(pLab, D) {
  const pal = D.labPoints;
  let bestQ = null, bestIdx = null, bestD = Infinity;
  for (let f = 0; f < D.labHull.length; f++) {
    const hi = D.labHull[f];
    const q = closestOnTri(pLab, pal[hi[0]], pal[hi[1]], pal[hi[2]]);
    const d = (q[0]-pLab[0])**2 + (q[1]-pLab[1])**2 + (q[2]-pLab[2])**2;
    if (d < bestD) { bestD = d; bestQ = q; bestIdx = hi; }
  }
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
  return { idx: [bestIdx[0], bestIdx[1], bestIdx[2]], w: [1-w1-w2, w1, w2] };
}

// 面内精修：投影点落在三角面上之后，换 Lab 目标再挤几 % 的位置。
// 2 轮是甜点，4 轮起逐像素更优但 8x8 块平均反而回升（指标错位）。
function refineOnFaceLab(pLab, idx, w, D, rounds) {
  if (!rounds) return w;
  const pal = D.labPoints;
  const a0 = pal[idx[0]], e1 = [pal[idx[1]][0]-a0[0], pal[idx[1]][1]-a0[1], pal[idx[1]][2]-a0[2]],
        e2 = [pal[idx[2]][0]-a0[0], pal[idx[2]][1]-a0[1], pal[idx[2]][2]-a0[2]];
  let u = Math.min(1, Math.max(0, w[1])), v = Math.min(1, Math.max(0, w[2]));
  const dist = (uu, vv) => {
    const q = [a0[0]+uu*e1[0]+vv*e2[0], a0[1]+uu*e1[1]+vv*e2[1], a0[2]+uu*e1[2]+vv*e2[2]];
    return Math.sqrt((q[0]-pLab[0])**2 + (q[1]-pLab[1])**2 + (q[2]-pLab[2])**2);
  };
  let best = dist(u, v);
  let step = 0.14;
  for (let r = 0; r < rounds; r++) {
    for (let du = -1; du <= 1; du++) for (let dv = -1; dv <= 1; dv++) {
      let uu = Math.min(1, Math.max(0, u + du*step));
      let vv = Math.min(1, Math.max(0, v + dv*step));
      if (uu + vv > 1) { const s2 = uu + vv; uu /= s2; vv /= s2; }
      const d = dist(uu, vv);
      if (d < best) { best = d; u = uu; v = vv; }
    }
    step *= 0.5;
  }
  return [1-u-v, u, v];
}

// 权重截断后归一化
function pruneWeights(w, minWeight) {
  let s = 0;
  for (let k = 0; k < w.length; k++) { if (w[k] < minWeight) w[k] = 0; s += w[k]; }
  if (s > 0) for (let k = 0; k < w.length; k++) w[k] /= s;
  return w;
}

// 预计算数据（delaunay_data.js）里没有 Lab 凸包时按需补算，只算一次
function ensureLabData(D) {
  if (D.labHull) return;
  const labPoints = D.palette.map(c => rgbToLab(c));
  const labTetras = computeDelaunay(labPoints);
  const cnt = new Map();
  for (const t of labTetras) for (const f of tetraFaces(t)) {
    const k = f.slice().sort().join(',');
    cnt.set(k, (cnt.get(k) || 0) + 1);
  }
  const labHull = [];
  for (const t of labTetras) for (const f of tetraFaces(t)) {
    if (cnt.get(f.slice().sort().join(',')) === 1) labHull.push(f);
  }
  D.labPoints = labPoints;
  D.labHull = labHull;
  D.labL = labPoints.map(c => c[0]);
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
  const projLab = opts.projLab !== false;      // 凸包投影在 Lab（ΔE76）里算，默认开
  const refine = opts.refine ?? 2;              // 投影落定后的面内精修轮数
  const mixSpan = opts.mixSpan ?? 35;           // 凸包内混合允许的最大顶点亮度跨度
  const nearMax = 0.9;
  if (projLab) ensureLabData(D);
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
        // 混合跨度：四面体 4 个顶点的 Lab L 跨度太大时，块平均虽正确，但块内就是
        // 「亮底 + 暗点」的棋盘格（LightGreen L=90 与 Olive L=56 混出 L=78 正是一例）。
        // 亮区这类像素退回单一近色，暗部不受影响（暗部本来就该有暗点）。
        let wideMix = false;
        if (mixSpan > 0 && maxW < nearMax) {
          const tet = D.tetras[t];
          let lo = Infinity, hi = -Infinity;
          for (let k = 0; k < 4; k++) {
            const L = D.labL[tet[k]];
            if (L < lo) lo = L;
            if (L > hi) hi = L;
          }
          const luma = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
          wideMix = (hi - lo) > mixSpan && luma > 76;
        }
        if (maxW >= 0.9 || wideMix) {
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
          let pr, w3;
          if (projLab) {
            const pLab = rgbToLab(p);
            pr = projectToHullLab(pLab, D);
            w3 = refineOnFaceLab(pLab, pr.idx, pr.w.slice(), D, refine);
          } else {
            pr = projectToHull(p, D);
            w3 = pr.w.slice();
          }
          w3 = pruneWeights(w3, minWeight);
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

// ============================================================
// 分层编辑器引擎
// 分析（分层 + 聚类）与渲染（投影色 → 抖动色）彻底分开：
//   analyzeProjection  只依赖「色阶 + 分界 + 每层色数」
//   projectionDither   吃一份 plan + 手动颜色槽 + 手绘标签，渲染任一阶段
// 这样拖分界、改色、画笔涂抹都不需要重跑 k-means。
// ============================================================

// 色阶：PS 输入白场语义 —— 先按 black/white 拉伸，再施加 gamma 中间调
function lumaCurve(L, black, white, gamma) {
  const span = Math.max(1e-6, white - black);
  let v = (L - black) / span;
  v = v < 0 ? 0 : (v > 1 ? 1 : v);
  if (gamma && gamma !== 1) v = Math.pow(v, gamma);
  return v;
}

// 分析：分层、分桶、每层聚类出投影色与其混合配方
// opts 支持：bands, colors, clusterSpace, edgeMode, black, white, gamma, edges(覆盖分界)
function analyzeProjection(img, D, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const npix = width * height;
  const nBands = Math.max(2, Math.min(64, opts.bands ?? 24));
  const K = Math.max(2, Math.min(32, opts.colors ?? 8));
  const useLab = (opts.clusterSpace || 'Lab') === 'Lab';
  const black = Math.min(opts.black ?? 0, 0.98);
  const white = Math.max(opts.white ?? 1, black + 0.02);
  const gamma = opts.gamma ?? 1;

  const luma = new Float32Array(npix);
  const labCache = new Float32Array(npix * 3);
  const opaque = new Uint8Array(npix);
  for (let i = 0; i < npix; i++) {
    const o = i * 4;
    if (data[o + 3] === 0) continue;
    opaque[i] = 1;
    const lab = rgbToLab([data[o], data[o + 1], data[o + 2]]);
    luma[i] = lumaCurve(lab[0] / 100, black, white, gamma);
    labCache[i * 3] = lab[0]; labCache[i * 3 + 1] = lab[1]; labCache[i * 3 + 2] = lab[2];
  }

  let edges;
  if (opts.edges && opts.edges.length === nBands + 1) {
    edges = Float64Array.from(opts.edges);
  } else {
    edges = new Float64Array(nBands + 1);
    if (opts.edgeMode === 'quantile') {
      const sorted = Float32Array.from(luma).sort();
      for (let k = 0; k <= nBands; k++) {
        const idx = Math.min(npix - 1, Math.max(0, Math.round(k / nBands * (npix - 1))));
        edges[k] = sorted[idx];
      }
      edges[0] = 0; edges[nBands] = 1;
    } else {
      for (let k = 0; k <= nBands; k++) edges[k] = k / nBands;
    }
  }
  for (let k = 1; k <= nBands; k++) if (edges[k] < edges[k - 1]) edges[k] = edges[k - 1];

  const plan = { width, height, npix, nBands, edges, bands: [], opaque, luma,
                 labCache, useLab, K, black, white, gamma, bandOf: new Int32Array(npix) };
  bucketBands(plan, img, true, D);
  return plan;
}

// 把像素按当前 edges 分进各层。cluster=true 时重新聚类投影色（分析 / 重分析），
// 否则只重分桶（用户拖了分界，保留已聚出的投影色与配方）
function bucketBands(plan, img, cluster, D) {
  const { npix, nBands, edges, luma, opaque } = plan;
  const src = img ? img.data : null;
  const buckets = Array.from({ length: nBands }, () => []);
  for (let i = 0; i < npix; i++) {
    if (!opaque[i]) { plan.bandOf[i] = -1; continue; }
    let b = 0;
    while (b < nBands - 1 && luma[i] >= edges[b + 1]) b++;
    plan.bandOf[i] = b;
    buckets[b].push(i);
  }
  const SAMPLE = 2000;
  const old = plan.bands || [];
  const bands = [];
  for (let b = 0; b < nBands; b++) {
    const members = buckets[b];
    const prev = old[b];
    const info = { index: b, lo: edges[b], hi: edges[b + 1], members,
                   count: members.length, ratio: npix ? members.length / npix : 0,
                   projRGB: [], mixes: [] };
    if (members.length) {
      // 需要重新聚类的情形：显式重分析；或这一层上一轮是空的（拖分界把像素推进
      // 了原本没有投影色的层），此时没有可复用的 projRGB，必须补一次局部聚类，
      // 否则这一层的像素会没有任何颜色可落。
      const needCluster = cluster || !prev || !prev.projRGB || !prev.projRGB.length;
      if (needCluster) {
        const step = Math.max(1, Math.floor(members.length / SAMPLE));
        const samples = [];
        for (let i = 0; i < members.length; i += step) {
          const q = members[i];
          samples.push(plan.useLab
            ? [plan.labCache[q * 3], plan.labCache[q * 3 + 1], plan.labCache[q * 3 + 2]]
            : [src[q * 4], src[q * 4 + 1], src[q * 4 + 2]]);
        }
        const centers = kmeans(samples, plan.K, b * 7919 + 13, 20);
        info.projRGB = centers.map(c => plan.useLab ? labToRgb(c) : c);
        info.mixes = info.projRGB.map(c => mixWeights(c, D));
      } else {
        // 只重分桶：投影色和它们的混合配方都不变，直接复用（这一步省掉凸包求值）
        info.projRGB = prev.projRGB;
        info.mixes = (prev.mixes && prev.mixes.length)
          ? prev.mixes
          : info.projRGB.map(c => mixWeights(c, D));
      }
    }
    bands.push(info);
  }
  plan.bands = bands;
  return plan;
}

// 槽的目标色：色号 → 调色板色；数组 → 自定义投影色
function slotRGB(sv, pal) {
  return (typeof sv.color === 'number') ? pal[sv.color] : sv.color;
}

// 槽级混合权重：凸包内直接用 RGB 四面体，凸包外按需走 Lab 投影 + 面内精修
function slotMixWeights(target, D, sv) {
  if (sv.projLab === false) return mixWeights(target, D);
  ensureLabData(D);
  const pLab = rgbToLab(target);
  const gsize = D.gridSize, step = 256 / gsize;
  const c0 = Math.min(gsize - 1, Math.max(0, (target[0] / step) | 0));
  const c1 = Math.min(gsize - 1, Math.max(0, (target[1] / step) | 0));
  const c2 = Math.min(gsize - 1, Math.max(0, (target[2] / step) | 0));
  let t = -1;
  if (D.grid) {
    t = D.grid[(c0 * gsize + c1) * gsize + c2];
    if (t < 0) t = findContaining(target, D);
  } else {
    t = findContaining(target, D);
  }
  if (t >= 0) return mixWeights(target, D);
  if (sv.gamut === false) return mixWeights(target, D);
  const pr = projectToHullLab(pLab, D);
  const w = refineOnFaceLab(pLab, pr.idx, pr.w.slice(), D, sv.refine ?? 2);
  let sum = 0;
  const ww = [];
  for (let k = 0; k < 3; k++) { const x = Math.max(0, w[k]); ww.push(x); sum += x; }
  if (sum <= 0) return { idx: pr.idx, w: [1 / 3, 1 / 3, 1 / 3] };
  return { idx: pr.idx, w: ww.map(v => v / sum) };
}

// 限制混合用到的板内色个数：4=四面体（原样）/ 3=丢最小权 / 2=留最大两权。
// 与桌面版 layered_dither._mix_weights_k 的枚举式实现不完全等价（这里直接截断
// 已有的四面体权重，省掉 C(n,3) 枚举），但「用几个色」这个语义一致，
// 落盘内也仍然成立。k=2 时端点跨度太大会退成单色平涂，避免异色互补。
function restrictMix(m, k, pal, spanLimit) {
  if (!m || !m.w || k >= 4 || m.w.length <= k) return m;
  const order = m.w.map((w, i) => [w, i]).sort((a, b) => b[0] - a[0]);
  const keep = order.slice(0, Math.max(1, k));
  if (keep.length === 2 && spanLimit > 0 && pal) {
    // 两色混合：两端颜色 Lab 距离太大就退成单色平涂，避免异色互补出噪点
    const la = rgbToLab(pal[m.idx[keep[0][1]]] || [0, 0, 0]);
    const lb = rgbToLab(pal[m.idx[keep[1][1]]] || [0, 0, 0]);
    if (Math.hypot(la[0] - lb[0], la[1] - lb[1], la[2] - lb[2]) > spanLimit) {
      return { idx: [m.idx[keep[0][1]]], w: [1] };
    }
  }
  const sum = keep.reduce((acc, o) => acc + o[0], 0);
  if (sum <= 0) return { idx: [m.idx[keep[0][1]]], w: [1] };
  return { idx: keep.map(o => m.idx[o[1]]), w: keep.map(o => o[0] / sum) };
}

function pruneMix(m, minWeight) {
  if (!m || !(minWeight > 0)) return m;
  let sum = 0;
  for (let k = 0; k < m.w.length; k++) { if (m.w[k] < minWeight) m.w[k] = 0; sum += m.w[k]; }
  if (sum > 0) for (let k = 0; k < m.w.length; k++) m.w[k] /= sum;
  return m;
}

// ============================================================
// 2 色取色：枚举色对 × 图案能实现的档位（与桌面版 layered_dither 同一套判据）
// ============================================================
// 图案有 N 个阈值 → 2 色混合只能实现 step/N 这些比例（2×2=4 档、4×4=16 档）。
// 桌面版实测：按档位选色对比"连续最优再被阈值截断"低 6.7%（2×2）~14%（4×4）。
// 网页版原来是从四面体 4 色权重里截掉最小两权（restrictMix），那是另一种算法，
// 2 色下色差明显更高（实测 DE8 3.80 vs 1.81），这里改成同一套。
const PAIR_GRID_CACHE = new Map();

function pairGrid(D, levels) {
  const hit = PAIR_GRID_CACHE.get(levels);
  if (hit) return hit;
  const pal = D.palette;
  const n = pal.length;
  const lab = pal.map(rgbToLab);
  const pairs = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  const steps = levels + 1;
  const ts = [];
  for (let s = 0; s < steps; s++) ts.push(s / levels);
  const Q = new Float64Array(pairs.length * steps * 3);
  for (let p = 0; p < pairs.length; p++) {
    const A = lab[pairs[p][0]], Bc = lab[pairs[p][1]];
    for (let s = 0; s < steps; s++) {
      const t = ts[s], o = (p * steps + s) * 3;
      Q[o] = A[0] + (Bc[0] - A[0]) * t;
      Q[o + 1] = A[1] + (Bc[1] - A[1]) * t;
      Q[o + 2] = A[2] + (Bc[2] - A[2]) * t;
    }
  }
  const g = { pairs, steps, ts, Q, lab };
  if (PAIR_GRID_CACHE.size > 4) PAIR_GRID_CACHE.clear();
  PAIR_GRID_CACHE.set(levels, g);
  return g;
}

// 返回 { idx:[a,b], w:[wa,wb] }；权重一定落在 step/levels 上
function pairMix2(target, D, levels, penalty) {
  const g = pairGrid(D, levels);
  const p = rgbToLab(target);
  const { pairs, steps, ts, Q, lab } = g;
  let bestI = 0, bestS = 0, bestCost = Infinity;
  for (let i = 0; i < pairs.length; i++) {
    let bd = Infinity, bs = 0;
    const base = i * steps * 3;
    for (let s = 0; s < steps; s++) {
      const o = base + s * 3;
      const d = (Q[o] - p[0]) ** 2 + (Q[o + 1] - p[1]) ** 2 + (Q[o + 2] - p[2]) ** 2;
      if (d < bd) { bd = d; bs = s; }
    }
    let cost = Math.sqrt(bd);
    if (penalty > 0) {
      const A = lab[pairs[i][0]], B = lab[pairs[i][1]];
      cost += penalty * (Math.hypot(A[0] - p[0], A[1] - p[1], A[2] - p[2])
                       + Math.hypot(B[0] - p[0], B[1] - p[1], B[2] - p[2]));
    }
    if (cost < bestCost) { bestCost = cost; bestI = i; bestS = bs; }
  }
  const t = ts[bestS];
  return { idx: [pairs[bestI][0], pairs[bestI][1]], w: [1 - t, t] };
}

// ============================================================
// 线稿保护：在高分辨率原图上找「细线」，不是找「深色」
// ============================================================
// 只用「够暗」当判据会把成片的暗部（头发、翅膀）也当成线，结果整块被涂死。
// 真正的线是**局部暗谷**：比周围一圈都暗，且暗不少。用
//     resp = max(邻域 L) - L
// 描述这件事：1px 的线 resp 上百，大片暗部内部 resp 只有个位数。
// 必须在原图上算再下采样：20% 的图里细线已经被平均成浅灰，暗谷没了。
const LINE_CACHE = new Map();

// ---------- 线稿清理：骨架化 ----------
// 2px 宽的斜线在像素网格上会变成「这行 4 个、下行 5 个、列上重叠 1~2 个」的阶梯块，
// 看起来脏。Zhang-Suen 细化把细线压成 8 连通 1px，换行换列只走对角，不再有正交重叠。
// 粗块（翅膀暗部那种）不参与：先腐蚀找出宽度 > 2 的芯，把那片原样保留，只细化其余。
function morphPass(m, w, h, dilate) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hit = dilate ? 0 : 1;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) { if (!dilate) hit = 0; continue; }
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const v = (xx < 0 || xx >= w) ? 0 : m[yy * w + xx];
          if (dilate) { if (v) hit = 1; } else if (!v) hit = 0;
        }
      }
      out[y * w + x] = hit;
    }
  }
  return out;
}

function thinMask(mask, w, h, maxIter) {
  const img = Uint8Array.from(mask);
  for (let it = 0; it < (maxIter || 40); it++) {
    let changed = false;
    for (let step = 0; step < 2; step++) {
      const del = [];
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          if (!img[i]) continue;
          const p2 = img[i - w], p3 = img[i - w + 1], p4 = img[i + 1], p5 = img[i + w + 1];
          const p6 = img[i + w], p7 = img[i + w - 1], p8 = img[i - 1], p9 = img[i - w - 1];
          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;
          const seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2];
          let A = 0;
          for (let k = 0; k < 8; k++) if (!seq[k] && seq[k + 1]) A++;
          if (A !== 1) continue;
          if (step === 0) { if (p2 * p4 * p6 || p4 * p6 * p8) continue; }
          else { if (p2 * p4 * p8 || p2 * p6 * p8) continue; }
          del.push(i);
        }
      }
      if (del.length) { for (let k = 0; k < del.length; k++) img[del[k]] = 0; changed = true; }
    }
    if (!changed) break;
  }
  return img;
}

// L 形拐角（2x2 里恰好 3 个线像素）切对角：删掉拐角那个，剩下的一对走斜线。
// 只在「删了不会断开连接」时才切 —— 否则线会被切成碎段（实测不加保护会从 41 段炸到 215 段）。
const NB8 = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];

function nbConnected(m, w, h, y, x) {
  const alive = [];
  for (let k = 0; k < 8; k++) {
    const yy = y + NB8[k][0], xx = x + NB8[k][1];
    if (yy >= 0 && yy < h && xx >= 0 && xx < w && m[yy * w + xx]) alive.push([yy, xx]);
  }
  if (alive.length <= 1) return true;
  const key = (p) => p[0] * w + p[1];
  const set = new Set(alive.map(key));
  const seen = new Set([key(alive[0])]);
  const stack = [alive[0]];
  while (stack.length) {
    const [cy, cx] = stack.pop();
    for (let k = 0; k < 8; k++) {
      const yy = cy + NB8[k][0], xx = cx + NB8[k][1];
      const kk = yy * w + xx;
      if (set.has(kk) && !seen.has(kk)) { seen.add(kk); stack.push([yy, xx]); }
    }
  }
  return seen.size === alive.length;
}

// ---------- 线稿分段：把线按 8 连通切成对象 ----------
// 每段可以在 UI 里单独设「用固定色」还是「跟随投影」。
function labelLines(mask, w, h) {
  const lab = new Int32Array(w * h).fill(-1);
  const stack = [];
  let id = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || lab[i] >= 0) continue;
    stack.length = 0;
    stack.push(i);
    lab[i] = id;
    while (stack.length) {
      const p = stack.pop();
      const y = (p / w) | 0, x = p % w;
      for (let k = 0; k < 8; k++) {
        const yy = y + NB8[k][0], xx = x + NB8[k][1];
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        const q = yy * w + xx;
        if (mask[q] && lab[q] < 0) { lab[q] = id; stack.push(q); }
      }
    }
    id++;
  }
  return { lab, count: id };
}

function cutCorners(mask, w, h, maxPass) {
  const m = Uint8Array.from(mask);
  for (let pass = 0; pass < (maxPass || 6); pass++) {
    const cand = [];
    for (let y = 0; y < h - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const a = m[y*w+x], b = m[y*w+x+1], c = m[(y+1)*w+x], d = m[(y+1)*w+x+1];
        if (a + b + c + d !== 3) continue;
        if (!d) cand.push([y, x]);
        else if (!c) cand.push([y, x + 1]);
        else if (!b) cand.push([y + 1, x]);
        else cand.push([y + 1, x + 1]);
      }
    }
    let changed = false;
    for (let k = 0; k < cand.length; k++) {
      const [y, x] = cand[k];
      if (!m[y * w + x]) continue;
      if (nbConnected(m, w, h, y, x)) { m[y * w + x] = 0; changed = true; }
    }
    if (!changed) break;
  }
  return m;
}

function cleanLineMask(mask, w, h) {
  // 全量骨架化 + 切角：粗块也一起收成 1px 线，否则「宽度 ≥3 的阶梯」会原样保留，
  // 看起来还是有直角。想保留粗块就把「线稿清理」关掉。
  return cutCorners(thinMask(mask, w, h, 60), w, h, 8);
}

function lineRespMap(orig, tw, th) {
  // key 必须区分「不同的图」：只按尺寸会让换图后复用上一张的线稿
  const key = [(orig.__rev || 0), orig.width, orig.height, tw, th].join('|');
  const hit = LINE_CACHE.get(key);
  if (hit) return hit;
  const ow = orig.width, oh = orig.height, data = orig.data;
  const np0 = ow * oh;
  const L = new Float32Array(np0);
  for (let i = 0; i < np0; i++) {
    const o = i * 4;
    L[i] = (data[o + 3] === 0) ? 255 : rgbToLab([data[o], data[o + 1], data[o + 2]])[0];
  }
  // 分离式 max 滤波（半径 3，够盖住 2~3px 的线）
  const R = 3;
  const tmp = new Float32Array(np0), M = new Float32Array(np0);
  for (let y = 0; y < oh; y++) {
    const row = y * ow;
    for (let x = 0; x < ow; x++) {
      let m = 0;
      const a = Math.max(0, x - R), b = Math.min(ow - 1, x + R);
      for (let k = a; k <= b; k++) if (L[row + k] > m) m = L[row + k];
      tmp[row + x] = m;
    }
  }
  for (let x = 0; x < ow; x++) {
    for (let y = 0; y < oh; y++) {
      let m = 0;
      const a = Math.max(0, y - R), b = Math.min(oh - 1, y + R);
      for (let k = a; k <= b; k++) { const v = tmp[k * ow + x]; if (v > m) m = v; }
      M[y * ow + x] = m;
    }
  }
  // 反相：min 滤波得到「周围最暗」，用来抓比周围亮的线（高光描边）
  const tmpN = new Float32Array(np0), Mn = new Float32Array(np0);
  for (let y = 0; y < oh; y++) {
    const row = y * ow;
    for (let x = 0; x < ow; x++) {
      let m = 999;
      const a = Math.max(0, x - R), b = Math.min(ow - 1, x + R);
      for (let k = a; k <= b; k++) if (L[row + k] < m) m = L[row + k];
      tmpN[row + x] = m;
    }
  }
  for (let x = 0; x < ow; x++) {
    for (let y = 0; y < oh; y++) {
      let m = 999;
      const a = Math.max(0, y - R), b = Math.min(oh - 1, y + R);
      for (let k = a; k <= b; k++) { const v = tmpN[k * ow + x]; if (v < m) m = v; }
      Mn[y * ow + x] = m;
    }
  }
  // 每格取「最强响应」的样本（线色也一并带回来）
  const n = tw * th;
  const resp = new Float32Array(n);
  const respLight = new Float32Array(n);
  const rgb = new Float32Array(n * 3);
  const rgbLight = new Float32Array(n * 3);
  const minL = new Float32Array(n).fill(999);
  const minRGB = new Float32Array(n * 3);
  const fx = ow / tw, fy = oh / th;
  for (let y = 0; y < oh; y++) {
    const ty = Math.min(th - 1, (y / fy) | 0);
    for (let x = 0; x < ow; x++) {
      const i = y * ow + x, o = i * 4;
      if (data[o + 3] === 0) continue;
      const t = ty * tw + Math.min(tw - 1, (x / fx) | 0);
      const r = M[i] - L[i];
      if (r > resp[t]) {
        resp[t] = r;
        rgb[t * 3] = data[o]; rgb[t * 3 + 1] = data[o + 1]; rgb[t * 3 + 2] = data[o + 2];
      }
      const rl = L[i] - Mn[i];
      if (rl > respLight[t]) {
        respLight[t] = rl;
        rgbLight[t * 3] = data[o]; rgbLight[t * 3 + 1] = data[o + 1]; rgbLight[t * 3 + 2] = data[o + 2];
      }
      if (L[i] < minL[t]) {
        minL[t] = L[i];
        minRGB[t * 3] = data[o]; minRGB[t * 3 + 1] = data[o + 1]; minRGB[t * 3 + 2] = data[o + 2];
      }
    }
  }
  const out = { resp, rgb, respLight, rgbLight, minL, minRGB, tw, th };
  if (LINE_CACHE.size > 3) LINE_CACHE.clear();
  LINE_CACHE.set(key, out);
  return out;
}

// 整图二维误差扩散。结果只作为「ed 方法」像素的贴图来源：
// 扩散本身在整图上正常进行（误差不会在像素间丢失），其它方法各自的像素不受影响。
function diffuseImage2D(img, palette, opts) {
  opts = opts || {};
  const { width, height, data } = img;
  const weights = ED_FILTERS[opts.filter || 'FS'];
  const alpha = opts.alpha ?? 1.0;
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
      if (alphaCh[i] === 0) continue;
      const old = [arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]];
      const c = palette[nearestRGBIdx(old, palette)];
      out[i * 4] = c[0]; out[i * 4 + 1] = c[1]; out[i * 4 + 2] = c[2]; out[i * 4 + 3] = 255;
      const qe = [(old[0] - c[0]) * alpha, (old[1] - c[1]) * alpha, (old[2] - c[2]) * alpha];
      for (const wgt of weights) {
        const ny = y + wgt[0], nx = x + wgt[1];
        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
          const j = (ny * width + nx) * 3;
          arr[j] += qe[0] * wgt[2]; arr[j + 1] += qe[1] * wgt[2]; arr[j + 2] += qe[2] * wgt[2];
        }
      }
    }
  }
  return out;
}

// 层内一维误差扩散：同一槽的像素在画面上是分散的，只能按扫描序累积误差
const SEQ_KERNELS = {
  FS:     [7 / 16, 3 / 16, 5 / 48],
  Jarvis: [7 / 48, 5 / 48, 3 / 48],
  Stucki: [8 / 42, 4 / 42, 2 / 42],
};

function diffuseSeq(seq, palette, opts, out) {
  opts = opts || {};
  const alpha = opts.alpha ?? 1;
  const gain = (opts.strength ?? 64) / 64;
  const kern = SEQ_KERNELS[opts.filter] || SEQ_KERNELS.FS;
  const pend = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of seq) {
    const o = p * 4;
    const want = [out[o] + pend[0][0], out[o + 1] + pend[0][1], out[o + 2] + pend[0][2]];
    const c = palette[nearestRGBIdx(want, palette)];
    out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 255;
    const e = [(want[0] - c[0]) * alpha * gain,
               (want[1] - c[1]) * alpha * gain,
               (want[2] - c[2]) * alpha * gain];
    pend[0] = [pend[1][0] + e[0] * kern[0], pend[1][1] + e[1] * kern[0], pend[1][2] + e[2] * kern[0]];
    pend[1] = [pend[2][0] + e[0] * kern[1], pend[2][1] + e[1] * kern[1], pend[2][2] + e[2] * kern[1]];
    pend[2] = [e[0] * kern[2], e[1] * kern[2], e[2] * kern[2]];
  }
}

// 把一个颜色槽预处理成渲染状态（同槽只算一次权重与矩阵，逐像素零查找）
function slotState(sv, pal, D, mvOf, width, kind, mixK, defaultMatrix) {
  const c = slotRGB(sv, pal);
  const method = sv.method || 'mix';
  const st = { kind, c, method, sv, mx: null, mv: null, spanCap: 0, edKey: null };
  // 槽的矩阵默认「跟随全局面板」（matrixAuto），否则导入槽时写死的 bayer4 会把
  // 「阈值矩阵 2×2 / 4×4」这个面板参数彻底盖掉 —— 表现就是怎么切都没区别。
  const matName = (sv.matrixAuto === false)
    ? (sv.matrix || 'bayer4')
    : (DITHER_MATRICES[defaultMatrix] ? defaultMatrix : 'bayer4');
  if (method === 'mix') {
    const k = Math.min(4, Math.max(2, sv.mixColors ?? mixK ?? 4));
    st.mv = mvOf(matName, 0);
    if (k === 2) {
      st.mx = pairMix2(c, D, st.mv.m * st.mv.m, 0.10);
    } else {
      st.mx = pruneMix(restrictMix(slotMixWeights(c, D, sv), k, pal, sv.mixSpanLimit ?? 60),
                       sv.minWeight ?? 0);
    }
    st.spanCap = sv.mixSpan ?? 0;
  } else if (method === 'ordered') {
    st.mv = mvOf(matName, sv.strength ?? 64);
  } else if (method === 'ed') {
    st.edKey = (sv.filter || 'FS') + '|' + (sv.alpha ?? 1);
  }
  return st;
}

// 渲染。opts:
//   plan         analyzeProjection 的结果（缺省则现场分析）
//   slots        {层号: [槽,...]}，空 = 该层走自动投影
//   labels       Int32Array(npix)，值为槽 uid，-1 = 未手绘
//   labelSlots   {uid: 槽}，手绘标签的解析表
//   stage        'projection' | 'final'
//   onlyBand     只渲染某层（其余透明），-1 = 全部
function projectionDither(img, palette, D, opts) {
  opts = opts || {};
  const plan = opts.plan || analyzeProjection(img, D, opts);
  const { width, height, npix, opaque, bands, bandOf } = plan;
  const slotsMap = opts.slots || {};
  const labels = opts.labels || null;
  const labelSlots = opts.labelSlots || null;
  const only = (opts.onlyBand === undefined || opts.onlyBand === null) ? -1 : opts.onlyBand;
  const doFinal = (opts.stage || 'final') !== 'projection';
  const mixK = Math.min(4, Math.max(2, opts.mixColors ?? 4));
  let lineGroups = null, lineSlotMap = null;   // 线稿分段结果（供返回时报告段数）
  // 自动路径的默认阈值矩阵：必须来自面板参数，不能写死 bayer，
  // 否则「矩阵阶数 2×2」这类设置对成品图完全没有影响。
  const defaultMatrix = DITHER_MATRICES[opts.matrix] ? opts.matrix : 'bayer4';
  const src = img.data;
  const out = new Uint8ClampedArray(npix * 4);
  // 调色板可能在建 plan 之后变过（用户取消勾选），此时 mixes 里存的下标越界。
  // 与其抛 TypeError 进 alert，不如落到最近的合法板内色 —— 数据层另有 plan.palSig
  // 校验会在不匹配时重建 plan，这里是最后一道防线。
  const safeColor = (i, fb) => palette[i] || palette[nearestRGBIdx(fb || [0, 0, 0], palette)]
                          || palette[0] || [0, 0, 0];
  const put = (o, c) => {
    const cc = c || palette[0] || [0, 0, 0];
    out[o] = cc[0]; out[o + 1] = cc[1]; out[o + 2] = cc[2]; out[o + 3] = 255;
  };
  ensureLabData(D);

  const mvCache = new Map();
  const mvOf = (matName, strength) => {
    const key = matName + '|' + (strength ?? 0);
    let v = mvCache.get(key);
    if (!v) { v = matrixVals(DITHER_MATRICES[matName] || DITHER_MATRICES.bayer, strength ?? 0); mvCache.set(key, v); }
    return v;
  };
  const thAt = (mv, p) => mv.ths[(((p / width) | 0) % mv.m) * mv.m + (p % width) % mv.m];

  const nearestIdx = (r, g, b, targets) => {
    let bj = 0, bd = Infinity;
    for (let j = 0; j < targets.length; j++) {
      const c = targets[j];
      const d = (r - c[0]) ** 2 + (g - c[1]) ** 2 + (b - c[2]) ** 2;
      if (d < bd) { bd = d; bj = j; }
    }
    return bj;
  };

  // 颜色槽只作用于「手绘标签指向它」的像素，不参与任何自动分配。
  // 这保证了「改一个槽的颜色」只影响画笔画过的那片区域 ——
  // 曾经有过「按最近的槽自动分配」的版本，改色会让整层一起变。
  const stateOf = new Map();
  const stateFor = (sv, kind) => {
    let st = stateOf.get(sv);
    if (!st) { st = slotState(sv, D.palette, D, mvOf, width, kind, mixK, defaultMatrix); stateOf.set(sv, st); }
    return st;
  };
  const autoStateOf = new Map();   // 层号:色位 → 自动投影色的渲染状态

  const pick = new Array(npix);            // 每像素的渲染状态（null = 不渲染）
  const seqs = new Map();                  // 槽 → 层内 1D 误差扩散像素序列
  const edOpts = new Map();                // filter|alpha → 槽（2D 贴图共享）
  let needED = false;
  const forceBand = opts.forceBand || null;
  for (let p = 0; p < npix; p++) {
    if (!opaque[p]) continue;
    // 强制层覆盖：跨层涂色后，这块像素在显示上要算进被涂的那一层
    const b = (forceBand && forceBand[p] >= 0) ? forceBand[p] : bandOf[p];
    if (only >= 0 && only !== b) continue;
    const o = p * 4, r = src[o], g = src[o + 1], bl = src[o + 2];
    let st = null;
    if (labels && labelSlots) {
      const uid = labels[p];
      if (uid >= 0) { const sv = labelSlots[uid]; if (sv) st = stateFor(sv, 'slot'); }
    }
    // 该层有颜色槽时，像素按「最近的投影色」归到对应槽上。
    // 归属由投影色决定（建 plan 时就固定了），所以改某个槽的颜色/方法只影响
    // 它自己那一块，不会像「按最近槽色吸附」那样把整层都带跑。
    if (!st) {
      const band0 = bands[b];
      const ss = slotsMap[b];
      if (band0 && band0.projRGB.length && ss && ss.length) {
        let sv = null;
        if (ss.length === band0.projRGB.length) {
          sv = ss[nearestIdx(r, g, bl, band0.projRGB)];   // 与导入自动时一一对应
        } else {
          let bd0 = Infinity;                            // 用户增删过颜色：按颜色最近
          for (const cand of ss) {
            const c0 = slotRGB(cand, palette);
            if (!c0) continue;
            const d0 = (r - c0[0]) ** 2 + (g - c0[1]) ** 2 + (bl - c0[2]) ** 2;
            if (d0 < bd0) { bd0 = d0; sv = cand; }
          }
        }
        if (sv) st = stateFor(sv, 'slot');
      }
    }
    if (!st) {
      const band = bands[b];
      if (!band || !band.projRGB.length) continue;
      // 注意：下标是「该层第几个投影色」bj，不是层号 b。用层号索引 mixes 会在
      // 层号 >= 每层色数时取到 undefined，把整层像素当空处理（会留下透明洞）。
      const bj = nearestIdx(r, g, bl, band.projRGB);
      const key = b + ':' + bj;
      st = autoStateOf.get(key);
      if (!st) {
        // 配方缺失不丢像素：渲染阶段对 mx === null 走最近板内色平涂兜底
        const mv0 = mvOf(defaultMatrix, 0);
        const lv0 = mv0.m * mv0.m;
        let mx0 = null;
        if (mixK === 2) {
          // 2 色：直接按图案档位枚举色对，不再从四面体权重里截断
          mx0 = pairMix2(band.projRGB[bj], D, lv0, opts.mix2Penalty ?? 0.10);
        } else {
          const rawMix = (band.mixes && band.mixes[bj]) || null;
          mx0 = rawMix ? restrictMix(rawMix, mixK, palette, 60) : null;
        }
        st = { kind: 'auto', c: band.projRGB[bj], method: 'mix', sv: null,
               mx: mx0, mv: mv0, spanCap: 0, edKey: null };
        autoStateOf.set(key, st);
      }
    }
    pick[p] = st;
    if (st.method === 'error' && st.sv) {
      if (!seqs.has(st.sv)) seqs.set(st.sv, []);
      seqs.get(st.sv).push(p);
    } else if (st.method === 'ed' && st.sv) {
      needED = true; edOpts.set(st.edKey, st.sv);
    }
  }

  // 2D 误差扩散贴图（只在确有 ed 像素时计算；参数相同的槽共用一张）
  const edMaps = new Map();
  if (needED && doFinal) {
    for (const [key, sv] of edOpts) {
      edMaps.set(key, diffuseImage2D(img, palette, { filter: sv.filter || 'FS', alpha: sv.alpha ?? 1 }));
    }
  }

  for (let p = 0; p < npix; p++) {
    const st = pick[p];
    if (!st) continue;
    const o = p * 4;
    if (!doFinal) { put(o, st.c); continue; }
    if (st.method === 'direct') { put(o, palette[nearestRGBIdx(st.c, palette)]); continue; }
    if (st.method === 'error') {      // 先放原色，最后统一扩散
      out[o] = src[o]; out[o + 1] = src[o + 1]; out[o + 2] = src[o + 2]; out[o + 3] = 255;
      continue;
    }
    if (st.method === 'ed') {
      const em = edMaps.get(st.edKey);
      if (em) { out[o] = em[o]; out[o + 1] = em[o + 1]; out[o + 2] = em[o + 2]; out[o + 3] = 255; }
      continue;
    }
    if (st.method === 'ordered') {
      const r = src[o], g = src[o + 1], bl = src[o + 2];
      const off = (thAt(st.mv, p) - 0.5) * (st.sv.strength ?? 64);
      put(o, palette[nearestRGBIdx([r + off, g + off, bl + off], palette)]);
      continue;
    }
    if (!st.mx) { put(o, palette[nearestRGBIdx(st.c, palette)]); continue; }
    if (st.spanCap > 0) {
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < st.mx.idx.length; k++) {
        const L = D.labL ? D.labL[st.mx.idx[k]] : 0;
        if (L < lo) lo = L;
        if (L > hi) hi = L;
      }
      const luma = 0.299 * src[o] + 0.587 * src[o + 1] + 0.114 * src[o + 2];
      if ((hi - lo) > st.spanCap && luma > 76) { put(o, palette[nearestRGBIdx(st.c, palette)]); continue; }
    }
    const tt = thAt(st.mv, p);
    let acc = 0, sel = st.mx.w.length - 1;
    for (let k = 0; k < st.mx.w.length; k++) { acc += st.mx.w[k]; if (acc >= tt) { sel = k; break; } }
    put(o, safeColor(st.mx.idx[sel], st.c));
  }

  // 层内一维误差扩散（同槽像素按扫描序）
  for (const [sv, seq] of seqs) {
    if (seq.length) diffuseSeq(seq, palette, sv, out);
  }

  // 线稿保护：轮廓是 1~2px 的暗线，2 色抖动会把它们打成散点。只对「局部暗谷」
  // 足够强的格子动手，成片暗部（resp 很小）不受影响 —— 否则整块暗部会被涂死。
  // 只作用在成品阶段：投影阶段要能看到真实的投影色，不能提前被线覆盖。
  if (doFinal && opts.lineMask && (opts.lineKeep ?? 0) > 0) {
    const lm = opts.lineMask;
    const dark = opts.lineMode === 'dark';       // 整片暗部（风格化），不是细线
    const thr = opts.lineThresh ?? 35;
    const mix = Math.max(0, Math.min(1, opts.lineMix ?? 0));   // 0=纯线色 1=完全不留
    let cleanMask = null;
    // 「整片暗部」是风格化的整块纯色，骨架化会把它收成细线，效果很糟 —— 该模式不做清理
    if (opts.lineClean && !dark) {
      const raw = new Uint8Array(npix);
      for (let p = 0; p < npix; p++) {
        raw[p] = (dark ? (lm.minL[p] < thr) : (lm.resp[p] >= thr)) ? 1 : 0;
      }
      cleanMask = cleanLineMask(raw, width, height);
    }
    const light = opts.lineDir === 'light';          // 反相：抓比周围亮的线
    const RESPMAP = light ? lm.respLight : lm.resp;
    const RGBMAP = light ? lm.rgbLight : lm.rgb;
    const hit = (p) => cleanMask ? !!cleanMask[p]
      : (dark ? (lm.minL[p] < thr) : (RESPMAP[p] >= thr));
    const colOf = (p) => dark
      ? [lm.minRGB[p * 3], lm.minRGB[p * 3 + 1], lm.minRGB[p * 3 + 2]]
      : [RGBMAP[p * 3], RGBMAP[p * 3 + 1], RGBMAP[p * 3 + 2]];
    // 分段：每段可以被单独指定「固定色 / 跟随投影」
    const builtMask = cleanMask || (() => {
      const m = new Uint8Array(npix);
      for (let p = 0; p < npix; p++) {
        m[p] = (dark ? (lm.minL[p] < thr) : (RESPMAP[p] >= thr)) ? 1 : 0;
      }
      return m;
    })();
    lineGroups = labelLines(builtMask, width, height);
    const lineSlots = opts.lineSlots || null;
    lineSlotMap = lineSlots;
    const darkL = opts.lineDarkL ?? 55;
    // 暗线：线色只能取「比阈值暗」的板内色；亮线：只能取「比阈值亮」的
    const cand = [];
    for (let i = 0; i < palette.length; i++) {
      if (light ? (D.labL[i] > darkL) : (D.labL[i] < darkL)) cand.push(i);
    }
    const use = cand.length ? cand : palette.map((_, i) => i);
    for (let p = 0; p < npix; p++) {
      if (!hit(p)) continue;
      const o = p * 4;
      if (out[o + 3] === 0) continue;
      // 线色向该格原本的画面色靠拢：mix 越大线越淡，视觉上越"细"
      const base = colOf(p);
      const t0 = base[0], t1 = base[1], t2 = base[2];
      const tgt = rgbToLab([t0 * (1 - mix) + src[o] * mix,
                            t1 * (1 - mix) + src[o + 1] * mix,
                            t2 * (1 - mix) + src[o + 2] * mix]);
      let bi = use[0], bd = Infinity;
      for (const ci of use) {
        const P = D.labPoints[ci];
        const d = (P[0] - tgt[0]) ** 2 + (P[1] - tgt[1]) ** 2 + (P[2] - tgt[2]) ** 2;
        if (d < bd) { bd = d; bi = ci; }
      }
      let c = palette[bi] || [0, 0, 0];
      const gid = lineGroups.lab[p];
      const sv = (lineSlotMap && lineSlotMap[gid]) || (lineSlotMap && lineSlotMap.__all) || null;
      if (sv && sv.mode === 'fixed' && palette[sv.color]) c = palette[sv.color];
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2];
    }
  }
  return { width, height, data: out, lineCount: lineGroups ? lineGroups.count : 0 };
}

// 统一入口
function runDither(img, method, palette, D, opts) {
  if (method === 'ordered') return orderedDither(img, palette, opts);
  if (method === 'error_diffusion') return errorDiffusion(img, palette, opts);
  if (method === 'barycentric') return barycentricDither(img, palette, D, opts);
  if (method === 'projection') return projectionDither(img, palette, D, opts);
  throw new Error('未知方法: ' + method);
}
