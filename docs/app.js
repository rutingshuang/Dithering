// ============================================================
// app.js — 网页版 UI 逻辑
// 第一部分：公共状态、参数收集、尺寸换算
// ============================================================
const D = DITHER_DATA;
const palette = D.palette;
let srcData = null;          // 原图 ImageData（未缩放）

// 主画布的缓存在这里声明一次，另一半在编辑器段使用（唯一出口是 mainResult）
let mainCache = { key: '', im: null };

const $ = id => document.getElementById(id);

// ---------- 调色板选择 ----------
let selected = new Set(palette.map((_, i) => i));   // 初始全选
let baryData = DITHER_DATA;                          // 全选时用离线预计算
let lastSelSig = '';

function getSelectedPalette() {
  return [...selected].sort((a, b) => a - b).map(i => palette[i]);
}
function getSelSig() {
  return [...selected].sort((a, b) => a - b).join(',');
}
function ensureBaryData() {
  const sig = getSelSig();
  if (sig === lastSelSig) return;
  lastSelSig = sig;
  const sel = getSelectedPalette();
  // 全选 63 色 → 用离线预计算；否则运行时 Delaunay
  baryData = (sel.length === palette.length) ? DITHER_DATA : buildBaryData(sel);
}

// ---------- 尺寸 ----------
function computeTarget(w, h) {
  const wTxt = $('widthInput').value.trim();
  const hTxt = $('heightInput').value.trim();
  let nw, nh;
  if (wTxt && hTxt) { nw = +wTxt; nh = +hTxt; }
  else if (wTxt) { nw = +wTxt; nh = Math.round(h * nw / w); }
  else if (hTxt) { nh = +hTxt; nw = Math.round(w * nh / h); }
  else { const pct = +$('compression').value / 100; nw = Math.round(w * pct); nh = Math.round(h * pct); }
  return [Math.max(1, nw), Math.max(1, nh)];
}

function resizeData(imgData, tw, th) {
  if (imgData.width === tw && imgData.height === th) return imgData;
  const cv = document.createElement('canvas');
  cv.width = imgData.width; cv.height = imgData.height;
  cv.getContext('2d').putImageData(imgData, 0, 0);
  const ocv = document.createElement('canvas');
  ocv.width = tw; ocv.height = th;
  ocv.getContext('2d').drawImage(cv, 0, 0, tw, th);
  return ocv.getContext('2d').getImageData(0, 0, tw, th);
}

// ---------- 参数收集 ----------
function gatherOpts() {
  const m = $('method').value;
  if (m === 'ordered') {
    return { matrix: $('matrix').value, strength: +$('strength').value,
             colorSpace: $('colorSpace').value };
  }
  if (m === 'error_diffusion') {
    return { filter: $('filter').value, alpha: +$('alpha').value };
  }
  if (m === 'projection') {
    const black = +($('gradeBlack') || {}).value || 0;
    const white = +($('gradeWhite') || {}).value || 1;
    const gamma = +($('gradeGamma') || {}).value || 1;
    return {
      bands: Math.max(2, Math.min(64, +$('projBands').value || 24)),
      colors: Math.max(2, Math.min(32, +$('projColors').value || 8)),
      mixColors: Math.max(2, Math.min(4, +($('projMixColors') || {}).value || 4)),
      edgeMode: $('projEdge').value,
      clusterSpace: $('projSpace').value,
      matrix: $('projMatrix').value || 'bayer4',
      black: Math.min(black, white - 0.02),
      white: Math.max(white, black + 0.02),
      gamma,
    };
  }
  return {
    matrix: $('baryMatrix').value,
    strength: +$('baryStrength').value,
    gamut: $('gamut').checked,
    minWeight: +$('minWeight').value,
    edgeGuard: +$('edgeGuard').value,
    projLab: $('projLab').checked,
    refine: +$('refine').value,
    mixSpan: +$('mixSpan').value,
  };
}

// 需要 Delaunay 数据的方法（全选时用离线预计算，子集时运行时构建）
function needsBaryData(m) { return m === 'barycentric' || m === 'projection'; }

// 凸包/四面体混合要求 >= 4 色：少于 4 色时 projectToHull 拿不到有效面
const MIN_HULL_COLORS = 4;
// 纯判断：渲染路径（ensurePlan / refreshAll）用这个，不弹窗
function paletteUnsuitable(method) {
  return needsBaryData(method) && getSelectedPalette().length < MIN_HULL_COLORS;
}
// 带提示：只给「用户主动点分析 / 处理」的入口用
function paletteTooSmall(method) {
  if (!paletteUnsuitable(method)) return false;
  alert('重心混合/分层投影至少需要选中 ' + MIN_HULL_COLORS + ' 个调色板色（当前 ' +
        getSelectedPalette().length + ' 个）。\n四面体混合在少于 4 色时无解。');
  return true;
}

// ============================================================
// 分层编辑器
// 三个阶段共用一个 plan（分层 + 每层聚类），但「改什么」彼此独立：
//   ① 分层：色阶 / 分界 → 重新算 luma 与分桶（不重跑 k-means 之外的语义）
//   ② 投影：颜色槽 + 画笔标签 → 只对手绘像素生效
//   ③ 抖动：槽级方法参数（常驻可见，不需要点开）
// 主画布只有一个出口 paintCanvas()，视图只是它的入参。
// ============================================================
const ED = {
  plan: null,        // analyzeProjection 的结果
  src: null,         // 与 plan 同一份缩放后的源图 ImageData
  slots: {},         // {层号: [槽]}
  labels: null,      // Int32Array(npix)，值 = 槽 uid，-1 = 未手绘
  labelSlots: {},    // uid -> 槽
  uidSeq: 1,
  curBand: 0,
  curSlotUid: -1,
  lastSlotOfBand: {},// 每层上次用的槽，切层不丢画笔色
  editing: false,    // 是否处于编辑器模式
  view: 'result',    // result | source | projection | final
  layerOnly: false,  // 投影色/抖动色是否只看当前层
  highlightUid: -1,  // 只看某个颜色的区域（-1 = 不高亮）
  highlightOff: false,   // 「高亮当前色」开关关掉后的状态
  highlightHit: 0,       // 上一次高亮命中的像素数（给提示用）
  highlightEdge: 0,      // 上一次描边的边界像素数
  highlightPainted: 0,       // 该颜色里被手绘过的像素数（0 = 全靠自动分配）
  highlightSpeckle: false,   // 该颜色是否是「单像素椒盐」区域
  highlightThick: 1,         // 上一次描边用的线粗（图像像素）
  brush: 6,
  eraser: false,
  crossBand: false,
  painting: false,
  resizing: null,     // Alt+右键调笔刷大小时的拖动状态
  forceBand: null,    // Int32Array | null：跨层涂色后强制层（-1 = 按亮度分层）
  zoom: 0,           // 0 = 适配；否则为百分比
  pan: { x: 0, y: 0 },   // 中键拖动产生的平移量
  scale: 1,              // 当前画布显示倍率（fitCanvas 维护）
  rev: 0,
};

// 给自动化判据用的只读句柄（生产逻辑不读它）
window.__ditherEditor = ED;

const SLOT_METHODS = [
  ['mix', '重心混合'], ['direct', '直接投影'], ['ordered', '有序抖动'],
  ['error', '层内扩散'], ['ed', '二维扩散'],
];
const MATRIX_NAMES = [
  ['bayer', 'bayer'], ['clustered', 'clustered'], ['diagonal', 'diagonal'], ['spiral', 'spiral'],
];
// 界面上可见的矩阵键（4×4 与 2×2 各四个）
const MATRIX_KEYS = ['bayer4', 'clustered4', 'diagonal4', 'spiral4',
                     'bayer2', 'clustered2', 'diagonal2', 'spiral2'];

const onEl = (id, ev, fn) => {
  const el = $(id);
  if (el) el.addEventListener(ev, fn); else console.warn('missing element #' + id);
};
const hex2 = c => c.map(v => Math.max(0, Math.min(255, v | 0)).toString(16).padStart(2, '0')).join('');
const rgbCss = c => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
function slotColor(sv) {
  if (!sv) return null;
  if (typeof sv.color === 'number') {
    const sel = getSelectedPalette();
    return sel[sv.color] || palette[sv.color] || null;
  }
  return sv.color || null;
}

// used 语义已改成「槽只作用于手绘像素」，所以不再需要 used 参与自动分配；
// 保留字段是为了工程文件兼容与「待涂」提示。
// 这些默认值必须与「自动投影」的渲染路径一致（projLab=false 走与 mixWeights 相同的
// RGB 投影、mixSpan=0 不做跨度截断），否则一进编辑器、槽一建出来，成品就变了。
function slotDefaults(color, uid) {
  return { uid: uid ?? ED.uidSeq++,
           color: (color === undefined || color === null) ? 0 : color,
           method: 'mix', matrix: 'bayer4', strength: 64,
           filter: 'FS', alpha: 1, projLab: false, refine: 2,
           mixSpan: 0, gamut: true, minWeight: 0, used: false };
}

// ---------- 参数指纹 ----------
function optsSig(opts) {
  const o = opts || {};
  return [o.bands, o.colors, o.mixColors, o.edgeMode, o.clusterSpace,
          o.black, o.white, o.gamma].join('|');
}

function bumpRev() {
  ED.rev++;
  mainCache = { key: '', im: null };
}

function invalidatePlan() {
  ED.plan = null;
  ED.src = null;
  bumpRev();
}

function renderBands() {
  const box = $('bandList');
  if (!box) return;
  if (!ED.plan) {
    box.innerHTML = '<div class="hint">进入编辑器后自动分析</div>';
    if ($('bandCount')) $('bandCount').textContent = '-';
    return;
  }
  const plan = ED.plan;
  const painted = new Array(plan.nBands).fill(0);
  if (ED.labels) {
    for (let i = 0; i < ED.labels.length; i++) {
      const u = ED.labels[i];
      if (u < 0) continue;
      const sv = ED.labelSlots[u];
      const b = bandOfPx(i);
      if (sv && b >= 0) painted[b]++;
    }
  }
  if ($('bandCount')) $('bandCount').textContent = plan.nBands + ' 层';
  let html = '';
  for (const band of plan.bands) {
    const slots = ED.slots[band.index];
    const tag = (slots && slots.length) ? slots.length + ' 色' : band.projRGB.length + ' 自动';
    const pct = Math.round(band.ratio * 1000) / 10;
    const l0 = Math.round(band.lo * 100), l1 = Math.round(band.hi * 100);
    // 这层的代表色（前 4 个颜色），用来一眼区分「每层的颜色确实不一样」
    const repr = (ED.slots[band.index] || []).slice(0, 4)
      .map(sv => { const c = slotColor(sv); return c
        ? `<i style="background:${rgbCss(c)}"></i>` : ''; }).join('');
    html += `<div class="band-item${band.index === ED.curBand ? ' active' : ''}" data-band="${band.index}">
      <span class="name">第 ${band.index + 1} 层</span>
      <span class="bar" title="L ${l0}–${l1}">
        <i style="width:${Math.min(100, pct)}%"></i>
      </span>
      <span class="pct">${pct}%</span>
      <span class="tag">${repr}<em>L ${l0}–${l1} · ${tag}${painted[band.index] ? ' · 手绘 ' + painted[band.index] + 'px' : ''}</em></span>
    </div>`;
  }
  box.innerHTML = html;
}

function renderSlots(i) {
  const el = $('slotList');
  if (!el) return;
  if (!ED.plan) { el.innerHTML = ''; return; }
  const band = ED.plan.bands[i];
  if (!band) { el.innerHTML = ''; return; }
  const sel = getSelectedPalette();
  const slots = ED.slots[i];
  if (!slots || !slots.length) {
    el.innerHTML = `<div class="hint">这一层还没有颜色槽，点上面的「导入自动」按投影色生成。</div>`;
    return;
  }
  const sumOf = sv => {
    const m = sv.method || 'mix';
    const name = (SLOT_METHODS.find(x => x[0] === m) || [m, m])[1];
    const mat = (m === 'mix' || m === 'ordered') ? ' · ' + (sv.matrix || 'bayer4') : '';
    const extra = (m === 'error' || m === 'ed') ? ' · ' + (sv.filter || 'FS') : '';
    return name + mat + extra;
  };
  let html = '';
  slots.forEach((sv, j) => {
    const c = slotColor(sv) || [0, 0, 0];
    const m = sv.method || 'mix';
    html += `<div class="slot-card${sv.uid === ED.curSlotUid ? ' sel' : ''}${sv.used ? '' : ' pending'}"
                  data-uid="${sv.uid}">
      <div class="slot-main">
        <span class="swatch" data-act="pick" data-i="${i}" data-j="${j}"
              style="background:${rgbCss(c)}" title="设为画笔颜色"></span>
        <select data-act="color" data-i="${i}" data-j="${j}">
          ${sel.map((pc, k) => `<option value="${k}"${k === sv.color ? ' selected' : ''}>${k} #${hex2(pc)}</option>`).join('')}
          <option value="-1"${typeof sv.color !== 'number' ? ' selected' : ''}>自定义</option>
        </select>
        <select data-act="method" data-i="${i}" data-j="${j}" title="这个颜色的抖动方法">
          ${SLOT_METHODS.map(([v, t]) => `<option value="${v}"${v === m ? ' selected' : ''}>${t}</option>`).join('')}
        </select>
        <button data-act="opts" data-i="${i}" data-j="${j}" title="参数微调">参数</button>
        <button data-act="del" data-i="${i}" data-j="${j}" title="删除这个颜色">×</button>
      </div>
      <div class="slot-sum">${sumOf(sv)}${sv.used ? '' : ' · 待涂'}</div>
    </div>`;
  });
  el.innerHTML = html;
}

// 单色参数弹窗：方法 / 矩阵 / 强度 / 核 / alpha / 精修 / 跨度 / 凸包 / Lab / 权重截断
let dialogSlot = null;   // { i, j }

function openSlotDialog(i, j) {
  if (!ED.plan || !ED.slots[i] || !ED.slots[i][j]) return;
  dialogSlot = { i, j };
  const sv = ED.slots[i][j];
  const m = sv.method || 'mix';
  const isMix = m === 'mix', isOrdered = m === 'ordered', isDiff = (m === 'error' || m === 'ed');
  const matOpts = MATRIX_KEYS
    .map(k => `<option value="${k}"${k === (sv.matrix || 'bayer4') ? ' selected' : ''}>${k}</option>`).join('');
  const core = (isMix || isOrdered)
    ? `<label>矩阵阶数
        <select data-act="matrixSize" data-i="${i}" data-j="${j}">
          <option value="4"${((sv.matrix || 'bayer4').endsWith('4') ? ' selected' : '')}>4×4</option>
          <option value="2"${((sv.matrix || 'bayer4').endsWith('2') ? ' selected' : '')}>2×2（粗颗粒）</option>
        </select></label>
      <label>图案
        <select data-act="matrixName" data-i="${i}" data-j="${j}">
          ${MATRIX_NAMES.map(x => x[0]).map(k => `<option value="${k}"${(sv.matrix || 'bayer4').startsWith(k) ? ' selected' : ''}>${k}</option>`).join('')}
        </select></label>`
    : '';
  const strength = isOrdered
    ? `<label class="full">强度 <input type="range" min="0" max="128" value="${sv.strength ?? 64}"
        data-act="strength" data-i="${i}" data-j="${j}"> <span class="val">${sv.strength ?? 64}</span></label>`
    : '';
  const diff = isDiff
    ? `<label>扩散核
        <select data-act="filter" data-i="${i}" data-j="${j}">
          ${SLOT_FILTERS.map(v => `<option${v === (sv.filter || 'FS') ? ' selected' : ''}>${v}</option>`).join('')}
        </select></label>
      <label>alpha <input type="range" min="0" max="1" step="0.05" value="${sv.alpha ?? 1}"
        data-act="alpha" data-i="${i}" data-j="${j}"></label>
      <label class="full">强度 <input type="range" min="0" max="128" value="${sv.strength ?? 64}"
        data-act="strength" data-i="${i}" data-j="${j}"></label>`
    : '';
  const advanced = isMix
    ? `<label>混合色数
        <select data-act="mixColors" data-i="${i}" data-j="${j}">
          ${[4, 3, 2].map(k => `<option value="${k}"${(sv.mixColors ?? 4) === k ? ' selected' : ''}>${k} 色</option>`).join('')}
        </select></label>
      <label>面内精修 <input type="number" min="0" max="4" value="${sv.refine ?? 2}"
        data-act="refine" data-i="${i}" data-j="${j}"></label>
      <label>混合跨度 <input type="number" min="0" max="60" value="${sv.mixSpan ?? 35}"
        data-act="mixSpan" data-i="${i}" data-j="${j}"></label>
      <label>权重截断 <input type="number" min="0" max="0.5" step="0.05" value="${sv.minWeight ?? 0}"
        data-act="minWeight" data-i="${i}" data-j="${j}"></label>
      <label><input type="checkbox" data-act="projLab" data-i="${i}" data-j="${j}"
        ${sv.projLab !== false ? ' checked' : ''}> Lab 投影</label>
      <label><input type="checkbox" data-act="gamut" data-i="${i}" data-j="${j}"
        ${sv.gamut !== false ? ' checked' : ''}> 凸包外投影</label>`
    : '';

  const body = `
    <div class="method-tabs full">
      ${SLOT_METHODS.map(([v, t]) => `<button class="mt${v === m ? ' active' : ''}"
          data-setmethod="${v}">${t}</button>`).join('')}
    </div>
    <label class="full">颜色
      <select data-act="color" data-i="${i}" data-j="${j}">
        ${getSelectedPalette().map((pc, k) => `<option value="${k}"${k === sv.color ? ' selected' : ''}>${k} #${hex2(pc)}</option>`).join('')}
        <option value="-1"${typeof sv.color !== 'number' ? ' selected' : ''}>自定义</option>
      </select>
      <input type="color" value="#${hex2(slotColor(sv) || [0, 0, 0])}" data-act="custom" data-i="${i}" data-j="${j}">
    </label>
    ${core}${strength}${diff}
    ${advanced ? `<details class="full adv"><summary>高级参数</summary><div class="adv-grid">${advanced}</div></details>` : ''}
    <div class="hint full">${
      isMix ? '重心混合：目标色可以是板外色，用板内色按比例混出来；色号是板内色时必然是平涂。'
      : m === 'ordered' ? '有序抖动：以这个颜色为基色做阈值抖动，纹理最明显。'
      : m === 'direct' ? '直接投影：全部量化到最近板内色，平涂无纹理。'
      : m === 'ed' ? '二维扩散：整图按扫描序做误差扩散，本颜色的像素取扩散结果。'
      : '层内一维扩散：同色像素按扫描序累积误差，纹理比二维弱。'
    }</div>`;
  $('slotDialogBody').innerHTML = body;
  $('slotDialogTitle').textContent = '第 ' + (i + 1) + ' 层 · 颜色 ' + (j + 1);
  $('slotDialog').classList.remove('hidden');
}

// 参数改动（列表与弹窗共用）
function applySlotParam(t, sv) {
  const act = t.dataset.act;
  if (act === 'color') {
    if (+t.value < 0) {
      sv.color = Array.isArray(sv.color) ? sv.color : (getSelectedPalette()[sv.color] || [128, 128, 128]).slice();
    } else sv.color = +t.value;
  } else if (act === 'custom') {
    const h = t.value.replace('#', '');
    sv.color = [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  } else if (act === 'method') sv.method = t.value;
  else if (act === 'matrix') sv.matrix = t.value;
  else if (act === 'matrixSize') sv.matrix = (sv.matrix || 'bayer4').replace(/[24]$/, '') + t.value;
  else if (act === 'matrixName') sv.matrix = t.value + ((sv.matrix || 'bayer4').slice(-1) === '2' ? '2' : '4');
  else if (act === 'mixColors') sv.mixColors = +t.value;
  else if (act === 'filter') sv.filter = t.value;
  else if (act === 'projLab') sv.projLab = t.checked;
  else if (act === 'gamut') sv.gamut = t.checked;
  else if (['strength', 'alpha', 'refine', 'mixSpan', 'minWeight'].includes(act)) sv[act] = +t.value;
  else return false;
  if (sv.uid === ED.curSlotUid) {
    const c = slotColor(sv);
    if ($('brushSwatch')) $('brushSwatch').style.background = c ? rgbCss(c) : 'transparent';
  }
  bumpRev();
  return true;
}

// ---------- 撤销栈 ----------
const UNDO = { stack: [], redo: [], cur: null };
function beginStroke() { UNDO.cur = { px: [], old: [], add: [] }; }
function recordPixel(p, oldVal, newVal) {
  if (!UNDO.cur) return;
  UNDO.cur.px.push(p); UNDO.cur.old.push(oldVal); UNDO.cur.add.push(newVal);
}
function endStroke() {
  if (UNDO.cur && UNDO.cur.px.length) {
    UNDO.stack.push(UNDO.cur);
    if (UNDO.stack.length > 60) UNDO.stack.shift();
    UNDO.redo.length = 0;
  }
  UNDO.cur = null;
  updateUndoButtons();
}
function applyOp(op, dir) {
  if (!ED.labels || !op) return;
  for (let k = 0; k < op.px.length; k++) {
    ED.labels[op.px[k]] = dir > 0 ? op.add[k] : op.old[k];
  }
  bumpRev();
}
function undo() {
  const op = UNDO.stack.pop();
  if (!op) return;
  applyOp(op, -1);
  UNDO.redo.push(op);
  afterEdit();
  updateUndoButtons();
}
function redo() {
  const op = UNDO.redo.pop();
  if (!op) return;
  applyOp(op, 1);
  UNDO.stack.push(op);
  afterEdit();
  updateUndoButtons();
}
function updateUndoButtons() {
  if ($('undoBtn')) $('undoBtn').disabled = !UNDO.stack.length;
  if ($('redoBtn')) $('redoBtn').disabled = !UNDO.redo.length;
}

// ---------- plan 构建 ----------
function buildPlan(opts, keepEdges) {
  ensureBaryData();
  const [tw, th] = computeTarget(srcData.width, srcData.height);
  const src = resizeData(srcData, tw, th);
  const nBands = opts.bands;
  const oldN = ED.plan ? ED.plan.nBands : 0;
  const oldK = ED.plan ? ED.plan.K : 0;

  if (!ED.plan || ED.plan.npix !== src.width * src.height) {
    ED.labels = new Int32Array(src.width * src.height).fill(-1);
    UNDO.stack.length = 0; UNDO.redo.length = 0;
  }
  const o = Object.assign({}, opts);
  if (keepEdges && ED.plan && oldN === nBands) o.edges = Array.from(ED.plan.edges);

  const plan = analyzeProjection(src, baryData, o);
  plan.sig = optsSig(opts);
  plan.palSig = getSelSig();
  ED.plan = plan;
  ED.src = src;
  if (ED.labels && ED.labels.length !== plan.npix) {
    ED.labels = new Int32Array(plan.npix).fill(-1);
  }
  // 层数变了：槽按在层内的相对位置重新归位，手绘标签保留（uid 不变）
  if (oldN && oldN !== plan.nBands) regrowSlots(oldN, plan.nBands);
  ED.curBand = Math.min(ED.curBand, plan.nBands - 1);

  // 层数 / 每层色数变了：颜色槽必须跟着重建。
  // 以前只重排旧槽（regrowSlots），结果「槽数 ≠ 投影色数」，渲染退回按颜色最近匹配，
  // 用旧层的颜色去画新层的像素 —— 实测 ΔE8 从 6.56 暴涨到 20.76、整体暗掉 45%。
  const bandsChanged = (plan.nBands !== oldN) || (plan.K !== oldK);
  if (bandsChanged || !Object.keys(ED.slots).length) {
    // 重建之前把用户改过的抖动方式与参数记下来，重建后按「同层同序号」还原 ——
    // 否则一次意外的 plan 重建就会把调好的方法/参数全部打回默认。
    const keep = {};
    const KEYS = ['method', 'matrix', 'strength', 'filter', 'alpha',
                  'projLab', 'refine', 'mixSpan', 'gamut', 'minWeight', 'mixColors', 'used'];
    for (const k in ED.slots) {
      keep[k] = ED.slots[k].map(sv => {
        const o = {};
        for (const kk of KEYS) if (sv[kk] !== undefined) o[kk] = sv[kk];
        return o;
      });
    }
    ED.slots = {};
    ED.labelSlots = {};
    ED.uidSeq = 1;
    for (let bi = 0; bi < plan.nBands; bi++) {
      importAutoSlots(bi, true);
      const old = keep[bi];
      if (old && ED.slots[bi]) {
        ED.slots[bi].forEach((sv, j) => {
          if (old[j]) Object.assign(sv, old[j]);
          ED.labelSlots[sv.uid] = sv;
        });
      }
    }
    // 层结构变了，旧的手绘标签无处可依
    if (ED.labels) ED.labels.fill(-1);
    if (ED.forceBand) ED.forceBand.fill(-1);
    UNDO.stack.length = 0;
    UNDO.redo.length = 0;
    if (ED.slots[ED.curBand] && ED.slots[ED.curBand].length) {
      ED.curSlotUid = ED.slots[ED.curBand][0].uid;
      ED.lastSlotOfBand = {};
    }
  }

  if (ED.forceBand && ED.forceBand.length !== plan.npix) ED.forceBand = null;
  if (ED.labels) {
    if (!ED.forceBand || ED.forceBand.length !== plan.npix) {
      ED.forceBand = new Int32Array(plan.npix).fill(-1);
    }
    reconcileLabels();
  }
  bumpRev();
  return plan;
}

function regrowSlots(oldN, newN) {
  const out = {};
  for (const k in ED.slots) {
    const idx = Math.min(newN - 1, Math.round((+k + 0.5) / oldN * newN - 0.5));
    if (!out[idx]) out[idx] = [];
    out[idx].push(...ED.slots[k]);
  }
  ED.slots = out;
}

function ensurePlan(opts) {
  if (!srcData) return null;
  if (paletteUnsuitable($('method').value)) return null;
  const o = opts || gatherOpts();
  const [tw, th] = computeTarget(srcData.width, srcData.height);
  const stale = !ED.plan || !ED.src
    || ED.plan.sig !== optsSig(o)
    || ED.plan.palSig !== getSelSig()
    || ED.src.width !== tw || ED.src.height !== th;
  if (stale) {
    buildPlan(o, true);
    renderBands(); renderSlots(ED.curBand);
    if (ED.editing) renderEdgeBar();
  }
  return ED.plan;
}

function analyzeNow(keepEdges) {
  if (!srcData) { alert('请先选择图片'); return false; }
  if (paletteTooSmall($('method').value)) return false;
  buildPlan(gatherOpts(), keepEdges !== false);
  renderBands(); renderSlots(ED.curBand); renderEdgeBar();
  paintCanvas();
  return true;
}

function rebucketNow() {
  if (!ED.plan || !ED.src) return;
  bucketBands(ED.plan, ED.src, false, baryData);
  bumpRev();
  renderBands(); renderEdgeBar();
  paintCanvas();
}

let rebucketTimer = null;
function rebucketDebounced() {
  clearTimeout(rebucketTimer);
  rebucketTimer = setTimeout(rebucketNow, 120);
}

// ---------- 渲染：唯一出口 ----------

function mainKey() {
  const [tw, th] = srcData ? computeTarget(srcData.width, srcData.height) : [0, 0];
  return [$('method').value, JSON.stringify(gatherOpts()), getSelSig(),
          tw + 'x' + th, ED.rev].join('@');
}

// 成品图：按当前方法算，与编辑器状态无关的走 runDither，投影走编辑器管线
function mainResult() {
  if (!srcData) return null;
  const m = $('method').value;
  if (m === 'projection') ensurePlan();
  const key = mainKey();
  if (mainCache.key === key && mainCache.im) return mainCache.im;
  let im = null;
  if (m === 'projection') {
    if (!ED.plan || !ED.src) return null;
    im = renderStage('final', -1);
  } else {
    if (needsBaryData(m)) ensureBaryData();
    const [tw, th] = computeTarget(srcData.width, srcData.height);
    const src = resizeData(srcData, tw, th);
    im = runDither(src, m, getSelectedPalette(), baryData, gatherOpts());
  }
  mainCache = { key, im };
  return im;
}

function renderStage(stage, onlyBand) {
  if (!ED.plan || !ED.src) return null;
  if (stage === 'source') return extractBand(ED.src, ED.plan, onlyBand);
  ensureBaryData();
  return runDither(ED.src, 'projection', getSelectedPalette(), baryData,
                   Object.assign({}, gatherOpts(), {
                     plan: ED.plan, slots: ED.slots, labels: ED.labels,
                     labelSlots: ED.labelSlots, stage,
                     forceBand: ED.forceBand,
                     onlyBand: (onlyBand === undefined ? -1 : onlyBand),
                   }));
}

// 高亮：只保留选中颜色（手绘到它的像素），其余压暗，方便看「这个颜色到底占了哪」
// 一个像素「归哪个颜色」：
//   手绘过 → 按标签；
//   没涂过 → 先找它在**投影色**里最近的那个，再把该投影色映射到对应槽。
// 不能直接按「离哪个槽最近」判：槽之间颜色很近，相邻像素会来回跳，
// 区域碎成椒盐点，描边就会把整块涂满。
function slotOfPixel(p) {
  if (ED.labels && ED.labels[p] >= 0) return ED.labels[p];
  const b = bandOfPx(p);
  const band = ED.plan.bands[b];
  const slots = ED.slots[b];
  if (!band || !slots || !slots.length || !band.projRGB.length) return -1;
  const o = p * 4;
  const r = ED.src.data[o], g = ED.src.data[o + 1], bl = ED.src.data[o + 2];
  let bj = 0, bd = Infinity;
  for (let j = 0; j < band.projRGB.length; j++) {
    const c = band.projRGB[j];
    const d = (r - c[0]) ** 2 + (g - c[1]) ** 2 + (bl - c[2]) ** 2;
    if (d < bd) { bd = d; bj = j; }
  }
  // 槽是「导入自动」一一对应生成时，直接按索引映射（区域与投影块完全一致）
  if (slots.length === band.projRGB.length && slots[bj]) return slots[bj].uid;
  // 用户增删过颜色：退回按最近颜色
  let best = -1, bestD = Infinity;
  for (const sv of slots) {
    const c = slotColor(sv);
    if (!c) continue;
    const d = (r - c[0]) ** 2 + (g - c[1]) ** 2 + (bl - c[2]) ** 2;
    if (d < bestD) { bestD = d; best = sv.uid; }
  }
  return best;
}

const HL_EDGE = [0, 220, 235];   // 选中颜色的边界线（青色）

// 像素实际所属的层（跨层涂过的按强制层算）
function bandOfPx(p) {
  if (ED.forceBand && ED.forceBand[p] >= 0) return ED.forceBand[p];
  return ED.plan.bandOf[p];
}

// plan 被重建（改层数/尺寸/色阶）后，labels 里可能还指着「已经不在这一层」的槽。
// 不清掉的话渲染会用错槽的颜色，画面就乱了。作为兜底保留，正常路径由上面的锁挡住。
function reconcileLabels() {
  if (!ED.labels || !ED.plan) return 0;
  const slotBand = {};
  for (const k in ED.slots) for (const sv of ED.slots[k]) slotBand[sv.uid] = +k;
  let cleared = 0;
  for (let p = 0; p < ED.labels.length; p++) {
    const uid = ED.labels[p];
    if (uid < 0) continue;
    const owner = (ED.forceBand && ED.forceBand[p] >= 0) ? ED.forceBand[p] : ED.plan.bandOf[p];
    if (slotBand[uid] === undefined || slotBand[uid] !== owner) {
      ED.labels[p] = -1;
      if (ED.forceBand) ED.forceBand[p] = -1;
      cleared++;
    }
  }
  return cleared;
}

// 有手绘之后锁住分层与尺寸：改这些会重排图层，手绘数据必然对不上
function layersLocked() {
  return ED.editing && paintedCount() > 0;
}

function refreshLockState() {
  const locked = layersLocked();
  const n = paintedCount();
  ['projBands', 'projColors', 'projMixColors', 'compression', 'compNum',
   'widthInput', 'heightInput', 'gradeBlack', 'gradeWhite', 'gradeGamma',
   'openEdge'].forEach(id => { const el = $(id); if (el) el.disabled = locked; });
  if ($('projEdge')) $('projEdge').disabled = locked;
  if ($('projSpace')) $('projSpace').disabled = locked;
  if ($('projMatrix')) $('projMatrix').disabled = locked;
  if ($('projMatrixSize')) $('projMatrixSize').disabled = locked;
  if ($('lockBadge')) {
    $('lockBadge').textContent = locked ? '已锁定 · ' + n + ' px 手绘' : '未锁定';
    $('lockBadge').style.color = locked ? 'var(--warn)' : '';
  }
  if ($('unlockBtn')) $('unlockBtn').classList.toggle('hidden', !locked);
  if ($('lockHint')) {
    $('lockHint').textContent = locked
      ? '分层参数与色阶已锁定：改它们会重排图层，手绘数据对不上。要改先点「解锁并清除上色」。'
      : '';
  }
}

// 已有多少手绘像素
function paintedCount() {
  if (!ED.labels) return 0;
  let n = 0;
  for (let i = 0; i < ED.labels.length; i++) if (ED.labels[i] >= 0) n++;
  return n;
}

// 3×3 多数滤波：邻居过半则置 1，消除孤立噪点，让区域连成块
function smoothMask(mask, w, h, rounds) {
  let cur = mask;
  for (let r = 0; r < rounds; r++) {
    const next = new Uint8Array(cur.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let ones = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            ones += cur[ny * w + nx]; n++;
          }
        }
        next[y * w + x] = (ones * 2 > n) ? 1 : 0;
      }
    }
    cur = next;
  }
  return cur;
}

function renderHighlighted(uid) {
  const npix = ED.plan.npix, w = ED.plan.width, h = ED.plan.height;
  const im = renderStage(ED.view, ED.layerOnly ? ED.curBand : -1);
  if (!im) return null;
  // 归属规则与渲染一致：手绘优先，否则按投影色索引映射到槽。
  // 所以「还没涂过」的颜色也有区域 —— 那正是它现在被自动分配到的像素。
  const mask = new Uint8Array(npix);
  let paintedN = 0;
  for (let p = 0; p < npix; p++) {
    mask[p] = (slotOfPixel(p) === uid) ? 1 : 0;
    if (ED.labels && ED.labels[p] === uid) paintedN++;
  }

  // 这个颜色的像素是不是连成片的？分层是按亮度切的，某些颜色本身就是
  // 「单像素椒盐」—— 那种区域描边只会糊成一片，得换一种显示方式。
  let hit = 0, joined = 0;
  for (let p = 0; p < npix; p++) {
    if (!mask[p]) continue;
    hit++;
    const x = p % w;
    let nb = 0;
    if (x > 0 && mask[p - 1]) nb++;
    if (x < w - 1 && mask[p + 1]) nb++;
    if (p >= w && mask[p - w]) nb++;
    if (p + w < npix && mask[p + w]) nb++;
    if (nb >= 2) joined++;
  }
  const continuity = hit ? joined / hit : 0;
  const speckle = continuity < 0.55;      // 椒盐：邻居大多也是自己人的比例很低

  // 线粗随显示倍率走：缩小时 1px 线根本看不见
  const scale = ED.scale || 1;
  const thick = scale >= 1 ? 1 : Math.min(6, Math.max(1, Math.round(1 / scale)));

  const out = new Uint8ClampedArray(im.data);
  let edge = 0;
  for (let p = 0; p < npix; p++) {
    const o = p * 4;
    if (out[o + 3] === 0) continue;
    if (!mask[p]) {
      // 不属于这个颜色：轻微压暗（0.55）—— 压太黑就看不出抖动的纹理差别了，
      // 而「比较不同颜色用不同方法的优劣」正是高亮的用途之一
      out[o] = out[o] * 0.55; out[o + 1] = out[o + 1] * 0.55; out[o + 2] = out[o + 2] * 0.55;
      continue;
    }
    if (speckle) {
      // 椒盐区域：半透明青叠加 —— 既标出位置，又不完全盖掉该像素的实际颜色
      out[o] = HL_EDGE[0] * 0.38 + out[o] * 0.62;
      out[o + 1] = HL_EDGE[1] * 0.38 + out[o + 1] * 0.62;
      out[o + 2] = HL_EDGE[2] * 0.38 + out[o + 2] * 0.62;
      continue;
    }
    // 连片区域：只描边界，线粗 = thick（缩小显示时变粗才看得见）
    const x = p % w, y = (p / w) | 0;
    let isEdge = false;
    for (let dy = -thick; dy <= thick && !isEdge; dy++) {
      const ny = y + dy;
      for (let dx = -thick; dx <= thick; dx++) {
        const nx = x + dx;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) { isEdge = y === 0 || x === 0 || y === h - 1 || x === w - 1; break; }
        if (!mask[ny * w + nx]) { isEdge = true; break; }
      }
    }
    if (isEdge) {
      out[o] = HL_EDGE[0]; out[o + 1] = HL_EDGE[1]; out[o + 2] = HL_EDGE[2];
      edge++;
    }
  }
  ED.highlightHit = hit;
  ED.highlightEdge = edge;
  ED.highlightPainted = paintedN;
  ED.highlightSpeckle = speckle;
  ED.highlightThick = thick;
  return { width: im.width, height: im.height, data: out };
}

function extractBand(src, plan, index) {
  if (index === undefined || index === null || index < 0) {
    // 整图原色（不改动像素，直接拷一份）
    return { width: src.width, height: src.height, data: new Uint8ClampedArray(src.data) };
  }
  const out = new Uint8ClampedArray(src.data.length);
  const band = plan.bands[index];
  if (!band) return { width: src.width, height: src.height, data: out };
  for (const q of band.members) {
    const o = q * 4;
    out[o] = src.data[o]; out[o + 1] = src.data[o + 1];
    out[o + 2] = src.data[o + 2]; out[o + 3] = 255;
  }
  return { width: src.width, height: src.height, data: out };
}

// 当前视图应当显示的图。视图只是入参，不存在第二份缓存。
// 投影色 / 抖动色给**全图**：工作区要能上色就必须看得见整张画面，
// 当前层的范围只用来限制画笔落点（见 brushStamp 的 bandOf 过滤）。
// 原色视图例外 —— 它本来就是「只看这一层」的观察工具。
function viewImage() {
  if (ED.view === 'result') return mainResult();
  if (!ED.plan || !ED.src) return null;
  if (ED.view !== 'source' && !ED.highlightOff) {
    const hl = ED.highlightUid;
    if (hl >= 0) {
      const hi = renderHighlighted(hl);
      if (hi) return hi;
      // 这个颜色还没涂过 → 退回普通渲染（不进高亮、不清画面）
    }
  }
  // 三个视图一视同仁：默认整图，勾了「只看当前层」才只画这一层。
  // （以前 source 写死成 ED.curBand，导致原色视图永远是单层、跟开关无关）
  if (ED.layerOnly) return renderStage(ED.view, ED.curBand);
  return renderStage(ED.view, -1);
}

// 把 viewImage() 画到唯一画布上
function paintCanvas() {
  const cv = $('dstCanvas');
  if (!cv) return;
  const im = viewImage();
  if (!im) {
    if ($('stageHint')) $('stageHint').textContent = srcData ? '点「处理」出图' : '先选择一张图片';
    return;
  }
  cv.width = im.width; cv.height = im.height;
  cv.getContext('2d').putImageData(new ImageData(im.data, im.width, im.height), 0, 0);
  fitCanvas();
  const hints = {
    result: '成品图（当前方法的最终结果）',
    source: '原色：整图原色（勾「只看当前层」只看第 N 层）',
    projection: '投影色：整图的投影结果；画笔只改当前层',
    final: '抖动色：整图按各颜色槽的方法渲染后的结果',
  };
  let hlTip = '';
  if (ED.highlightUid >= 0 && !ED.highlightOff && ED.view !== 'source') {
    const src0 = ED.highlightPainted > 0 ? '（手绘 ' + ED.highlightPainted + ' px）' : '（自动分配）';
    hlTip = ED.highlightHit > 0
      ? ' · 高亮 ' + ED.highlightHit + ' px' + src0 + (ED.highlightSpeckle
          ? '，该区域是零散像素，用青色点标出'
          : '，青色描边' + (ED.highlightThick > 1
              ? '（线粗 ' + ED.highlightThick + 'px，当前缩放看不清 1px）' : '') + '是它的边界')
      : ' · 这个颜色在图上没有像素';
  }
  if ($('stageHint')) {
    $('stageHint').textContent = ED.view === 'result'
      ? hints.result
      : (ED.layerOnly ? '只看第 ' + (ED.curBand + 1) + ' 层 · ' : '第 ' + (ED.curBand + 1) + ' 层可涂 · ')
        + hints[ED.view] + hlTip;
  }
}

function updateViewButtons() {
  document.querySelectorAll('#viewTabs button').forEach(b => {
    b.classList.toggle('active', b.dataset.view === ED.view);
    b.disabled = (b.dataset.view !== 'result') && !ED.plan;
  });
}

function refreshAll() {
  updateViewButtons();
  paintCanvas();
}

// ---------- 画布缩放 ----------
// 缩放因子取整数倍或整数分之一：非整数缩放会重采样出规则纹理，
// 看起来像深色棋盘格（真实原因是缩小，不是抖动算法）。
function niceScale(raw) {
  if (raw >= 1) return Math.max(1, Math.floor(raw + 1e-6));
  return 1 / Math.max(1, Math.ceil(1 / raw - 1e-6));
}

// 平移：中键拖动（1:1 或放大后画布会超出视口）
function applyPan() {
  const fit = $('canvasFit');
  if (!fit) return;
  fit.style.transform = 'translate(' + ED.pan.x + 'px,' + ED.pan.y + 'px)';
}

function bindPan() {
  const cv = $('dstCanvas');
  if (!cv) return;
  let dragging = false, sx = 0, sy = 0, px = 0, py = 0;
  cv.addEventListener('mousedown', ev => {
    if (ev.button !== 1) return;          // 中键
    ev.preventDefault();
    dragging = true;
    sx = ev.clientX; sy = ev.clientY; px = ED.pan.x; py = ED.pan.y;
  });
  cv.addEventListener('auxclick', ev => { if (ev.button === 1) ev.preventDefault(); });
  window.addEventListener('mousemove', ev => {
    if (!dragging) return;
    ED.pan.x = px + (ev.clientX - sx);
    ED.pan.y = py + (ev.clientY - sy);
    applyPan();
  });
  window.addEventListener('mouseup', ev => { if (ev.button === 1) dragging = false; });
  // 中键滚轮也被浏览器当作自动滚动，挡掉
  cv.addEventListener('wheel', ev => {
    if (ev.button === 1) ev.preventDefault();
  }, { passive: false });
}

// 缩放停下后补一次重绘：描边线粗是按显示倍率算的，而 fitCanvas 在渲染之后
// 才更新 ED.scale，不补这一下的话线粗永远停在上一次的档位。
let repaintTimer = null;
let repainting = false;
function scheduleRepaint() {
  clearTimeout(repaintTimer);
  repaintTimer = setTimeout(() => {
    if (repainting) return;
    repainting = true;
    paintCanvas();
    repainting = false;
  }, 180);
}

function fitCanvas() {
  const cv = $('dstCanvas'), fit = $('canvasFit'), body = $('stageBody');
  if (!cv || !fit || !body || !cv.width) return;
  const pad = 24;
  const availW = body.clientWidth - pad * 2;
  const availH = body.clientHeight - pad * 2;
  let pct;
  if (ED.zoom > 0) pct = ED.zoom / 100;
  else pct = niceScale(Math.min(availW / cv.width, availH / cv.height, 16));
  cv.style.width = Math.max(1, Math.round(cv.width * pct)) + 'px';
  cv.style.height = Math.max(1, Math.round(cv.height * pct)) + 'px';
  cv.style.maxWidth = 'none';                     // 交给这里的尺寸；CSS 的 max-width 会
  cv.style.maxHeight = 'none';                    // 只压宽度、放任高度 → 竖向拉伸
  ED.scale = pct;                                 // 描边线粗按显示倍率自适应
  cv.classList.toggle('zoom-in', pct >= 1);       // 只有放大才用最近邻
  if ($('zoomLabel')) {
    $('zoomLabel').textContent = ED.zoom > 0 ? Math.round(pct * 100) + '%'
      : (Math.round(pct * 100) + '% (适配)');
  }
  const cl = $('cursorLayer');
  if (cl) { cl.width = cv.width; cl.height = cv.height; }
  applyPan();
  // 高亮开着时，倍率变了要按新倍率重算线粗
  if (!repainting && ED.highlightUid >= 0 && !ED.highlightOff
      && ED.view !== 'result' && ED.view !== 'source') {
    scheduleRepaint();
  }
}

// ---------- 画笔 ----------
function brushStamp(x, y) {
  if (!ED.plan || !ED.labels) return;
  const plan = ED.plan, w = plan.width, h = plan.height, r = Math.max(1, ED.brush);
  const uid = ED.eraser ? -1 : ED.curSlotUid;
  if (!ED.eraser && uid < 0) return;
  const sv = ED.labelSlots[uid];
  if (sv) sv.used = true;
  const r2 = r * r;
  for (let dy = -r; dy <= r; dy++) {
    const py = y + dy;
    if (py < 0 || py >= h) continue;
    for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r2) continue;
      const px = x + dx;
      if (px < 0 || px >= w) continue;
      const p = py * w + px;
      if (!ED.crossBand && plan.bandOf[p] !== ED.curBand) continue;
      const oldVal = ED.labels[p];
      if (oldVal === uid) continue;
      recordPixel(p, oldVal, uid);
      ED.labels[p] = uid;
      // 涂到了别的层的像素上：把这像素在显示上并进当前层，否则它仍算原来那层
      if (ED.forceBand && plan.bandOf[p] !== ED.curBand) ED.forceBand[p] = ED.curBand;
    }
  }
  bumpRev();
}

function canvasPos(cv, ev) {
  const r = cv.getBoundingClientRect();
  return [Math.floor((ev.clientX - r.left) * cv.width / r.width),
          Math.floor((ev.clientY - r.top) * cv.height / r.height)];
}

function drawCursor(x, y) {
  const cl = $('cursorLayer'), cv = $('dstCanvas');
  if (!cl || !cv || ED.painting) return;    // 拖动中保留笔迹，别清屏
  const g = cl.getContext('2d');
  g.clearRect(0, 0, cl.width, cl.height);
  if (x === undefined) return;
  const scale = cl.width / cv.width;
  g.strokeStyle = ED.eraser ? 'rgba(255,150,150,.95)' : 'rgba(255,255,255,.95)';
  g.lineWidth = 1.5;
  g.beginPath();
  g.arc(x * scale, y * scale, Math.max(2, ED.brush * scale), 0, Math.PI * 2);
  g.stroke();
  g.strokeStyle = 'rgba(0,0,0,.6)';
  g.lineWidth = 1;
  g.beginPath();
  g.arc(x * scale, y * scale, Math.max(2, ED.brush * scale) + 1, 0, Math.PI * 2);
  g.stroke();
}

// 拖动中在 overlay 上画笔迹：全量重绘一张 810K px 的图要 200ms+，
// 挂在 mousemove 上就会把事件吞掉，表现出来就是「只能点出离散的点」。
function paintStrokeDot(x, y) {
  const cl = $('cursorLayer');
  if (!cl) return;
  const g = cl.getContext('2d');
  const sv = ED.labelSlots[ED.curSlotUid];
  const c = ED.eraser ? [40, 40, 60] : (sv ? slotColor(sv) : [255, 255, 255]);
  g.globalAlpha = ED.eraser ? 0.45 : 0.85;
  g.fillStyle = rgbCss(c || [255, 255, 255]);
  g.beginPath();
  g.arc(x, y, Math.max(1, ED.brush), 0, Math.PI * 2);
  g.fill();
  g.globalAlpha = 1;
}

function clearOverlay() {
  const cl = $('cursorLayer');
  if (cl) cl.getContext('2d').clearRect(0, 0, cl.width, cl.height);
}

function bindPaint() {
  const cv = $('dstCanvas');
  if (!cv) return;
  cv.addEventListener('mousedown', ev => {
    if (ev.button !== 0) return;      // 只有左键才画笔；中键留给平移
    if (!ED.editing || ED.view === 'result') return;
    if (!ED.plan) return;
    if (!ED.eraser && ED.curSlotUid < 0) {
      flashHint('先在这一层的颜色列表里点一个颜色，再涂');
      return;
    }
    ED.painting = true;
    beginStroke();
    const [x, y] = canvasPos(cv, ev);
    brushStamp(x, y);
    paintStrokeDot(x, y);        // 即时反馈走 overlay，不动主画布
    ev.preventDefault();
  });
  // Alt + 右键拖动 = 调笔刷大小（不必去右栏找滑块）
  cv.addEventListener('contextmenu', ev => { if (ED.editing) ev.preventDefault(); });
  cv.addEventListener('mousedown', ev => {
    if (ev.button !== 2 || !ev.altKey || !ED.editing) return;
    ev.preventDefault();
    ED.resizing = { x: ev.clientX, size: ED.brush };
  });
  window.addEventListener('mousemove', ev => {
    if (!ED.resizing) return;
    const d = Math.round((ev.clientX - ED.resizing.x) / 3);
    ED.brush = Math.max(1, Math.min(60, ED.resizing.size + d));
    if ($('brushSize')) $('brushSize').value = ED.brush;
    if ($('brushSizeVal')) $('brushSizeVal').textContent = ED.brush;
    drawCursor(...(() => { const [x, y] = canvasPos(cv, ev); return [x, y]; })());
  });
  window.addEventListener('mouseup', ev => {
    if (ev.button === 2) ED.resizing = null;
  });

  cv.addEventListener('mousemove', ev => {
    if (!ED.editing) return;
    const [x, y] = canvasPos(cv, ev);
    if (ED.painting) {
      brushStamp(x, y);
      paintStrokeDot(x, y);
    } else {
      drawCursor(x, y);
    }
  });
  cv.addEventListener('mouseleave', () => { if (!ED.painting) drawCursor(undefined); });
  window.addEventListener('mouseup', () => {
    if (!ED.painting) return;
    ED.painting = false;
    endStroke();
    clearOverlay();
    afterEdit();                 // 松手才真正重渲一次
  });
  // 滚轮缩放
  cv.addEventListener('wheel', ev => {
    if (!ED.editing) return;
    ev.preventDefault();
    const cur = ED.zoom > 0 ? ED.zoom : 100;
    ED.zoom = Math.max(25, Math.min(800, Math.round(cur * (ev.deltaY < 0 ? 1.15 : 1 / 1.15))));
    fitCanvas();
  }, { passive: false });
}

// 改抖动方法时，如果正停在「投影色」视图，自动切到「抖动色」——
// 投影阶段本来就只显示槽颜色、不走方法分支，停在那儿改方法会以为没生效
function ensureFinalView() {
  if (ED.view === 'projection') {
    ED.view = 'final';
    updateViewButtons();
  }
}

function afterEdit() {
  refreshLockState();
  renderBands();
  renderSlots(ED.curBand);
  paintCanvas();
}

let hintTimer = null;
function flashHint(msg) {
  if ($('stageHint')) {
    $('stageHint').textContent = msg;
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => paintCanvas(), 2400);
  }
}

// ---------- 颜色槽操作 ----------
function nearestSlotUid(i, color, excludeUid) {
  const mine = (ED.slots[i] || []).filter(sv => sv.uid !== excludeUid);
  if (!mine.length) return -1;
  const lab = rgbToLab(color);
  let best = mine[0], bd = Infinity;
  for (const sv of mine) {
    const c = slotColor(sv) || [0, 0, 0];
    const l2 = rgbToLab(c);
    const d = (lab[0] - l2[0]) ** 2 + (lab[1] - l2[1]) ** 2 + (lab[2] - l2[2]) ** 2;
    if (d < bd) { bd = d; best = sv; }
  }
  best.used = true;
  return best.uid;
}

function deleteSlot(i, j) {
  const slots = ED.slots[i];
  if (!slots || !slots[j]) return;
  const sv = slots.splice(j, 1)[0];
  const to = nearestSlotUid(i, slotColor(sv) || [0, 0, 0], sv.uid);
  delete ED.labelSlots[sv.uid];
  if (ED.labels) {
    const op = { px: [], old: [], add: [] };
    for (let p = 0; p < ED.labels.length; p++) {
      if (ED.labels[p] === sv.uid) { op.px.push(p); op.old.push(sv.uid); op.add.push(to); ED.labels[p] = to; }
    }
    if (op.px.length) { UNDO.stack.push(op); UNDO.redo.length = 0; }
  }
  if (!slots.length) delete ED.slots[i];
  if (ED.curSlotUid === sv.uid) ED.curSlotUid = ED.lastSlotOfBand[i] || -1;
  bumpRev();
  updateUndoButtons();
}

function clearBandSlots(i) {
  for (const sv of (ED.slots[i] || [])) delete ED.labelSlots[sv.uid];
  const uids = new Set((ED.slots[i] || []).map(sv => sv.uid));
  if (ED.labels) {
    const op = { px: [], old: [], add: [] };
    for (let p = 0; p < ED.labels.length; p++) {
      if (uids.has(ED.labels[p])) { op.px.push(p); op.old.push(ED.labels[p]); op.add.push(-1); ED.labels[p] = -1; }
    }
    if (op.px.length) { UNDO.stack.push(op); UNDO.redo.length = 0; }
  }
  delete ED.slots[i];
  ED.curSlotUid = -1;
  bumpRev();
  updateUndoButtons();
}

function selectSlot(uid) {
  ED.curSlotUid = uid;
  ED.lastSlotOfBand[ED.curBand] = uid;
  ED.highlightUid = uid;        // 选中即高亮该颜色的区域
  const sv = ED.labelSlots[uid];
  const c = sv ? slotColor(sv) : null;
  if ($('brushSwatch')) $('brushSwatch').style.background = c ? rgbCss(c) : 'transparent';
  if ($('brushLabel')) {
    $('brushLabel').textContent = sv
      ? (typeof sv.color === 'number' ? '板内色 #' + sv.color + ' ' + hex2(c || [0, 0, 0])
                                      : '自定义 ' + hex2(c || [0, 0, 0]))
      : '未选颜色（点下面的色块）';
  }
  renderSlots(ED.curBand);
  paintCanvas();          // 选中即高亮：画布要立刻反映，不能等下次涂完才更新
}

// ---------- 图层切换 ----------
function selectBand(idx) {
  if (!ED.plan) return;
  ED.curBand = Math.max(0, Math.min(ED.plan.nBands - 1, idx));
  const keep = ED.lastSlotOfBand[ED.curBand];
  let uid = (keep !== undefined && ED.labelSlots[keep]) ? keep : -1;
  // 这层还没用过画笔：优先「保持当前画笔颜色」——在本层找颜色最接近的槽，
  // 而不是粗暴跳到第一个（那样每换一层都得重新选色）
  if (uid < 0 && ED.slots[ED.curBand] && ED.slots[ED.curBand].length) {
    const prev = slotColor(ED.labelSlots[ED.curSlotUid]);
    let best = ED.slots[ED.curBand][0];
    if (prev) {
      let bd = Infinity;
      for (const sv of ED.slots[ED.curBand]) {
        const c = slotColor(sv);
        if (!c) continue;
        const d = (c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2 + (c[2] - prev[2]) ** 2;
        if (d < bd) { bd = d; best = sv; }
      }
    }
    uid = best.uid;
  }
  ED.curSlotUid = uid;
  ED.highlightUid = uid;
  const sv = ED.labelSlots[ED.curSlotUid];
  if ($('brushSwatch')) $('brushSwatch').style.background = sv ? rgbCss(slotColor(sv)) : 'transparent';
  if ($('brushLabel')) {
    $('brushLabel').textContent = sv ? '板内色 #' + sv.color : '未选颜色（点下面的色块）';
  }
  renderBands(); renderSlots(ED.curBand); paintCanvas();
}

// ---------- 色阶 ----------
function renderGradeHist() {
  const cv = $('gradeHist');
  if (!cv) return;
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#0d0d18';
  g.fillRect(0, 0, W, H);
  const src = ED.src || (srcData ? resizeData(srcData, ...computeTarget(srcData.width, srcData.height)) : null);
  if (!src) return;
  const hist = new Float32Array(128);
  for (let i = 0; i < src.width * src.height; i++) {
    if (src.data[i * 4 + 3] === 0) continue;
    const L = rgbToLab([src.data[i * 4], src.data[i * 4 + 1], src.data[i * 4 + 2]])[0] / 100;
    hist[Math.min(127, (L * 128) | 0)]++;
  }
  let hmax = 0;
  for (let i = 0; i < 128; i++) if (hist[i] > hmax) hmax = hist[i];
  g.fillStyle = '#3f5e9c';
  for (let x = 0; x < W; x++) {
    const v = hmax ? hist[Math.min(127, (x / W * 128) | 0)] / hmax : 0;
    const hh = Math.max(1, v * (H - 6));
    g.fillRect(x, H - hh, 1, hh);
  }
  // 每层的阈值线（分界）：让「分层」这件事在色阶面板里直接看得见
  if (ED.plan && ED.plan.edges) {
    for (let k = 0; k < ED.plan.edges.length; k++) {
      const inner = (k > 0 && k < ED.plan.edges.length - 1);
      g.fillStyle = inner ? 'rgba(120,200,255,.85)' : 'rgba(200,200,220,.55)';
      g.fillRect(Math.round(ED.plan.edges[k] * W), 0, inner ? 1 : 1, H);
    }
  }
  // 黑/白场标记
  const bBlack = +$('gradeBlack').value, bWhite = +$('gradeWhite').value;
  g.fillStyle = 'rgba(255,213,79,.95)';
  g.fillRect(bBlack * W, 0, 2, H);
  g.fillRect(bWhite * W - 2, 0, 2, H);
  g.fillStyle = 'rgba(0,0,0,.45)';
  g.fillRect(0, 0, bBlack * W, H);
  g.fillRect(bWhite * W, 0, W - bWhite * W, H);
}

// 直方图上直接拖：靠近黄线拖黑/白场，靠近蓝线拖该层分界
function gradeHistDrag(startEv) {
  const cv = $('gradeHist');
  if (!cv) return;
  if (layersLocked()) { flashHint('色阶与分层已锁定：先「解锁并清除上色」'); return; }
  const r = cv.getBoundingClientRect();
  const x0 = (startEv.clientX - r.left) / r.width;
  const cands = [
    { kind: 'black', v: +$('gradeBlack').value },
    { kind: 'white', v: +$('gradeWhite').value },
  ];
  if (ED.plan && ED.plan.edges) {
    for (let k = 1; k < ED.plan.nBands; k++) cands.push({ kind: 'edge', k, v: ED.plan.edges[k] });
  }
  let best = null, bd = Infinity;
  for (const c of cands) {
    const d = Math.abs(c.v - x0);
    if (d < bd) { bd = d; best = c; }
  }
  // 命中热区：画布宽度的 2%
  if (!best || bd > 0.02) return;
  cv.classList.add('dragging');
  const move = ev => {
    const v = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    if (best.kind === 'black') {
      const w = +$('gradeWhite').value;
      $('gradeBlack').value = Math.min(v, w - 0.02);
      $('gradeBlackVal').textContent = (+$('gradeBlack').value).toFixed(2);
      updateGradeSummary();
    } else if (best.kind === 'white') {
      const b = +$('gradeBlack').value;
      $('gradeWhite').value = Math.max(v, b + 0.02);
      $('gradeWhiteVal').textContent = (+$('gradeWhite').value).toFixed(2);
      updateGradeSummary();
    } else {
      const lo = ED.plan.edges[best.k - 1] + 0.002;
      const hi = ED.plan.edges[best.k + 1] - 0.002;
      ED.plan.edges[best.k] = Math.max(lo, Math.min(hi, v));
      // 拖动中只移动这条线，不重建分桶 —— 重建会连带重渲整图（200ms+），
      // 线就会卡住不动。松手后统一重算（见 up）。
    }
    renderGradeHist();      // 轻量重绘：线跟着鼠标走
  };
  const up = () => {
    cv.classList.remove('dragging');
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
    if (best.kind === 'edge') {
      rebucketNow();                            // 分界线：松手才重分桶 + 重渲
    } else {
      applyGrade();                             // 黑/白场：松手重新分析
    }
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}

function applyGrade() {
  if (layersLocked()) { flashHint('色阶已锁定：先「解锁并清除上色」'); return; }
  ED.zoom = 0;
  // 色阶改变的是 luma 本身，旧的手调分界是「相对旧 luma」的位置，沿用会错位 ——
  // 表现出来就是「调了色阶画面几乎没变」。所以这里重新按分层依据生成分界。
  if (ED.editing) analyzeNow(false);
  else { invalidatePlan(); paintCanvas(); }
  updateGradeSummary();
  renderGradeHist();
}

function updateGradeSummary() {
  if (!$('gradeSummary')) return;
  const b = +($('gradeBlack') || {}).value || 0;
  const w = +($('gradeWhite') || {}).value || 1;
  const g = +($('gradeGamma') || {}).value || 1;
  $('gradeSummary').textContent = `黑场 ${b.toFixed(2)} / 白场 ${w.toFixed(2)} / 中间调 ${g.toFixed(2)}`;
}

// ---------- 分界条 ----------
function renderEdgeBar() {
  const cv = $('edgeBar');
  if (!cv) return;
  if (!ED.plan) { if ($('edgeVals')) $('edgeVals').textContent = ''; return; }
  const plan = ED.plan, W = cv.width, H = cv.height;
  const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  g.fillStyle = '#0d0d18';
  g.fillRect(0, 0, W, H);
  const hist = new Float32Array(128);
  for (let i = 0; i < plan.npix; i++) {
    if (!plan.opaque[i]) continue;
    hist[Math.min(127, (plan.luma[i] * 128) | 0)]++;
  }
  let hmax = 0;
  for (let i = 0; i < 128; i++) if (hist[i] > hmax) hmax = hist[i];
  g.fillStyle = '#33507f';
  for (let x = 0; x < W; x++) {
    const v = hmax ? hist[Math.min(127, (x / W * 128) | 0)] / hmax : 0;
    const hh = Math.max(1, v * (H - 10));
    g.fillRect(x, H - hh, 1, hh);
  }
  // 层带底色
  for (let k = 0; k < plan.nBands; k++) {
    const x0 = plan.edges[k] * W, x1 = plan.edges[k + 1] * W;
    g.fillStyle = (k === ED.curBand) ? 'rgba(106,159,255,.20)' : (k % 2 ? 'rgba(255,255,255,.03)' : 'transparent');
    g.fillRect(x0, 0, x1 - x0, H - 10);
  }
  // 分界：1px 线（视觉细），热区在 mousedown 里用 ±5px 判定
  for (let k = 0; k <= plan.nBands; k++) {
    const x = plan.edges[k] * W;
    const inner = (k > 0 && k < plan.nBands);
    g.fillStyle = inner ? 'rgba(255,213,79,.95)' : 'rgba(150,150,180,.7)';
    g.fillRect(Math.round(x), 0, inner ? 1 : 1, H - 10);
  }
  if ($('edgeVals')) {
    $('edgeVals').textContent = '分界：' + Array.from(plan.edges)
      .map(v => (v * 100).toFixed(0)).join(' / ') + '（拖动黄色细线）';
  }
}

function edgeBarDrag(startEv) {
  if (!ED.plan) return;
  if (layersLocked()) { flashHint('分层已锁定：改分界会重排图层。先「解锁并清除上色」'); return; }
  const cv = $('edgeBar');
  const r = cv.getBoundingClientRect();
  const v0 = (startEv.clientX - r.left) / r.width;
  // 热区 ±5px（画布坐标），命中最近的内部把手
  let best = -1, bd = Infinity;
  const hot = 5 / r.width;
  for (let k = 1; k < ED.plan.nBands; k++) {
    const d = Math.abs(ED.plan.edges[k] - v0);
    if (d < bd) { bd = d; best = k; }
  }
  if (best < 0 || bd > hot) return;
  const move = ev => {
    const v = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    const lo = ED.plan.edges[best - 1] + 0.002;
    const hi = ED.plan.edges[best + 1] - 0.002;
    ED.plan.edges[best] = Math.max(lo, Math.min(hi, v));
    renderEdgeBar();
    rebucketDebounced();
  };
  const up = () => {
    document.removeEventListener('mousemove', move);
    document.removeEventListener('mouseup', up);
  };
  document.addEventListener('mousemove', move);
  document.addEventListener('mouseup', up);
}

// ---------- 编辑器模式 ----------
function enterEditorMode() {
  if (!srcData) { alert('请先选择图片'); return; }
  if (paletteTooSmall($('method').value)) return;
  const wasMethod = $('method').value;
  if (wasMethod !== 'projection') {
    $('method').value = 'projection';
    updateMethodParams();
  }
  if (!analyzeNow(true)) return;
  ED.editing = true;
  document.body.classList.add('editing');
  $('enterEditor').classList.add('hidden');
  $('exitEditor').classList.remove('hidden');
  ED.view = 'projection';
  ED.zoom = 0;
  ED.layerOnly = false;
  if ($('layerOnly')) $('layerOnly').checked = false;
  // 颜色槽默认就绪：**每一层**都按自动投影色生成槽（以前只给当前层生成，
  // 用户切到别的层会发现没颜色可涂）
  for (let b = 0; b < ED.plan.nBands; b++) {
    if (!ED.slots[b] || !ED.slots[b].length) importAutoSlots(b, true);
  }
  if (ED.slots[ED.curBand] && ED.slots[ED.curBand].length) {
    selectSlot(ED.slots[ED.curBand][0].uid);
  }
  // 进编辑器默认不高亮：高亮会把非该颜色的像素压到 55%，
  // 一进来就整图变暗，观感像「打开编辑器把成品弄坏了」
  ED.highlightUid = -1;
  if (ED.labels) ED.forceBand = new Int32Array(ED.plan.npix).fill(-1);
  refreshLockState();
  renderBands(); renderSlots(ED.curBand); refreshAll(); updateUndoButtons();
  updateGradeSummary(); renderGradeHist();
  // 色阶是分层的入口，进来就先摆出来（可关闭）
  $('gradePanel').classList.remove('hidden');
}

function exitEditorMode() {
  ED.editing = false;
  document.body.classList.remove('editing');
  updateMethodParams();          // 解除方法下拉的 disabled，否则退出后换不了方法
  $('enterEditor').classList.remove('hidden');
  $('exitEditor').classList.add('hidden');
  ED.view = 'result';
  ED.zoom = 0;
  refreshLockState();
  $('gradePanel').classList.add('hidden');
  $('edgePanel').classList.add('hidden');
  refreshAll();
}

// ============================================================
// 事件绑定与主流程
// ============================================================
const SLOT_FILTERS = ['FS', 'Jarvis', 'Stucki'];

let debounceTimer = null;
function scheduleProcess() {
  if (!srcData) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(doProcess, 250);
}

// ---------- 矩阵下拉 ----------
function matrixOptions(size) {
  const n = +size || 4;
  return MATRIX_NAMES.map(([key, label]) => [key + n, n + '×' + n + ' ' + label]);
}
function fillMatrixSelect(sel, size) {
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = matrixOptions(size)
    .map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
}

function updateMethodParams() {
  const m = $('method').value;
  $('orderedParams').classList.toggle('hidden', m !== 'ordered');
  $('edParams').classList.toggle('hidden', m !== 'error_diffusion');
  $('baryParams').classList.toggle('hidden', m !== 'barycentric');
  $('projParams').classList.toggle('hidden', m !== 'projection');
  $('method').disabled = ED.editing;
}

// ---------- 参数变化 ----------
function onParamChange() {
  // 锁只拦「会重建 plan」的参数（层数/色数/尺寸/分层依据/空间）。
  // 非分层参数（矩阵、滤波核等）照常放行。UI 上这些控件已经 disabled，
  // 这里再拦一次，防脚本/快捷键绕过。
  const opts = gatherOpts();
  const [tw, th] = srcData ? computeTarget(srcData.width, srcData.height) : [0, 0];
  const willRebuild = !ED.plan || ED.plan.sig !== optsSig(opts)
    || !ED.src || ED.src.width !== tw || ED.src.height !== th;
  if (willRebuild && layersLocked()) {
    flashHint('分层与尺寸已锁定：改它们会重排图层。先「解锁并清除上色」');
    refreshLockState();
    return;
  }
  invalidatePlan();
  if (ED.editing) {
    // 编辑器里改参数：立刻重建 plan，保持画面与右栏一致。
    // 注意不要动 ED.view —— 这里曾经把「成品」强行切成「投影色」，
    // 投影色是未量化的中间结果，用户会以为成品被改坏了（实测 ΔE8 从 6.56 掉到 0.96）。
    if (!analyzeNow(true)) return;
    refreshAll();
  } else {
    scheduleProcess();
  }
}

function bindRange(id, valId, digits) {
  const el = $(id);
  if (!el) return;
  el.addEventListener('input', () => {
    if ($(valId)) {
      const v = +el.value;
      $(valId).textContent = digits ? v.toFixed(digits) : el.value;
    }
    if (id.startsWith('grade') && ED.editing) renderGradeHist();
    if (id.startsWith('grade')) { updateGradeSummary(); return; }
  });
}

['strength:strengthVal', 'baryStrength:baryStrengthVal', 'alpha:alphaVal',
 'minWeight:minWeightVal', 'edgeGuard:edgeGuardVal', 'refine:refineVal',
 'mixSpan:mixSpanVal'].forEach(pair => {
  const [id, val] = pair.split(':');
  bindRange(id, val, id === 'minWeight' || id === 'alpha' ? 2 : 0);
});

['strength', 'baryStrength', 'alpha', 'minWeight', 'edgeGuard', 'refine', 'mixSpan']
  .forEach(id => onEl(id, 'change', scheduleProcess));
['matrix', 'colorSpace', 'filter', 'baryMatrix', 'projEdge', 'projSpace']
  .forEach(id => onEl(id, 'change', onParamChange));

// 数字输入框：input 事件防抖生效（边输入边看效果），change 立刻生效
let paramTimer = null;
function paramDebounced() {
  clearTimeout(paramTimer);
  paramTimer = setTimeout(onParamChange, 180);
}
['projBands', 'projColors', 'projMixColors'].forEach(id => {
  onEl(id, 'input', paramDebounced);
  onEl(id, 'change', onParamChange);
});
onEl('projMatrixSize', 'change', () => { fillMatrixSelect($('projMatrix'), $('projMatrixSize').value); onParamChange(); });
onEl('projMatrix', 'change', onParamChange);

onEl('method', 'change', () => {
  updateMethodParams();
  if (ED.editing) { exitEditorMode(); }
  scheduleProcess();
});

// ---------- 上传 ----------
onEl('fileInput', 'change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
onEl('dropZone', 'click', () => $('fileInput').click());
onEl('dropZone', 'dragover', e => e.preventDefault());
onEl('dropZone', 'drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) { alert('请选择图片文件'); return; }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    let w = img.width, h = img.height;
    const MAX_PIXELS = 4000000;
    if (w * h > MAX_PIXELS) {
      const s = Math.sqrt(MAX_PIXELS / (w * h));
      w = Math.max(1, Math.round(w * s));
      h = Math.max(1, Math.round(h * s));
    }
    const cv = $('srcCanvas');
    cv.width = w; cv.height = h;
    cv.getContext('2d').drawImage(img, 0, 0, w, h);
    srcData = cv.getContext('2d').getImageData(0, 0, w, h);
    cv.classList.remove('hidden');
    if ($('imgMeta')) $('imgMeta').textContent = file.name + ' · ' + w + '×' + h + ' px';
    URL.revokeObjectURL(url);
    if (ED.editing) { invalidatePlan(); analyzeNow(true); refreshAll(); }
    else { invalidatePlan(); scheduleProcess(); }
  };
  img.onerror = () => alert('无法读取该图片');
  img.src = url;
}

// ---------- 尺寸 ----------
function clearWH() { $('widthInput').value = ''; $('heightInput').value = ''; }
onEl('compression', 'input', () => {
  clearWH();
  $('compNum').value = $('compression').value;
  onParamChange();
});
onEl('compNum', 'input', () => {
  clearWH();
  let v = +$('compNum').value || 100;
  v = Math.min(100, Math.max(1, v));
  $('compression').value = v;
  onParamChange();
});
function syncCompFromWH() {
  if (!srcData) return;
  const ow = srcData.width, oh = srcData.height;
  const wTxt = $('widthInput').value.trim(), hTxt = $('heightInput').value.trim();
  let pct = null;
  if (wTxt && hTxt) pct = Math.min(+wTxt / ow, +hTxt / oh) * 100;
  else if (wTxt) pct = +wTxt / ow * 100;
  else if (hTxt) pct = +hTxt / oh * 100;
  if (pct != null) {
    pct = Math.min(100, Math.max(1, Math.round(pct)));
    $('compression').value = pct; $('compNum').value = pct;
  }
}
['widthInput', 'heightInput'].forEach(id => {
  onEl(id, 'input', () => { syncCompFromWH(); onParamChange(); });
});

// ---------- 调色板 ----------
function renderPalette() {
  const box = $('paletteSwatches');
  box.innerHTML = '';
  for (let i = 0; i < palette.length; i++) {
    const c = palette[i];
    const sw = document.createElement('div');
    sw.className = 'swatch' + (D.paid[i] ? ' paid' : '') + (selected.has(i) ? '' : ' off');
    sw.style.background = rgbCss(c);
    sw.title = (D.paid[i] ? '[付费] ' : '') + (D.names[i] || '') + ' (' + c.join(',') + ')';
    sw.addEventListener('click', () => {
      if (selected.has(i)) selected.delete(i); else selected.add(i);
      renderPalette();
      onParamChange();
      if (ED.editing) { renderSlots(ED.curBand); }
    });
    box.appendChild(sw);
  }
  if ($('palCount')) $('palCount').textContent = selected.size;
}
function setAll(cond) {
  selected = new Set(palette.map((_, i) => i).filter(cond));
  renderPalette();
  onParamChange();
  if (ED.editing) renderSlots(ED.curBand);
}
onEl('selAll', 'click', () => setAll(() => true));
onEl('selNone', 'click', () => setAll(() => false));
onEl('selFree', 'click', () => setAll(i => !D.paid[i]));
onEl('selPaid', 'click', () => setAll(i => D.paid[i]));

// ---------- 视图页签 ----------
onEl('viewTabs', 'click', e => {
  const b = e.target.closest('button[data-view]');
  if (!b || b.disabled) return;
  ED.view = b.dataset.view;
  ED.zoom = 0;
  updateViewButtons();
  paintCanvas();
});

// ---------- 缩放 ----------
onEl('zoomFit', 'click', () => { ED.zoom = 0; ED.pan = { x: 0, y: 0 }; fitCanvas(); });
onEl('zoom100', 'click', () => { ED.zoom = 100; ED.pan = { x: 0, y: 0 }; fitCanvas(); });
window.addEventListener('resize', () => fitCanvas());

// ---------- 进入 / 退出编辑器 ----------
onEl('enterEditor', 'click', enterEditorMode);
onEl('exitEditor', 'click', exitEditorMode);

// ---------- 图层 ----------
onEl('bandList', 'click', e => {
  const it = e.target.closest('.band-item');
  if (!it) return;
  selectBand(+it.dataset.band);
});

// ---------- 颜色槽 ----------
// 按自动投影色生成当前层的颜色槽（进编辑器时默认就绪）
function importAutoSlots(i, silent) {
  if (!ED.plan || !ED.plan.bands[i]) return;
  // 该层没有投影色（像素为空 / 聚类退化）时没有颜色可导入，留空即可
  if (!ED.plan.bands[i].projRGB.length) return;
  ED.slots[i] = ED.plan.bands[i].projRGB.map(c => {
    // 保留原始浮点投影色：取整会让混合权重变一点点，于是「一进编辑器成品就变」。
    // 显示用的 hex 在渲染卡片时再取整。
    const sv = slotDefaults(c.slice());
    ED.labelSlots[sv.uid] = sv;
    return sv;
  });
  if (!silent && ED.slots[i].length) selectSlot(ED.slots[i][0].uid);
  bumpRev();
}

onEl('addSlot', 'click', () => {
  if (!ED.plan) return;
  const i = ED.curBand, band = ED.plan.bands[i];
  const seed = band.projRGB.length ? band.projRGB[0] : [128, 128, 128];
  const sv = slotDefaults(seed.map(v => Math.max(0, Math.min(255, Math.round(v)))));
  if (!ED.slots[i]) ED.slots[i] = [];
  ED.slots[i].push(sv);
  ED.labelSlots[sv.uid] = sv;
  selectSlot(sv.uid);
  bumpRev(); afterEdit();
});
onEl('importAuto', 'click', () => {
  if (!ED.plan) return;
  importAutoSlots(ED.curBand);
  afterEdit();
});

onEl('slotList', 'click', e => {
  const t = e.target.closest('[data-act]');
  if (!t) {
    // 点卡片任何空白处都算选中这个颜色（以前只有点色块才行）
    const card = e.target.closest('.slot-card');
    if (card && ED.plan) {
      const uid = +card.dataset.uid;
      if (ED.labelSlots[uid]) selectSlot(uid);
    }
    return;
  }
  if (!ED.plan) return;
  const act = t.dataset.act, i = +t.dataset.i, j = +t.dataset.j;
  if (!ED.plan.bands[i]) return;
  if (act === 'pick') { selectSlot(ED.slots[i][j].uid); return; }
  if (act === 'opts') { openSlotDialog(i, j); return; }
  if (act === 'del') { deleteSlot(i, j); afterEdit(); return; }
});

onEl('slotList', 'change', e => {
  const t = e.target.closest('[data-act]');
  if (!t || !ED.plan) return;
  const i = +t.dataset.i, j = +t.dataset.j;
  if (!ED.slots[i] || !ED.slots[i][j]) return;
  const act0 = t.dataset.act;
  if (!applySlotParam(t, ED.slots[i][j])) return;
  if (act0 === 'method') ensureFinalView();
  renderSlots(i);
  afterEdit();
});

// 弹窗内的参数改动
onEl('slotDialogBody', 'click', e => {
  const btn = e.target.closest('[data-setmethod]');
  if (!btn || !dialogSlot) return;
  const { i, j } = dialogSlot;
  if (!ED.slots[i] || !ED.slots[i][j]) return;
  ED.slots[i][j].method = btn.dataset.setmethod;
  ensureFinalView();
  bumpRev();
  openSlotDialog(i, j);
  renderSlots(i);
  afterEdit();
});
onEl('slotDialogBody', 'change', e => {
  const t = e.target.closest('[data-act]');
  if (!t || !dialogSlot) return;
  const { i, j } = dialogSlot;
  if (!ED.slots[i] || !ED.slots[i][j]) return;
  const sv = ED.slots[i][j];
  const before = sv.method;
  if (!applySlotParam(t, sv)) return;
  // 方法换了要重画弹窗（参数组不同）
  if (sv.method !== before) { ensureFinalView(); openSlotDialog(i, j); }
  renderSlots(i);
  afterEdit();
});
onEl('slotDialogBody', 'input', e => {
  const t = e.target.closest('input[type="range"]');
  if (!t || !dialogSlot) return;
  const { i, j } = dialogSlot;
  if (!ED.slots[i] || !ED.slots[i][j]) return;
  if (applySlotParam(t, ED.slots[i][j])) { renderSlots(i); afterEdit(); }
});
onEl('slotDialogClose', 'click', () => { $('slotDialog').classList.add('hidden'); dialogSlot = null; });
onEl('slotDialogOk', 'click', () => { $('slotDialog').classList.add('hidden'); dialogSlot = null; });
onEl('slotDialogDel', 'click', () => {
  if (!dialogSlot) return;
  deleteSlot(dialogSlot.i, dialogSlot.j);
  $('slotDialog').classList.add('hidden'); dialogSlot = null;
  afterEdit();
});

// 只看当前层
onEl('layerOnly', 'change', e => {
  ED.layerOnly = e.target.checked;
  ED.zoom = 0;
  paintCanvas();
});

// ---------- 画笔 ----------
onEl('brushSize', 'input', e => {
  ED.brush = +e.target.value;
  if ($('brushSizeVal')) $('brushSizeVal').textContent = ED.brush;
});
onEl('brushErase', 'change', e => {
  ED.eraser = e.target.checked;
  if ($('brushState')) $('brushState').textContent = ED.eraser ? '橡皮' : '画笔';
});
onEl('brushCrossBand', 'change', e => { ED.crossBand = e.target.checked; });
onEl('highlightOn', 'change', e => {
  ED.highlightOff = !e.target.checked;
  ED.highlightUid = ED.highlightOff ? -1 : ED.curSlotUid;
  paintCanvas();
});
onEl('brushClear', 'click', () => {
  if (!ED.labels) return;
  const op = { px: [], old: [], add: [] };
  for (let p = 0; p < ED.labels.length; p++) {
    if (ED.labels[p] >= 0 && ED.plan && ED.plan.bandOf[p] === ED.curBand) {
      op.px.push(p); op.old.push(ED.labels[p]); op.add.push(-1); ED.labels[p] = -1;
    }
  }
  if (op.px.length) { UNDO.stack.push(op); UNDO.redo.length = 0; bumpRev(); }
  updateUndoButtons();
  afterEdit();
  flashHint('已清除第 ' + (ED.curBand + 1) + ' 层的手绘');
});
onEl('undoBtn', 'click', undo);
onEl('redoBtn', 'click', redo);

window.addEventListener('keydown', e => {
  if (!ED.editing) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || ''));
  if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !typing) {
    e.preventDefault();
    if (e.shiftKey) redo(); else undo();
  } else if (e.key === 'b' && !typing) { $('brushErase').checked = false; ED.eraser = false; $('brushState').textContent = '画笔'; }
  else if (e.key === 'e' && !typing) { $('brushErase').checked = true; ED.eraser = true; $('brushState').textContent = '橡皮'; }
  else if (e.key === '[' && !typing) selectBand(ED.curBand - 1);
  else if (e.key === ']' && !typing) selectBand(ED.curBand + 1);
});

// ---------- 色阶面板 ----------
onEl('openGrade', 'click', () => {
  $('gradePanel').classList.toggle('hidden');
  if (!$('gradePanel').classList.contains('hidden')) renderGradeHist();
});
onEl('gradeClose', 'click', () => $('gradePanel').classList.add('hidden'));
['gradeBlack', 'gradeWhite', 'gradeGamma'].forEach(id => {
  onEl(id, 'input', () => {
    const v = +$(id).value;
    const label = id.replace('grade', '').toLowerCase();
    const ids = { black: 'gradeBlackVal', white: 'gradeWhiteVal', gamma: 'gradeGammaVal' };
    if ($(ids[label])) $(ids[label]).textContent = v.toFixed(2);
    renderGradeHist();
    updateGradeSummary();
  });
});
onEl('gradeApply', 'click', () => { applyGrade(); renderGradeHist(); });
onEl('gradeReset', 'click', () => {
  $('gradeBlack').value = 0; $('gradeWhite').value = 1; $('gradeGamma').value = 1;
  $('gradeBlackVal').textContent = '0.00';
  $('gradeWhiteVal').textContent = '1.00';
  $('gradeGammaVal').textContent = '1.00';
  updateGradeSummary();
  renderGradeHist();
  applyGrade();
});

// ---------- 阈值面板 ----------
onEl('openEdge', 'click', () => {
  $('edgePanel').classList.toggle('hidden');
  if (!$('edgePanel').classList.contains('hidden')) renderEdgeBar();
});
onEl('edgeClose', 'click', () => $('edgePanel').classList.add('hidden'));
onEl('unlockBtn', 'click', () => {
  const n = paintedCount();
  if (!confirm('解锁会清除全部 ' + n + ' px 手绘数据（颜色槽保留），确定？')) return;
  if (ED.labels) ED.labels.fill(-1);
  if (ED.forceBand) ED.forceBand.fill(-1);
  UNDO.stack.length = 0; UNDO.redo.length = 0;
  bumpRev();
  afterEdit();
  updateUndoButtons();
  flashHint('已解锁并清除手绘');
});

onEl('edgeBar', 'mousedown', ev => { edgeBarDrag(ev); ev.preventDefault(); });
onEl('gradeHist', 'mousedown', ev => { gradeHistDrag(ev); ev.preventDefault(); });

// ---------- 浮层可拖动 + 记住位置 ----------
function makePanelDraggable(panelId) {
  const panel = $(panelId);
  if (!panel) return;
  const bar = panel.querySelector('.fp-title');
  if (!bar) return;
  try {
    const saved = JSON.parse(localStorage.getItem('dither.pos.' + panelId) || 'null');
    if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
      panel.style.left = saved.left + 'px';
      panel.style.top = saved.top + 'px';
      panel.style.bottom = 'auto';
      panel.style.transform = 'none';
    }
  } catch (e) { /* 隐私模式忽略 */ }
  bar.addEventListener('mousedown', ev => {
    if (ev.target.classList && ev.target.classList.contains('fp-close')) return;
    const r = panel.getBoundingClientRect();
    const ox = ev.clientX - r.left, oy = ev.clientY - r.top;
    panel.style.transform = 'none';
    panel.style.bottom = 'auto';
    const move = e2 => {
      const left = Math.max(0, Math.min(window.innerWidth - 140, e2.clientX - ox));
      const top = Math.max(0, Math.min(window.innerHeight - 60, e2.clientY - oy));
      panel.style.left = left + 'px';
      panel.style.top = top + 'px';
    };
    const up = () => {
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
      try {
        localStorage.setItem('dither.pos.' + panelId, JSON.stringify({
          left: parseInt(panel.style.left, 10) || 0,
          top: parseInt(panel.style.top, 10) || 0,
        }));
      } catch (e) { /* 忽略 */ }
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
    ev.preventDefault();
  });
}

// ---------- 处理 / 下载 ----------
function doProcess() {
  if (!srcData) return;
  if (paletteTooSmall($('method').value)) return;
  $('processBtn').disabled = true;
  $('processBtn').textContent = '处理中…';
  setTimeout(() => {
    try {
      bumpRev();
      if ($('method').value === 'projection') {
        if (!analyzeNow(true)) return;
        if (!ED.editing) ED.view = 'result';
        refreshAll();
      } else {
        refreshAll();
      }
    } catch (err) {
      alert('处理失败: ' + err.message);
    } finally {
      $('processBtn').textContent = '处理';
      $('processBtn').disabled = false;
    }
  }, 20);
}
onEl('processBtn', 'click', doProcess);

onEl('downloadBtn', 'click', () => {
  const im = mainResult();
  if (!im) { alert('请先选择图片'); return; }
  const cv = document.createElement('canvas');
  cv.width = im.width; cv.height = im.height;
  cv.getContext('2d').putImageData(new ImageData(im.data, im.width, im.height), 0, 0);
  const a = document.createElement('a');
  a.href = cv.toDataURL('image/png');
  a.download = 'dithered.png';
  a.click();
});

// ---------- 初始化 ----------
fillMatrixSelect($('matrix'), 4);
fillMatrixSelect($('baryMatrix'), 4);
fillMatrixSelect($('projMatrix'), 4);
updateMethodParams();
renderPalette();
bindPaint();
bindPan();
updateUndoButtons();
updateGradeSummary();
updateViewButtons();
fitCanvas();
refreshLockState();
['gradePanel', 'edgePanel', 'slotDialog'].forEach(makePanelDraggable);

// ---------- 页签切换：图像抖动 / 汉字像素图 ----------
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const t = btn.dataset.tab;
    document.getElementById('panel-image').classList.toggle('hidden', t !== 'image');
    document.getElementById('panel-text').classList.toggle('hidden', t !== 'text');
    if (t === 'image') fitCanvas();
  });
});

// ---------- 资源版本自检 ----------
(function autoReloadOnUpdate() {
  const tag = document.querySelector('script[src*="dither.js"]');
  const cur = tag && (tag.src.match(/[?&]v=([^&"']+)/) || [])[1];
  if (!cur) return;
  let reloading = false;
  async function check() {
    if (reloading) return;
    try {
      const r = await fetch(location.pathname + '?_=' + Date.now(), { cache: 'no-store' });
      const html = await r.text();
      const m = html.match(/dither\.js\?v=([^"&']+)/);
      if (m && m[1] !== cur) { reloading = true; location.reload(); }
    } catch (e) { /* 离线忽略 */ }
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) check(); });
})();
