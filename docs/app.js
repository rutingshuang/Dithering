// ============================================================
// app.js — 网页版 UI 逻辑
// ============================================================
const D = DITHER_DATA;
const palette = D.palette;
let srcData = null;   // 原图 ImageData
let resultData = null;

const $ = id => document.getElementById(id);

// ---------- 调色板选择 ----------
let selected = new Set(palette.map((_, i) => i));   // 初始全选
let baryData = DITHER_DATA;                          // 全选时用预计算数据
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

function renderPalette() {
  const box = $('paletteSwatches');
  box.innerHTML = '';
  for (let i = 0; i < palette.length; i++) {
    const c = palette[i];
    const sw = document.createElement('div');
    sw.className = 'swatch' + (D.paid[i] ? ' paid' : '') + (selected.has(i) ? '' : ' off');
    sw.style.background = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    sw.title = (D.paid[i] ? '[付费] ' : '') + (D.names[i] || '') + ' (' + c.join(',') + ')' + (selected.has(i) ? '' : '（未选中）');
    sw.addEventListener('click', () => {
      if (selected.has(i)) selected.delete(i); else selected.add(i);
      renderPalette();
      scheduleProcess();
    });
    box.appendChild(sw);
  }
  $('palCount').textContent = selected.size;
}
function setAll(cond) {
  selected = new Set(palette.map((_, i) => i).filter(cond));
  renderPalette();
  scheduleProcess();
}
$('selAll').addEventListener('click', () => setAll(() => true));
$('selNone').addEventListener('click', () => setAll(() => false));
$('selFree').addEventListener('click', () => setAll(i => !D.paid[i]));
$('selPaid').addEventListener('click', () => setAll(i => D.paid[i]));
renderPalette();

// ---------- 方法切换显示对应参数 ----------
$('method').addEventListener('change', () => { updateParams(); scheduleProcess(); });
function updateParams() {
  const m = $('method').value;
  $('orderedParams').classList.toggle('hidden', m !== 'ordered');
  $('edParams').classList.toggle('hidden', m !== 'error_diffusion');
  $('baryParams').classList.toggle('hidden', m !== 'barycentric');
}

// ---------- 滑块数值显示 + 自动处理（防抖） ----------
let debounceTimer = null;
function scheduleProcess() {
  if (!srcData) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(doProcess, 250);
}
function bindRange(id, valId) {
  $(id).addEventListener('input', () => { $(valId).textContent = $(id).value; scheduleProcess(); });
}
bindRange('strength', 'strengthVal');
bindRange('alpha', 'alphaVal');
bindRange('baryStrength', 'baryStrengthVal');
bindRange('minWeight', 'minWeightVal');
bindRange('edgeGuard', 'edgeGuardVal');

// 压缩：滑块 ↔ 数字输入 双向同步；调 % 时清空宽高（避免冲突）
function clearWH() { $('widthInput').value = ''; $('heightInput').value = ''; }
$('compression').addEventListener('input', () => {
  clearWH();
  $('compNum').value = $('compression').value;
  scheduleProcess();
});
$('compNum').addEventListener('input', () => {
  clearWH();
  let v = +$('compNum').value || 100;
  v = Math.min(100, Math.max(1, v));
  $('compression').value = v;
  scheduleProcess();
});
// 填宽/高 → 实时把压缩比例同步为该尺寸隐含的比例
function syncCompFromWH() {
  if (!srcData) return;
  const ow = srcData.width, oh = srcData.height;
  const wTxt = $('widthInput').value.trim();
  const hTxt = $('heightInput').value.trim();
  let pct = null;
  if (wTxt && hTxt) pct = Math.min(+wTxt / ow, +hTxt / oh) * 100;
  else if (wTxt) pct = +wTxt / ow * 100;
  else if (hTxt) pct = +hTxt / oh * 100;
  if (pct != null) {
    pct = Math.min(100, Math.max(1, Math.round(pct)));
    $('compression').value = pct;
    $('compNum').value = pct;
  }
}
['widthInput', 'heightInput'].forEach(id => {
  $(id).addEventListener('input', () => { syncCompFromWH(); scheduleProcess(); });
});
['colorSpace', 'matrix', 'filter', 'baryMatrix'].forEach(id => {
  $(id).addEventListener('change', scheduleProcess);
});

// 凸包投影：开/关 各存一套 权重截断/边缘保护 的值，切换时互换（与桌面版一致）
const BARY_CFG = {
  'on':  { minWeight: 0.1, edgeGuard: 10 },
  'off': { minWeight: 0,   edgeGuard: 0 },
};
let baryGamutState = false;
function onGamutToggle() {
  const newState = $('gamut').checked;
  const oldKey = baryGamutState ? 'on' : 'off';
  const newKey = newState ? 'on' : 'off';
  // 保存当前值到旧状态配置
  BARY_CFG[oldKey].minWeight = +$('minWeight').value;
  BARY_CFG[oldKey].edgeGuard = +$('edgeGuard').value;
  // 载入新状态配置
  $('minWeight').value = BARY_CFG[newKey].minWeight;
  $('edgeGuard').value = BARY_CFG[newKey].edgeGuard;
  $('minWeightVal').textContent = BARY_CFG[newKey].minWeight;
  $('edgeGuardVal').textContent = BARY_CFG[newKey].edgeGuard;
  baryGamutState = newState;
}
$('gamut').addEventListener('change', () => { onGamutToggle(); scheduleProcess(); });

// ---------- 图片上传 ----------
$('fileInput').addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
$('dropZone').addEventListener('click', () => $('fileInput').click());
$('dropZone').addEventListener('dragover', e => e.preventDefault());
$('dropZone').addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) { alert('请选择图片文件'); return; }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    // 限制处理尺寸上限，避免大图卡顿
    let w = img.width, h = img.height;
    const MAX = 1500;
    if (Math.max(w, h) > MAX) { const s = MAX / Math.max(w, h); w = Math.round(w * s); h = Math.round(h * s); }
    const cv = $('srcCanvas'); cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    srcData = ctx.getImageData(0, 0, w, h);
    URL.revokeObjectURL(url);
    scheduleProcess();
  };
  img.onerror = () => alert('无法读取该图片');
  img.src = url;
}

// ---------- 压缩 / 指定尺寸 ----------
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

// ---------- 处理 ----------
function gatherOpts() {
  const m = $('method').value;
  if (m === 'ordered') {
    return { matrix: $('matrix').value, strength: +$('strength').value, colorSpace: $('colorSpace').value };
  }
  if (m === 'error_diffusion') {
    return { filter: $('filter').value, alpha: +$('alpha').value };
  }
  return {
    matrix: $('baryMatrix').value,
    strength: +$('baryStrength').value,
    gamut: $('gamut').checked,
    minWeight: +$('minWeight').value,
    edgeGuard: +$('edgeGuard').value,
  };
}

function doProcess() {
  if (!srcData) return;
  $('processBtn').disabled = true;
  $('processBtn').textContent = '处理中…';
  setTimeout(() => {
    try {
      const opts = gatherOpts();
      const sel = getSelectedPalette();
      if ($('method').value === 'barycentric') ensureBaryData();
      // 压缩/指定尺寸：处理前缩放源图
      const [tw, th] = computeTarget(srcData.width, srcData.height);
      const src = resizeData(srcData, tw, th);
      const res = runDither(src, $('method').value, sel, baryData, opts);
      const cv = $('dstCanvas');
      cv.width = res.width; cv.height = res.height;
      cv.getContext('2d').putImageData(new ImageData(res.data, res.width, res.height), 0, 0);
      resultData = res;
    } catch (err) {
      alert('处理失败: ' + err.message);
    }
    $('processBtn').textContent = '处理';
    $('processBtn').disabled = false;
  }, 20);
}
$('processBtn').addEventListener('click', doProcess);

// ---------- 下载 ----------
$('downloadBtn').addEventListener('click', () => {
  const cv = $('dstCanvas');
  if (cv.width === 0) { alert('请先处理图片'); return; }
  const a = document.createElement('a');
  a.href = cv.toDataURL('image/png');
  a.download = 'dithered.png';
  a.click();
});

updateParams();
