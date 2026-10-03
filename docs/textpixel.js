// ============================================================
// textpixel.js — 汉字转像素图
//
// 把文字按指定字体/字号渲染到 canvas，再逐像素把抗锯齿边缘二值化，
// 得到硬边的像素位图（适合做游戏内的像素字体 / UI 文字素材）。
//
// 颜色可选：前景色任意取，也能一键取 weplace 调色板里的色；背景可纯色或透明。
// ============================================================
(function () {
  'use strict';

  const $ = id => document.getElementById(id);

  // 常见中文字体（按平台回退；浏览器只认系统里装了的字体）
  const TEXT_FONTS = [
    ['"Microsoft YaHei","PingFang SC","Hiragino Sans GB",sans-serif', '微软雅黑 / 苹方（默认）'],
    ['SimHei,"Heiti SC","Noto Sans SC",sans-serif', '黑体'],
    ['SimSun,"Songti SC","Noto Serif SC",serif', '宋体'],
    ['KaiTi,"Kaiti SC",STKaiti,serif', '楷体'],
    ['FangSong,"STFangsong",serif', '仿宋'],
    ['"Microsoft JhengHei","PingFang TC",sans-serif', '微软正黑 / 繁体'],
    ['"Noto Sans SC",sans-serif', 'Noto Sans SC'],
    ['monospace', '等宽'],
    ['sans-serif', '系统无衬线'],
    ['serif', '系统衬线'],
  ];

  // ---------- 小工具 ----------
  function hexToRgb(hex) {
    const h = String(hex).replace('#', '');
    const v = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
    return [parseInt(v.slice(0, 2), 16) || 0,
            parseInt(v.slice(2, 4), 16) || 0,
            parseInt(v.slice(4, 6), 16) || 0];
  }
  function rgbToHex(c) {
    return '#' + c.slice(0, 3).map(v => Math.max(0, Math.min(255, v | 0))
      .toString(16).padStart(2, '0')).join('');
  }

  // ---------- 折行 ----------
  // manual：只按回车分；auto：按「每行最多 N 个全角字」拆（半角按实际宽度折算）
  function layoutLines(mctx, text, wrapMode, wrapChars) {
    const raw = String(text).split('\n');
    if (wrapMode !== 'auto') return raw;
    const full = Math.max(1, mctx.measureText('中').width);
    const limit = full * Math.max(1, wrapChars);
    const out = [];
    for (const line of raw) {
      if (!line) { out.push(''); continue; }
      let cur = '';
      for (const ch of line) {
        const t = cur + ch;
        if (cur && mctx.measureText(t).width > limit + 0.5) { out.push(cur); cur = ch; }
        else cur = t;
      }
      out.push(cur);
    }
    return out;
  }

  // ---------- 渲染 ----------
  // 关键：先在 size×SS 的高分辨率上渲染，再按 SS×SS 子块求平均覆盖率降到目标尺寸。
  // 低字号下普通字体笔画很细，抗锯齿后边缘覆盖度只有三四成，直接在小尺寸上按
  // 阈值二值化会把笔画砍断；超采样让覆盖率估得准，配合较低阈值就能保住笔画。
  function renderTextPixel(o) {
    const SS = Math.max(1, Math.min(8, o.supersample || 4));
    const hiSize = o.size * SS;

    const probe = document.createElement('canvas').getContext('2d');
    probe.font = hiSize + 'px ' + o.fontFamily;
    const lines = layoutLines(probe, o.text, o.wrapMode, o.wrapChars);
    if (!lines.length) return null;

    // 用字体实际 ascent/descent 定行高，避免汉字上下沿被裁掉
    const fm = probe.measureText('中');
    const ascent = fm.fontBoundingBoxAscent || hiSize * 0.88;
    const descent = fm.fontBoundingBoxDescent || hiSize * 0.18;
    const gapHi = o.gap * SS;
    const lineH = Math.ceil(ascent + descent) + gapHi;
    let maxW = 1;
    for (const l of lines) maxW = Math.max(maxW, probe.measureText(l).width);
    const W_hi = Math.max(1, Math.ceil(maxW));
    const H_hi = Math.max(1, lines.length * lineH - gapHi);

    // 高分辨率绘制（透明底，alpha 即笔画覆盖度；颜色在此步无关紧要）
    const hi = document.createElement('canvas');
    hi.width = W_hi; hi.height = H_hi;
    const hctx = hi.getContext('2d', { willReadFrequently: true });
    hctx.font = hiSize + 'px ' + o.fontFamily;
    hctx.textBaseline = 'alphabetic';
    hctx.fillStyle = '#000';
    lines.forEach((l, i) => hctx.fillText(l, 0, i * lineH + ascent));
    const hiData = hctx.getImageData(0, 0, W_hi, H_hi).data;

    // 降到目标尺寸：每个像素取 SS×SS 子块的平均覆盖率
    const W = Math.max(1, Math.ceil(W_hi / SS));
    const H = Math.max(1, Math.ceil(H_hi / SS));
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d');
    const out = ctx.createImageData(W, H);
    const d = out.data;
    const fg = hexToRgb(o.color);
    const bg = hexToRgb(o.bg);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sum = 0, cnt = 0;
        for (let dy = 0; dy < SS; dy++) {
          const sy = y * SS + dy;
          if (sy >= H_hi) break;
          const row = sy * W_hi;
          for (let dx = 0; dx < SS; dx++) {
            const sx = x * SS + dx;
            if (sx >= W_hi) break;
            sum += hiData[(row + sx) * 4 + 3];
            cnt++;
          }
        }
        const cov = cnt ? sum / cnt : 0;
        const i = (y * W + x) * 4;
        if (o.levels > 1) {
          // 多级：把覆盖率量化成 N 档，再据它在前景/背景之间插值。
          // 低字号下笔画只有一像素宽，覆盖度大量落在 30~70%，二值化会整条砍掉；
          // 用灰阶把"半覆盖"画出来，可读性远好于二值 —— 这才是超采样该有的用法。
          const N = o.levels;
          const lv = Math.round(cov / 255 * (N - 1)) / (N - 1);   // 0..1
          if (o.bgTransparent) {
            d[i] = fg[0]; d[i + 1] = fg[1]; d[i + 2] = fg[2];
            d[i + 3] = Math.round(lv * 255);
          } else {
            d[i] = Math.round(bg[0] + (fg[0] - bg[0]) * lv);
            d[i + 1] = Math.round(bg[1] + (fg[1] - bg[1]) * lv);
            d[i + 2] = Math.round(bg[2] + (fg[2] - bg[2]) * lv);
            d[i + 3] = 255;
          }
        } else if (cov >= o.threshold) {          // 二值硬边（传统像素风）
          d[i] = fg[0]; d[i + 1] = fg[1]; d[i + 2] = fg[2]; d[i + 3] = 255;
        } else if (o.bgTransparent) {
          d[i] = 0; d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 0;
        } else {
          d[i] = bg[0]; d[i + 1] = bg[1]; d[i + 2] = bg[2]; d[i + 3] = 255;
        }
      }
    }
    ctx.putImageData(out, 0, 0);
    return cv;
  }

  // 最近邻放大（导出用）
  function scaleCanvas(cv, n) {
    if (!cv || n <= 1) return cv;
    const out = document.createElement('canvas');
    out.width = cv.width * n; out.height = cv.height * n;
    const ctx = out.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cv, 0, 0, out.width, out.height);
    return out;
  }

  // ---------- UI ----------
  const fontSel = $('txtFont');
  if (!fontSel) return;                     // 面板不存在则不做任何事

  TEXT_FONTS.forEach(([val, label]) => {
    const o = document.createElement('option');
    o.value = val; o.textContent = label;
    fontSel.appendChild(o);
  });

  // 调色板色块：按分组显示，点一下取该色为前景色
  // 分组是为了低字号场景：黑白灰对比最强、最容易辨认；免费/付费分组则便于
  // 做一个"只用免费色"的方案。
  const PAL_GROUPS = {
    all: () => true,
    gray: i => i < 6,                                       // 调色板前 6 个是黑白灰
    free: i => !(DITHER_DATA.paid && DITHER_DATA.paid[i]),
    paid: i => !!(DITHER_DATA.paid && DITHER_DATA.paid[i]),
  };

  function renderTextPalette() {
    const box = $('txtPalette');
    if (!box) return;
    box.innerHTML = '';
    if (typeof DITHER_DATA === 'undefined' || !DITHER_DATA.palette) return;
    const key = $('txtPalGroup') ? $('txtPalGroup').value : 'all';
    const test = PAL_GROUPS[key] || PAL_GROUPS.all;
    const paid = DITHER_DATA.paid || [];
    const names = DITHER_DATA.names || [];
    DITHER_DATA.palette.forEach((c, i) => {
      if (!test(i)) return;
      const hex = rgbToHex(c);
      const d = document.createElement('div');
      d.className = 'pswatch' + (paid[i] ? ' paid' : '');
      d.style.background = hex;
      d.title = (paid[i] ? '[付费] ' : '') + (names[i] || '') + ' ' + hex;
      d.addEventListener('click', () => { $('txtColor').value = hex; onInput(); });
      box.appendChild(d);
    });
  }
  renderTextPalette();
  if ($('txtPalGroup')) {
    $('txtPalGroup').addEventListener('change', renderTextPalette);
  }

  function gather() {
    const size = Math.max(6, Math.min(64, +$('txtSize').value || 16));
    return {
      text: $('txtInput').value,
      fontFamily: fontSel.value || TEXT_FONTS[0][0],
      size: size,
      wrapMode: $('txtWrap').value,
      wrapChars: Math.max(1, Math.min(64, +$('txtWrapChars').value || 8)),
      gap: Math.max(0, Math.min(16, +$('txtGap').value || 0)),
      color: $('txtColor').value,
      bg: $('txtBg').value,
      bgTransparent: $('txtBgTransparent').checked,
      threshold: Math.max(1, Math.min(254, +$('txtThr').value || 128)),
      supersample: Math.max(1, Math.min(8, +($('txtSupersample') || {}).value || 4)),
      levels: Math.max(1, Math.min(16, +($('txtLevels') || {}).value || 1)),
    };
  }

  let lastCanvas = null;

  function render() {
    const o = gather();
    const cv = renderTextPixel(o);
    lastCanvas = cv;
    const pv = $('txtCanvas');
    if (!cv) {
      pv.width = 1; pv.height = 1;
      $('txtDims').textContent = '（没有内容）';
      return;
    }
    // 预览按整数倍放大到高度约 160px：小字看得清，放大用最近邻所以不糊
    const zoom = Math.max(1, Math.min(16, Math.round(160 / cv.height) || 1));
    const pvcv = scaleCanvas(cv, zoom);
    pv.width = pvcv.width; pv.height = pvcv.height;
    pv.getContext('2d').drawImage(pvcv, 0, 0);
    $('txtDims').textContent = cv.width + ' × ' + cv.height + ' px（预览 ' + zoom + '×）';
  }

  let timer = null;
  function onInput() {
    // 滑块/输入时立即反馈，文本输入稍作防抖
    clearTimeout(timer);
    timer = setTimeout(render, 60);
  }

  // 字号：滑块 ↔ 数字框 双向同步
  $('txtSizeRange').addEventListener('input', () => {
    $('txtSize').value = $('txtSizeRange').value;
    $('txtSizeVal').textContent = $('txtSizeRange').value;
    onInput();
  });
  $('txtSize').addEventListener('input', () => {
    let v = Math.max(6, Math.min(64, +$('txtSize').value || 16));
    $('txtSizeRange').value = v;
    $('txtSizeVal').textContent = v;
    onInput();
  });
  $('txtGap').addEventListener('input', () => {
    $('txtGapVal').textContent = $('txtGap').value; onInput();
  });
  $('txtThr').addEventListener('input', () => {
    $('txtThrVal').textContent = $('txtThr').value; onInput();
  });
  $('txtColor').addEventListener('input', () => {
    $('txtColorHex').textContent = $('txtColor').value; onInput();
  });
  ['txtInput', 'txtFont', 'txtWrap', 'txtWrapChars', 'txtBg', 'txtBgTransparent',
   'txtSupersample', 'txtLevels'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', onInput);
    el.addEventListener('change', onInput);
  });
  $('txtWrap').addEventListener('change', () => {
    $('wrapRow').classList.toggle('hidden', $('txtWrap').value !== 'auto');
    onInput();
  });

  // 载入自定义字体文件
  $('txtFontUpload').addEventListener('click', () => $('txtFontFile').click());
  $('txtFontFile').addEventListener('change', async e => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      const name = 'UserFont' + Date.now();
      const ff = new FontFace(name, 'url(' + URL.createObjectURL(f) + ')');
      await ff.load();
      document.fonts.add(ff);
      const o = document.createElement('option');
      o.value = '"' + name + '"';
      o.textContent = '自定义：' + f.name;
      fontSel.appendChild(o);
      fontSel.value = o.value;
      $('txtFontName').textContent = f.name;
      render();
    } catch (err) {
      alert('字体载入失败：' + (err && err.message ? err.message : err));
    }
  });

  // 下载
  $('txtDownload').addEventListener('click', () => {
    if (!lastCanvas) { alert('没有可下载的内容'); return; }
    const n = Math.max(1, Math.min(8, +$('txtScale').value || 1));
    const out = scaleCanvas(lastCanvas, n);
    const a = document.createElement('a');
    a.href = out.toDataURL('image/png');
    a.download = 'text_pixel_' + n + 'x.png';
    a.click();
  });

  // 透明模式下背景色选择器没有意义，灰掉避免困惑
  function syncBgEnable() {
    const t = $('txtBgTransparent').checked;
    $('txtBg').disabled = t;
    $('txtBg').style.opacity = t ? 0.35 : 1;
  }
  $('txtBgTransparent').addEventListener('change', syncBgEnable);

  // 多级灰度模式下阈值不参与运算，灰掉以免误解
  function syncThrEnable() {
    const el = $('txtLevels');
    const binary = !el || +el.value === 1;
    const thr = $('txtThr');
    if (thr) {
      thr.disabled = !binary;
      thr.style.opacity = binary ? 1 : 0.35;
    }
    const h = $('txtThrHint');
    if (h) {
      h.textContent = binary
        ? '仅「1 级（二值）」模式生效；小字号建议 50~80'
        : '当前是多级灰度模式，阈值不参与运算';
    }
  }
  if ($('txtLevels')) $('txtLevels').addEventListener('change', syncThrEnable);

  render();
  syncBgEnable();
  syncThrEnable();
})();
