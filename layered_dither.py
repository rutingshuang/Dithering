# -*- coding: utf-8 -*-
"""
layered_dither.py —— 分层投影混色抖动

把 Krita「按明暗分成 N 层 → 每层套一次色板化滤镜、各调各的参数 → 叠加合成」
这套手工工作流，做成可复现、可批量、可存预设的引擎。

原理
----
核心是两步分离：

  1. 投影：按明暗分层后，每层聚类出少量投影色（允许不在调色板内），
     层内像素投影到最接近的那个 —— 每层只有几种颜色，大片区域同色。
  2. 抖动：对每一个投影色单独求"用调色板色把它混出来"的权重，再按图案阈值分配
     —— 输出全部落在调色板内，且同一个投影色的抖动图案完全一致。

色差由第 1 步的投影质量（每层色数、层数与分层方式）决定，第 2 步只负责忠实还原。
关键在第 2 步面对的是**单个目标色**（单点问题），不需要所选颜色构成覆盖整个层内
分布的凸包 —— 固定调色板做不到那件事。

依赖 scipy（Delaunay / 凸包）。

用法
----
    import layered_dither as ld
    pal = ld.load_palette('palette.txt')
    img = Image.open('in.png')
    prof = ld.make_projection_profile(n_bands=24, colors_per_band=8)
    ld.render(img, pal, prof).save('out.png')

命令行：
    python layered_dither.py in.png out.png --bands 24 --colors 8
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ============================================================================
# 一、阈值图案库
# ============================================================================
# 所有图案最终都归一化成 [0,1) 的阈值矩阵，越大表示该位置越"晚"被点亮。

def bayer_matrix(n: int = 8) -> np.ndarray:
    """任意 2 的幂阶 Bayer 矩阵（递归构造）。"""
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("bayer 阶数必须是 >=2 的 2 的幂")
    m = np.array([[0]], dtype=np.int64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2],
                      [4 * m + 3, 4 * m + 1]])
    return m


def clustered_matrix(n: int = 8) -> np.ndarray:
    """点簇（clustered-dot）：从中心向外逐环生长，印刷网点观感。

    环用切比雪夫距离，所以生长出来是**方形**的。
    """
    c = (n - 1) / 2.0
    ys, xs = np.mgrid[0:n, 0:n]
    dy, dx = ys - c, xs - c
    r = np.maximum(np.abs(dy), np.abs(dx))          # 切比雪夫环
    ang = np.arctan2(dy, dx)
    order = np.lexsort((ang.ravel(), r.ravel()))    # 主键 r，次键角度
    m = np.empty(n * n, dtype=np.int64)
    m[order] = np.arange(n * n)
    return m.reshape(n, n)


def round_dot_matrix(n: int = 8) -> np.ndarray:
    """圆形点簇：环用欧氏距离，生长出来是**圆点**——真正的调幅网点（AM 屏）。

    和 clustered_matrix 的区别只在距离度量：切比雪夫给方形点、欧氏给圆形点。
    圆形点更接近印刷/漫画网点的观感，同阶数下色差也比方形点低约 13%。
    周期别超过 8：更大的点会跨越感知块，块内只剩寥寥几环，局部平均会严重偏离
    （实测 32 阶时 8x8 块内的色比偏差达 0.31）。
    """
    c = (n - 1) / 2.0
    ys, xs = np.mgrid[0:n, 0:n]
    dy, dx = ys - c, xs - c
    r = np.hypot(dy, dx)                            # 欧氏环 → 圆点
    ang = np.arctan2(dy, dx)
    order = np.lexsort((ang.ravel(), r.ravel()))
    m = np.empty(n * n, dtype=np.int64)
    m[order] = np.arange(n * n)
    return m.reshape(n, n)


def diagonal_matrix(n: int = 8) -> np.ndarray:
    """45° 斜线图案。"""
    ys, xs = np.mgrid[0:n, 0:n]
    return (xs + ys) % n


def spiral_matrix(n: int = 8) -> np.ndarray:
    """螺旋图案（角度为主键、半径为次键）。"""
    c = (n - 1) / 2.0
    ys, xs = np.mgrid[0:n, 0:n]
    r = np.hypot(ys - c, xs - c)
    ang = (np.arctan2(ys - c, xs - c) + 2 * np.pi) % (2 * np.pi)
    order = np.lexsort((r.ravel(), ang.ravel()))    # 主键 ang
    m = np.empty(n * n, dtype=np.int64)
    m[order] = np.arange(n * n)
    return m.reshape(n, n)


def checker_matrix(n: int = 8) -> np.ndarray:
    """棋盘（50% 硬阈值）。"""
    ys, xs = np.mgrid[0:n, 0:n]
    return ((ys + xs) % 2).astype(np.int64) * (n * n // 2)


def noise_matrix(n: int = 8, seed: int = 0) -> np.ndarray:
    """白噪声阈值（Krita 色板化滤镜 Random 模式的等价物）。"""
    rng = np.random.default_rng(int(seed))
    return (rng.random((n, n)) * (n * n - 1)).astype(np.int64)


def blue_noise_matrix(n: int = 8, seed: int = 0, candidates: int = 24) -> np.ndarray:
    """蓝噪声阈值矩阵（Mitchell 最佳候选法，环面格点上构造）。

    白噪声抖动在大面积平色区会出现明显的"点聚簇/空洞"，蓝噪声把这些点推开，
    观感细腻很多 —— 复刻参考图里那种均匀散点最接近的做法。
    逐个选点，每个新点从若干个随机候选里挑"离已有点集最远"的那个。
    """
    rng = np.random.default_rng(int(seed))
    total = n * n
    ys, xs = np.mgrid[0:n, 0:n]
    pts = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float64)
    d = np.abs(pts[:, None, :] - pts[None, :, :])
    d = np.minimum(d, n - d)                    # 环面：图案平铺后边界要相接
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    chosen = np.zeros(total, dtype=bool)
    order = np.empty(total, dtype=np.int64)
    chosen_idx = np.empty(total, dtype=np.int64)
    cur = int(rng.integers(total))
    for i in range(total):
        chosen[cur] = True
        order[cur] = i
        chosen_idx[i] = cur
        if i == total - 1:
            break
        k = min(int(candidates), total - i - 1)
        cand = rng.integers(0, total, size=k)
        cand = cand[~chosen[cand]]
        if cand.size == 0:
            cand = np.nonzero(~chosen)[0]
        dd = dist[np.ix_(cand, chosen_idx[:i + 1])].min(axis=1)
        cur = int(cand[int(np.argmax(dd))])
    return order.reshape(n, n)


def _rank01(m: np.ndarray) -> np.ndarray:
    """秩归一化（并列取平均秩）→ [0,1) 且分布均匀、均值 0.5。

    不要用 (m+0.5)/m.size 这种线性缩放：diagonal 的值域是 0..n-1、
    checker 只有两个取值，线性缩放会得到一个均值不对的阈值场，
    导致棋盘/斜线图案整体偏亮或偏暗。秩归一化对所有图案都成立。
    """
    flat = np.asarray(m, dtype=np.float64).ravel()
    n = flat.size
    order = np.argsort(flat, kind="stable")
    sv = flat[order]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:                       # 并列 → 取平均秩，保持图元对称
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ((ranks + 0.5) / n).reshape(np.asarray(m).shape)


_MATRIX_FUNCS = {
    "bayer": bayer_matrix,
    "clustered": clustered_matrix,
    "rounddot": round_dot_matrix,
    "diagonal": diagonal_matrix,
    "spiral": spiral_matrix,
    "checker": checker_matrix,
    "noise": noise_matrix,
    "bluenoise": blue_noise_matrix,
}

PATTERN_NAMES = tuple(_MATRIX_FUNCS.keys()) + ("pattern",)


def pattern_matrix(pattern: str, size: int = 8, seed: int = 0) -> np.ndarray:
    """取归一化到 [0,1) 的阈值矩阵。pattern='pattern' 时读入外部图片。"""
    key = pattern.lower()
    if key == "pattern":
        raise ValueError("外部图案请用 load_pattern_image() 直接传入数组")
    fn = _MATRIX_FUNCS.get(key)
    if fn is None:
        raise ValueError(f"未知图案: {pattern}（可用: {', '.join(PATTERN_NAMES)}）")
    size = max(2, int(size))
    if key == "bayer" and (size & (size - 1)) != 0:
        size = 1 << (size - 1).bit_length()          # 自动补到最近的 2 的幂
    if key in ("noise", "bluenoise"):
        m = fn(size, seed)
    else:
        m = fn(size)
    return _rank01(m)


def load_pattern_image(path: str, size: Optional[int] = None) -> np.ndarray:
    """读入外部图案（如 Krita 的 .pat / 任意图片），灰度化并归一化到 [0,1)。

    未指定 size 时保留原尺寸；指定则按最近邻缩放到 size×size 并保持平铺无缝。
    """
    im = Image.open(path).convert("L")
    if size:
        im = im.resize((int(size), int(size)), Image.NEAREST)
    arr = np.asarray(im, dtype=np.float64)
    if arr.max() > 0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0 - 1e-6)


# ============================================================================
# 二、色彩空间与最近色
# ============================================================================

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0-255) → CIELAB(D65)。输入 (...,3)，输出 (...,3)。"""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(xyz[..., 0] / Xn), f(xyz[..., 1] / Yn), f(xyz[..., 2] / Zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def compute_luma(arr: np.ndarray, space: str = "lab") -> np.ndarray:
    """算归一化明度场 (h,w) ∈ [0,1]。space: 'lab' | 'hsv' | 'gray'。"""
    s = space.lower()
    if s in ("gray", "luma", "rec709"):
        y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        return np.clip(y / 255.0, 0.0, 1.0)
    if s == "hsv":
        v = np.max(arr, axis=-1) / 255.0
        return np.clip(v, 0.0, 1.0)
    # 默认 Lab 的 L*（感知均匀，分层最贴近人手划分明暗的习惯）
    lab = _rgb_to_lab(arr.reshape(-1, 3)).reshape(arr.shape[0], arr.shape[1], 3)
    return np.clip(lab[..., 0] / 100.0, 0.0, 1.0)


# ============================================================================
# 三、Profile / Band 数据模型
# ============================================================================

@dataclass
class Band:
    """一个亮度层（= Krita 里的一个图层）的抖动参数。"""
    lo: float = 0.0                 # 该层明度下界 [0,1]
    hi: float = 1.0                 # 该层明度上界
    pattern: str = "bayer"          # bayer/clustered/diagonal/spiral/checker/noise/bluenoise
    size: int = 16                  # 图案阶数
    cell: int = 1                   # 每个图案单元放大到几像素（>1 → 图案变粗）
    seed: int = 0                   # 噪声图案种子
    flat: Optional[list] = None     # 强制平涂色（None = 不强制）
    invert: bool = False            # 反转图案阈值
    enabled: bool = True            # 关掉该层 = 该层不参与抖动
    colors: Optional[list] = None   # 手改的颜色槽（见 ColorSlot）；None = 自动 k-means 投影

    def wide(self) -> Tuple[float, float]:
        return (min(self.lo, self.hi), max(self.lo, self.hi))

    def slots(self) -> List["ColorSlot"]:
        """把 colors 字段解析成 ColorSlot 列表（容错：允许只写 color）。"""
        if not self.colors:
            return []
        out = []
        for c in self.colors:
            if isinstance(c, ColorSlot):
                out.append(c)
            elif isinstance(c, dict):
                out.append(ColorSlot(**{k: v for k, v in c.items()
                                        if k in ColorSlot.__dataclass_fields__}))
            else:                                   # 裸值：当调色板索引或 RGB
                out.append(ColorSlot(color=c))
        return out


@dataclass
class ColorSlot:
    """一层里的一个颜色槽：用哪个颜色、按哪种方式还原成调色板色。

    color: int  -> 调色板索引（63 色里的第几个）
           list -> 自定义投影色 [r,g,b]，允许不在调色板内，靠混合还原
    method:
      mix     用调色板色重心混合出 color（默认；color 若是调色板色则退化为平涂）
      direct  最近调色板色平涂
      ordered 加有序抖动偏移后取最近色（纹理最强）
      error   层内 1D 误差扩散（同一层的像素在空间上是分散的，只能按扫描序扩散，
              效果比整图误差扩散差）
    strength: ordered/error 的扰动强度，0-128，64 为默认
    """
    color: object = 0
    method: str = "mix"
    strength: int = 64
    note: str = ""



@dataclass
class Profile:
    """整套分层方案，可直接序列化成 JSON 存成预设。"""
    bands: List[Band] = field(default_factory=list)
    space: str = "lab"              # 分层用的明度空间：lab / hsv / gray
    match_space: str = "LAB"        # 投影聚类与调色板混合用的色彩空间：LAB / RGB
    gamma: float = 1.0              # 明度预处理：L' = L ** gamma
    black: float = 0.0              # 明度重映射（色阶黑场）
    white: float = 1.0              # 明度重映射（色阶白场）
    edge_mode: str = "linear"       # 分层依据：linear(按明度等分) | quantile(按图像分布等频) | hybrid
    edge_mix: float = 0.5           # hybrid 模式下向 linear 靠拢的比例
    colors_per_band: int = 8        # 每层投影色数量（越大投影越准、色差越低）
    mix_colors: int = 4             # 每个投影色用几个板内色混合：4=重心(无损) 3=三角形 2=线段
    adaptive_k: bool = True         # 按层内颜色散度分配色数（总预算 = 层数 x colors_per_band）
    kmeans_iters: int = 20          # k-means 迭代轮数
    kmeans_sample: int = 60000      # k-means 采样上限（加速用）
    name: str = "projection"

    # ---- 序列化 ----
    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str) -> "Profile":
        """读回预设；旧版本多出来的字段会被忽略，因此旧 JSON 仍可用。"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        band_fields = set(Band.__dataclass_fields__)
        prof_fields = set(Profile.__dataclass_fields__)
        bands = [Band(**{k: v for k, v in b.items() if k in band_fields})
                 for b in d.get("bands", [])]
        rest = {k: v for k, v in d.items() if k in prof_fields and k != "bands"}
        return Profile(bands=bands, **rest)


def _curve(points: Sequence[Tuple[float, float]], t: float) -> float:
    """分段线性求值：points 为 [(t, v), ...]，按 t 升序。"""
    if not points:
        return 0.0
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        if t0 <= t <= t1:
            if t1 - t0 < 1e-12:
                return v1
            k = (t - t0) / (t1 - t0)
            return v0 + (v1 - v0) * k
    return points[-1][1]


def make_profile(n_bands: int = 13,
                 pattern: str = "bayer",
                 size: int = 16,
                 cell: int = 1,
                 seed: int = 0,
                 space: str = "lab",
                 match_space: str = "LAB",
                 size_curve: Optional[Sequence[Tuple[float, float]]] = None,
                 cell_curve: Optional[Sequence[Tuple[float, float]]] = None,
                 pattern_segments: Optional[Sequence[Tuple[float, float, str]]] = None,
                 edge_mode: str = "linear",
                 edge_mix: float = 0.5,
                 colors_per_band: int = 8,
                 name: str = "custom") -> Profile:
    """按"参数随亮度变化的曲线"生成 n 层配置。

    把手工逐层调参换成几条曲线，每层参数 = 曲线在该层位置的取值（离散化），
    因此可复现、可批量。

    曲线的自变量 t 是**层序归一化位置**（0=最暗层，1=最亮层），不是绝对明度值 ——
    这样同一份预设换一张明暗分布完全不同的图，语义仍然成立
    （配合 edge_mode='quantile' 按图像明暗分布自适应分层）。

    size_curve / cell_curve: [(t, 值), ...]
    pattern_segments: [(t_lo, t_hi, 图案名), ...]，未覆盖区间用 pattern 参数
    """
    bands: List[Band] = []
    for i in range(n_bands):
        lo = i / n_bands
        hi = (i + 1) / n_bands
        t = (i + 0.5) / n_bands                    # 层序归一化位置
        sz = _curve(size_curve, t) if size_curve else size
        ce = _curve(cell_curve, t) if cell_curve else cell
        pat = pattern
        if pattern_segments:
            for t0, t1, p in pattern_segments:
                if t0 <= t < t1:
                    pat = p
                    break
        bands.append(Band(lo=lo, hi=hi, pattern=pat, size=int(round(sz)),
                          cell=max(1, int(round(ce))), seed=seed))
    return Profile(bands=bands, space=space, match_space=match_space,
                   edge_mode=edge_mode, edge_mix=edge_mix,
                   colors_per_band=colors_per_band, name=name)


def make_projection_profile(n_bands: int = 24,
                            colors_per_band: int = 8,
                            match_space: str = "LAB",
                            space: str = "lab",
                            edge_mode: str = "linear",
                            pattern: str = "bayer",
                            size: int = 16,
                            cell: int = 1) -> Profile:
    """投影混色的标准预设：每层投影到少量投影色，再逐个用调色板色混合出该投影色。

    "大面积同色 + 输出落盘内"同时成立：
      - 每层自由聚类 K 个投影色（可不在调色板内）→ 大片区域只有一种颜色
      - 对每个投影色单独用调色板色混合 → 输出 100% 落在调色板内
      - 同一个投影色的像素共用一组权重 → 该区域的抖动图案完全一致，整齐不乱

    色差由投影质量决定，主要旋钮是 colors_per_band 与 n_bands（实测 24 层 x 8 色
    区域平均 ΔE 0.95，32 层 x 12 色降到 0.86）。
    """
    return make_profile(
        n_bands=n_bands,
        space=space,
        match_space=match_space,
        edge_mode=edge_mode,
        pattern=pattern,
        size=size,
        cell=cell,
        colors_per_band=int(colors_per_band),
        name="投影混色（大面积同色·落盘内）",
    )


def make_reference_profile(n_bands: int = 24,
                           colors_per_band: int = 2,
                           pattern: str = "rounddot",
                           size: int = 8,
                           match_space: str = "LAB",
                           space: str = "lab",
                           edge_mode: str = "linear",
                           cell: int = 1) -> Profile:
    """参考图风格：每层只给极少量的色，明暗过渡交给圆点网点。

    对应 Krita 手工流程里「每层只挑两个色、用网点抖出中间调」的做法。分层让每层的
    明暗范围很窄，2 个色就够覆盖，所以色差损失有限（实测每层 2 色 ΔE 1.125，
    每层 12 色 0.797，绝对差 0.33），换来的是网点结构清晰、画面干净。

    注意网点尺寸要和题材匹配：pattern='rounddot' 且 size=8 适合大色块的图（动画、
    像素画）；细节密集的插画用 size=4 或改用 bayer，否则 8x8 的圆点会盖掉细节。
    """
    return make_profile(
        n_bands=n_bands,
        space=space,
        match_space=match_space,
        edge_mode=edge_mode,
        pattern=pattern,
        size=int(size),
        cell=cell,
        colors_per_band=int(colors_per_band),
        name="参考图风格（每层限色·圆点网点）",
    )


PRESETS = {
    "projection": make_projection_profile,
    "reference": make_reference_profile,
}


def resolve_edges(profile: Profile, luma: Optional[np.ndarray],
                  bands: Sequence[Band]) -> np.ndarray:
    """算出本次实际使用的分带边界。

    linear   : 用 profile 里写死的 lo/hi（绝对明度等分）
    quantile : 按当前图像的明度分位数等频分带 —— 每层像素数相同，
               于是"最暗层"永远是这张图最暗的那批像素，换图不用重调。
    hybrid   : 两者按 edge_mix 加权。
    """
    n_bands = len(bands)
    e = np.array([b.lo for b in bands] + [bands[-1].hi], dtype=np.float64)
    e = np.maximum.accumulate(e)
    if luma is None or profile.edge_mode == "linear" or n_bands < 1:
        return e

    q = np.linspace(0.0, 1.0, n_bands + 1)
    eq = np.quantile(luma, q)
    if profile.edge_mode == "hybrid":
        lin = np.linspace(float(luma.min()), float(luma.max()), n_bands + 1)
        m = float(np.clip(profile.edge_mix, 0.0, 1.0))
        eq = eq * (1.0 - m) + lin * m
    eq[0], eq[-1] = 0.0, 1.0
    # 大片纯色会让分位点重合，累积取大保证单调不减
    return np.maximum.accumulate(np.asarray(eq, dtype=np.float64))


# ============================================================================
# 四、引擎
# ============================================================================

_FIELD_CACHE: dict = {}


def threshold_field(pattern: str, size: int, cell: int, seed: int,
                    h: int, w: int, invert: bool = False,
                    custom: Optional[np.ndarray] = None) -> np.ndarray:
    """在整图坐标系上平铺阈值图案，返回 (h,w) ∈ [0,1)。

    关键：所有层共用同一个原点 (0,0)，因此跨层图案相位天然对齐，
    不会出现手工分层时层与层之间纹路错位的问题。
    """
    if custom is not None:
        base = custom
    else:
        key = (pattern, size, cell, seed, invert)
        base = _FIELD_CACHE.get(key)
        if base is None:
            base = pattern_matrix(pattern, size, seed)
            if invert:
                base = 1.0 - base - 1e-9
            if cell > 1:
                base = np.repeat(np.repeat(base, cell, axis=0), cell, axis=1)
            _FIELD_CACHE[key] = base
    mh, mw = base.shape
    reps = (int(math.ceil(h / mh)), int(math.ceil(w / mw)))
    return np.tile(base, reps)[:h, :w]


def _split_alpha(image: Image.Image):
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.float32)
        alpha = arr[:, :, 3].copy()
        return arr[:, :, :3].astype(np.float64), alpha
    return np.array(image.convert("RGB"), dtype=np.float64), None


def _merge_alpha(rgb: np.ndarray, alpha) -> Image.Image:
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    if alpha is None:
        return Image.fromarray(out, "RGB")
    a = np.where(alpha < 128, 0, 255).astype(np.uint8)
    out[alpha == 0] = 0
    return Image.fromarray(np.dstack([out, a]), "RGBA")


# ============================================================================
# 四·B、重心混合版分层抖动（低色差）
# ============================================================================
# 有序分层的选色是"逐像素最近色"，区域平均色必然偏离源色 —— 这就是色差的来源。
# 重心混合换了选色规则：把像素所在的调色板四面体顶点按重心权重分配，
# 区域内选色比例 == 权重 == 源色在凸包中的坐标，因此区域平均色误差趋近 0。
#
# 两者结合的方式很干净：重心混合用 tt ∈ [0,1) 决定"按什么比例在顶点间分配"。
# 把 tt 换成由分层图案场压缩而来：
#       tt = 0.5 + (图案阈值 - 0.5) * s        s = clip(strength / 128, 0, 1)
# - s = 1：tt 在区域内均匀铺满 [0,1)，选色比例精确等于权重 → 色差最低，
#          而图案样式（点簇/网纹/散点）仍决定纹理长什么样
# - s = 0：tt 全部压到 0.5 → 该层退化为平涂（对应参考图的高光区）
# 于是"暗部有纹理、高光平涂"的分层手感完全保留，色差却显著下降。
# 强度参数在这里的含义变成「混合度」。

BARY_STRENGTH_FULL = 128.0      # 混合度归一化基准：strength 达到此值即为全力混合


def _require_scipy():
    try:
        from scipy.spatial import Delaunay
        return Delaunay
    except Exception as e:
        raise RuntimeError(
            "重心混合模式需要 scipy（python -m pip install scipy）；"
            "分层有序抖动不需要它。") from e


def _to_space(flat_rgb: np.ndarray, space: str) -> np.ndarray:
    """把 (N,3) 的 0-255 RGB 转换到目标色彩空间（LAB / RGB）。

    HSV 不做支持：H 是环形通道，无法参与凸包混合。
    """
    s = space.upper()
    if s == "LAB":
        return _rgb_to_lab(np.clip(flat_rgb, 0, 255))
    return np.asarray(flat_rgb, dtype=np.float64)


def _closest_on_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """向量化：p(M,3) 到三角形 (a,b,c) 的最近点（Ericson 算法）。"""
    ab = b - a; ac = c - a
    ap = p - a; bp = p - b; cp = p - c
    d1 = ap @ ab; d2 = ap @ ac; d3 = bp @ ab; d4 = bp @ ac; d5 = cp @ ab; d6 = cp @ ac
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    q = np.empty_like(p)
    mA = (d1 <= 0) & (d2 <= 0); mB = (d3 >= 0) & (d4 <= d3); mC = (d6 >= 0) & (d5 <= d6)
    mAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0); mAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    mBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    mIn = ~(mA | mB | mC | mAB | mAC | mBC)
    q[mA] = a; q[mB] = b; q[mC] = c
    if mAB.any():
        v = d1[mAB] / (d1[mAB] - d3[mAB]); q[mAB] = a + v[:, None] * ab
    if mAC.any():
        w = d2[mAC] / (d2[mAC] - d6[mAC]); q[mAC] = a + w[:, None] * ac
    if mBC.any():
        w = (d4[mBC] - d3[mBC]) / ((d4[mBC] - d3[mBC]) + (d5[mBC] - d6[mBC]))
        q[mBC] = b + w[:, None] * (c - b)
    if mIn.any():
        denom = 1.0 / (va[mIn] + vb[mIn] + vc[mIn]); vv = vb[mIn] * denom; ww = vc[mIn] * denom
        q[mIn] = a + vv[:, None] * ab + ww[:, None] * ac
    return q


def _project_to_hull(p: np.ndarray, hull_verts: np.ndarray, hull_idx: np.ndarray):
    """凸包外点投影到凸包表面，返回所在面的 3 个调色板索引 + 重心权重。"""
    M = p.shape[0]
    best_q = np.empty_like(p)
    best_face = np.empty((M, 3, 3))
    best_d2 = np.full(M, np.inf)
    best_idx = np.empty((M, 3), dtype=np.intp)
    for f in range(hull_verts.shape[0]):
        a, b, c = hull_verts[f]
        qf = _closest_on_tri(p, a, b, c)
        d2 = np.sum((qf - p) ** 2, axis=1)
        better = d2 < best_d2
        if better.any():
            best_d2[better] = d2[better]; best_q[better] = qf[better]
            best_face[better] = hull_verts[f]; best_idx[better] = hull_idx[f]
    v0 = best_face[:, 1] - best_face[:, 0]
    v1 = best_face[:, 2] - best_face[:, 0]
    v2 = best_q - best_face[:, 0]
    d00 = np.einsum('ij,ij->i', v0, v0); d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0); d21 = np.einsum('ij,ij->i', v2, v1)
    den = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    return best_idx, np.stack([1 - w1 - w2, w1, w2], axis=1)


# ============================================================================
# 四·C、每层限色：自适应代表色 + 重心混合
# ============================================================================
# 前面几条路都受制于固定调色板：颜色必须来自 palette.txt 那几十个色。
# 这里换一条 —— 每一层自己聚类出 K 个代表色（允许不在原调色板里）：
#   层内只有 K 种颜色  -> 大面积同色、画面干净
#   层内再做重心混合   -> 用这 K 个色的空间混合把区域平均色补回源色，色差不掉
# 层数越细，每层颜色范围越窄，同样 K 个色就覆盖得越好。实测（24 层 x 每层 6 色，
# LAB 聚类）区域平均 ΔE 比固定 63 色板还低两成，纹理能量只有后者的五分之一 ——
# 也就是"更干净"和"色差更低"同时成立。

def _kmeans(X: np.ndarray, K: int, iters: int = 20, sample: int = 60_000,
            seed: int = 0) -> np.ndarray:
    """k-means++ 初始化 + Lloyd 迭代；在采样集上收敛，返回 (K,d) 中心。

    层内像素动辄几十万，全量迭代没必要 —— 颜色分布的形状用几万点就够刻画了。
    """
    rng = np.random.default_rng(int(seed))
    n = X.shape[0]
    K = int(min(K, n))
    if K <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float64)
    Xs = X if n <= sample else X[rng.choice(n, int(sample), replace=False)]
    Xs = np.asarray(Xs, dtype=np.float64)

    centers = [Xs[rng.integers(len(Xs))]]
    for _ in range(K - 1):
        d = np.min(((Xs[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        tot = float(d.sum())
        centers.append(Xs[rng.choice(len(Xs), p=d / tot)] if tot > 0
                       else Xs[rng.integers(len(Xs))])
    C = np.asarray(centers, dtype=np.float64)

    for _ in range(int(iters)):
        lab = ((Xs[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        newC = C.copy()
        for j in range(K):
            m = lab == j
            if m.any():
                newC[j] = Xs[m].mean(0)
        done = np.allclose(newC, C, atol=1e-3)
        C = newC
        if done:
            break
    return C


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB(D65) → sRGB(0-255)。"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t):
        t3 = t ** 3
        return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

    x = 0.95047 * finv(fx)
    y = 1.0 * finv(fy)
    z = 1.08883 * finv(fz)
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    lin = np.stack([x, y, z], -1) @ M.T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255)


def _palette_tetra(pal: np.ndarray, space: str = "LAB"):
    """调色板的 Delaunay 四面体结构，供"用调色板色混合出某个目标色"使用。

    space: 'LAB'（默认，感知均匀，混合结果更贴近人眼）或 'RGB'。
    注意 HSV 的 H 是环形通道，不能直接做凸包混合，调用方需自行回退到 LAB。
    """
    Delaunay = _require_scipy()
    pal_space = _to_space(np.asarray(pal, dtype=np.float64), space)
    tri = Delaunay(pal_space)
    Vs = pal_space[tri.simplices]
    A = np.concatenate([np.transpose(Vs, (0, 2, 1)),
                        np.ones((Vs.shape[0], 1, 4))], axis=1)
    Ainv = np.linalg.inv(A)
    return pal_space, tri, Ainv, pal_space[tri.convex_hull], tri.convex_hull


_TRIU_CACHE = {}


def _mix_weights_k(target_rgb: np.ndarray, pal_space, tri, Ainv,
                   hull_verts, hull_idx, space: str = "LAB",
                   k: int = 4, cand: int = 16):
    """用 k 个调色板色混合出目标色。k=4 走四面体重心（凸包内精确无损）。

    k=2/3 是"少色方案"：2 色就是把目标投影到某两个板内色连成的线段上。
    分解本身是准的，但要清楚代价——端点往往离目标很远，会选出异色做补偿
    （实测近白色被分解成 85% 白 + 15% 深橄榄），权重虽对，块内却平均不干净：
    实测区域平均 ΔE 从 0.85 升到 3.51，3 色是 2.66。
    """
    if k >= 4:
        return _mix_weights(target_rgb, pal_space, tri, Ainv, hull_verts, hull_idx, space)

    p = _to_space(np.asarray(target_rgb, dtype=np.float64).reshape(1, 3), space)
    n = len(pal_space)
    if k == 2:
        key = n
        if key not in _TRIU_CACHE:
            I, J = np.triu_indices(n, k=1)
            AB = pal_space[J] - pal_space[I]
            _TRIU_CACHE[key] = (I, J, pal_space[I], pal_space[J], AB,
                                np.maximum((AB * AB).sum(1), 1e-12))
        I, J, A, B, AB, den = _TRIU_CACHE[key]
        ap = p[0][None, :] - A
        t = np.clip((ap * AB).sum(1) / den, 0.0, 1.0)
        d = np.linalg.norm(A + t[:, None] * AB - p[0][None, :], axis=1)
        m = int(np.argmin(d))
        # 混合点 Q = (1-t)*A + t*B，所以 A 的权重是 1-t
        return np.array([I[m], J[m]], dtype=np.intp), np.array([1.0 - t[m], t[m]])

    # k == 3：取最近的 cand 个色，枚举三元组，投影到三角形
    dd = np.linalg.norm(pal_space - p[0][None, :], axis=1)
    top = np.argsort(dd)[:cand]
    best = (np.inf, None, None)
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            for c in range(b + 1, len(top)):
                V = pal_space[[top[a], top[b], top[c]]]
                q = _closest_on_tri(p, V[0], V[1], V[2])[0]
                e = float(np.linalg.norm(q - p[0]))
                if e < best[0]:
                    best = (e, [int(top[a]), int(top[b]), int(top[c])], q)
    e, idx, q = best
    V = pal_space[idx]
    v0, v1, v2 = V[1] - V[0], V[2] - V[0], q - V[0]
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1; d20 = v2 @ v0; d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    den = den if abs(den) > 1e-12 else 1e-12
    w1 = max(0.0, (d11 * d20 - d01 * d21) / den)
    w2 = max(0.0, (d00 * d21 - d01 * d20) / den)
    w0 = max(0.0, 1.0 - w1 - w2)
    ssum = w0 + w1 + w2
    return np.array(idx, dtype=np.intp), np.array([w0, w1, w2]) / (ssum if ssum > 0 else 1.0)


def _mix_weights(target_rgb: np.ndarray, pal_space, tri, Ainv,
                 hull_verts, hull_idx, space: str = "LAB"):
    """求"用调色板色混合出目标色"的权重，返回 (调色板索引, 权重)，权重和为 1。

    目标色落在凸包内 -> 取所在四面体的重心坐标；
    落在凸包外   -> 先投影到凸包表面，再取该面的重心坐标。
    """
    p = _to_space(np.asarray(target_rgb, dtype=np.float64).reshape(1, 3), space)
    sidx = int(tri.find_simplex(p)[0])
    if sidx >= 0:
        w = Ainv[sidx] @ np.append(p[0], 1.0)
        w = np.clip(w, 0.0, None)
        s = float(w.sum())
        return tri.simplices[sidx], (w / s if s > 0 else np.full(4, 0.25))
    bi, w3 = _project_to_hull(p, hull_verts, hull_idx)
    w3 = np.clip(w3[0], 0.0, None)
    s = float(w3.sum())
    return bi[0], (w3 / s if s > 0 else np.full(3, 1.0 / 3.0))


def _nearest_indices(pts: np.ndarray, pal: np.ndarray, space: str = "LAB"):
    """每个点最近的调色板索引（在该层的投影空间里比）。"""
    d = ((_to_space(pts, space)[:, None, :] - _to_space(pal, space)[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def _error_diffuse_1d(pts: np.ndarray, pal: np.ndarray, strength: int = 64):
    """层内一维误差扩散。

    分层之后同一层的像素在画面上是分散的（亮部东一块西一块），没有稳定的空间邻居，
    所以只能在「该层像素的扫描序」上做 1D 累积扩散。它是真的误差扩散，但不具备
    整图 FS 的空间分布特性，观感上更容易结成条带。
    """
    n = len(pts)
    out = np.empty_like(pts)
    gain = float(strength) / 64.0
    err = np.zeros(3, dtype=np.float64)
    for i in range(n):
        want = pts[i] + err
        k = int(((pal - want) ** 2).sum(axis=1).argmin())
        out[i] = pal[k]
        err = (want - pal[k]) * gain
    return out


def _apply_slots(px: np.ndarray, field: np.ndarray, slots: List[ColorSlot],
                 pal: np.ndarray, pal_lab, tri, Ainv, hull_verts, hull_idx,
                 space: str = "LAB") -> np.ndarray:
    """按用户指定的颜色槽渲染一层像素。

    先把层内像素归类到最近的颜色槽（在投影空间里比），再让每个槽按自己的 method
    还原。归类而不是逐像素混合，是为了让「同一个槽的像素共用一套抖动」，与分层投影
    的设计一致。
    """
    if not slots:
        return px.copy()

    targets = np.asarray([
        pal[int(s.color)] if isinstance(s.color, (int, float)) else np.asarray(s.color, dtype=np.float64)
        for s in slots
    ], dtype=np.float64)

    d = ((_to_space(px, space)[:, None, :]
          - _to_space(targets, space)[None, :, :]) ** 2).sum(-1)
    assign = d.argmin(axis=1)

    out = np.empty_like(px)
    for j, slot in enumerate(slots):
        m = assign == j
        if not m.any():
            continue
        tgt = targets[j]
        method = str(slot.method or "mix").lower()
        st = int(slot.strength)

        if method == "direct":
            out[m] = tgt
        elif method == "ordered":
            # 加均值 0 的有序偏移后取最近色：纹理最强，但会偏离正确色
            pert = px[m] + (field[m][:, None] - 0.5) * float(st)
            out[m] = pal[_nearest_indices(pert, pal, space)]
        elif method == "error":
            out[m] = _error_diffuse_1d(px[m], pal, st)
        else:  # mix
            idx, wgt = _mix_weights(tgt, pal_lab, tri, Ainv, hull_verts, hull_idx, space)
            cum = np.cumsum(wgt)
            pick = np.argmax(cum[None, :] >= field[m][:, None], axis=1)
            out[m] = pal[np.asarray(idx, dtype=np.intp)[pick]]
    return out


def _band_index_map(image: Image.Image, profile: Profile):
    """公共前置：算出每个像素属于哪一层。供渲染与分析复用。"""
    arr, alpha = _split_alpha(image)
    luma = compute_luma(arr, profile.space)
    if profile.gamma != 1.0:
        luma = np.power(np.clip(luma, 0, 1), profile.gamma)
    if profile.black != 0.0 or profile.white != 1.0:
        span = max(1e-6, profile.white - profile.black)
        luma = np.clip((luma - profile.black) / span, 0.0, 1.0)
    bands = [b for b in profile.bands if b.enabled]
    if not bands:
        return arr, alpha, np.zeros(arr.shape[:2], dtype=int), []
    edges = resolve_edges(profile, luma, bands)
    idx = np.clip(np.searchsorted(edges, luma, side="right") - 1, 0, len(bands) - 1)
    return arr, alpha, idx, bands


def analyze_bands(image: Image.Image, palette: Sequence[Sequence[int]], profile: Profile,
                  only: Optional[int] = None):
    """逐层分析：像素数、自动聚出的投影色、每个投影色的调色板混合配方。

    返回 [{index, lo, hi, pixels, ratio, flat, colors:[{color, weight, recipe}]}, ...]
    color 是自动投影色（RGB，可能不在板内），recipe 是它由哪些调色板色混出来。
    这个结果只用于展示与编辑起点，不参与渲染。only 指定层号可只分析一层（GUI 打开
    单层颜色编辑器时用，避免为了一张图跑满 24 次 k-means）。
    """
    pal = np.asarray(palette, dtype=np.float64)
    arr, _alpha, band_idx, bands = _band_index_map(image, profile)
    flat = arr.reshape(-1, 3)
    flat_band = band_idx.reshape(-1)

    space = str(getattr(profile, "match_space", "LAB")).upper()
    if space not in ("RGB", "LAB"):
        space = "LAB"
    pal_lab, tri, Ainv, hull_verts, hull_idx = _palette_tetra(pal, space)
    K = max(1, int(getattr(profile, "colors_per_band", 6)))
    iters = int(getattr(profile, "kmeans_iters", 20))
    sample = int(getattr(profile, "kmeans_sample", 60_000))
    total = max(1, flat.shape[0])

    out = []
    for k, band in enumerate(bands):
        if only is not None and k != only:
            continue
        sel = np.nonzero(flat_band == k)[0]
        item = {"index": k, "lo": band.lo, "hi": band.hi,
                "pixels": int(sel.size), "ratio": sel.size / total,
                "flat": list(band.flat) if band.flat is not None else None,
                "manual": bool(band.colors), "colors": []}
        if sel.size and band.flat is None:
            px = flat[sel]
            if space == "LAB":
                proj_space = _kmeans(_rgb_to_lab(px), K, iters, sample, band.seed + k)
                proj_rgb = _lab_to_rgb(proj_space)
            else:
                proj_rgb = _kmeans(px, K, iters, sample, band.seed + k)
                proj_space = proj_rgb
            assign = np.linalg.norm(
                _to_space(px, space)[:, None, :] - proj_space[None, :, :], axis=-1).argmin(axis=1)
            for j, c in enumerate(proj_rgb):
                idx, wgt = _mix_weights(c, pal_lab, tri, Ainv, hull_verts, hull_idx, space)
                order = np.argsort(-wgt)
                item["colors"].append({
                    "color": [int(round(v)) for v in c],
                    "weight": float((assign == j).mean()),
                    "recipe": [{"index": int(idx[i]), "weight": float(wgt[i])}
                               for i in order if wgt[i] > 0.005],
                })
        out.append(item)
    return out


def render_band_preview(image: Image.Image, palette: Sequence[Sequence[int]],
                        profile: Profile, index: int) -> Tuple[Image.Image, Image.Image]:
    """渲染单层预览：返回 (该层像素的原色图, 该层像素的抖动图)，其余区域透明。

    用透明而不是置灰，这样在 UI 里叠在原图上看边界最直观。
    """
    pal = np.asarray(palette, dtype=np.float64)
    arr, _alpha, band_idx, bands = _band_index_map(image, profile)
    if not bands or index < 0 or index >= len(bands):
        empty = Image.new("RGBA", image.size, (0, 0, 0, 0))
        return empty, empty
    full = Image.fromarray(arr.reshape(arr.shape[0], arr.shape[1], 3).astype(np.uint8), "RGB")
    out = render(image, pal, profile)
    mask = (band_idx == index).astype(np.uint8) * 255

    src_rgba = np.dstack([np.asarray(full), mask])
    out_rgba = np.dstack([np.asarray(out.convert("RGB")), mask])
    return (Image.fromarray(src_rgba, "RGBA"), Image.fromarray(out_rgba, "RGBA"))


def layered_projection_dither(image: Image.Image,
                              palette: Sequence[Sequence[int]],
                              profile: Profile,
                              custom_patterns: Optional[dict] = None,
                              progress=None) -> Image.Image:
    """投影 + 单独混色：把"大面积同色"和"输出落在调色板内"两件事拆开做。

      第 1 步 投影：按明暗分层后，每层自由聚类 K 个投影色（**允许不在调色板内**），
                    层内像素投影到最接近的那个投影色。
                    → 每层只有 K 种颜色，大片区域同色。
      第 2 步 抖动：对**每一个投影色单独**求解"用调色板色把它混出来"的权重，
                    再用图案阈值按权重分配。同一个投影色的像素共用同一组权重。
                    → 输出全部落在调色板内，且同一区域的抖动图案完全一致。

    色差不在第 2 步补偿（第 2 步只忠实还原投影色），而是由第 1 步的投影质量
    （每层色数 K、层数与分层方式）决定 —— 所以想降色差就调 K 和层数。

    与 layered_barycentric_dither 的关键区别：那边每个像素各自找凸包、权重随位置
    变化，既带来偏色也让纹理杂乱；这里一个投影色只有一组权重，纹理整齐划一。
    """
    pal = np.asarray(palette, dtype=np.float64)
    if pal.ndim != 2 or pal.shape[1] != 3 or len(pal) < 4:
        raise ValueError("调色板至少需要 4 个颜色（要构成四面体）")

    # 色彩空间：RGB / LAB。HSV 的 H 是环形通道，做不了凸包混合，回退到 LAB。
    space = str(getattr(profile, "match_space", "LAB")).upper()
    if space not in ("RGB", "LAB"):
        space = "LAB"

    arr, alpha = _split_alpha(image)
    h, w, _ = arr.shape
    luma = compute_luma(arr, profile.space)
    if profile.gamma != 1.0:
        luma = np.power(np.clip(luma, 0, 1), profile.gamma)
    if profile.black != 0.0 or profile.white != 1.0:
        span = max(1e-6, profile.white - profile.black)
        luma = np.clip((luma - profile.black) / span, 0.0, 1.0)

    bands = [b for b in profile.bands if b.enabled]
    if not bands:
        return image.convert("RGB")
    edges = resolve_edges(profile, luma, bands)
    band_idx = np.clip(np.searchsorted(edges, luma, side="right") - 1, 0, len(bands) - 1)

    pal_lab, tri, Ainv, hull_verts, hull_idx = _palette_tetra(pal, space)
    K = max(1, int(getattr(profile, "colors_per_band", 6)))
    iters = int(getattr(profile, "kmeans_iters", 20))
    sample = int(getattr(profile, "kmeans_sample", 60_000))

    flat = arr.reshape(-1, 3)
    flat_band = band_idx.reshape(-1)
    out = np.empty_like(flat)

    # 每层用几个投影色：固定 K，或按层内颜色散度分配总预算。
    # 固定 K 会让颜色分布宽的层吃不饱、分布窄的层浪费预算；实测按散度分配
    # 在同样总色数下能把投影误差再降约 13%。
    band_sel = [np.nonzero(flat_band == k)[0] for k in range(len(bands))]
    if getattr(profile, "adaptive_k", True):
        weights = []
        for sel in band_sel:
            if sel.size == 0:
                weights.append(0.0)
            else:
                labs = _rgb_to_lab(flat[sel])
                weights.append(float(labs.std(0).mean() * np.sqrt(sel.size)))
        wts = np.asarray(weights, dtype=np.float64)
        budget = K * max(1, sum(1 for s in band_sel if s.size))
        if wts.sum() > 0:
            ks = np.maximum(2, np.round(wts / wts.sum() * budget)).astype(int)
        else:
            ks = np.full(len(bands), K, dtype=int)
    else:
        ks = np.full(len(bands), K, dtype=int)

    for k, band in enumerate(bands):
        sel = band_sel[k]
        if sel.size == 0:
            if progress:
                progress(k + 1, len(bands))
            continue
        px = flat[sel]

        if band.flat is not None:
            out[sel] = np.asarray(band.flat, dtype=np.float64)[None, :]
            if progress:
                progress(k + 1, len(bands))
            continue

        cp = custom_patterns.get(band.pattern) if custom_patterns else None
        field = threshold_field(band.pattern, band.size, band.cell, band.seed,
                                h, w, band.invert, cp).reshape(-1)[sel]

        # ---- 手动颜色槽：跳过 k-means，按用户指定的颜色与还原方式处理 ----
        slots = band.slots()
        if slots:
            out[sel] = _apply_slots(px, field, slots, pal, pal_lab, tri, Ainv,
                                    hull_verts, hull_idx, space)
            if progress:
                progress(k + 1, len(bands))
            continue

        # ---- 第 1 步：每层聚类投影色，把层内像素投影上去 ----
        if space == "LAB":
            px_lab = _rgb_to_lab(px)
            proj_lab = _kmeans(px_lab, max(1, int(ks[k])), iters, sample, band.seed + k)
            proj_rgb = _lab_to_rgb(proj_lab)
            proj_space = proj_lab
        else:
            proj_rgb = _kmeans(px, K, iters, sample, band.seed + k)
            proj_space = proj_rgb
        assign = np.linalg.norm(
            _to_space(px, space)[:, None, :] - proj_space[None, :, :],
            axis=-1).argmin(axis=1)

        # ---- 第 2 步：每个投影色单独求权重，同一权重负责该颜色的全部像素 ----
        for j in range(len(proj_rgb)):
            m = assign == j
            if not m.any():
                continue
            idx, wgt = _mix_weights_k(proj_rgb[j], pal_lab, tri, Ainv,
                                      hull_verts, hull_idx, space,
                                      k=int(getattr(profile, 'mix_colors', 4)))
            cum = np.cumsum(wgt)
            pick = np.argmax(cum[None, :] >= field[m][:, None], axis=1)
            out[sel[m]] = pal[np.asarray(idx, dtype=np.intp)[pick]]

        if progress:
            progress(k + 1, len(bands))

    return _merge_alpha(out.reshape(h, w, 3), alpha)


def render(image: Image.Image,
           palette: Sequence[Sequence[int]],
           profile: Profile,
           custom_patterns: Optional[dict] = None,
           progress=None) -> Image.Image:
    """渲染入口：按 Profile 做分层投影混色抖动。"""
    return layered_projection_dither(image, palette, profile,
                                     custom_patterns, progress)


def load_palette(file_path: str) -> List[Tuple[int, int, int]]:
    """读调色板，兼容 dithers.py 的 "Name:[r, g, b]" / "*Name:[r,g,b]" 格式。"""
    palette: List[Tuple[int, int, int]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("*"):
                line = line[1:].strip()
            if ":" in line:
                line = line.split(":", 1)[1]
            line = line.replace("[", " ").replace("]", " ").replace(",", " ")
            parts = line.split()
            if len(parts) >= 3:
                try:
                    palette.append(tuple(int(x) for x in parts[:3]))
                except ValueError:
                    continue
    return palette


def analyze_image(image: Image.Image, n_bands: int = 13, space: str = "lab",
                  edge_mode: str = "quantile") -> dict:
    """统计各亮度带的像素占比，用于自动规划分层（哪层该细、哪层该平）。"""
    arr, _ = _split_alpha(image)
    luma = compute_luma(arr, space)
    flat = luma.reshape(-1)
    if edge_mode == "linear":
        edges = np.linspace(0.0, 1.0, n_bands + 1)
    else:
        edges = np.quantile(flat, np.linspace(0.0, 1.0, n_bands + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.maximum.accumulate(edges)
    hist, _ = np.histogram(flat, bins=edges)
    total = max(1, flat.size)
    return {
        "counts": hist.tolist(),
        "ratios": (hist / total).tolist(),
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "edges": edges.tolist(),
    }


# ============================================================================
# 五、命令行
# ============================================================================

def _cli():
    import argparse
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台中文输出
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="分层投影混色抖动")
    ap.add_argument("src", help="输入图片")
    ap.add_argument("dst", nargs="?", default=None, help="输出图片")
    ap.add_argument("--palette", default="palette.txt", help="调色板文件")
    ap.add_argument("--profile", default=None, help="Profile JSON（优先生效）")
    ap.add_argument("--bands", type=int, default=24, help="层数")
    ap.add_argument("--colors", type=int, default=8, help="每层投影色数量")
    ap.add_argument("--space", default="lab", choices=["lab", "hsv", "gray"],
                    help="分层用的明度空间")
    ap.add_argument("--match-space", default="LAB", choices=["LAB", "RGB"],
                    help="投影聚类与调色板混合用的色彩空间")
    ap.add_argument("--edge-mode", default=None,
                    choices=["linear", "quantile", "hybrid"],
                    help="分层依据：linear 按明度等分（保色彩），"
                         "quantile 按图像分布等频（保细节）")
    ap.add_argument("--pattern", default=None,
                    choices=list(PATTERN_NAMES),
                    help="抖动图案（默认 bayer）")
    ap.add_argument("--size", type=int, default=None, help="图案阶数")
    ap.add_argument("--cell", type=int, default=None, help="图案放大倍数")
    ap.add_argument("--scale", type=float, default=1.0, help="输出缩放")
    ap.add_argument("--save-profile", default=None, help="把本次 Profile 存成 JSON")
    ap.add_argument("--analyze", action="store_true", help="只打印各层像素占比")
    args = ap.parse_args()

    img = Image.open(args.src)
    if args.scale != 1.0:
        img = img.resize((max(1, int(img.width * args.scale)),
                          max(1, int(img.height * args.scale))), Image.LANCZOS)

    if args.analyze:
        mode = args.edge_mode or "linear"
        info = analyze_image(img, args.bands, args.space, mode)
        print(f"平均明度 {info['mean']:.3f}  中位 {info['median']:.3f}  分层依据 {mode}")
        for i, r in enumerate(info["ratios"]):
            bar = "#" * int(r * 60)
            print(f"  层{i + 1:>2}  L* {info['edges'][i] * 100:5.1f}~{info['edges'][i + 1] * 100:5.1f}  "
                  f"{r * 100:5.1f}%  {bar}")
        return

    if args.profile:
        profile = Profile.from_json(args.profile)
    else:
        profile = make_projection_profile(
            n_bands=args.bands, colors_per_band=args.colors,
            pattern=args.pattern or "bayer",
            size=args.size if args.size else 16,
            cell=args.cell if args.cell else 1)
    profile.match_space = args.match_space
    profile.space = args.space
    if args.edge_mode:
        profile.edge_mode = args.edge_mode

    palette = load_palette(args.palette)
    if not palette:
        raise SystemExit(f"调色板为空: {args.palette}")

    out = render(img, palette, profile,
                 progress=lambda d, t: print(f"\r分层投影混色 {d}/{t}", end=""))
    print()
    dst = args.dst or os.path.splitext(args.src)[0] + "_layered.png"
    out.save(dst)
    if args.save_profile:
        profile.to_json(args.save_profile)
    print(f"已保存 {dst}（{len(profile.bands)} 层 x 每层 {profile.colors_per_band} 色，"
          f"{profile.match_space}，{profile.edge_mode}）")


if __name__ == "__main__":
    _cli()
