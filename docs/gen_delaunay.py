# -*- coding: utf-8 -*-
"""离线预计算：调色板的 Delaunay 四面体、重心逆矩阵、凸包面、粗网格查找表。
生成 web/delaunay.js，供浏览器端重心混合直接查表（避免在 JS 里实现 3D Delaunay）。"""
import json
import os
import numpy as np
from scipy.spatial import Delaunay

HERE = os.path.dirname(os.path.abspath(__file__))
PAL_TXT = os.path.join(HERE, '..', 'palette.txt')


def load_palette(path):
    """解析 palette.txt：返回 (rgb列表, 名称列表, 付费标记列表)"""
    rgbs, names, paid = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or ':' not in line:
                continue
            is_paid = line.startswith('*')
            name_part, rgb_part = line.lstrip('*').split(':', 1)
            rgb_part = rgb_part.strip().strip('[]')
            try:
                rgb = tuple(int(x.strip()) for x in rgb_part.split(','))
            except ValueError:
                continue
            if len(rgb) == 3:
                rgbs.append(rgb)
                names.append(name_part.strip())
                paid.append(is_paid)
    return rgbs, names, paid


rgbs, names, paid = load_palette(PAL_TXT)
pal = np.array(rgbs, dtype=np.float64)
tri = Delaunay(pal)

# 1) 四面体（调色板索引）
tetras = tri.simplices.tolist()

# 2) 每个四面体的重心逆矩阵 Ainv：w = Ainv @ [r,g,b,1]，A = [v0 v1 v2 v3; 1 1 1 1]
ainv = []
for t in tri.simplices:
    V = pal[t]  # (4,3)
    A = np.concatenate([V.T, np.ones((1, 4))], axis=0)  # (4,4)
    ainv.append(np.linalg.inv(A).tolist())

# 3) 凸包面（三角形，调色板索引）
hull = tri.convex_hull.tolist()

# 4) 粗网格查找表：RGB 0-255，步长 4 → 64^3 格，存"含该格中心的四面体索引"(-1=凸包外)
# 运行时查表若该四面体不含像素(权重<0)，回退完整遍历——必然精确
GSIZE = 64
STEP = 256 / GSIZE
grid = []
for ri in range(GSIZE):
    r = ri * STEP + STEP / 2
    for gi in range(GSIZE):
        g = gi * STEP + STEP / 2
        for bi in range(GSIZE):
            b = bi * STEP + STEP / 2
            s = tri.find_simplex(np.array([[r, g, b]], dtype=np.float64))[0]
            grid.append(int(s))  # -1 = 凸包外

data = {
    'palette': rgbs,
    'names': names,
    'paid': paid,
    'tetras': tetras,
    'ainv': ainv,
    'hull': hull,
    'gridSize': GSIZE,
    'grid': grid,
}

out = os.path.join(HERE, 'delaunay.js')
with open(out, 'w', encoding='utf-8') as f:
    f.write('// 自动生成（web/gen_delaunay.py）。不要手动编辑。\n')
    f.write('const DITHER_DATA = ' + json.dumps(data, separators=(',', ':')) + ';\n')

print(f'调色板 {len(rgbs)} 色 | 四面体 {len(tetras)} | 凸包面 {len(hull)} | 网格 {GSIZE}^3')

# ===== 验证 1：网格查表 → 重心权重 → 重建 =====
rng = np.random.default_rng(0)
pal_np = pal
ok = 0
total = 0
for _ in range(2000):
    p = rng.integers(0, 256, size=3)
    cell = (p // STEP).astype(int)
    if np.any(cell >= GSIZE): continue
    idx = (cell[0] * GSIZE + cell[1]) * GSIZE + cell[2]
    t = grid[idx]
    if t < 0:
        continue  # 凸包外（不计入分母）
    total += 1
    w = np.array(ainv[t]) @ np.append(p, 1.0)
    recon = pal_np[tri.simplices[t]].T @ np.clip(w, 0, None)  # (3,)
    if np.abs(recon - p).max() < 1.5:
        ok += 1
print(f'验证: 网格查表→重心重建 误差<1.5 的像素 {ok}/{total} ({100*ok/total:.1f}%)')

# ===== 验证 2：与 scipy find_simplex 的一致性 =====
agree = 0
for _ in range(500):
    p = rng.integers(0, 256, size=3)
    cell = (p // STEP).astype(int)
    if np.any(cell >= GSIZE): continue
    idx = (cell[0] * GSIZE + cell[1]) * GSIZE + cell[2]
    t = grid[idx]
    s = tri.find_simplex(p[None, :])[0]
    # 允许：凸包外(-1) 与 找到最近四面体 的差异；网格内的应基本一致
    if t == s or (s < 0 and t >= 0 and np.all(np.array(ainv[t]) @ np.append(p, 1.0) >= -0.01)):
        agree += 1
print(f'验证: 网格四面体 vs scipy 一致率 {agree}/500 ({100*agree/500:.1f}%)')
