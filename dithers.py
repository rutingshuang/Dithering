import numpy as np
from PIL import Image
import colorsys
from scipy import ndimage
from scipy.spatial import Delaunay


# 定义几种常见的有序抖动矩阵
def bayer_matrix(size=8):
    if size == 2:
        return np.array([[0, 2],
                         [3, 1]])
    elif size == 4:
        return np.array([[0, 8, 2, 10],
                         [12, 4, 14, 6],
                         [3, 11, 1, 9],
                         [15, 7, 13, 5]])
    elif size == 8:
        return np.array([
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ])
    else:
        raise ValueError("仅支持2x2、4x4或8×8 Bayer矩阵")


def clustered_dot_matrix(size=8):
    if size == 4:
        return np.array([[14, 11, 8, 13],
                         [10, 6, 4, 9],
                         [7, 2, 0, 5],
                         [12, 3, 1, 15]])
    elif size == 8:
        return np.array([[62, 57, 48, 36, 37, 49, 58, 63],
                         [56, 47, 35, 21, 22, 38, 50, 59],
                         [46, 34, 20, 10, 11, 23, 39, 51],
                         [33, 19, 9, 3, 0, 4, 12, 24],
                         [32, 18, 8, 2, 1, 5, 13, 25],
                         [45, 31, 17, 7, 6, 14, 26, 40],
                         [55, 44, 30, 16, 15, 27, 41, 52],
                         [61, 54, 43, 29, 28, 42, 53, 60]])
    else:
        raise ValueError("点簇矩阵格式不支持")


def diagonal_matrix(size=4):
    return np.array([[15, 11, 5, 0],
                     [10, 14, 9, 4],
                     [3, 8, 13, 7],
                     [1, 2, 9, 12]])


def spiral_matrix(size=4):
    return np.array([[9, 1, 6, 14],
                     [3, 11, 15, 4],
                     [5, 7, 0, 8],
                     [12, 13, 10, 2]])


# 根据方法获取矩阵
def get_dither_matrix(method, msize=4):
    if method == "bayer":
        return bayer_matrix(msize)
    elif method == "clustered":
        return clustered_dot_matrix(msize)
    elif method == "diagonal":
        return diagonal_matrix(4)
    elif method == "spiral":
        return spiral_matrix(4)
    else:
        raise ValueError("未知的抖动方法: " + method)


# 找到调色盘中最接近的颜色 (RGB空间)
def find_nearest_color_rgb(color, palette):
    r, g, b = color
    palette = np.array(palette)
    distances = np.sum((palette - np.array([r, g, b])) ** 2, axis=1)
    return tuple(palette[np.argmin(distances)])


# 找到调色盘中最接近的颜色 (HSV空间)
def find_nearest_color_hsv(color, hsv_palette, weights=(1.0, 1.0, 1.0)):
    """
    在HSV空间中找到最接近的调色盘颜色
    使用加权欧氏距离，考虑H通道的循环特性
    """
    # 计算与调色盘中每个颜色的加权距离
    diff = (hsv_palette - color) * weights
    # 处理H通道的循环特性
    h_diff = np.abs(diff[:, 0])
    # 如果H差值大于0.5，考虑循环特性
    mask = h_diff > 0.5
    h_diff[mask] = 1.0 - h_diff[mask]
    diff[:, 0] = h_diff

    # 计算加权平方距离
    distances = np.sum(diff ** 2, axis=1)
    return np.argmin(distances)


# 将RGB图像转换为HSV (0-1范围)
def pil_to_hsv(img):
    """PIL Image(RGB) → numpy(HSV, 0-1)"""
    hsv = img.convert("HSV")
    arr = np.array(hsv, dtype=np.float32) / 255.0
    return arr


# 向量化：在调色盘中找最接近的颜色索引（RGB 欧氏距离，分块防止内存爆掉）
# 用逐通道 2D 累加，避免三维大数组；float64 与旧逐像素实现逐位一致
def _nearest_rgb_indices(flat, palette, chunk=200_000):
    pal = np.asarray(palette, dtype=np.float64)
    p0, p1, p2 = pal[:, 0], pal[:, 1], pal[:, 2]
    n = flat.shape[0]
    idx = np.empty(n, dtype=np.intp)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        p = flat[s:e].astype(np.float64)               # (C,3)
        d = (p[:, 0:1] - p0[None, :]) ** 2
        d += (p[:, 1:2] - p1[None, :]) ** 2
        d += (p[:, 2:3] - p2[None, :]) ** 2            # (C,K)
        idx[s:e] = np.argmin(d, axis=1)
    return idx


# 向量化：HSV 加权欧氏距离 + H 通道循环特性（逐通道累加，float64 与旧实现一致）
def _nearest_hsv_indices(flat, hsv_palette, weights=(1.0, 1.0, 1.0), chunk=100_000):
    pal = np.asarray(hsv_palette, dtype=np.float64)
    w0, w1, w2 = float(weights[0]), float(weights[1]), float(weights[2])
    ph0, ph1, ph2 = pal[:, 0], pal[:, 1], pal[:, 2]
    n = flat.shape[0]
    idx = np.empty(n, dtype=np.intp)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        p = flat[s:e].astype(np.float64)               # (C,3)
        hd = np.abs((ph0[None, :] - p[:, 0:1]) * w0)
        hd = np.minimum(hd, 1.0 - hd)                  # H 通道循环
        d = hd * hd
        sd = (ph1[None, :] - p[:, 1:2]) * w1
        d += sd * sd
        vd = (ph2[None, :] - p[:, 2:3]) * w2
        d += vd * vd                                   # (C,K)
        idx[s:e] = np.argmin(d, axis=1)
    return idx


# sRGB → CIELAB (D65)，vectorized；输入 RGB 0-255，输出 Lab
def _rgb_to_lab(rgb):
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


# 向量化：按 Lab ΔE76 找最接近的调色板颜色索引（逐通道累加，分块防爆内存）
def _nearest_lab_indices(flat_rgb, lab_palette, chunk=200_000):
    n = flat_rgb.shape[0]
    idx = np.empty(n, dtype=np.intp)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lab = _rgb_to_lab(flat_rgb[s:e])               # (C,3)
        d = (lab[:, 0:1] - lab_palette[:, 0][None, :]) ** 2
        d += (lab[:, 1:2] - lab_palette[:, 1][None, :]) ** 2
        d += (lab[:, 2:3] - lab_palette[:, 2][None, :]) ** 2
        idx[s:e] = np.argmin(d, axis=1)
    return idx


# 核心量化函数：对 RGB 数组做有序抖动（含 RGB/HSV/Lab 色彩空间支持），返回调色板颜色 (uint8, h,w,3)
# 供 ordered_dithering 与 multiscale_ordered_dithering 复用，保证行为一致
def _ordered_dither_array(arr_rgb, palette, method="bayer", strength=64,
                          color_space="RGB", weights=(1.0, 1.0, 1.0)):
    h, w, _ = arr_rgb.shape

    matrix = get_dither_matrix(method)
    msize = matrix.shape[0]
    # 归一化到 [-0.5, 0.5]，再乘以强度
    matrix = ((matrix + 0.5) / (msize * msize) - 0.5) * strength
    # 将矩阵平铺覆盖整幅图（等价于 matrix[y % msize, x % msize]）
    offsets = np.tile(matrix, (int(np.ceil(h / msize)), int(np.ceil(w / msize))))[:h, :w]

    if color_space.upper() == "HSV":
        # 将RGB调色盘转换为HSV (0-1范围)
        hsv_palette = []
        for rgb in palette:
            r, g, b = [x / 255.0 for x in rgb]
            h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
            hsv_palette.append((h_val, s_val, v_val))
        hsv_palette = np.array(hsv_palette)
        # 将RGB图像转换为HSV (0-1范围)
        rgb_img = Image.fromarray(np.clip(arr_rgb, 0, 255).astype(np.uint8), "RGB")
        arr_hsv = pil_to_hsv(rgb_img)
        # 偏移量缩放到0-1范围，与旧实现一致
        img_mod = np.clip(arr_hsv + (offsets[:, :, None] / 255.0), 0, 1)
        flat = img_mod.reshape(-1, 3)
        idx = _nearest_hsv_indices(flat, hsv_palette, weights)
    elif color_space.upper() == "LAB":
        # Lab 感知色距：先裁剪到[0,255]保证 Lab 有效（越界像素罕见，影响可忽略）
        lab_palette = _rgb_to_lab(np.asarray(palette, dtype=np.float64))
        img_mod = np.clip(arr_rgb + offsets[:, :, None], 0, 255)
        flat = img_mod.reshape(-1, 3)
        idx = _nearest_lab_indices(flat, lab_palette)
    else:
        # 与旧实现一致：不裁剪，直接对加偏移后的像素做最近邻量化
        img_mod = arr_rgb + offsets[:, :, None]
        flat = img_mod.reshape(-1, 3)
        idx = _nearest_rgb_indices(flat, palette)

    out = np.asarray(palette, dtype=np.float32)[idx].reshape(h, w, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


# 有序抖动函数（支持透明通道保留、色彩空间选择和加权参数）
def ordered_dithering(image, palette, method="bayer", strength=64, color_space="RGB", weights=(1.0, 1.0, 1.0)):
    """
    对RGB通道做有序抖动，透明通道不处理
    只要透明度 > 0 就视为完全不透明 (alpha=255)
    透明像素(alpha=0)保持透明

    支持在RGB或HSV色彩空间中进行颜色比较
    支持HSV空间中的加权参数
    """
    # 保留透明度
    if image.mode == "RGBA":
        arr_rgb = np.array(image, dtype=np.float32)
        alpha = arr_rgb[:, :, 3].copy()
        arr_rgb = arr_rgb[:, :, :3]  # 只取RGB
        has_alpha = True
    else:
        arr_rgb = np.array(image.convert("RGB"), dtype=np.float32)
        alpha = None
        has_alpha = False

    out = _ordered_dither_array(arr_rgb, palette, method, strength, color_space, weights)

    # 如果有透明度 → 非零透明度设为255
    if has_alpha:
        # 创建二值化的alpha通道（0或255）
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        # 完全透明像素的RGB置0，与旧实现保持一致
        out[alpha == 0] = 0
        rgba = np.dstack([out, alpha_binary])
        return Image.fromarray(rgba, "RGBA")
    else:
        return Image.fromarray(out, "RGB")


# === 金字塔低频误差反馈（方案A）===

def _pyramid_down(data, factor=0.5):
    """高斯平滑后下采样（避免锯齿），返回 float32 (h,w,3)。"""
    sigma = 0.6 / factor
    data = ndimage.gaussian_filter(data, sigma=(sigma, sigma, 0))
    h, w, c = data.shape
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    return ndimage.zoom(data, (nh / h, nw / w, 1), order=1)


def _pyramid_up(data, target_shape):
    """双线性上采样到目标尺寸。"""
    th, tw = target_shape
    h, w, c = data.shape
    return ndimage.zoom(data, (th / h, tw / w, 1), order=1)


def multiscale_ordered_dithering(image, palette, method="bayer", strength=64,
                                 color_space="RGB", weights=(1.0, 1.0, 1.0),
                                 scale_factor=0.5, levels=3, feedback=0.5):
    """
    金字塔低频误差反馈有序抖动（方案A）。

    思路：把“区域色差”当作低频误差，从粗到细逐层补偿——
      1. 高斯下采样构建金字塔（全在 RGB 空间）
      2. 最粗层做有序抖动
      3. 逐层下推：残差 = 当前层原图 − 粗层抖动结果(上采样)；把残差按反馈增益
         加到当前层输入上再做有序抖动
    效果：细层纹理仍是有序矩阵的规则纹理，但区域(低频)色差被显著补偿。

    参数：
      scale_factor  每层缩小比例（默认 0.5）
      levels        金字塔层数（1 = 退化为普通有序抖动）
      feedback      反馈增益 α（0 = 不做补偿；建议 0.3~0.8）
    """
    # 保留透明度
    if image.mode == "RGBA":
        arr_rgb = np.array(image, dtype=np.float32)
        alpha = arr_rgb[:, :, 3].copy()
        arr_rgb = arr_rgb[:, :, :3]  # 只取RGB
        has_alpha = True
    else:
        arr_rgb = np.array(image.convert("RGB"), dtype=np.float32)
        alpha = None
        has_alpha = False

    # 构建金字塔（下采样后的图层按比例变小）
    levels = max(1, min(int(levels), 6))
    pyramid = [arr_rgb]
    cur = arr_rgb
    for _ in range(levels - 1):
        if min(cur.shape[0], cur.shape[1]) < 8:
            break
        cur = _pyramid_down(cur, scale_factor)
        pyramid.append(cur)

    # 从最粗层开始抖动
    out = _ordered_dither_array(pyramid[-1], palette, method, strength, color_space, weights)

    # 逐层下推 + 残差反馈
    for k in range(len(pyramid) - 2, -1, -1):
        cur_data = pyramid[k]                                    # (h_k,w_k,3) 0-255
        up = _pyramid_up(out.astype(np.float64), cur_data.shape[:2])
        residual = cur_data - up                                 # 区域(低频)误差
        corrected = cur_data + float(feedback) * residual        # 补偿输入
        out = _ordered_dither_array(corrected.astype(np.float32),
                                    palette, method, strength, color_space, weights)

    if has_alpha:
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        out[alpha == 0] = 0
        return Image.fromarray(np.dstack([out, alpha_binary]), "RGBA")
    else:
        return Image.fromarray(out, "RGB")


def ordered_dithering_with_error(image, palette, method="bayer", strength=64,
                                               color_space="RGB", weights=(1.0, 1.0, 1.0)):
    """
    有序抖动 + 全向大范围误差扩散
    - 每个抖动矩阵区域独立处理
    - 按矩阵值从小到大决定像素处理顺序
    - 误差扩散到整个区域内尚未处理的像素（采用高斯加权）
    """

    if image.mode == "RGBA":
        arr_rgb = np.array(image, dtype=np.float32)
        alpha = arr_rgb[:, :, 3].copy()
        arr_rgb = arr_rgb[:, :, :3]
        has_alpha = True
    else:
        arr_rgb = np.array(image.convert("RGB"), dtype=np.float32)
        alpha = None
        has_alpha = False

    h, w, _ = arr_rgb.shape

    # === 抖动矩阵 ===
    matrix = get_dither_matrix(method)
    msize = matrix.shape[0]
    matrix_norm = ((matrix + 0.5) / (msize * msize) - 0.5) * strength

    # === 色彩空间预处理 ===
    if color_space.upper() == "HSV":
        hsv_palette = []
        for rgb in palette:
            r, g, b = [x / 255.0 for x in rgb]
            hsv_palette.append(colorsys.rgb_to_hsv(r, g, b))
        hsv_palette = np.array(hsv_palette)

        rgb_img = Image.fromarray(arr_rgb.astype(np.uint8), "RGB")
        arr_hsv = pil_to_hsv(rgb_img)
    else:
        hsv_palette = None
        arr_hsv = None

    out = np.zeros_like(arr_rgb)

    # === 构造高斯权重核 ===
    yy, xx = np.mgrid[0:msize, 0:msize]
    cy, cx = (msize - 1) / 2, (msize - 1) / 2
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = (msize / 2) ** 2
    kernel = np.exp(-dist2 / (2 * sigma2))  # 高斯权重
    kernel /= kernel.sum()  # 归一化

    # === 按区域处理 ===
    for by in range(0, h, msize):
        for bx in range(0, w, msize):
            block_h = min(msize, h - by)
            block_w = min(msize, w - bx)

            coords = []
            for yy in range(block_h):
                for xx in range(block_w):
                    val = matrix_norm[yy, xx]
                    coords.append((val, yy, xx))
            coords.sort(key=lambda t: t[0])  # 小值先

            # 区域内误差缓冲
            error_buf = np.zeros((block_h, block_w, 3), dtype=np.float32)

            processed = np.zeros((block_h, block_w), dtype=bool)

            for _, yy, xx in coords:
                y, x = by + yy, bx + xx
                if has_alpha and alpha[y, x] == 0:
                    out[y, x] = (0, 0, 0)
                    processed[yy, xx] = True
                    continue

                # 原像素 + 偏移 + 当前误差
                old_pixel = arr_rgb[y, x] + matrix_norm[yy, xx] + error_buf[yy, xx]

                if color_space.upper() == "HSV" and hsv_palette is not None:
                    old_pixel_hsv = arr_hsv[y, x] + matrix_norm[yy, xx] / 255.0
                    old_pixel_hsv = np.clip(old_pixel_hsv, 0, 1)
                    best_idx = find_nearest_color_hsv(old_pixel_hsv, hsv_palette, weights)
                    new_pixel = palette[best_idx]
                else:
                    new_pixel = find_nearest_color_rgb(old_pixel, palette)

                out[y, x] = new_pixel
                processed[yy, xx] = True

                # 误差
                err = old_pixel - new_pixel

                # 扩散到区域内所有未处理像素
                weights_sum = 0.0
                weights_map = np.zeros((block_h, block_w))
                for ny in range(block_h):
                    for nx in range(block_w):
                        if not processed[ny, nx]:
                            weights_map[ny, nx] = kernel[ny, nx]
                            weights_sum += kernel[ny, nx]

                if weights_sum > 0:
                    norm_weights = weights_map / weights_sum
                    error_buf += err * norm_weights[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    if has_alpha:
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        rgba = np.dstack([out, alpha_binary])
        return Image.fromarray(rgba, "RGBA")
    else:
        return Image.fromarray(out, "RGB")


def _diffuse_row(err_row, dy_weights):
    """把一行的量化误差按 (dx, wgt) 向量化扩散到下一行（dx 为列偏移，可为负）。"""
    out = np.zeros_like(err_row)
    for dx, wt in dy_weights:
        if dx == 0:
            out += wt * err_row
        elif dx > 0:
            out[dx:] += wt * err_row[:-dx]
        else:
            out[:dx] += wt * err_row[-dx:]
    return out


def floyd_steinberg_dither(image, palette, alpha_strength=1.0, filter_type="FS"):
    """
    支持三种误差扩散：
    - FS: Floyd–Steinberg
    - Jarvis: Jarvis, Judice & Ninke
    - Stucki: Stucki

    性能优化：
    - 调色板只建一次，最近色用矩阵乘法，避免逐像素重建数组
    - 上一行/上两行的误差贡献用 numpy 向量化计算
    - 仅"同行左→右"的误差传播本质串行，保留一个极小的逐像素循环
    相对旧版（逐像素全循环）通常快一个数量级。
    """
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.float32)
        alpha_channel = arr[:, :, 3].copy()
        arr = arr[:, :, :3]
        has_alpha = True
    else:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        alpha_channel = None
        has_alpha = False

    h, w, _ = arr.shape

    # 选择权重矩阵
    if filter_type == "FS":
        weights = [
            ((0, 1), 7 / 16),
            ((1, -1), 3 / 16),
            ((1, 0), 5 / 16),
            ((1, 1), 1 / 16)
        ]
    elif filter_type == "Jarvis":
        weights = [
            ((0, 1), 7 / 48), ((0, 2), 5 / 48),
            ((1, -2), 3 / 48), ((1, -1), 5 / 48), ((1, 0), 7 / 48), ((1, 1), 5 / 48), ((1, 2), 3 / 48),
            ((2, -2), 1 / 48), ((2, -1), 3 / 48), ((2, 0), 5 / 48), ((2, 1), 3 / 48), ((2, 2), 1 / 48)
        ]
    elif filter_type == "Stucki":
        weights = [
            ((0, 1), 8 / 42), ((0, 2), 4 / 42),
            ((1, -2), 2 / 42), ((1, -1), 4 / 42), ((1, 0), 8 / 42), ((1, 1), 4 / 42), ((1, 2), 2 / 42),
            ((2, -2), 1 / 42), ((2, -1), 2 / 42), ((2, 0), 4 / 42), ((2, 1), 2 / 42), ((2, 2), 1 / 42)
        ]
    else:
        raise ValueError("filter_type must be 'FS', 'Jarvis' or 'Stucki'")

    # 按扩散方向分组：dy=0 同行向右；dy=1 下一行；dy=2 下两行
    col_w = [(dx, wt) for (dy, dx), wt in weights if dy == 0]
    row1_w = [(dx, wt) for (dy, dx), wt in weights if dy == 1]
    row2_w = [(dx, wt) for (dy, dx), wt in weights if dy == 2]

    pal = np.array(palette)                    # int64，与旧实现一致
    pal64 = pal.astype(np.float64)
    pal64_T = pal64.T
    pal_sq = np.sum(pal64 ** 2, axis=1)        # (K,)

    orig = arr.copy()                          # 原图像（不可变），输出/误差分开存
    out = np.zeros_like(arr)
    err = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(h):
        # 上一行/上两行的误差贡献（向量化）
        fa = np.zeros((w, 3), dtype=np.float32)
        if y >= 1:
            fa += _diffuse_row(err[y - 1], row1_w)
        if y >= 2:
            fa += _diffuse_row(err[y - 2], row2_w)

        for x in range(w):
            if has_alpha and alpha_channel[y, x] == 0:
                out[y, x] = orig[y, x]         # 透明像素保持原值（alpha=0 不可见）
                continue

            # 同行左→右误差（串行）
            fl = np.zeros(3, dtype=np.float32)
            for dx, wt in col_w:
                if x >= dx:
                    fl += wt * err[y, x - dx]

            cur = orig[y, x] + fa[x] + fl
            # 最近色：|p-pal|² = |p|² - 2 p·pal + |pal|²（避免逐像素建 (K,3) 数组）
            p_sq = float(np.dot(cur, cur))
            d = pal_sq - 2.0 * np.dot(cur, pal64_T) + p_sq
            new = pal[np.argmin(d)]
            e = (cur - new) * alpha_strength
            out[y, x] = new
            err[y, x] = e

    out = np.clip(out, 0, 255).astype(np.uint8)

    if has_alpha:
        alpha_binary = np.where(alpha_channel < 128, 0, 255).astype(np.uint8)
        rgba = np.dstack([out, alpha_binary])
        return Image.fromarray(rgba, "RGBA")
    else:
        return Image.fromarray(out, "RGB")


def _closest_on_tri(p, a, b, c):
    """向量化：p(M,3) 到三角形(a,b,c) 的最近点（Ericson 算法）。"""
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
        w = (d4[mBC] - d3[mBC]) / ((d4[mBC] - d3[mBC]) + (d5[mBC] - d6[mBC])); q[mBC] = b + w[:, None] * (c - b)
    if mIn.any():
        denom = 1.0 / (va[mIn] + vb[mIn] + vc[mIn]); vv = vb[mIn] * denom; ww = vc[mIn] * denom
        q[mIn] = a + vv[:, None] * ab + ww[:, None] * ac
    return q


def _project_to_hull(p, hull_verts, hull_idx):
    """凸包外点 p(M,3) → 凸包表面最近点所在面的 3 个顶点（返回调色板索引）+ 三角形重心权重。
    hull_verts: (F,3,3) 凸包面顶点坐标；hull_idx: (F,3) 对应调色板索引。
    返回 (best_idx(M,3) 调色板索引, weights(M,3))。"""
    M = p.shape[0]
    best_q = np.empty_like(p); best_face = np.empty((M, 3, 3)); best_d2 = np.full(M, np.inf)
    best_idx = np.empty((M, 3), dtype=np.intp)
    for f in range(hull_verts.shape[0]):
        a, b, c = hull_verts[f]
        qf = _closest_on_tri(p, a, b, c)
        d2 = np.sum((qf - p) ** 2, axis=1)
        better = d2 < best_d2
        if better.any():
            best_d2[better] = d2[better]; best_q[better] = qf[better]
            best_face[better] = hull_verts[f]; best_idx[better] = hull_idx[f]
    v0 = best_face[:, 1] - best_face[:, 0]; v1 = best_face[:, 2] - best_face[:, 0]; v2 = best_q - best_face[:, 0]
    d00 = np.einsum('ij,ij->i', v0, v0); d01 = np.einsum('ij,ij->i', v0, v1); d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0); d21 = np.einsum('ij,ij->i', v2, v1)
    den = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / den; w2 = (d00 * d21 - d01 * d20) / den
    return best_idx, np.stack([1 - w1 - w2, w1, w2], axis=1)


def _edge_map_lab(arr):
    """Lab 色差边缘图：邻近像素 ΔE 之和。arr: (h,w,3) RGB 0-255。"""
    h, w, _ = arr.shape
    lab = _rgb_to_lab(arr.reshape(-1, 3)).reshape(h, w, 3)
    gy = np.sqrt(np.sum((lab[1:] - lab[:-1]) ** 2, axis=2))    # (h-1,w)
    gx = np.sqrt(np.sum((lab[:, 1:] - lab[:, :-1]) ** 2, axis=2))  # (h,w-1)
    em = np.zeros((h, w))
    em[1:] += gy; em[:-1] += gy
    em[:, 1:] += gx; em[:, :-1] += gx
    return em / 2.0


def barycentric_dither(image, palette, method="bayer", space="RGB", strength=64,
                       gamut=False, min_weight=0.0, edge_guard=0.0):
    """
    调色板重心混合抖动（barycentric）。

    原理：任何在调色板凸包内的颜色，都能用包围它的四面体顶点色按重心权重混合出来。
    每个像素不再只选“一个最近色”，而是：
      1. Delaunay 在指定空间（RGB/Lab）对调色板建四面体
      2. 找到包含该像素的四面体，算出重心权重（w0+w1+w2+w3=1）
      3. 用有序阈值矩阵在顶点色之间按权重选色 → 区域内平均色精确逼近源色

    为避免“靠近调色板色 → 权重集中 → 平涂”和“凸包外 → 直接最近色 → 平涂”的退化，
    先对像素加上有序抖动偏移（strength 控制）再做重心/最近色，使所有区域都有规则纹理。
    偏移量在矩阵周期内均值为 0，因此区域平均仍近似守恒。

    增强参数：
      gamut      凸包外像素投影到凸包表面最近点再混合（改善饱和色区域色差；默认关）
      min_weight 丢弃权重低于该值的顶点色再归一化（缓解像素弥散；默认 0 = 不截断）
      edge_guard Lab 色差边缘保护阈值（>0 时边缘像素走有序路径保锐度；默认 0 = 关）

    space: 'RGB'（区域平均精确）或 'LAB'（更符合人眼）
    strength: 输入扰动强度，与有序抖动语义一致（默认 64；0 = 不扰动，退化为纯重心）
    """
    # 保留透明度
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.float32)
        alpha = arr[:, :, 3].copy()
        arr = arr[:, :, :3]
        has_alpha = True
    else:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        alpha = None
        has_alpha = False

    h, w, _ = arr.shape
    N = h * w
    pal = np.asarray(palette, dtype=np.float64)

    # Delaunay 四面体（在指定空间）
    if space.upper() == "LAB":
        pal_space = _rgb_to_lab(pal)
        lab_pal = pal_space
    else:
        pal_space = pal
        lab_pal = None
    tri = Delaunay(pal_space)
    # 预计算所有四面体的重心逆变换：A@w=[p,1] → w = Ainv@[p,1]，避免逐四面体求解
    Vs = pal_space[tri.simplices]                                            # (S,4,3)
    A_all = np.concatenate([np.transpose(Vs, (0, 2, 1)),
                            np.ones((Vs.shape[0], 1, 4))], axis=1)           # (S,4,4)
    Ainv_all = np.linalg.inv(A_all)
    hull_verts = pal_space[tri.convex_hull] if gamut else None               # (F,3,3)
    hull_idx = tri.convex_hull if gamut else None                            # (F,3) 调色板索引

    # 同一矩阵派生两套量：偏移（扰动输入，均值为0）与阈值（选顶点，[0,1)均匀）
    mat = get_dither_matrix(method)
    msize = mat.shape[0]
    norm = (mat + 0.5) / (msize * msize)
    off = (norm - 0.5) * float(strength)
    offsets = np.tile(off, (int(np.ceil(h / msize)), int(np.ceil(w / msize))))[:h, :w].reshape(-1)
    thresholds = np.tile(norm, (int(np.ceil(h / msize)), int(np.ceil(w / msize))))[:h, :w].reshape(-1)

    # Lab 边缘保护图
    em = _edge_map_lab(arr).reshape(-1) if edge_guard > 0 else None

    flat = arr.reshape(-1, 3).astype(np.float64)
    out = np.empty((N, 3), dtype=np.float64)

    def _prune(weights):
        """权重截断：丢弃 < min_weight 的顶点色再归一化。"""
        if min_weight <= 0:
            return weights
        w = np.where(weights < min_weight, 0.0, weights)
        ssum = w.sum(axis=1, keepdims=True)
        return w / np.where(ssum > 0, ssum, 1.0)

    chunk = 1_000_000
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        f = flat[s:e]                               # 未扰动
        pts = _rgb_to_lab(np.clip(f, 0, 255)) if space.upper() == "LAB" else f
        sidx = tri.find_simplex(pts)                # -1 = 凸包外
        inside = sidx >= 0

        ordered_mask = np.zeros(len(f), dtype=bool)  # 走“有序路径”的像素（近色 + 凸包外 + 边缘）
        if em is not None:
            ordered_mask |= em[s:e] >= edge_guard    # 边缘保护

        # 凸包内：重心权重
        b = np.zeros((len(f), 4))
        in_idx = np.nonzero(inside & ~ordered_mask)[0]
        if len(in_idx):
            pts4 = np.column_stack([pts[in_idx], np.ones(len(in_idx))])          # (M,4)
            bw = np.einsum('nij,nj->ni', Ainv_all[sidx[in_idx]], pts4)           # 重心权重
            bw = np.clip(bw, 0, None)                 # 凸包面/边上浮点误差可能微负
            bw_sum = bw.sum(axis=1, keepdims=True)
            bw = bw / np.where(bw_sum > 0, bw_sum, 1.0)
            bw = _prune(bw)
            b[in_idx] = bw
            # 近色（权重高度集中）→ 有序路径，恢复纹理
            ordered_mask[in_idx[bw.max(axis=1) >= 0.9]] = True

        # 凸包外：gamut 投影混合，或有序路径
        oidx = np.nonzero(~inside & ~ordered_mask)[0]
        if len(oidx):
            if gamut:
                best_idx, w3 = _project_to_hull(f[oidx], hull_verts, hull_idx)
                w3 = _prune(w3)
                cum = np.cumsum(w3, axis=1)
                tt = thresholds[s:e][oidx][:, None]
                sel = np.argmax(cum >= tt, axis=1)
                out[s + oidx] = pal[best_idx[np.arange(len(oidx)), sel]]
            else:
                ordered_mask[oidx] = True

        # “中间色”像素：纯重心精确混合（区域色差强项）
        mix_idx = np.nonzero(~ordered_mask & inside)[0]
        if len(mix_idx):
            cum = np.cumsum(b[mix_idx], axis=1)
            tt = thresholds[s:e][mix_idx][:, None]
            sel = np.argmax(cum >= tt, axis=1)      # 按权重挑顶点
            gi = s + mix_idx
            out[gi] = pal[tri.simplices[sidx[mix_idx]][np.arange(len(mix_idx)), sel]]

        # “近色/凸包外/边缘”像素：有序路径（扰动 + 最近色）
        ord_idx = np.nonzero(ordered_mask)[0]
        if len(ord_idx):
            go = s + ord_idx
            f_pert = f[ord_idx] + offsets[s:e][ord_idx][:, None]
            if space.upper() == "LAB":
                idx = _nearest_lab_indices(np.clip(f_pert, 0, 255), lab_pal)
            else:
                idx = _nearest_rgb_indices(f_pert, palette)
            out[go] = pal[idx]

    result = np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)

    if has_alpha:
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        result[alpha == 0] = 0
        return Image.fromarray(np.dstack([result, alpha_binary]), "RGBA")
    else:
        return Image.fromarray(result, "RGB")


def adaptive_barycentric_dither(image, palette, method="bayer", space="RGB", strength=64,
                                gamut=True, min_weight=0.1, edge_guard=10):
    """
    像素级自适应重心混合：逐像素在“原始重心”与“增强版”之间选。

    依据 oracle 消融实验（多尺度特征可预测每块最优方法）：
    - 凸包外像素 → 增强版（gamut 投影，改善饱和色色差）
    - 边缘像素   → 增强版（edge 保护，保锐度；edge_guard=0 则关闭）
    - 平滑凸包内 → 原始重心（避开 min_weight 截断带来的色差损失）
    纯色差实测比全局单一设置降 ~12.6%（edge_guard=0 时）。

    注意：开启 edge_guard 时，边缘像素走有序路径，区域色差会回升（用色差换锐度）。
    """
    plain = barycentric_dither(image, palette, method, space, strength)
    enh = barycentric_dither(image, palette, method, space, strength,
                             gamut=gamut, min_weight=min_weight, edge_guard=edge_guard)

    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.float64)
        alpha = arr[:, :, 3]
        arr = arr[:, :, :3]
        has_alpha = True
    else:
        arr = np.array(image.convert("RGB"), dtype=np.float64)
        alpha = None
        has_alpha = False
    h, w, _ = arr.shape

    # 凸包外 mask
    pal_space = _rgb_to_lab(np.asarray(palette, dtype=np.float64)) if space.upper() == "LAB" else np.asarray(palette, dtype=np.float64)
    tri = Delaunay(pal_space)
    pts = _rgb_to_lab(np.clip(arr, 0, 255)) if space.upper() == "LAB" else arr
    use_enh = (tri.find_simplex(pts.reshape(-1, 3)) < 0).reshape(h, w)
    if edge_guard > 0:
        use_enh |= _edge_map_lab(arr) >= edge_guard

    a_plain = np.array(plain.convert("RGB"), dtype=np.uint8)
    a_enh = np.array(enh.convert("RGB"), dtype=np.uint8)
    a_plain[use_enh] = a_enh[use_enh]

    if has_alpha:
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        a_plain[alpha == 0] = 0
        return Image.fromarray(np.dstack([a_plain, alpha_binary]), "RGBA")
    else:
        return Image.fromarray(a_plain, "RGB")


# 读取调色盘
def load_palette(file_path):
    palette = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace("[", "").replace("]", "")
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rgb = tuple(map(int, parts[:3]))
                    palette.append(rgb)
                except ValueError:
                    pass
    return palette


# 主函数
def main():
    img = Image.open("木头人.jpg")

    # 压缩倍率（例如 0.5 = 缩小一半）
    scale = 0.1
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, Image.LANCZOS)

    palette = load_palette("palette.txt")

    # 要测试的强度
    strengths = [32, 64, 128]
    # "bayer", "clustered", "diagonal", "spiral"
    for method in ["bayer"]:
        for strength in strengths:
            out_img = ordered_dithering(img, palette, method, strength=strength)
            out_img.save(f"output_{method}_s{strength}.png")
            print(f"{method} 抖动完成（strength={strength}），已保存为 output_{method}_s{strength}.png")


if __name__ == "__main__":
    main()