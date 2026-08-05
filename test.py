import numpy as np
from PIL import Image
import colorsys

# 定义几种常见的有序抖动矩阵
def bayer_matrix(size=8):
    if size == 2:
        return np.array([[0, 2],
                         [3, 1]])
    elif size == 4:
        return np.array([[ 0,  8,  2, 10],
                         [12,  4, 14,  6],
                         [ 3, 11,  1,  9],
                         [15,  7, 13,  5]])
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
        return np.array([[14, 11,  8, 13],
                         [10,  6,  4,  9],
                         [ 7,  2,  0,  5],
                         [12,  3,  1, 15]])
    elif size == 8:
        return np.array([[62, 57, 48, 36, 37, 49, 58, 63],
                         [56, 47, 35, 21, 22, 38, 50, 59],
                         [46, 34, 20, 10, 11, 23, 39, 51],
                         [33, 19,  9,  3,  0,  4, 12, 24],
                         [32, 18,  8,  2,  1,  5, 13, 25],
                         [45, 31, 17,  7,  6, 14, 26, 40],
                         [55, 44, 30, 16, 15, 27, 41, 52],
                         [61, 54, 43, 29, 28, 42, 53, 60]])
    else:
        raise ValueError("点簇矩阵格式不支持")

def diagonal_matrix(size=4):
    return np.array([[15, 11,  5,  0],
                     [10, 14,  9,  4],
                     [ 3,  8, 13,  7],
                     [ 1,  2,  9, 12]])

def spiral_matrix(size=4):
    return np.array([[ 9, 1, 6, 14],
                     [3, 11, 15, 4],
                     [5, 7, 0, 8],
                     [ 12, 13, 10, 2]])

# 根据方法获取矩阵
def get_dither_matrix(method):
    if method == "bayer":
        return bayer_matrix(8)
    elif method == "clustered":
        return clustered_dot_matrix(4)
    elif method == "diagonal":
        return diagonal_matrix(4)
    elif method == "spiral":
        return spiral_matrix(4)
    else:
        raise ValueError("未知的抖动方法: " + method)

# 找到调色盘中最接近的颜色
def find_nearest_color(color, palette):
    r, g, b = color
    palette = np.array(palette)
    distances = np.sum((palette - np.array([r, g, b]))**2, axis=1)
    return tuple(palette[np.argmin(distances)])


# 有序抖动函数（支持透明通道保留、色彩空间选择和加权参数）
# def ordered_dithering(image, palette, method="bayer", strength=64, color_space="RGB", weights=(1.0, 1.0, 1.0)):
#     """
#     对RGB通道做有序抖动，透明通道不处理
#     只要透明度 > 0 就视为完全不透明 (alpha=255)
#     透明像素(alpha=0)保持透明
#
#     支持在RGB或HSV色彩空间中进行颜色比较
#     支持HSV空间中的加权参数
#     """
#     # 保留透明度
#     if image.mode == "RGBA":
#         arr_rgb = np.array(image, dtype=np.float32)
#         alpha = arr_rgb[:, :, 3].copy()
#         arr_rgb = arr_rgb[:, :, :3]  # 只取RGB
#         has_alpha = True
#     else:
#         arr_rgb = np.array(image.convert("RGB"), dtype=np.float32)
#         alpha = None
#         has_alpha = False
#
#     h, w, _ = arr_rgb.shape
#
#     # 获取矩阵
#     matrix = get_dither_matrix(method)
#     msize = matrix.shape[0]
#
#     # 归一化到 [-0.5, 0.5]，再乘以强度
#     matrix = ((matrix + 0.5) / (msize * msize) - 0.5) * strength
#
#     # 根据色彩空间准备调色盘
#     if color_space.upper() == "HSV":
#         # 将RGB调色盘转换为HSV (0-1范围)
#         hsv_palette = []
#         for rgb in palette:
#             r, g, b = [x / 255.0 for x in rgb]
#             h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
#             hsv_palette.append((h_val, s_val, v_val))
#         hsv_palette = np.array(hsv_palette)
#
#         # 将RGB图像转换为HSV (0-1范围)
#         arr_hsv = np.zeros_like(arr_rgb)
#         for y in range(h):
#             for x in range(w):
#                 r, g, b = arr_rgb[y, x] / 255.0
#                 arr_hsv[y, x] = colorsys.rgb_to_hsv(r, g, b)
#     else:
#         # 默认使用RGB色彩空间
#         hsv_palette = None
#         arr_hsv = None
#
#     out = np.zeros_like(arr_rgb)
#
#     for y in range(h):
#         for x in range(w):
#             if has_alpha and alpha[y, x] == 0:
#                 # 完全透明像素 → 保持透明
#                 out[y, x] = (0, 0, 0)
#                 continue
#
#             # 获取抖动偏移量
#             offset = matrix[y % msize, x % msize]
#
#             if color_space.upper() == "HSV" and hsv_palette is not None:
#                 # 在HSV空间中进行处理
#                 old_pixel_hsv = arr_hsv[y, x] + offset / 255.0  # 缩放偏移量到0-1范围
#                 old_pixel_hsv = np.clip(old_pixel_hsv, 0, 1)
#
#                 # 在HSV空间中找到最接近的颜色
#                 min_dist = float('inf')
#                 best_idx = 0
#                 for i, hsv_color in enumerate(hsv_palette):
#                     # 计算HSV空间中的加权距离（注意H是循环的）
#                     h_dist = min(abs(old_pixel_hsv[0] - hsv_color[0]),
#                                  1 - abs(old_pixel_hsv[0] - hsv_color[0]))
#                     s_dist = abs(old_pixel_hsv[1] - hsv_color[1])
#                     v_dist = abs(old_pixel_hsv[2] - hsv_color[2])
#
#                     # 应用权重
#                     dist = h_dist**2 * weights[0] + s_dist**2 * weights[1] + v_dist**2 * weights[2]
#
#                     if dist < min_dist:
#                         min_dist = dist
#                         best_idx = i
#
#                 # 使用原始RGB调色盘中的颜色
#                 new_pixel = palette[best_idx]
#             else:
#                 # 在RGB空间中找到最接近的颜色
#                 old_pixel = arr_rgb[y, x] + offset
#                 new_pixel = find_nearest_color(old_pixel, palette)
#
#             out[y, x] = new_pixel
#
#     out = np.clip(out, 0, 255).astype(np.uint8)
#
#     # 如果有透明度 → 非零透明度设为255
#     if has_alpha:
#         # 创建二值化的alpha通道（0或255）
#         alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
#         rgba = np.dstack([out, alpha_binary])
#         return Image.fromarray(rgba, "RGBA")
#     else:
#         return Image.fromarray(out, "RGB")

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

    h, w, _ = arr_rgb.shape

    # 获取矩阵
    matrix = get_dither_matrix(method)
    msize = matrix.shape[0]

    # 归一化到 [-0.5, 0.5]，再乘以强度
    matrix = ((matrix + 0.5) / (msize * msize) - 0.5) * strength

    # 根据色彩空间准备调色盘
    if color_space.upper() == "HSV":
        # 将RGB调色盘转换为HSV (0-1范围)
        hsv_palette = []
        for rgb in palette:
            r, g, b = [x / 255.0 for x in rgb]
            h_val, s_val, v_val = colorsys.rgb_to_hsv(r, g, b)
            hsv_palette.append((h_val, s_val, v_val))
        hsv_palette = np.array(hsv_palette)

        # 将RGB图像转换为HSV (0-1范围)
        arr_hsv = np.zeros_like(arr_rgb)
        for y in range(h):
            for x in range(w):
                r, g, b = arr_rgb[y, x] / 255.0
                arr_hsv[y, x] = colorsys.rgb_to_hsv(r, g, b)
    else:
        # 默认使用RGB色彩空间
        hsv_palette = None
        arr_hsv = None

    out = np.zeros_like(arr_rgb)

    for y in range(h):
        for x in range(w):
            if has_alpha and alpha[y, x] == 0:
                # 完全透明像素 → 保持透明
                out[y, x] = (0, 0, 0)
                continue

            # 获取抖动偏移量
            offset = matrix[y % msize, x % msize]

            if color_space.upper() == "HSV" and hsv_palette is not None:
                # 在HSV空间中进行处理
                old_pixel_hsv = arr_hsv[y, x] + offset / 255.0  # 缩放偏移量到0-1范围
                old_pixel_hsv = np.clip(old_pixel_hsv, 0, 1)

                # 在HSV空间中找到最接近的颜色
                min_dist = float('inf')
                best_idx = 0
                for i, hsv_color in enumerate(hsv_palette):
                    # 计算HSV空间中的加权距离（注意H是循环的）
                    h_dist = min(abs(old_pixel_hsv[0] - hsv_color[0]),
                                 1 - abs(old_pixel_hsv[0] - hsv_color[0]))
                    s_dist = abs(old_pixel_hsv[1] - hsv_color[1])
                    v_dist = abs(old_pixel_hsv[2] - hsv_color[2])

                    # 应用权重
                    dist = h_dist**2 * weights[0] + s_dist**2 * weights[1] + v_dist**2 * weights[2]

                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i

                # 使用原始RGB调色盘中的颜色
                new_pixel = palette[best_idx]
            else:
                # 在RGB空间中找到最接近的颜色
                old_pixel = arr_rgb[y, x] + offset
                new_pixel = find_nearest_color(old_pixel, palette)
                arr_rgb[y, x] = new_pixel - arr_rgb[y, x]
                quant_error = (old_pixel - new_pixel)

                # 传播误差
                if x + 4 < w:
                    arr_rgb[y, x + 4] += quant_error * 7 / 16
                if y + 4 < h:
                    if x - 4 >= 0:
                        arr_rgb[y + 4, x - 4] += quant_error * 3 / 16
                    arr_rgb[y + 1, x] += quant_error * 5 / 16
                    if x + 4 < w:
                        arr_rgb[y + 4, x + 4] += quant_error * 1 / 16

            out[y, x] = new_pixel

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 如果有透明度 → 非零透明度设为255
    if has_alpha:
        # 创建二值化的alpha通道（0或255）
        alpha_binary = np.where(alpha < 128, 0, 255).astype(np.uint8)
        rgba = np.dstack([out, alpha_binary])
        return Image.fromarray(rgba, "RGBA")
    else:
        return Image.fromarray(out, "RGB")

import numpy as np
from PIL import Image

def floyd_steinberg_dither(image, palette, alpha_strength=1.0, filter_type="Jarvis"):
    """
    支持三种误差扩散：
    - FS: Floyd–Steinberg
    - Jarvis: Jarvis, Judice & Ninke
    - Stucki: Stucki
    """
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.float32)
        alpha_channel = arr[:, :, 3].copy()
        arr = arr[:, :, :3]
    else:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        alpha_channel = None

    h, w, _ = arr.shape

    # 选择权重矩阵
    if filter_type == "FS":
        # Floyd–Steinberg
        weights = [
            ((0, 1), 7/16),
            ((1, -1), 3/16),
            ((1, 0), 5/16),
            ((1, 1), 1/16)
        ]
    elif filter_type == "Jarvis":
        weights = [
            ((0, 1), 7/48), ((0, 2), 5/48),
            ((1, -2), 3/48), ((1, -1), 5/48), ((1, 0), 7/48), ((1, 1), 5/48), ((1, 2), 3/48),
            ((2, -2), 1/48), ((2, -1), 3/48), ((2, 0), 5/48), ((2, 1), 3/48), ((2, 2), 1/48)
        ]
    elif filter_type == "Stucki":
        weights = [
            ((0, 1), 8/42), ((0, 2), 4/42),
            ((1, -2), 2/42), ((1, -1), 4/42), ((1, 0), 8/42), ((1, 1), 4/42), ((1, 2), 2/42),
            ((2, -2), 1/42), ((2, -1), 2/42), ((2, 0), 4/42), ((2, 1), 2/42), ((2, 2), 1/42)
        ]
    else:
        raise ValueError("filter_type must be 'FS', 'Jarvis' or 'Stucki'")

    for y in range(h):
        for x in range(w):
            if alpha_channel is not None and alpha_channel[y, x] == 0:
                continue

            old_pixel = arr[y, x].copy()
            new_pixel = find_nearest_color(old_pixel, palette)
            arr[y, x] = new_pixel
            quant_error = (old_pixel - new_pixel) * alpha_strength

            # 传播误差
            for (dy, dx), wgt in weights:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    arr[ny, nx] += quant_error * wgt

    out = np.clip(arr, 0, 255).astype(np.uint8)

    if alpha_channel is not None:
        alpha_binary = np.where(alpha_channel < 128, 0, 255).astype(np.uint8)
        rgba = np.dstack([out, alpha_binary])
        return Image.fromarray(rgba, "RGBA")
    else:
        return Image.fromarray(out, "RGB")

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
