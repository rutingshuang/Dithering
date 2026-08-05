import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import argparse
import os


def load_palette(palette_file):
    """从 palette.txt 读取调色盘，返回 Nx3 的 numpy 数组"""
    palette = []
    with open(palette_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            name, rgb_str = line.strip().split(":")
            rgb = rgb_str.strip().strip("[]").split(",")
            rgb = [int(x) for x in rgb]
            palette.append(rgb)
    return np.array(palette, dtype=np.float32)


def bayer_4x4():
    """返回 Bayer 4x4 阈值矩阵"""
    bayer = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)
    return bayer


def ordered_dither(img_tensor, threshold_matrix, palette, temp=50.0, train=True, strength=64.0):
    """
    修改后的有序抖动 + 调色盘映射，更接近 dithering.py 的实现
    img_tensor: (3,H,W), 归一化到[0,1]
    threshold_matrix: (n,n), 原始的阈值矩阵（未归一化）
    palette: (K,3), 0-255
    strength: 抖动强度，类似 dithering.py 中的 strength 参数
    """
    C, H, W = img_tensor.shape
    n = threshold_matrix.shape[0]

    # 将图像从 [0,1] 转换到 [0,255] 范围，与 dithering.py 一致
    img_255 = img_tensor * 255.0

    # 归一化阈值矩阵到 [-0.5, 0.5]，再乘以强度（类似 dithering.py）
    threshold_normalized = ((threshold_matrix + 0.5) / (n * n) - 0.5) * strength

    # 扩展矩阵以覆盖整张图
    tiled_matrix = threshold_normalized.repeat(H // n + 1, W // n + 1)
    tiled_matrix = tiled_matrix[:H, :W].to(img_tensor.device)

    # 加入阈值扰动（与 dithering.py 一致的方式）
    img_mod = img_255 + tiled_matrix.unsqueeze(0)  # 广播到 3 通道

    # 将图像限制在 [0, 255] 范围内
    img_mod = torch.clamp(img_mod, 0, 255)

    # 调色盘 tensor (K,3)，保持在 [0,255] 范围
    palette_t = torch.tensor(palette, device=img_tensor.device, dtype=torch.float32)

    if train:
        # Soft 最近邻，用 softmin 近似（保持可微性）
        img_flat = img_mod.permute(1, 2, 0).reshape(-1, 3)  # (HW,3)
        dist = torch.cdist(img_flat, palette_t)  # (HW,K)
        weights = torch.softmax(-dist * temp, dim=1)  # soft assignment
        output_flat = weights @ palette_t  # (HW,3)
        output = output_flat.reshape(H, W, 3).permute(2, 0, 1)  # (3,H,W)

        # 转换回 [0,1] 范围
        output = output / 255.0
    else:
        # 硬最近邻（推理时使用）
        img_flat = img_mod.permute(1, 2, 0).reshape(-1, 3)
        dist = torch.cdist(img_flat, palette_t)
        indices = dist.argmin(dim=1)
        output_flat = palette_t[indices]
        output = output_flat.reshape(H, W, 3).permute(2, 0, 1)

        # 转换回 [0,1] 范围
        output = output / 255.0

    return output


def perceptual_loss(original, dithered, palette):
    """修正后的感知损失函数，处理多通道图像"""
    C, H, W = original.shape

    # 1. 将调色盘转换为tensor
    palette_t = torch.tensor(palette / 255.0, device=original.device, dtype=torch.float32)

    # 2. 找到每个像素最接近的调色盘颜色（作为目标）
    original_flat = original.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    dist = torch.cdist(original_flat * 255.0, palette_t)  # 注意：调色盘是0-255范围
    nearest_indices = dist.argmin(dim=1)
    target_flat = palette_t[nearest_indices]  # 最接近的调色盘颜色
    target = target_flat.reshape(H, W, 3).permute(2, 0, 1)  # 恢复形状 (3, H, W)

    # 3. 颜色一致性损失 - 确保抖动结果接近目标颜色
    color_loss = (dithered - target).pow(2).mean()

    # 4. 纹理多样性损失 - 鼓励适当的局部变化
    # 对每个通道分别计算局部方差，然后取平均
    kernel = torch.ones(1, 1, 3, 3, device=dithered.device) / 9.0

    # 为每个通道计算局部方差
    channel_variances = []
    for c in range(C):
        channel = dithered[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        local_mean = torch.nn.functional.conv2d(channel, kernel, padding=1)
        local_var = (channel - local_mean).pow(2).mean()
        channel_variances.append(local_var)

    local_var = torch.stack(channel_variances).mean()

    # 我们希望局部方差适中
    target_var = 0.01  # 适中的方差目标
    texture_loss = (local_var - target_var).pow(2)

    return color_loss + 0.1 * texture_loss


def train_dither(img_path, palette_file, output_dir, epochs=200, lr=0.05, c=0.1, strength=64.0):
    # 打开图片，不缩放，转为 [0,1]
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    # 调色盘
    palette = load_palette(palette_file)

    # 初始化阈值矩阵
    init_matrix = bayer_4x4()
    threshold_matrix = torch.tensor(init_matrix, requires_grad=True, device=img_tensor.device)

    # 优化器
    optimizer = optim.Adam([threshold_matrix], lr=lr)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        output = ordered_dither(img_tensor, threshold_matrix, palette, train=True, strength=strength)

        # 使用修正的感知损失
        diff = perceptual_loss(img_tensor, output, palette)
        var = output.var()
        loss = diff + c * var

        loss.backward()
        optimizer.step()

        # 每 50 epoch 保存一次结果
        if epoch % 50 == 0 or epoch == epochs:
            with torch.no_grad():
                out_img = ordered_dither(img_tensor, threshold_matrix, palette, train=False, strength=strength)
                out_img = (out_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(out_img).save(os.path.join(output_dir, f"epoch_{epoch}.png"))
            print(f"Epoch {epoch}/{epochs}, Loss={loss.item():.6f}")

    print("优化后的阈值矩阵：")
    print(threshold_matrix.detach().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--palette", type=str, required=True, help="调色盘文件路径 palette.txt")
    parser.add_argument("--output", type=str, default="results", help="输出文件夹")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.05, help="学习率")
    parser.add_argument("--c", type=float, default=0.1, help="方差项权重")
    parser.add_argument("--strength", type=float, default=64.0, help="抖动强度")

    args = parser.parse_args()

    train_dither(args.image, args.palette, args.output, args.epochs, args.lr, args.c, args.strength)