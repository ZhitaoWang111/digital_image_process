#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_landscape.m 的 Python 等价实现（局部四分区优化）。

默认读取：
  part_sky_0.jpg / part_dam_0.jpg / part_water_0.jpg / part_slope_0.jpg

默认输出：
  optimized_sky.jpg / optimized_dam.jpg / optimized_water.jpg / optimized_slope.jpg

依赖：
  numpy, opencv-python, imageio, scipy
（不依赖 skimage）

运行：
  python optimize_landscape.py               # 默认 workdir=当前目录
  python optimize_landscape.py --workdir .   # 指定目录
  python optimize_landscape.py --preview     # 预览（需要 matplotlib）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
import imageio.v2 as imageio
from scipy import ndimage as ndi


# -------------------------
# 基础 I/O 与工具
# -------------------------
def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        out = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        out = img.astype(np.float32) / 65535.0
    else:
        out = img.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def imread_rgb(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    return img


def imwrite_rgb(path: Path, img01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, _to_uint8(img01))


def ensure_rgb(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        return np.stack([img01, img01, img01], axis=-1)
    if img01.ndim == 3 and img01.shape[2] == 1:
        return np.repeat(img01, 3, axis=2)
    return img01[:, :, :3]


def rgb2gray(img01: np.ndarray) -> np.ndarray:
    img01 = ensure_rgb(img01)
    # OpenCV expects RGB? We'll compute in float:
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def imsharpen(img01: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    img01 = np.clip(img01.astype(np.float32), 0.0, 1.0)
    blurred = gaussian_blur(img01, sigma=max(0.0, float(radius)))
    detail = img01 - blurred
    if threshold and threshold > 0:
        m = (np.abs(detail) > float(threshold)).astype(np.float32)
        detail = detail * m
    out = img01 + float(amount) * detail
    return np.clip(out, 0.0, 1.0)


# -------------------------
# RGB <-> LAB（近似 MATLAB rgb2lab/lab2rgb）
# -------------------------
def rgb_to_lab_matlab(rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 MATLAB 风格：L∈[0,100], a,b≈[-128,127]."""
    rgb_u8 = _to_uint8(rgb01)
    lab_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab_u8[..., 0] * (100.0 / 255.0)
    a = lab_u8[..., 1] - 128.0
    b = lab_u8[..., 2] - 128.0
    return L, a, b


def lab_matlab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lab_u8 = np.empty((*L.shape, 3), dtype=np.float32)
    lab_u8[..., 0] = np.clip(L * (255.0 / 100.0), 0.0, 255.0)
    lab_u8[..., 1] = np.clip(a + 128.0, 0.0, 255.0)
    lab_u8[..., 2] = np.clip(b + 128.0, 0.0, 255.0)
    lab_u8 = lab_u8.astype(np.uint8)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


# -------------------------
# CLAHE（近似 MATLAB adapthisteq）
# -------------------------
def clahe_on_L01(L01: np.ndarray, num_tiles: Tuple[int, int], clip_limit_matlab: float) -> np.ndarray:
    """
    MATLAB adapthisteq(L,'NumTiles',[m n],'ClipLimit',x) 的近似。
    OpenCV 的 clipLimit 尺度不同，这里用经验映射。
    """
    L01 = np.clip(L01.astype(np.float32), 0.0, 1.0)
    L_u8 = _to_uint8(L01)

    # 经验映射：0.005 -> ~2.0, 0.02 -> ~4.0
    if clip_limit_matlab <= 0.006:
        clip_cv = 2.0
    elif clip_limit_matlab <= 0.021:
        clip_cv = 4.0
    else:
        clip_cv = max(2.0, float(clip_limit_matlab) * 200.0)

    clahe = cv2.createCLAHE(clipLimit=float(clip_cv), tileGridSize=(int(num_tiles[1]), int(num_tiles[0])))
    out_u8 = clahe.apply(L_u8)
    return out_u8.astype(np.float32) / 255.0


# -------------------------
# 去雾（替代 imreducehaze）
# -------------------------
def guided_filter(gray_I: np.ndarray, p: np.ndarray, radius: int = 5, eps: float = 1e-3) -> np.ndarray:
    I = gray_I.astype(np.float32)
    P = p.astype(np.float32)
    r = int(max(1, radius))
    k = 2 * r + 1

    mean_I = cv2.blur(I, (k, k))
    mean_P = cv2.blur(P, (k, k))
    mean_IP = cv2.blur(I * P, (k, k))
    cov_IP = mean_IP - mean_I * mean_P

    mean_II = cv2.blur(I * I, (k, k))
    var_I = mean_II - mean_I * mean_I

    a = cov_IP / (var_I + float(eps))
    b = mean_P - a * mean_I

    mean_a = cv2.blur(a, (k, k))
    mean_b = cv2.blur(b, (k, k))

    q = mean_a * I + mean_b
    return q.astype(np.float32)


def _min_filter(img: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel, borderType=cv2.BORDER_REFLECT)


def reduce_haze_dcp(img01: np.ndarray, amount: float = 0.95, patch: int = 15, t0: float = 0.08) -> np.ndarray:
    """暗通道先验去雾（近似 imreducehaze）。"""
    I = ensure_rgb(_to_float01(img01))
    dark = np.min(I, axis=2)
    dark = _min_filter(dark, int(max(3, patch)))

    flat = dark.reshape(-1)
    topk = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -topk)[-topk:]
    A = I.reshape(-1, 3)[idx].mean(axis=0)

    omega = float(np.clip(amount, 0.0, 0.99))
    I_div = I / np.maximum(A.reshape(1, 1, 3), 1e-6)
    dark_div = np.min(I_div, axis=2)
    dark_div = _min_filter(dark_div, int(max(3, patch)))
    t = 1.0 - omega * dark_div
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    gray = rgb2gray(I)
    t_ref = guided_filter(gray, t, radius=5, eps=1e-3)
    t_ref = np.clip(t_ref, t0, 1.0)

    J = (I - A.reshape(1, 1, 3)) / t_ref[:, :, None] + A.reshape(1, 1, 3)
    return np.clip(J, 0.0, 1.0).astype(np.float32)


# -------------------------
# 四分区处理（对应 MATLAB: process_*）
# -------------------------
def process_dam(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    # 1) 去雾
    try:
        I = reduce_haze_dcp(I, amount=0.95, patch=15)
    except Exception:
        # 退化方案：拉伸 + gamma
        lo = float(np.quantile(I, 0.0))
        hi = float(np.quantile(I, 1.0))
        I = np.clip((I - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        I = np.clip(I ** 1.4, 0.0, 1.0)

    # 2) 空间权重（越靠上越强）
    rows, cols = I.shape[:2]
    Y = (np.arange(rows, dtype=np.float32).reshape(-1, 1) + 1.0)  # 1..rows
    spatial_weight = 1.0 - (Y / float(rows))
    spatial_weight = spatial_weight ** 2

    # HSV（float 输入会输出 H[0..360], S/V[0..1]）
    hsv = cv2.cvtColor(I.astype(np.float32), cv2.COLOR_RGB2HSV)
    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    # 3) 发白区域：亮度高 & 饱和度低
    mask_pale = (V > 0.35) & (S < 0.4)

    # 4) 结合位置信息
    final_mask = mask_pale.astype(np.float32) * (spatial_weight * 0.8 + 0.2)

    # 5) 压暗（Gamma）
    V_dark = np.clip(V ** 2.0, 0.0, 1.0)
    V_new = V * (1.0 - final_mask) + V_dark * final_mask

    # 6) 增饱和
    S_new = S.copy()
    S_new[mask_pale] = np.clip(S[mask_pale] * 1.5 + 0.1, 0.0, 1.0)

    # 7) 纹理增强：base + detail
    try:
        base = guided_filter(V_new, V_new, radius=5, eps=1e-3)
    except Exception:
        base = gaussian_blur(V_new, 2.0)
    detail = V_new - base
    V_final = np.clip(base + detail * 3.5, 0.0, 1.0)

    hsv[..., 2] = V_final
    hsv[..., 1] = S_new
    out_rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)

    # 8) 锐化
    out = imsharpen(out_rgb, radius=2.0, amount=1.5, threshold=0.0)
    return np.clip(out, 0.0, 1.0)


def process_sky(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    L, a, b = rgb_to_lab_matlab(I)
    L01 = np.clip(L / 100.0, 0.0, 1.0)

    # 增强蓝色：b<0 更蓝
    b2 = b.copy()
    mask_blue = b2 < 0
    b2[mask_blue] = b2[mask_blue] * 1.3

    # CLAHE 增强亮度层次（NumTiles [8 8], ClipLimit 0.005）
    L_enh = clahe_on_L01(L01, num_tiles=(8, 8), clip_limit_matlab=0.005)

    I_enh = lab_matlab_to_rgb(L_enh * 100.0, a, b2)

    # 模拟太阳光晕
    rows, cols = I.shape[:2]
    sun_x = int(round(cols * 0.2))
    sun_y = int(round(rows * 0.2))
    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float32),
                       np.arange(1, rows + 1, dtype=np.float32))
    dist_sq = (X - float(sun_x)) ** 2 + (Y - float(sun_y)) ** 2
    sigma = float(min(rows, cols)) * 0.4
    sun_glow = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)

    sun_color = np.array([1.0, 0.9, 0.6], dtype=np.float32).reshape(1, 1, 3)

    luminance_mask = np.max(I, axis=2).astype(np.float32)
    bloom_weight = 1.0 / (1.0 + np.exp(-10.0 * (luminance_mask - 0.3))).astype(np.float32)

    bloom = sun_glow[..., None] * 0.5 * sun_color * bloom_weight[..., None]
    out = np.clip(I_enh + bloom, 0.0, 1.0)
    return out


def process_water(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    # 偏蓝一点
    I2 = I.copy()
    I2[..., 0] *= 0.85
    I2[..., 1] *= 1.10
    I2[..., 2] *= 1.20
    I2 = np.clip(I2, 0.0, 1.0)

    blurred = gaussian_blur(I2, 2.0)
    high_freq = I2 - blurred
    I_ripples = np.clip(I2 + high_freq * 3.0, 0.0, 1.0)

    luminance = rgb2gray(I_ripples)
    mask_high = luminance > 0.6
    I_ripples[mask_high, :] = np.clip(I_ripples[mask_high, :] * 1.1, 0.0, 1.0)

    return np.clip(I_ripples, 0.0, 1.0)


def process_slope(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    L, a, b = rgb_to_lab_matlab(I)
    L01 = np.clip(L / 100.0, 0.0, 1.0)

    # CLAHE（NumTiles[16 16], ClipLimit 0.02）
    L_detail = clahe_on_L01(L01, num_tiles=(16, 16), clip_limit_matlab=0.02)
    L_new = np.clip(L01 * 0.3 + L_detail * 0.7, 0.0, 1.0)

    b2 = b * 1.15  # 稍暖

    out_rgb = lab_matlab_to_rgb(L_new * 100.0, a, b2)

    # 锐化
    out = imsharpen(out_rgb, radius=1.0, amount=1.5, threshold=0.02)
    return np.clip(out, 0.0, 1.0)


# -------------------------
# 顶层入口（对应 MATLAB optimize_landscape）
# -------------------------
def optimize_landscape(workdir: Path = Path("."), preview: bool = False) -> Dict[str, Path]:
    workdir = Path(workdir)

    imgs = {
        "sky": imread_rgb(workdir / "part_sky_0.jpg"),
        "dam": imread_rgb(workdir / "part_dam_0.jpg"),
        "water": imread_rgb(workdir / "part_water_0.jpg"),
        "slope": imread_rgb(workdir / "part_slope_0.jpg"),
    }

    out_sky = process_sky(imgs["sky"])
    out_dam = process_dam(imgs["dam"])
    out_water = process_water(imgs["water"])
    out_slope = process_slope(imgs["slope"])

    outputs = {
        "optimized_sky": workdir / "optimized_sky.jpg",
        "optimized_dam": workdir / "optimized_dam.jpg",
        "optimized_water": workdir / "optimized_water.jpg",
        "optimized_slope": workdir / "optimized_slope.jpg",
    }

    imwrite_rgb(outputs["optimized_sky"], out_sky)
    imwrite_rgb(outputs["optimized_dam"], out_dam)
    imwrite_rgb(outputs["optimized_water"], out_water)
    imwrite_rgb(outputs["optimized_slope"], out_slope)

    if preview:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.ravel()
        axs[0].imshow(out_sky); axs[0].set_title("天空"); axs[0].axis("off")
        axs[1].imshow(out_dam); axs[1].set_title("大坝"); axs[1].axis("off")
        axs[2].imshow(out_water); axs[2].set_title("水流"); axs[2].axis("off")
        axs[3].imshow(out_slope); axs[3].set_title("岸边"); axs[3].axis("off")
        plt.tight_layout()
        plt.show()

    return outputs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local optimization for landscape parts (MATLAB -> Python).")
    ap.add_argument("--workdir", type=str, default=".", help="图片所在目录（默认当前目录）")
    ap.add_argument("--preview", action="store_true", help="预览结果（需要 matplotlib）")
    args = ap.parse_args(argv)

    outs = optimize_landscape(Path(args.workdir), preview=args.preview)
    print("程序运行结束，图片已保存：")
    for k, v in outs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
