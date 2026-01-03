#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 optimize_landscape.m 与 global_landscape_correction.m 的主要功能完整迁移到 Python。

依赖：
  - numpy
  - imageio
  - opencv-python
  - scikit-image
  - scipy
（可选）matplotlib：用于 --preview 展示

用法示例：
  1) 先做局部优化（生成 optimized_*.jpg）：
     python landscape_pipeline.py optimize --workdir .

  2) 再做全局拼接与整体调色（生成 final_masterpiece.jpg）：
     python landscape_pipeline.py global --workdir .

默认文件名与 MATLAB 脚本一致：
  canvas: 9.jpg
  parts: part_sky_0.jpg / part_dam_0.jpg / part_water_0.jpg / part_slope_0.jpg
  outputs: optimized_sky.jpg / optimized_dam.jpg / optimized_water.jpg / optimized_slope.jpg / final_masterpiece.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import imageio.v2 as imageio
import cv2
from scipy import ndimage as ndi
from skimage import color, exposure


# -----------------------------
# I/O helpers
# -----------------------------
def _to_float01(img: np.ndarray) -> np.ndarray:
    """MATLAB im2double 对应：uint8 -> float [0,1]；float保持并裁剪。"""
    if img is None:
        raise ValueError("Empty image.")
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
    # imageio 读 jpg 通常是 RGB；如果是灰度则扩展为 (H,W)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    return img


def imwrite_rgb(path: Path, img01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, _to_uint8(img01))


# -----------------------------
# Basic image ops (MATLAB-like)
# -----------------------------
def ensure_rgb(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        return np.stack([img01, img01, img01], axis=-1)
    if img01.ndim == 3 and img01.shape[2] == 1:
        return np.repeat(img01, 3, axis=2)
    return img01[:, :, :3]


def rgb2gray(img01: np.ndarray) -> np.ndarray:
    img01 = ensure_rgb(img01)
    g = color.rgb2gray(img01)  # returns float [0,1]
    return g.astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """近似 imgaussfilt；sigma=0则返回原图。"""
    if sigma <= 0:
        return img
    # OpenCV 支持 (0,0) 自动核大小；对 float32 OK
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)


def disk_kernel(radius: int) -> np.ndarray:
    r = int(max(0, radius))
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask).astype(bool)


def imsharpen(img01: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    """
    近似 MATLAB imsharpen：
      detail = img - blur(img, radius)
      if threshold>0: 仅增强 |detail|>threshold 的像素
      out = img + amount*detail
    """
    img01 = np.clip(img01.astype(np.float32), 0.0, 1.0)
    blurred = gaussian_blur(img01, sigma=max(0.0, float(radius)))
    detail = img01 - blurred
    if threshold and threshold > 0:
        mask = np.abs(detail) > float(threshold)
        detail = detail * mask.astype(np.float32)
    out = img01 + float(amount) * detail
    return np.clip(out, 0.0, 1.0)


def guided_filter(gray_I: np.ndarray, p: np.ndarray, radius: int = 5, eps: float = 1e-3) -> np.ndarray:
    """
    简易 guided filter（单通道引导/输出），用于替代 imguidedfilter(V_new)。
    """
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


def stretchlim(img01: np.ndarray, low: float = 0.0, high: float = 1.0) -> Tuple[float, float]:
    """简化版 stretchlim：返回全局分位点范围。"""
    lo = np.quantile(img01, low)
    hi = np.quantile(img01, high)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def imadjust(img01: np.ndarray, in_range: Tuple[float, float], out_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """近似 MATLAB imadjust：线性拉伸到目标范围。"""
    lo, hi = in_range
    o0, o1 = out_range
    out = (img01 - lo) / (hi - lo)
    out = out * (o1 - o0) + o0
    return np.clip(out, min(o0, o1), max(o0, o1)).astype(np.float32)


# -----------------------------
# Dehaze (approx imreducehaze)
# -----------------------------
def _min_filter(img: np.ndarray, k: int) -> np.ndarray:
    """2D min filter via erosion."""
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel, borderType=cv2.BORDER_REFLECT)


def reduce_haze_dcp(img01: np.ndarray, amount: float = 0.95, patch: int = 15, t0: float = 0.08) -> np.ndarray:
    """
    Dark Channel Prior 去雾近似，作为 imreducehaze 的替代。
    amount ~ omega（越大去雾越强）。
    """
    I = ensure_rgb(_to_float01(img01))
    H, W, _ = I.shape

    # dark channel
    dark = np.min(I, axis=2)
    dark = _min_filter(dark, k=int(max(3, patch)))

    # airlight estimate (top 0.1% brightest in dark channel)
    flat_dark = dark.reshape(-1)
    n = flat_dark.size
    topk = max(1, int(n * 0.001))
    idxs = np.argpartition(flat_dark, -topk)[-topk:]
    flat_I = I.reshape(-1, 3)
    A = flat_I[idxs].mean(axis=0)  # (3,)

    # transmission
    omega = float(np.clip(amount, 0.0, 0.99))
    I_div = I / np.maximum(A.reshape(1, 1, 3), 1e-6)
    dark_div = np.min(I_div, axis=2)
    dark_div = _min_filter(dark_div, k=int(max(3, patch)))
    t = 1.0 - omega * dark_div
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    # refine transmission with guided filter
    gray = rgb2gray(I)
    t_ref = guided_filter(gray, t, radius=5, eps=1e-3)
    t_ref = np.clip(t_ref, t0, 1.0)

    # recover
    J = (I - A.reshape(1, 1, 3)) / t_ref[:, :, None] + A.reshape(1, 1, 3)
    return np.clip(J, 0.0, 1.0).astype(np.float32)


# -----------------------------
# Local optimization (optimize_landscape.m)
# -----------------------------
def process_dam(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    # 1) 去雾/提对比
    try:
        I = reduce_haze_dcp(I, amount=0.95, patch=15)
    except Exception:
        lo, hi = stretchlim(I, 0.0, 1.0)
        I = imadjust(I, (lo, hi), (0.0, 1.0))
        I = np.clip(I ** 1.4, 0.0, 1.0)

    # 2) 空间权重（越靠上权重越大）
    rows, cols = I.shape[:2]
    Y = np.arange(1, rows + 1, dtype=np.float32).reshape(-1, 1)
    spatial_weight = 1.0 - (Y / float(rows))
    spatial_weight = spatial_weight ** 2  # (H,1)

    hsv = color.rgb2hsv(I).astype(np.float32)
    V = hsv[:, :, 2]
    S = hsv[:, :, 1]

    # 3) 发白区域：亮度高、饱和度低
    mask_pale = (V > 0.35) & (S < 0.4)

    # 4) 结合位置信息，只更强地作用在上半部分
    final_mask = mask_pale.astype(np.float32) * (spatial_weight * 0.8 + 0.2)

    # 5) 压暗：Gamma
    V_dark = np.clip(V ** 2.0, 0.0, 1.0)
    V_new = V * (1.0 - final_mask) + V_dark * final_mask

    # 6) 给发白区域增加饱和度
    S_new = S.copy()
    S_new[mask_pale] = np.clip(S[mask_pale] * 1.5 + 0.1, 0.0, 1.0)

    # 7) 纹理增强（guided filter / gaussian fallback）
    try:
        base = guided_filter(V_new, V_new, radius=5, eps=1e-3)
    except Exception:
        base = gaussian_blur(V_new, 2.0)
    detail = V_new - base
    V_final = np.clip(base + detail * 3.5, 0.0, 1.0)

    hsv[:, :, 2] = V_final
    hsv[:, :, 1] = S_new
    out_rgb = color.hsv2rgb(hsv)

    # 8) 轻微锐化
    out = imsharpen(out_rgb, radius=2.0, amount=1.5, threshold=0.0)
    return np.clip(out, 0.0, 1.0)


def process_sky(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))
    lab = color.rgb2lab(I).astype(np.float32)
    L = lab[:, :, 0] / 100.0
    b = lab[:, :, 2]

    # 增强蓝色部分（b<0）
    mask_blue = b < 0
    b2 = b.copy()
    b2[mask_blue] = b2[mask_blue] * 1.3

    # CLAHE 增强亮度层次（NumTiles [8 8]）
    rows, cols = L.shape
    tile_h = max(1, rows // 8)
    tile_w = max(1, cols // 8)
    L_enh = exposure.equalize_adapthist(L, kernel_size=(tile_h, tile_w), clip_limit=0.005).astype(np.float32)

    lab_new = lab.copy()
    lab_new[:, :, 0] = np.clip(L_enh, 0.0, 1.0) * 100.0
    lab_new[:, :, 2] = b2
    I_enh = np.clip(color.lab2rgb(lab_new), 0.0, 1.0).astype(np.float32)

    # 模拟太阳光晕
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

    bloom = sun_glow[:, :, None] * 0.5 * sun_color * bloom_weight[:, :, None]
    out = np.clip(I_enh + bloom, 0.0, 1.0)
    return out


def process_water(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))

    # 偏蓝一点：R降低、G/B提升
    I2 = I.copy()
    I2[:, :, 0] *= 0.85
    I2[:, :, 1] *= 1.10
    I2[:, :, 2] *= 1.20
    I2 = np.clip(I2, 0.0, 1.0)

    # 提取高频波纹并增强
    blurred = gaussian_blur(I2, 2.0)
    high_freq = I2 - blurred
    I_ripples = np.clip(I2 + high_freq * 3.0, 0.0, 1.0)

    # 提亮高光反光
    luminance = rgb2gray(I_ripples)
    mask_high = luminance > 0.6
    I_ripples[mask_high, :] = np.clip(I_ripples[mask_high, :] * 1.1, 0.0, 1.0)

    return np.clip(I_ripples, 0.0, 1.0)


def process_slope(in_img: np.ndarray) -> np.ndarray:
    I = ensure_rgb(_to_float01(in_img))
    lab = color.rgb2lab(I).astype(np.float32)
    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    rows, cols = L.shape
    tile_h = max(1, rows // 16)
    tile_w = max(1, cols // 16)
    L_detail = exposure.equalize_adapthist(L, kernel_size=(tile_h, tile_w), clip_limit=0.02).astype(np.float32)
    L_new = np.clip(L * 0.3 + L_detail * 0.7, 0.0, 1.0)

    b2 = b * 1.15

    lab2 = lab.copy()
    lab2[:, :, 0] = L_new * 100.0
    lab2[:, :, 1] = a
    lab2[:, :, 2] = b2
    out_rgb = np.clip(color.lab2rgb(lab2), 0.0, 1.0).astype(np.float32)

    out = imsharpen(out_rgb, radius=1.0, amount=1.5, threshold=0.02)
    return np.clip(out, 0.0, 1.0)


def optimize_landscape(workdir: Path = Path("."), preview: bool = False) -> Dict[str, Path]:
    """
    读取分割后的四张图，对应处理并保存 optimized_*.jpg。
    返回输出文件路径字典。
    """
    workdir = Path(workdir)

    inputs = {
        "sky": workdir / "part_sky_0.jpg",
        "dam": workdir / "part_dam_0.jpg",
        "water": workdir / "part_water_0.jpg",
        "slope": workdir / "part_slope_0.jpg",
    }

    imgs = {k: imread_rgb(p) for k, p in inputs.items()}
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
        axs[0].imshow(out_sky); axs[0].set_title("天空")
        axs[1].imshow(out_dam); axs[1].set_title("大坝")
        axs[2].imshow(out_water); axs[2].set_title("水流")
        axs[3].imshow(out_slope); axs[3].set_title("岸边")
        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.show()

    return outputs


# -----------------------------
# Global correction (global_landscape_correction.m)
# -----------------------------
@dataclass
class Part:
    name: str
    raw: np.ndarray       # original part (for localization)
    optimized: np.ndarray # optimized part (to stitch)


def stitch_images(canvas: np.ndarray, parts: List[Part]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    自动定位 + 羽化拼接。返回 stitched_img(float01 RGB) 和 4个 masks(bool)。
    """
    final_img = ensure_rgb(_to_float01(canvas))
    gray_full = rgb2gray(final_img)

    H, W = gray_full.shape
    masks: List[np.ndarray] = [np.zeros((H, W), dtype=bool) for _ in range(4)]

    # 顺序：天空 -> 岸边 -> 水流 -> 大坝
    process_order = [0, 3, 2, 1]

    for idx in process_order:
        raw = parts[idx].raw
        opt = ensure_rgb(_to_float01(parts[idx].optimized))

        g_raw = rgb2gray(_to_float01(raw))

        # 内部区域定位模板（避免黑底/边缘干扰）
        bin_mask = g_raw > 0.02
        if not np.any(bin_mask):
            continue
        dist = ndi.distance_transform_edt(bin_mask)
        midx = int(np.argmax(dist))
        cr, cc = np.unravel_index(midx, dist.shape)
        max_d = float(dist[cr, cc])
        rad = int(max(10, np.floor(max_d * 0.8)))

        r1 = max(0, cr - rad)
        r2 = min(g_raw.shape[0] - 1, cr + rad)
        c1 = max(0, cc - rad)
        c2 = min(g_raw.shape[1] - 1, cc + rad)
        template = g_raw[r1:r2 + 1, c1:c2 + 1].astype(np.float32)
        full = gray_full.astype(np.float32)

        # match template
        res = cv2.matchTemplate(full, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)  # (x,y)
        tx, ty = int(max_loc[0]), int(max_loc[1])  # top-left in full (0-index)

        # raw top-left such that (r1,c1) aligns to (ty,tx)
        fy = ty - r1
        fx = tx - c1

        rs = int(np.round(fy))
        cs = int(np.round(fx))
        oh, ow = opt.shape[:2]
        re = rs + oh - 1
        ce = cs + ow - 1

        # edge snapping
        if abs(ce - (W - 1)) < 20:
            shift = (W - 1) - ce
            cs += shift
            ce = W - 1
        if abs(re - (H - 1)) < 20:
            shift = (H - 1) - re
            rs += shift
            re = H - 1
        if abs(cs - 0) < 20:
            cs = 0
        if abs(rs - 0) < 20:
            rs = 0

        rs_c = max(0, rs)
        re_c = min(H - 1, re)
        cs_c = max(0, cs)
        ce_c = min(W - 1, ce)

        l_rs = rs_c - rs
        l_re = l_rs + (re_c - rs_c)
        l_cs = cs_c - cs
        l_ce = l_cs + (ce_c - cs_c)

        if l_re >= oh or l_ce >= ow:
            continue

        opt_crop = opt[l_rs:l_re + 1, l_cs:l_ce + 1, :]
        raw_crop = g_raw[l_rs:l_re + 1, l_cs:l_ce + 1]

        # mask + fill holes + erode (去掉边缘杂色)
        mask_crop = raw_crop > 0.01
        mask_crop = binary_fill_holes(mask_crop)

        kernel = disk_kernel(3)
        mask_eroded = cv2.erode(mask_crop.astype(np.uint8), kernel, iterations=1).astype(bool)

        # feather alpha
        alpha = gaussian_blur(mask_eroded.astype(np.float32), 2.0)
        alpha3 = alpha[:, :, None]

        # record mask (use eroded mask as in MATLAB)
        current = np.zeros((H, W), dtype=bool)
        current[rs_c:re_c + 1, cs_c:ce_c + 1] = mask_eroded
        masks[idx] |= current

        # blend
        bg = final_img[rs_c:re_c + 1, cs_c:ce_c + 1, :]
        final_img[rs_c:re_c + 1, cs_c:ce_c + 1, :] = opt_crop * alpha3 + bg * (1.0 - alpha3)

    return np.clip(final_img, 0.0, 1.0), masks


def apply_global_effects(img01: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    I = ensure_rgb(_to_float01(img01))
    mask_sky = masks[0]
    mask_dam = masks[1]
    mask_ground = masks[1] | masks[2] | masks[3]

    # 1) 光线漫射：天空高光溢出到地景边缘（screen blend）
    sky_part = I * mask_sky[:, :, None].astype(np.float32)
    sky_bloom = gaussian_blur(sky_part.astype(np.float32), 20.0)

    sky_dilated = cv2.dilate(mask_sky.astype(np.uint8), disk_kernel(15), iterations=1).astype(bool)
    edge_zone = sky_dilated & mask_ground
    edge_zone3 = edge_zone[:, :, None]

    wrap_intensity = 0.5
    I_wrapped = 1.0 - (1.0 - I) * (1.0 - sky_bloom * wrap_intensity)
    I = np.where(edge_zone3, I_wrapped, I)

    # 2) 空气透视：大坝轻微带天空色
    sky_pixels = I[mask_sky]
    if sky_pixels.size == 0:
        sky_color = np.array([0.8, 0.9, 1.0], dtype=np.float32)
    else:
        sky_color = sky_pixels.mean(axis=0).astype(np.float32)
    sky_tint = sky_color.reshape(1, 1, 3)

    dam3 = mask_dam[:, :, None]
    tinted = I * 0.95 + sky_tint * 0.05
    I = np.where(dam3, tinted, I)

    # 3) 全局色调：轻微暖色 + S曲线
    lab = color.rgb2lab(I).astype(np.float32)
    lab[:, :, 1] = lab[:, :, 1] * 1.05 + 1.0  # a
    lab[:, :, 2] = lab[:, :, 2] * 1.05 + 3.0  # b

    L = lab[:, :, 0] / 100.0
    L = 1.0 / (1.0 + np.exp(-5.0 * (L - 0.5)))
    lab[:, :, 0] = np.clip(L, 0.0, 1.0) * 100.0

    out = np.clip(color.lab2rgb(lab), 0.0, 1.0).astype(np.float32)
    return out


def global_landscape_correction(workdir: Path = Path("."), preview: bool = False) -> Path:
    """
    读取 9.jpg + optimized_*.jpg + part_*.jpg，完成拼接与全局光影统一，输出 final_masterpiece.jpg。
    """
    workdir = Path(workdir)

    canvas = imread_rgb(workdir / "9.jpg")
    raw_paths = {
        "天空": workdir / "part_sky_0.jpg",
        "大坝": workdir / "part_dam_0.jpg",
        "水流": workdir / "part_water_0.jpg",
        "岸边": workdir / "part_slope_0.jpg",
    }
    opt_paths = {
        "天空": workdir / "optimized_sky.jpg",
        "大坝": workdir / "optimized_dam.jpg",
        "水流": workdir / "optimized_water.jpg",
        "岸边": workdir / "optimized_slope.jpg",
    }

    names = ["天空", "大坝", "水流", "岸边"]
    parts: List[Part] = []
    for n in names:
        parts.append(Part(name=n, raw=imread_rgb(raw_paths[n]), optimized=imread_rgb(opt_paths[n])))

    stitched, masks = stitch_images(canvas, parts)
    final_img = apply_global_effects(stitched, masks)

    out_path = workdir / "final_masterpiece.jpg"
    imwrite_rgb(out_path, final_img)

    if preview:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        axs[0].imshow(ensure_rgb(_to_float01(canvas))); axs[0].set_title("原始底图"); axs[0].axis("off")
        axs[1].imshow(final_img); axs[1].set_title("最终合成效果"); axs[1].axis("off")
        plt.tight_layout()
        plt.show()

    return out_path


# -----------------------------
# CLI
# -----------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Landscape pipeline: optimize parts + global stitch & grading."
    )

    # ✅ mode 改为可选位置参数，默认 all
    ap.add_argument(
        "mode",
        nargs="?",
        choices=["optimize", "global", "all"],
        default="all",
        help="运行模式：optimize / global / all（默认：all）",
    )

    ap.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="图片所在目录（默认当前目录）",
    )

    ap.add_argument(
        "--preview",
        action="store_true",
        help="用 matplotlib 预览结果（可选）",
    )

    args = ap.parse_args(argv)
    wd = Path(args.workdir)

    if args.mode in ("optimize", "all"):
        outs = optimize_landscape(wd, preview=args.preview)
        print("已生成：")
        for k, v in outs.items():
            print(f"  {k}: {v}")

    if args.mode in ("global", "all"):
        out = global_landscape_correction(wd, preview=args.preview)
        print(f"已生成：{out}")


if __name__ == "__main__":
    main()

