#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_landscape_correction.m 的 Python 等价实现（自动定位拼接 + 全局光影/色调统一）。

默认读取：
  canvas: 9.jpg
  optimized: optimized_sky.jpg / optimized_dam.jpg / optimized_water.jpg / optimized_slope.jpg
  raw parts: part_sky_0.jpg / part_dam_0.jpg / part_water_0.jpg / part_slope_0.jpg

默认输出：
  final_masterpiece.jpg

依赖：
  numpy, opencv-python, imageio, scipy
（不依赖 skimage）

运行：
  python global_landscape_correction.py
  python global_landscape_correction.py --workdir .
  python global_landscape_correction.py --preview     # 预览（需要 matplotlib）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def disk_kernel(radius: int) -> np.ndarray:
    r = int(max(0, radius))
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask).astype(bool)


# -------------------------
# RGB <-> LAB（近似 MATLAB rgb2lab/lab2rgb）
# -------------------------
def rgb_to_lab_matlab(rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# 数据结构
# -------------------------
@dataclass
class Part:
    name: str
    raw: np.ndarray
    optimized: np.ndarray


# -------------------------
# 拼接（对应 stitch_images）
# -------------------------
def stitch_images(canvas: np.ndarray, parts: List[Part]) -> Tuple[np.ndarray, List[np.ndarray]]:
    final_img = ensure_rgb(_to_float01(canvas))
    gray_full = rgb2gray(final_img)

    H, W = gray_full.shape
    masks: List[np.ndarray] = [np.zeros((H, W), dtype=bool) for _ in range(4)]

    # MATLAB: [1,4,3,2]（1-based） -> Python: [0,3,2,1]
    process_order = [0, 3, 2, 1]

    for idx in process_order:
        raw = parts[idx].raw
        opt = ensure_rgb(_to_float01(parts[idx].optimized))
        g_raw = rgb2gray(_to_float01(raw))

        # 内部区域定位模板（避免边缘黑底）
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

        # MATLAB normxcorr2(template, gray_full) -> 这里用 matchTemplate 直接找 top-left
        res = cv2.matchTemplate(gray_full.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        tx, ty = int(max_loc[0]), int(max_loc[1])  # template 在 canvas 上的左上角（0-based）

        # MATLAB: fy = ty - (r1-1); fx = tx - (c1-1) （1-based）
        # Python 0-based：raw 中 r1/c1 对应 template 左上角，应该对齐到 (ty,tx)
        fy = ty - r1
        fx = tx - c1

        rs = int(np.round(fy))
        cs = int(np.round(fx))
        oh, ow = opt.shape[:2]
        re = rs + oh - 1
        ce = cs + ow - 1

        # 边缘吸附（防止贴边空隙）
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

        # mask + fill holes + erode
        mask_crop = raw_crop > 0.01
        mask_crop = binary_fill_holes(mask_crop)

        mask_eroded = cv2.erode(mask_crop.astype(np.uint8), disk_kernel(3), iterations=1).astype(bool)

        # feather alpha
        alpha = gaussian_blur(mask_eroded.astype(np.float32), 2.0)
        alpha3 = alpha[..., None]

        # 保存 mask（用 eroded mask）
        current = np.zeros((H, W), dtype=bool)
        current[rs_c:re_c + 1, cs_c:ce_c + 1] = mask_eroded
        masks[idx] |= current

        # blend
        bg = final_img[rs_c:re_c + 1, cs_c:ce_c + 1, :]
        final_img[rs_c:re_c + 1, cs_c:ce_c + 1, :] = opt_crop * alpha3 + bg * (1.0 - alpha3)

    return np.clip(final_img, 0.0, 1.0), masks


# -------------------------
# 全局效果（对应 apply_global_effects）
# -------------------------
def apply_global_effects(img01: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    I = ensure_rgb(_to_float01(img01))

    mask_sky = masks[0]
    mask_dam = masks[1]
    mask_ground = masks[1] | masks[2] | masks[3]

    # 1) 光线漫射：天空高光溢出到地景边缘
    sky_part = I * mask_sky[..., None].astype(np.float32)
    sky_bloom = gaussian_blur(sky_part.astype(np.float32), 20.0)

    sky_dilated = cv2.dilate(mask_sky.astype(np.uint8), disk_kernel(15), iterations=1).astype(bool)
    edge_zone = sky_dilated & mask_ground
    edge_zone3 = edge_zone[..., None]

    wrap_intensity = 0.5
    I_wrapped = 1.0 - (1.0 - I) * (1.0 - sky_bloom * wrap_intensity)
    I = np.where(edge_zone3, I_wrapped, I)

    # 2) 空气透视：大坝轻混天空色
    sky_pixels = I[mask_sky]
    if sky_pixels.size == 0:
        sky_color = np.array([0.8, 0.9, 1.0], dtype=np.float32)
    else:
        sky_color = sky_pixels.mean(axis=0).astype(np.float32)
    sky_tint = sky_color.reshape(1, 1, 3)

    dam3 = mask_dam[..., None]
    I = np.where(dam3, (I * 0.95 + sky_tint * 0.05), I)

    # 3) 全局色调：轻微暖色 + S 曲线
    L, a, b = rgb_to_lab_matlab(I)
    a2 = a * 1.05 + 1.0
    b2 = b * 1.05 + 3.0

    L01 = np.clip(L / 100.0, 0.0, 1.0)
    L01 = 1.0 / (1.0 + np.exp(-5.0 * (L01 - 0.5))).astype(np.float32)
    L2 = np.clip(L01, 0.0, 1.0) * 100.0

    out = lab_matlab_to_rgb(L2, a2, b2)
    return np.clip(out, 0.0, 1.0)


# -------------------------
# 顶层入口（对应 global_landscape_correction）
# -------------------------
def global_landscape_correction(workdir: Path = Path("."), preview: bool = False) -> Path:
    workdir = Path(workdir)

    canvas = imread_rgb(workdir / "9.jpg")

    names = ["天空", "大坝", "水流", "岸边"]
    raw_paths = [
        workdir / "part_sky_0.jpg",
        workdir / "part_dam_0.jpg",
        workdir / "part_water_0.jpg",
        workdir / "part_slope_0.jpg",
    ]
    opt_paths = [
        workdir / "optimized_sky.jpg",
        workdir / "optimized_dam.jpg",
        workdir / "optimized_water.jpg",
        workdir / "optimized_slope.jpg",
    ]

    parts: List[Part] = []
    for n, rp, op in zip(names, raw_paths, opt_paths):
        parts.append(Part(name=n, raw=imread_rgb(rp), optimized=imread_rgb(op)))

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Global stitch & grading (MATLAB -> Python).")
    ap.add_argument("--workdir", type=str, default=".", help="图片所在目录（默认当前目录）")
    ap.add_argument("--preview", action="store_true", help="预览结果（需要 matplotlib）")
    args = ap.parse_args(argv)

    out = global_landscape_correction(Path(args.workdir), preview=args.preview)
    print(f"处理完成，图片已保存为 {out}")


if __name__ == "__main__":
    main()
