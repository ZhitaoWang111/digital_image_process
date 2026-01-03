# Image Processing Pipeline (Hard-coded Filenames)

本仓库包含若干脚本，组成一条“（可选）修正太阳/强光斑 → 交互式 GrabCut 抠图 → 无缝多类分割与边缘平滑 → 分区增强并合成最终成片”的流水线。

⚠️ **重要**：图像文件名基本都是**写死（hard-coded）**的（少量脚本支持指定工作目录，但读写的图片名仍固定）。请严格按下文列出的文件名准备输入文件，并按对应的输出文件名查看结果。

---

## Requirements

建议 Python 3.8+。

```bash
pip install opencv-python numpy scipy imageio matplotlib
```

> `GrabCut.py` 需要图形界面弹窗（OpenCV window）。无 GUI 的纯服务器环境可能无法运行。

---

## Scripts (Hard-coded I/O)

### 1) `reshaped.py`（可选：修正太阳/强光斑）

**功能**：自动检测图像中最亮区域（太阳/强光斑），使用 `inpaint` 修复背景，再用 `seamlessClone` 将“修圆后的太阳”无缝融合回图像。

**运行：**

```bash
python reshaped.py
```

**输入（写死）：**
- `9.jpg`

**输出（写死）：**
- `9_sun_reshaped.jpg`（修正后的整图）
- `reshaped_sun.jpg`（修圆后的太阳裁剪图/调试用）

---

### 2) `GrabCut.py`（交互式 GrabCut 分割）

> 该脚本每次只输出一个固定文件名。你需要运行多次并手动重命名结果文件（见下方运行顺序）。

**功能**：交互式 GrabCut 前景/背景分割（右键画框 + 左键涂抹前/背景 + `n` 迭代）。

**运行：**

```bash
python GrabCut.py
```

**输入（写死）：**
- `9_enhanced.jpg`

**输出（写死）：**
- `grabcut_result.jpg`（每次按 `s` 保存都会生成/覆盖）
- `result_mask.png`（掩膜可视化图片，生成/覆盖）

**交互操作：**
- 鼠标右键拖动：画矩形框（先框住前景的大致区域）
- 鼠标左键拖动：涂抹（配合 `0/1` 切换背景/前景笔刷）
- 键盘：
  - `0`：背景笔刷
  - `1`：前景笔刷
  - `n`：执行/更新 GrabCut（可多次迭代）
  - `s`：保存（输出 `grabcut_result.jpg`）
  - `r`：重置
  - `Esc`：退出

---

### 3) `imgenhanced.py`（Watershed + Soft Voting 平滑）

**功能**：读取原图与 3 张 GrabCut 部件图（sky/dam/slope），构造 markers 后执行 watershed 得到无缝分类，再通过 Soft Voting（高斯平滑概率投票）平滑边缘，并导出叠加可视化与各类扣图。

**运行：**

```bash
python imgenhanced.py
```

**输入（写死）：**
- 原图：`9.jpg`
- 天空：`GrabCut_Result_sky.jpg`
- 坝体：`grabcut_result_dam.jpg`
- 坡面：`grabcut_result_slope.jpg`

**输出（写死）：**
- `final_smooth_segmentation.jpg`（最终平滑分割叠加可视化）
- `part_sky_0.jpg`（天空扣图）
- `part_dam_0.jpg`（坝体扣图）
- `part_slope_0.jpg`（坡面扣图）
- `part_water_0.jpg`（水流扣图：由“原图减去 sky/dam/slope”推导得到）

---

### 4) `optimize_landscape.py`（局部四分区优化 / 分区增强）

**功能**：对 `imgenhanced.py` 输出的四个部件扣图（sky/dam/water/slope）分别做针对性的增强，输出对应的 `optimized_*.jpg`。

主要处理要点（按部件）：
- **sky**：LAB 空间增强蓝色（b<0 更蓝）+ 对 L 通道做 CLAHE + 添加轻微“太阳光晕”
- **dam**：暗通道先验去雾（DCP）→ 识别“发白区域”（亮度高 & 饱和度低）并压暗、提饱和 → 纹理细节增强 → 锐化
- **water**：整体偏蓝 + 强化高频“波纹”细节 + 对高亮区域略微增亮
- **slope**：L 通道 CLAHE 提细节 + 轻微暖色 + 锐化

**运行：**

```bash
python optimize_landscape.py
python optimize_landscape.py --workdir .
python optimize_landscape.py --preview   # 预览（需要 matplotlib）
```

**输入（写死，位于 workdir，默认当前目录）：**
- `part_sky_0.jpg`
- `part_dam_0.jpg`
- `part_water_0.jpg`
- `part_slope_0.jpg`

**输出（写死，位于 workdir）：**
- `optimized_sky.jpg`
- `optimized_dam.jpg`
- `optimized_water.jpg`
- `optimized_slope.jpg`

---

### 5) `global_landscape_correction.py`（自动定位拼接 + 全局光影/色调统一）

**功能**：把 `optimized_*.jpg` 自动定位并拼回底图 `9.jpg`，完成无缝融合，并叠加全局统一效果，输出最终成片 `final_masterpiece.jpg`。

处理要点：
- **自动定位/拼接**：对每个部件，从原始扣图 `part_*_0.jpg` 中找内部模板块（避开黑边），对底图做模板匹配定位；用原始扣图生成 mask（fill holes + 轻微腐蚀）并高斯羽化，按 alpha 融合到 `9.jpg` 上
- **全局统一效果**：天空高光“漫射/溢出”到地景边缘、坝体轻混天空色（空气透视）、整体 LAB 轻微暖色 + 亮度 S 曲线

**运行：**

```bash
python global_landscape_correction.py
python global_landscape_correction.py --workdir .
python global_landscape_correction.py --preview   # 预览（需要 matplotlib）
```

**输入（写死，位于 workdir，默认当前目录）：**
- 底图：`9.jpg`
- 四个部件原始扣图（用于定位与 mask）：
  - `part_sky_0.jpg`
  - `part_dam_0.jpg`
  - `part_water_0.jpg`
  - `part_slope_0.jpg`
- 四个部件增强结果（用于真正拼回）：
  - `optimized_sky.jpg`
  - `optimized_dam.jpg`
  - `optimized_water.jpg`
  - `optimized_slope.jpg`

**输出（写死，位于 workdir）：**
- `final_masterpiece.jpg`

---

> 说明：如果你另外有一个 `landscape_pipeline.py` 封装脚本，它通常只是按顺序调用 `optimize_landscape.py` → `global_landscape_correction.py`，便于一键跑完。

---


## Recommended Run Order (Hard-coded Filenames)

### Step 0（可选）：修正太阳/强光斑

1. 将待处理图片命名为 `9.jpg`
2. 运行：

```bash
python reshaped.py
```

输出：

```text
9_sun_reshaped.jpg
reshaped_sun.jpg
```

> 注意：后续脚本默认仍使用 `9.jpg` / `9_enhanced.jpg`，不会自动使用 `9_sun_reshaped.jpg`。

---

### Step 1：用 GrabCut 生成 sky / dam / slope（运行 3 次）

确保 `9_enhanced.jpg` 存在（GrabCut 固定读取它）。

#### 1) 抠 sky

```bash
python GrabCut.py
```

- 在窗口中完成分割后按 `s` 保存，生成 `grabcut_result.jpg`
- 将 `grabcut_result.jpg` 重命名为：

```text
GrabCut_Result_sky.jpg
```

#### 2) 抠 dam

```bash
python GrabCut.py
```

- 按 `s` 保存得到新的 `grabcut_result.jpg`
- 将其重命名为：

```text
grabcut_result_dam.jpg
```

#### 3) 抠 slope

```bash
python GrabCut.py
```

- 按 `s` 保存得到新的 `grabcut_result.jpg`
- 将其重命名为：

```text
grabcut_result_slope.jpg
```

完成后，目录中应包含：

```text
GrabCut_Result_sky.jpg
grabcut_result_dam.jpg
grabcut_result_slope.jpg
```

---

### Step 2：无缝分割与平滑（生成 `part_*_0.jpg`）

确保以下输入文件都存在（全部为写死文件名）：

```text
9.jpg
GrabCut_Result_sky.jpg
grabcut_result_dam.jpg
grabcut_result_slope.jpg
```

运行：

```bash
python imgenhanced.py
```

输出：

```text
final_smooth_segmentation.jpg
part_sky_0.jpg
part_dam_0.jpg
part_slope_0.jpg
part_water_0.jpg
```

---

### Step 3：分区增强并合成成片（生成 `final_masterpiece.jpg`）

确保以下输入文件都存在（默认都在当前目录）：

```text
9.jpg
part_sky_0.jpg
part_dam_0.jpg
part_water_0.jpg
part_slope_0.jpg
```

运行（先局部优化，再全局合成）：

```bash
python optimize_landscape.py
python global_landscape_correction.py
```

如需指定图片目录（但文件名仍固定），使用：

```bash
python optimize_landscape.py --workdir /path/to/images
python global_landscape_correction.py --workdir /path/to/images
```

输出（位于 workdir）：

```text
optimized_sky.jpg
optimized_dam.jpg
optimized_water.jpg
optimized_slope.jpg
final_masterpiece.jpg
```

---

## Notes / Gotchas

- `GrabCut.py` 需要 GUI 弹窗；无桌面环境可能无法运行。
- 默认情况下：
  - GrabCut 使用 `9_enhanced.jpg`
  - imgenhanced / global_landscape_correction 使用 `9.jpg`

  如果 `9_enhanced.jpg` 与 `9.jpg` 内容/尺寸不同，会影响“分割/拼接”的一致性，但脚本不一定会报错（只是结果可能不理想）。
- `global_landscape_correction.py` 的自动拼接依赖模板匹配定位。如果部件扣图与底图不对应（例如来自不同底图），会导致定位失败或拼接位置错误。
