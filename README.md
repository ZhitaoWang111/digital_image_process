# Image Processing Pipeline (Hard-coded Filenames)

本仓库包含 4 个脚本，组成一条“（可选）修正太阳/强光斑 → 交互式 GrabCut 抠图 → 无缝多类分割与边缘平滑 → 分区增强并合成最终成片”的流水线。

⚠️ **重要**：图像文件名基本都是**写死（hard-coded）**的（少量脚本支持指定工作目录，但读写的图片名仍固定）。请严格按下文列出的文件名准备输入文件，并按对应的输出文件名查看结果。

---

## Requirements

建议 Python 3.8+。

```bash
pip install opencv-python numpy matplotlib
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

### 4) `landscape_pipeline.py`（分区增强 + 自动拼回 + 全局调色，生成成片）

**功能**（整体流程）：
1. **局部优化（optimize）**：对 `part_*_0.jpg` 四个部件分别做增强处理（例如天空色彩/对比、地景去雾、白化区域压暗提饱和等），输出 `optimized_*.jpg`
2. **全局合成（global）**：将 `optimized_*.jpg` 自动定位并拼回原始底图 `9.jpg`，做羽化融合，并叠加全局统一效果，输出最终成片 `final_masterpiece.jpg`

> 脚本支持指定工作目录 `--workdir`，但在该目录内**读取/写入的图片名仍是固定的**。

**运行（默认 all：先 optimize 再 global）：**
```bash
python landscape_pipeline.py
```

**可选运行方式：**
```bash
python landscape_pipeline.py optimize
python landscape_pipeline.py global
python landscape_pipeline.py all
python landscape_pipeline.py all --workdir .
python landscape_pipeline.py all --workdir /path/to/images
```

**输入（写死的文件名，位于 workdir，默认当前目录）：**
- 底图：`9.jpg`
- 四个部件扣图（来自 `imgenhanced.py`）：  
  - `part_sky_0.jpg`
  - `part_dam_0.jpg`
  - `part_water_0.jpg`
  - `part_slope_0.jpg`

**中间输出（写死，位于 workdir）：**
- `optimized_sky.jpg`
- `optimized_dam.jpg`
- `optimized_water.jpg`
- `optimized_slope.jpg`

**最终输出（写死，位于 workdir）：**
- `final_masterpiece.jpg`

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

运行（默认 `all`）：

```bash
python landscape_pipeline.py
```

输出：

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
  - imgenhanced / landscape_pipeline 使用 `9.jpg`

  如果 `9_enhanced.jpg` 与 `9.jpg` 内容/尺寸不同，会影响“分割/拼接”的一致性，但脚本不一定会报错（只是结果可能不理想）。
- `landscape_pipeline.py` 的自动拼接依赖模板匹配定位。如果部件扣图与底图不对应（例如来自不同底图），会导致定位失败或拼接位置错误。
