import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
img_original = cv2.imread('9.jpg') 
img_sky_part = cv2.imread('GrabCut_Result_sky.jpg')
img_dam_part = cv2.imread('grabcut_result_dam.jpg')
img_slope_part = cv2.imread('grabcut_result_slope.jpg')

# 确保尺寸一致
h, w = img_original.shape[:2]
img_sky_part = cv2.resize(img_sky_part, (w, h))
img_dam_part = cv2.resize(img_dam_part, (w, h))
img_slope_part = cv2.resize(img_slope_part, (w, h))

# --- 基础工具函数 ---
def get_binary_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return mask

def keep_largest_component(mask):
    """只保留最大连通区域，去除噪点"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels < 2: return np.zeros_like(mask)
    max_area_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 
    new_mask = np.where(labels == max_area_idx, 255, 0).astype(np.uint8)
    return new_mask

print("Step 1: 预处理与分水岭计算...")

# 2. 准备基础 Mask
raw_mask_sky = get_binary_mask(img_sky_part)
raw_mask_dam = get_binary_mask(img_dam_part)
raw_mask_slope = get_binary_mask(img_slope_part)
# 推导水流 Mask
union_mask = cv2.bitwise_or(raw_mask_sky, cv2.bitwise_or(raw_mask_dam, raw_mask_slope))
raw_mask_water = cv2.bitwise_not(union_mask)

# 3. 过滤杂质 (保留最大主体)
mask_sky = keep_largest_component(raw_mask_sky)
mask_dam = keep_largest_component(raw_mask_dam)
mask_slope = keep_largest_component(raw_mask_slope)
mask_water = keep_largest_component(raw_mask_water)

# 4. 构建分水岭种子 (Markers)
kernel = np.ones((5,5), np.uint8)
# 腐蚀以确定核心区域
sure_sky = cv2.erode(mask_sky, kernel, iterations=3)
sure_dam = cv2.erode(mask_dam, kernel, iterations=3)
sure_slope = cv2.erode(mask_slope, kernel, iterations=3)
sure_water = cv2.erode(mask_water, kernel, iterations=3)

# 标记: 1=天, 2=坝, 3=坡, 4=水
markers = np.zeros((h, w), dtype=np.int32)
markers[sure_sky == 255] = 1
markers[sure_dam == 255] = 2
markers[sure_slope == 255] = 3
markers[sure_water == 255] = 4

# 运行分水岭 (解决空隙与归类)
cv2.watershed(img_original, markers)
# 此时 markers 包含了初步的无缝分割结果 (但边缘可能有锯齿)

print("Step 2: 边缘平滑与二次优化...")

# 5. 边缘平滑处理 (Soft Voting)
# 将分水岭结果拆分为概率图
# 形状: (H, W, 4)
prob_maps = []
label_ids = [1, 2, 3, 4] # 对应 天, 坝, 坡, 水

for label_id in label_ids:
    # 提取当前类的二值图 (0.0 或 1.0)
    current_class_mask = np.where(markers == label_id, 1.0, 0.0).astype(np.float32)
    
    # 关键步骤：高斯模糊
    # 核大小 (13, 13) 决定了边缘的软化程度，越大越平滑但可能模糊细节
    smoothed = cv2.GaussianBlur(current_class_mask, (13, 13), 2.0)
    prob_maps.append(smoothed)

# 堆叠所有概率图
prob_stack = np.dstack(prob_maps)

# Argmax 决策：在每个像素点，选择概率(模糊值)最大的那个类
# 结果索引是 0, 1, 2, 3，对应 label_ids 的位置
refined_indices = np.argmax(prob_stack, axis=2)

# 映射回 1, 2, 3, 4
refined_markers = np.zeros((h, w), dtype=np.uint8)
for i, label_id in enumerate(label_ids):
    refined_markers[refined_indices == i] = label_id

print("Step 3: 生成结果与保存各个部分...")

# 6. 保存独立部分 (使用平滑后的 refined_markers)
parts_info = [
    (1, 'part_sky_0.jpg', (230, 216, 173)),  # 浅蓝
    (2, 'part_dam_0.jpg', (0, 0, 255)),      # 红
    (3, 'part_slope_0.jpg', (0, 255, 0)),    # 绿
    (4, 'part_water_0.jpg', (255, 255, 0))   # 青 (水流)
]

# 准备合成大图
final_vis = img_original.copy()
overlay = np.zeros_like(final_vis)

for label_id, filename, color in parts_info:
    # 提取该部分的 Mask
    part_mask = np.where(refined_markers == label_id, 255, 0).astype(np.uint8)
    
    # 1. 保存独立图片 (扣图)
    part_img = cv2.bitwise_and(img_original, img_original, mask=part_mask)
    cv2.imwrite(filename, part_img)
    
    # 2. 绘制到合成图层
    overlay[part_mask == 255] = color

# 7. 合成显示图
# 边缘抗锯齿处理：不再有锯齿状的 watershed 边界线，而是自然的颜色过渡
alpha = 0.35
cv2.addWeighted(overlay, alpha, final_vis, 1 - alpha, 0, final_vis)

# 保存最终结果
output_final = 'final_smooth_segmentation.jpg'
cv2.imwrite(output_final, final_vis)

# 8. 展示
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title("Smoothed Seamless Result")
plt.imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 展示平滑后的水流 Mask 细节
water_mask_show = np.where(refined_markers == 4, 255, 0).astype(np.uint8)
plt.subplot(1, 2, 2)
plt.title("Refined Water Mask (Smoothed Edges)")
plt.imshow(water_mask_show, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"全部完成。")
print(f"1. 最终平滑分割图: {output_final}")
print(f"2. 各个部件已独立保存: part_sky.jpg, part_dam.jpg, part_slope.jpg, part_water.jpg")
