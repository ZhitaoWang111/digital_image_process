import cv2
import numpy as np
import os

def auto_correct_and_restore(image_path,
                             output_path,               
                             sun_corrected_path=None, 
                             crop_margin=1.2,
                             target_size_factor=1.0, min_target_size=50):

    rawimg = cv2.imread(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图像")
        return
    
    rows, cols = img.shape[:2]
    
    # 自动侦测太阳位置与形态
    print("正在分析图像光照分布...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去除噪点
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 寻找图像中的高亮区域（太阳可能存在的区域）
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    
    # 设定阈值：只提取亮度在最大亮度 95% 以上的区域
    thresh_val = max(245, max_val * 0.95)
    _, mask_sun = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_sun, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug_dir = "fixsun3_debug"
    os.makedirs(debug_dir, exist_ok=True)

    all_contours_vis = img.copy()
    cv2.drawContours(all_contours_vis, contours, -1, (0, 0, 255), 2)
    all_contours_path = os.path.join(debug_dir, "1_all_contours.jpg")
    cv2.imwrite(all_contours_path, all_contours_vis)
    print(f"可视化 1：所有轮廓图已保存 -> {all_contours_path}")
    
    if not contours:
        print("未检测到明显的太阳/强光区域。")
        return

    sun_contour = max(contours, key=cv2.contourArea)

    largest_contour_vis = img.copy()
    cv2.drawContours(largest_contour_vis, [sun_contour], -1, (0, 255, 0), 2)
    largest_contour_path = os.path.join(debug_dir, "2_largest_contour.jpg")
    cv2.imwrite(largest_contour_path, largest_contour_vis)
    print(f"可视化 2：最大区域轮廓图已保存 -> {largest_contour_path}")
    
    sun_region_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.drawContours(sun_region_mask, [sun_contour], -1, 255, -1)
    sun_region = cv2.bitwise_and(img, img, mask=sun_region_mask)
    sun_region_path = "9_auto_region.jpg"
    cv2.imwrite(sun_region_path, sun_region)
    print(f"太阳区域已保存: {sun_region_path}")
    
    # 拟合椭圆，获取几何参数
    ellipse_params = cv2.fitEllipse(sun_contour)
    (center_x, center_y), (d1, d2), angle = ellipse_params

    ellipse_vis = img.copy()
    cv2.ellipse(ellipse_vis, ellipse_params, (255, 0, 0), 2)
    ellipse_vis_path = os.path.join(debug_dir, "3_fitted_ellipse.jpg")
    cv2.imwrite(ellipse_vis_path, ellipse_vis)
    print(f"可视化 3：拟合椭圆图已保存 -> {ellipse_vis_path}")
    
    print(f"检测到太阳: 中心=({center_x:.1f}, {center_y:.1f}), "
          f"长轴={max(d1,d2):.1f}, 短轴={min(d1,d2):.1f}, 角度={angle:.1f}°")

    # 背景重构
    # 生成操作掩码
    inpainting_mask = np.zeros((rows, cols), dtype=np.uint8)
    # 绘制实心椭圆作为修补区域
    cv2.ellipse(inpainting_mask, ellipse_params, 255, -1)
    # 膨胀掩码，确保覆盖光晕边缘
    inpainting_mask = cv2.dilate(inpainting_mask, np.ones((21, 21), np.uint8), iterations=2)
    
    # 修复背景(Navier-Stokes算法)
    bg_clean = cv2.inpaint(img, inpainting_mask, 5, cv2.INPAINT_NS)
    
    # 太阳提取与整形
    # 构建旋转矩阵，以太阳中心为轴
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    img_rotated = cv2.warpAffine(img, M, (cols, rows))
    
    # 椭圆轴长：major_axis（长轴），minor_axis（短轴）
    major_axis = max(d1, d2)
    minor_axis = min(d1, d2)
    crop_w = int(minor_axis * crop_margin)  # 与短轴对齐的宽
    crop_h = int(major_axis * crop_margin)  # 与长轴对齐的高
    
    # 计算裁剪起始点
    x_start = max(0, int(center_x - crop_w / 2))
    y_start = max(0, int(center_y - crop_h / 2))
    x_end = min(cols, x_start + crop_w)
    y_end = min(rows, y_start + crop_h)
    
    # 提取太阳区域
    sun_crop = img_rotated[y_start:y_end, x_start:x_end]
    
    if sun_crop.size == 0 or sun_crop.shape[0] < 10 or sun_crop.shape[1] < 10:
        print("警告：裁剪区域过小，使用默认尺寸")
        crop_w = int(minor_axis)
        crop_h = int(major_axis)
        x_start = max(0, int(center_x - crop_w / 2))
        y_start = max(0, int(center_y - crop_h / 2))
        x_end = min(cols, x_start + crop_w)
        y_end = min(rows, y_start + crop_h)
        sun_crop = img_rotated[y_start:y_end, x_start:x_end]
    
    h_crop, w_crop = sun_crop.shape[:2]
    # 通过非等比缩放（拉伸短轴）把椭圆变为接近圆的形状
    target_side = max(h_crop, w_crop)
    sun_crop = cv2.resize(sun_crop, (target_side, target_side), interpolation=cv2.INTER_CUBIC)
    print(f"调整裁剪区域为近似方形（拉伸短轴）：{target_side}x{target_side}")

    adjusted_crop_path = os.path.join(debug_dir, "4_sun_crop_square.jpg")
    cv2.imwrite(adjusted_crop_path, sun_crop)
    print(f"可视化 4：调整后的太阳裁剪图已保存 -> {adjusted_crop_path}")
    
    # 使用长轴作为目标直径的基础，乘以缩放因子变成圆形
    base_diam = int(major_axis * target_size_factor)
    target_diam = max(base_diam, min_target_size)
    
    print(f"裁剪区域尺寸: {sun_crop.shape[1]}x{sun_crop.shape[0]}, 目标太阳直径: {target_diam}")
    
    sun_corrected = cv2.resize(sun_crop, (target_diam, target_diam), interpolation=cv2.INTER_CUBIC)
    
    if sun_corrected_path is None:
        base_name = os.path.splitext(output_path)[0]
        sun_corrected_path = f"{base_name}_sun_corrected.jpg"
    cv2.imwrite(sun_corrected_path, sun_corrected)
    print(f"重绘后的太阳图像已保存: {sun_corrected_path}")
    
    # 泊松融合
    # 制作新太阳的圆形掩码
    src_mask = np.zeros(sun_corrected.shape[:2], dtype=np.uint8)
    center_mask = (target_diam // 2, target_diam // 2)
    radius_mask = target_diam // 2 - 2  # 稍微缩小一点避免边界问题
    cv2.circle(src_mask, center_mask, radius_mask, 255, -1)
    
    # 将修正后的圆太阳融合回已修复背景的图像中
    center_int = (int(center_x), int(center_y))
    
    # 检查融合中心是否在图像范围内
    mask_h, mask_w = src_mask.shape[:2]
    min_x = center_int[0] - mask_w // 2
    min_y = center_int[1] - mask_h // 2
    max_x = center_int[0] + mask_w // 2
    max_y = center_int[1] + mask_h // 2
    
    # 如果超出边界，调整中心点
    if min_x < 0:
        center_int = (center_int[0] - min_x, center_int[1])
    if min_y < 0:
        center_int = (center_int[0], center_int[1] - min_y)
    if max_x >= cols:
        center_int = (center_int[0] - (max_x - cols + 1), center_int[1])
    if max_y >= rows:
        center_int = (center_int[0], center_int[1] - (max_y - rows + 1))
    
    try:
        final_composited = cv2.seamlessClone(
            sun_corrected, 
            bg_clean, 
            src_mask, 
            center_int, 
            cv2.NORMAL_CLONE
        )
    except Exception as e:
        print(f"融合失败: {e}")
        final_composited = bg_clean.copy()
        y1 = max(0, center_int[1] - mask_h // 2)
        y2 = min(rows, center_int[1] + mask_h // 2)
        x1 = max(0, center_int[0] - mask_w // 2)
        x2 = min(cols, center_int[0] + mask_w // 2)
        if y2 > y1 and x2 > x1:
            mask_resized = cv2.resize(src_mask, (x2 - x1, y2 - y1))
            sun_resized = cv2.resize(sun_corrected, (x2 - x1, y2 - y1))
            mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR) / 255.0
            final_composited[y1:y2, x1:x2] = (sun_resized * mask_3ch + final_composited[y1:y2, x1:x2] * (1 - mask_3ch)).astype(np.uint8)
    
    
    # 保存结果
    cv2.imwrite(output_path, final_composited)
    print(f"处理完成！图像已保存至: {output_path}")

# 执行函数
if __name__ == "__main__":
    
    auto_correct_and_restore(
        '9.jpg', 
        '9_sun_reshaped.jpg', 
        'reshaped_sun.jpg',
        crop_margin=1.2,        # 裁剪框放大倍数（1.0-2.0），越大包含越多边缘
        target_size_factor=0.9,  # 目标太阳大小因子（0.5-2.0），>1.0变大，<1.0变小
        min_target_size=500      # 最小太阳直径
    )