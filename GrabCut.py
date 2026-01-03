import cv2
import numpy as np

class GrabCutApp:
    def __init__(self, img_path, max_display_size=900):
        # 读取原始图像
        self.img_orig = cv2.imread(img_path)
        if self.img_orig is None:
            print(f"无法读取图片: {img_path}")
            return
        
        h_orig, w_orig = self.img_orig.shape[:2]
        
        # 【关键】等比例缩小用于交互显示
        scale = min(max_display_size / max(h_orig, w_orig), 1.0)
        self.scale = scale
        
        if scale < 1.0:
            w_disp = int(w_orig * scale)
            h_disp = int(h_orig * scale)
            self.img = cv2.resize(self.img_orig, (w_disp, h_disp), interpolation=cv2.INTER_AREA)
            print(f"原图尺寸: {w_orig}×{h_orig}, 显示尺寸: {w_disp}×{h_disp} (缩放比例: {scale:.3f})")
        else:
            self.img = self.img_orig.copy()
            print(f"原图尺寸: {w_orig}×{h_orig} (无需缩放)")
        
        self.img2 = self.img.copy()                               # 显示图像的备份
        self.mask_disp = np.zeros(self.img.shape[:2], dtype=np.uint8)  # 显示尺寸的掩模（用于交互）
        self.mask_orig = np.zeros(self.img_orig.shape[:2], dtype=np.uint8)  # 原图尺寸的掩模（用于处理）
        self.output = np.zeros(self.img_orig.shape, np.uint8)      # 输出图像（原图尺寸）

        # 交互状态变量
        self.rect = (0, 0, 1, 1)
        self.drawing = False         # 是否正在画框/涂抹
        self.rectangle = False       # 是否处于画矩形模式
        self.rect_over = False       # 矩形是否已画完
        self.thickness = 3           # 笔刷粗细
        self.value = {'color': (0, 0, 0), 'val': cv2.GC_BGD}  # 当前笔刷值 (默认背景)
                                     # GC_BGD=0 (背景), GC_FGD=1 (前景), 
                                     # GC_PR_BGD=2 (可能背景), GC_PR_FGD=3 (可能前景)

        print("操作说明:")
        print("1. 鼠标右键拖动: 画矩形框 (框选保留区域)")
        print("2. 按 'n': 执行/更新分割")
        print("3. 按 '0': 切换到【背景笔刷】(涂抹不需要的地方)")
        print("4. 按 '1': 切换到【前景笔刷】(涂抹需要保留的地方)")
        print("5. 按 's': 保存结果")
        print("6. 按 'r': 重置")
        print("7. 按 'ESC': 退出")

        # 设置鼠标回调
        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', self.on_mouse)
        self.run()

    def _to_orig_coords(self, x, y):
        """将显示坐标转换为原图坐标"""
        return int(x / self.scale), int(y / self.scale)
    
    def _to_disp_coords(self, x, y):
        """将原图坐标转换为显示坐标"""
        return int(x * self.scale), int(y * self.scale)
    
    def on_mouse(self, event, x, y, flags, param):
        # 将显示坐标转换为原图坐标
        x_orig, y_orig = self._to_orig_coords(x, y)
        
        # 鼠标右键：画初始矩形
        if event == cv2.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y
            self.ix_orig, self.iy_orig = x_orig, y_orig

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectangle:
                # 在显示图像上绘制矩形预览
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (255, 0, 0), 2)
                # 保存原图坐标的矩形
                self.rect = (min(self.ix_orig, x_orig), min(self.iy_orig, y_orig), 
                            abs(self.ix_orig - x_orig), abs(self.iy_orig - y_orig))
                self.rect_or_mask = 0 # 标记为使用矩形初始化
            elif self.drawing:
                # 鼠标左键涂抹修补 (Mask模式)
                # 在显示图像上绘制
                cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask_disp, (x, y), self.thickness, self.value['val'], -1)
                # 在原图尺寸的掩模上绘制
                thickness_orig = max(1, int(self.thickness / self.scale))
                cv2.circle(self.mask_orig, (x_orig, y_orig), thickness_orig, self.value['val'], -1)

        elif event == cv2.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            # 画出蓝色矩形示意（显示图像上）
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (255, 0, 0), 2)
            # 保存原图坐标的矩形
            self.rect = (min(self.ix_orig, x_orig), min(self.iy_orig, y_orig), 
                        abs(self.ix_orig - x_orig), abs(self.iy_orig - y_orig))
            self.rect_or_mask = 0
            print(f"矩形已选定 (显示坐标): ({self.ix}, {self.iy}) -> ({x}, {y})")
            print(f"矩形已选定 (原图坐标): {self.rect} -> 请按 'n' 开始分割")

        # 鼠标左键：涂抹细节
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.rect_over:
                print("请先用右键画一个矩形框！")
            else:
                self.drawing = True
                # 在显示图像上绘制
                cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask_disp, (x, y), self.thickness, self.value['val'], -1)
                # 在原图尺寸的掩模上绘制
                thickness_orig = max(1, int(self.thickness / self.scale))
                cv2.circle(self.mask_orig, (x_orig, y_orig), thickness_orig, self.value['val'], -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def run(self):
        while True:
            cv2.imshow('Input', self.img)
            # 显示缩小后的输出结果
            if self.scale < 1.0 and self.output.size > 0:
                output_disp = cv2.resize(self.output, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_AREA)
                cv2.imshow('Output', output_disp)
            else:
                cv2.imshow('Output', self.output)
            k = cv2.waitKey(1) & 0xFF

            # ESC退出
            if k == 27:
                break
            
            # 按 '0' 键：设置为画背景 (黑色笔刷)
            elif k == ord('0'): 
                print("切换笔刷: 标记背景 (去除区域)")
                self.value = {'color':(0,0,0), 'val':0} # 0 = GC_BGD
            
            # 按 '1' 键：设置为画前景 (白色笔刷)
            elif k == ord('1'):
                print("切换笔刷: 标记前景 (保留区域)")
                self.value = {'color':(255,255,255), 'val':1} # 1 = GC_FGD

            # 按 'n' 键：执行 GrabCut
            elif k == ord('n'):
                print("正在计算分割...请稍候（使用原图尺寸处理）")
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                if self.rect_or_mask == 0: # 第一次，使用矩形初始化
                    # 使用原图尺寸进行 GrabCut
                    cv2.grabCut(self.img_orig, self.mask_orig, self.rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                    self.rect_or_mask = 1 # 之后切换为 Mask 模式
                else: # 之后，使用 Mask 初始化 (包含用户的涂抹修改)
                    # 此时 GrabCut 会读取 mask_orig 中的 0(背景) 和 1(前景) 进行更新
                    cv2.grabCut(self.img_orig, self.mask_orig, self.rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

                # 生成原图尺寸的可视化结果
                mask2_orig = np.where((self.mask_orig == 2) | (self.mask_orig == 0), 0, 1).astype('uint8')
                self.output = self.img_orig * mask2_orig[:, :, np.newaxis]
                
                # 同步原图尺寸的掩模结果到显示尺寸的掩模（用于后续交互）
                if self.scale < 1.0:
                    # 将原图尺寸的掩模缩小到显示尺寸
                    mask2_disp = cv2.resize(mask2_orig, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # 更新显示掩模：保留用户手动标记的部分，其余使用 GrabCut 的结果
                    # 如果用户手动标记了前景(1)或背景(0)，保留；否则使用 GrabCut 的结果
                    user_marked = (self.mask_disp == 1) | (self.mask_disp == 0)
                    self.mask_disp = np.where(user_marked, self.mask_disp, mask2_disp).astype(np.uint8)
                
                print("分割完成。需要微调请按 0 或 1 涂抹，然后再次按 n")

            # 按 's' 键：保存
            elif k == ord('s'):
                cv2.imwrite('grabcut_result.jpg', self.output)
                print("结果已保存为 grabcut_result.jpg")

            # 按 'r' 键：重置
            elif k == ord('r'):
                self.img = self.img2.copy()
                self.mask_disp = np.zeros(self.img.shape[:2], dtype=np.uint8)
                self.mask_orig = np.zeros(self.img_orig.shape[:2], dtype=np.uint8)
                self.output = np.zeros(self.img_orig.shape, np.uint8)
                self.rect_over = False
                print("已重置")

        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 替换为您本地图片的路径
    GrabCutApp('9.jpg')
