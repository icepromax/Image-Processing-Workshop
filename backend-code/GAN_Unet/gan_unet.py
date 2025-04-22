import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from GAN_Unet.train import GeneratorUNet
import cv2



class AutoMaskGenerator:
    def __init__(self, detect_mode="hybrid"):
        """
        参数:
            detect_mode:
                'color'    - 基于颜色异常检测（适合纯色遮挡）
                'edge'     - 基于边缘不连续检测
                'hybrid'   - 混合检测（默认）
        """
        self.mode = detect_mode

    def generate_mask(self, image_np):
        """输入：numpy数组格式的RGB图像（0-255范围）
           输出：二值掩码（0/255）"""
        if self.mode == "color":
            return self._color_based_detection(image_np)
        elif self.mode == "edge":
            return self._edge_based_detection(image_np)
        else:
            return self._hybrid_detection(image_np)

    def _color_based_detection(self, img):
        """颜色异常区域检测（适合纯色遮挡）"""
        # 转换到HSV空间检测黑色区域
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _edge_based_detection(self, img):
        """边缘不连续区域检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 寻找轮廓异常区域
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # 过滤小噪点
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    def _hybrid_detection(self, img):
        """混合检测模式"""
        color_mask = self._color_based_detection(img)
        edge_mask = self._edge_based_detection(img)

        # 融合两种检测结果
        combined = cv2.bitwise_or(color_mask, edge_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)


# 加载训练好的生成器模型
def load_generator(model_path, device='cuda'):
    generator = GeneratorUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()  # 切换到推理模式
    return generator


# 执行修复的主函数

def auto_infer_with_merge(generator, image_path, device='cpu'):
    """带局部修复和图像拼接的完整流程"""
    # 1. 加载原始高分辨率图像
    orig_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = orig_image.size

    # 2. 生成低分辨率处理版本
    lr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    lr_tensor = lr_transform(orig_image).unsqueeze(0).to(device)

    # 3. 生成掩码（低分辨率）
    auto_masker = AutoMaskGenerator(detect_mode="hybrid")
    orig_np = np.array(orig_image.resize((128, 128)))
    mask_np = auto_masker.generate_mask(orig_np)
    mask_tensor = torch.from_numpy(mask_np / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)

    # 4. 生成低分辨率修复结果
    with torch.no_grad():
        corrupted = lr_tensor * (1 - mask_tensor)
        lr_restored = generator(corrupted, mask_tensor)

    # 5. 将结果上采样到原始分辨率
    hr_restored = transforms.Resize((orig_height, orig_width))(
        transforms.ToPILImage()(lr_restored.squeeze().cpu())
    )

    # 6. 生成高分辨率掩码
    hr_mask = transforms.Resize((orig_height, orig_width), interpolation=transforms.InterpolationMode.NEAREST)(
        transforms.ToPILImage()(mask_tensor.squeeze().cpu())
    )
    hr_mask_np = np.array(hr_mask) > 128

    # 7. 图像融合
    final_image = np.array(orig_image)
    restored_area = np.array(hr_restored)

    # 使用掩码混合图像（仅替换修复区域）
    final_image[hr_mask_np] = restored_area[hr_mask_np]

    return {
        "original": orig_image,
        "corrupted": transforms.ToPILImage()(corrupted.squeeze().cpu()),
        "mask": hr_mask,
        "restored_part": hr_restored,
        "final_result": Image.fromarray(final_image)
    }


# 修改后的主程序
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = load_generator('saved_models/60.pth', device=device)

    result = auto_infer_with_merge(generator, "output_images/img_1.png",device=device)

    # 可视化结果
    plt.figure(figsize=(24, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(result["original"])
    plt.title("Original Image"), plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(result["corrupted"])
    plt.title("Corrupted Area"), plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(result["mask"], cmap='gray')
    plt.title("Repair Mask"), plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(result["restored_part"])
    plt.title("Restored Part"), plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(result["final_result"])
    plt.title("Final Merged Result"), plt.axis('off')

    plt.tight_layout()
    plt.savefig("merged_result.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    # 新增：单独保存最佳结果
    best_result = result["final_result"]
    best_result.save("best_result.jpg",
                     quality=95,  # 设置JPEG质量（1-100）
                     subsampling=0,  # 保持最高色度采样
                     optimize=True)  # 启用压缩优化

    print("修复结果已保存：")
    print(f"- 对比图: merged_result.jpg")
    print(f"- 最佳结果: best_result.jpg")