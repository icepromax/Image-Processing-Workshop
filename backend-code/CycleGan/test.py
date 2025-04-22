import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.CycleGan.model import GeneratorResNet  # 确保路径与你的项目结构一致

def load_model(model_path, input_shape, n_residual_blocks=9):
    """加载预训练生成器模型"""
    model = GeneratorResNet(input_shape, n_residual_blocks)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path, img_size=(256, 256)):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度

def postprocess_image(tensor):
    """将模型输出转为PIL图像"""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor * 0.5 + 0.5  # 反归一化 [0,1]
    return transforms.ToPILImage()(tensor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output_image", type=str, default="monet_output.jpg", help="输出图片路径")
    parser.add_argument("--model_path", type=str, default="save/facades/G_BA_19.pth", help="G_BA模型路径")
    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数（需与训练时一致）
    input_shape = (3, 256, 256)
    n_residual_blocks = 9

    # 加载模型（G_BA: 从B域[照片]到A域[莫奈]）
    G_BA = load_model(args.model_path, input_shape, n_residual_blocks).to(device)

    # 处理输入图像
    input_tensor = preprocess_image(args.input_image).to(device)

    # 风格转换
    with torch.no_grad():
        output_tensor = G_BA(input_tensor)

    # 保存结果
    output_image = postprocess_image(output_tensor)
    output_image.save(args.output_image)
    print(f"风格转换完成！结果已保存至 {args.output_image}")

if __name__ == "__main__":
    main()