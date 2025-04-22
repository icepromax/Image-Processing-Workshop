import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from io import BytesIO
from .model import GeneratorResNet


class MonetStyleTransfer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = self.load_model(model_path).to(self.device)
        self.model.eval()

        self.supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_model(self, model_path):
        """加载预训练模型"""
        model = GeneratorResNet(input_shape=(3, 256, 256), num_residual_blocks=9)
        state_dict = torch.load(model_path, map_location='cpu')

        # 处理可能的权重键名不匹配
        if all(key.startswith('module.') for key in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 去掉 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        return model

    def _convert_to_rgb(self, img):
        """将图像转换为RGB格式"""
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

    def _validate_image(self, file_stream):
        """验证图像是否可读"""
        try:
            img = Image.open(file_stream)
            img.verify()  # 验证图像完整性
            file_stream.seek(0)  # 重置指针
            return True
        except Exception:
            return False

    def process_image(self, input_path, output_path=None, quality=95):
        """
        处理单张图片
        参数:
            input_path: 输入图片路径或文件对象
            output_path: 输出路径(可选)
            quality: 输出质量(1-100)
        返回:
            成功时返回PIL图像对象或保存路径
            失败时返回None
        """
        try:
            # 支持文件路径或文件对象
            if isinstance(input_path, (str, os.PathLike)):
                if not os.path.isfile(input_path):
                    raise ValueError("输入路径不是文件")

                ext = os.path.splitext(input_path)[1].lower()
                if ext not in self.supported_formats:
                    raise ValueError(f"不支持的图片格式: {ext}. 支持: {self.supported_formats}")

                with open(input_path, 'rb') as f:
                    if not self._validate_image(f):
                        raise ValueError("图片文件已损坏")
                    img = Image.open(input_path)
            else:  # 假设是文件对象
                if not self._validate_image(input_path):
                    raise ValueError("图片文件已损坏")
                img = Image.open(input_path)

            # 转换为RGB和处理
            img = self._convert_to_rgb(img)
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            # 后处理
            output_tensor = output_tensor.squeeze(0).cpu()
            output_tensor = output_tensor * 0.5 + 0.5
            result_img = transforms.ToPILImage()(output_tensor)

            # 保存或返回
            if output_path:
                # 根据扩展名选择保存格式
                ext = os.path.splitext(output_path)[1].lower()
                if ext in ('.jpg', '.jpeg'):
                    result_img.save(output_path, quality=quality, subsampling=0)
                elif ext == '.png':
                    result_img.save(output_path, compress_level=6)
                else:
                    result_img.save(output_path)
                return output_path
            return result_img

        except Exception as e:
            print(f"图片处理错误: {str(e)}")
            return None


# 示例用法
