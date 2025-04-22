import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import cv2

#模型训练代码

# 数据集类：加载CelebA数据集并生成带遮挡的图像
class CelebADataset(Dataset):
    def __init__(self, img_dir, csv_path, mask_func, transform=None):
        """
        参数:
            img_dir: 图像目录路径
            csv_path: 数据标注CSV文件路径
            mask_func: 遮挡生成函数
            transform: 图像预处理变换
        """
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)  # 加载数据标注
        self.mask_func = mask_func  # 遮挡生成函数
        self.transform = transform  # 图像预处理

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个样本：
        返回: (被遮挡图像, 原始图像, 遮挡掩码)
        """
        img_name = os.path.join(self.img_dir, f"{idx + 1:06d}.jpg")
        image = Image.open(img_name).convert("RGB")  # 加载RGB图像

        if self.transform:
            image = self.transform(image)  # 应用预处理

        mask = self.mask_func(image.shape[-2:])  # 生成随机遮挡掩码
        corrupted_image = image * (1 - mask)  # 应用遮挡
        return corrupted_image, image, mask


def irregular_mask(size, max_vertices=8, max_angle=2 * math.pi, mask_ratio_range=(0.1, 0.4)):
    """
    生成随机不规则多边形掩码
    参数:
        size: 图像尺寸 (h, w)
        max_vertices: 最大顶点数
        mask_ratio_range: 遮挡面积占比范围
    返回:
        [1, H, W] 二值掩码张量
    """
    h, w = size
    mask = torch.zeros((h, w))

    # 随机生成多边形顶点
    num_vertices = np.random.randint(3, max_vertices + 1)
    angles = np.sort(np.random.uniform(0, max_angle, num_vertices))

    # 计算顶点坐标
    center = (np.random.uniform(0, h), np.random.uniform(0, w))
    radius = h * np.random.uniform(*mask_ratio_range)  # 控制遮挡区域大小

    vertices = []
    for angle in angles:
        r = radius * np.random.uniform(0.5, 1.5)  # 添加径向随机性
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        vertices.append((x, y))

    # 生成多边形填充
    vertices = np.array(vertices).reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask.numpy(), [vertices], 1)

    return mask.unsqueeze(0)


# 混合掩码生成策略（在数据集中使用）
class MixedMaskGenerator:
    def __init__(self,
                 ratio=0.5,
                 irregular_params={'max_vertices': 12},
                 rectangle_params={'mask_ratio': 0.25}):
        """
        参数:
            ratio: 使用不规则掩码的概率
        """
        self.ratio = ratio
        self.irregular_params = irregular_params
        self.rectangle_params = rectangle_params

    def __call__(self, size):
        if np.random.rand() < self.ratio:
            return irregular_mask(size, **self.irregular_params)
        else:
            return random_mask(size,  ** self.rectangle_params)

        # 修改数据集初始化部分
        dataset = CelebADataset(
            img_dir='C:/Users/86182/Desktop/archive/img_align_celeba',
            csv_path='C:/Users/86182/Desktop/archive/list_eval_partition.csv',
            mask_func=MixedMaskGenerator(ratio=0.7),  # 70%概率使用不规则掩码
            transform=transform
        )
# 遮挡生成函数：创建随机矩形遮挡区域
def random_mask(size, mask_ratio=0.25):
    """
    参数:
        size: 图像尺寸 (h, w)
        mask_ratio: 遮挡区域占比
    返回:
        [1, H, W]大小的二值掩码张量
    """
    h, w = size
    mask = torch.zeros((h, w))
    # 计算遮挡区域大小
    mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
    # 随机生成遮挡位置
    top = np.random.randint(0, h - mask_h)
    left = np.random.randint(0, w - mask_w)
    mask[top:top + mask_h, left:left + mask_w] = 1  # 设置遮挡区域为1
    return mask.unsqueeze(0)  # 增加通道维度


# 改进的注意力模块：增强关键特征学习
class ImprovedAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        """
        参数:
            in_channels: 输入特征通道数
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),  # 通道压缩
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),  # 通道恢复
            nn.Sigmoid()  # 生成0-1的注意力权重
        )

    def forward(self, x):
        """前向传播：生成注意力权重并应用于特征图"""
        attention_map = self.attention(x)  # 计算注意力图
        return x * attention_map  # 特征图与注意力图相乘


# 生成器网络：基于U-Net结构
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器部分（下采样）
        self.encoder1 = self.conv_block(4, 64)  # 输入通道=4（RGB+mask）
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # 瓶颈层（包含注意力机制）
        self.bottleneck = self.conv_block(256, 512)
        self.attention = ImprovedAttentionBlock(512)

        # 解码器部分（上采样）
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # 输出层
        self.final = nn.Conv2d(64, 3, kernel_size=1)  # 输出RGB图像

    def conv_block(self, in_channels, out_channels):
        """下采样卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """上采样转置卷积块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        """前向传播流程"""
        # 1. 拼接输入图像和mask
        x = torch.cat([x, mask], dim=1)

        # 2. 编码器路径
        e1 = self.encoder1(x)  # 第一层特征
        e2 = self.encoder2(e1)  # 第二层特征
        e3 = self.encoder3(e2)  # 第三层特征

        # 3. 瓶颈层（应用注意力）
        bottleneck = self.bottleneck(e3)
        bottleneck = self.attention(bottleneck)

        # 4. 解码器路径（带跳跃连接）
        d3 = self.decoder3(bottleneck)
        d3 = self._resize(d3, e3) + e3  # 与编码器特征相加

        d2 = self.decoder2(d3)
        d2 = self._resize(d2, e2) + e2

        d1 = self.decoder1(d2)
        d1 = self._resize(d1, e1) + e1

        # 5. 最终输出（调整到128x128）
        return nn.functional.interpolate(
            self.final(d1),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )

    def _resize(self, x, target):
        """调整特征图尺寸匹配目标张量"""
        return nn.functional.interpolate(
            x,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=False
        )


# 判别器网络：区分真实/生成图像
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入: [3,128,128]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # [64,64,64]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # [128,32,32]
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # 输出0-1的判别概率
            # [1,1,1]
        )

    def forward(self, x):
        return self.model(x)


# 感知损失：基于VGG16的特征相似度计算
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练VGG16的前16层
        vgg = models.vgg16(pretrained=True).features[:16].eval()
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, real, fake):
        """计算真实与生成图像在特征空间的MSE损失"""
        real_features = self.vgg(real)
        fake_features = self.vgg(fake)
        return nn.functional.mse_loss(fake_features, real_features)


# 生成器损失函数
def generator_loss(pred_fake, real, fake, mask, perceptual_loss_fn=None):
    """
    参数:
        pred_fake: 判别器对生成图像的输出
        real: 真实图像
        fake: 生成图像
        mask: 遮挡区域
        perceptual_loss_fn: 感知损失计算函数
    返回:
        综合损失值
    """
    # 调整尺寸匹配
    fake_resized = nn.functional.interpolate(fake, size=real.shape[2:], mode='bilinear')
    real_resized = nn.functional.interpolate(real, size=real.shape[2:], mode='bilinear')

    # 1. L1像素损失（仅在遮挡区域计算）
    l1_loss = nn.L1Loss()(fake_resized * mask, real_resized * mask)

    # 2. 感知损失（可选）
    perceptual_loss = perceptual_loss_fn(fake_resized, real_resized) if perceptual_loss_fn else 0

    # 3. 对抗损失
    adv_loss = nn.BCELoss()(pred_fake, torch.ones_like(pred_fake))

    # 总损失 = L1损失 + 0.001*对抗损失 + 感知损失
    return l1_loss + 0.001 * adv_loss + perceptual_loss


# 判别器损失函数
def discriminator_loss(pred_real, pred_fake):
    """
    参数:
        pred_real: 对真实图像的判别输出
        pred_fake: 对生成图像的判别输出
    返回:
        判别器总损失
    """
    # 真实图像应判为1，生成图像应判为0
    real_loss = nn.BCELoss()(pred_real, torch.ones_like(pred_real))
    fake_loss = nn.BCELoss()(pred_fake, torch.zeros_like(pred_fake))
    return (real_loss + fake_loss) / 2  # 平均损失


# 主训练函数
if __name__ == '__main__':
    # 1. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    torch.backends.cudnn.benchmark = True  # 启用CuDNN加速

    # 2. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 统一缩放
        transforms.ToTensor(),  # 转为张量
    ])

    # 3. 加载数据集
    dataset = CelebADataset(
        img_dir='C:/Users/86182/Desktop/archive/img_align_celeba',
        csv_path='C:/Users/86182/Desktop/archive/list_eval_partition.csv',
        mask_func=random_mask,
        transform=transform
    )
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4. 初始化模型
    generator = GeneratorUNet().to(device)  # 生成器
    discriminator = Discriminator().to(device)  # 判别器
    perceptual_loss_fn = PerceptualLoss().to(device)  # 感知损失

    # 5. 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # 6. 训练循环
    epochs = 100
    output_dir = 'output_images'  # 结果保存目录
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = 'saved_models'  # 模型保存目录

    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (corrupted, real, mask) in progress_bar:
            corrupted, real, mask = corrupted.to(device), real.to(device), mask.to(device)

            # --- 训练生成器 ---
            fake = generator(corrupted, mask)  # 生成修复图像
            pred_fake = discriminator(fake)  # 判别器判断
            g_loss = generator_loss(pred_fake, real, fake, mask, perceptual_loss_fn)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --- 训练判别器 ---
            pred_real = discriminator(real)  # 真实图像判别
            pred_fake = discriminator(fake.detach())  # 生成图像判别（阻止梯度回传）
            d_loss = discriminator_loss(pred_real, pred_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 更新进度条显示
            progress_bar.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())

        # 7. 定期保存可视化结果
        if epoch % 5 == 0 or epoch == epochs - 1:
            generator.eval()
            with torch.no_grad():
                # 取一个测试batch
                example_batch = next(iter(train_loader))
                corrupted_example = example_batch[0].to(device)
                real_example = example_batch[1].to(device)
                mask_example = example_batch[2].to(device)
                fake_example = generator(corrupted_example, mask_example)
            generator.train()

            # 绘制对比图：原始/遮挡/修复
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            for i in range(5):
                axes[0, i].imshow(real_example[i].permute(1, 2, 0).cpu().numpy())
                axes[0, i].set_title("original image")
                axes[1, i].imshow(corrupted_example[i].permute(1, 2, 0).cpu().numpy())
                axes[1, i].set_title("Occlusion image")
                axes[2, i].imshow(fake_example[i].permute(1, 2, 0).cpu().numpy())
                axes[2, i].set_title("Repair results")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}.png"))
            plt.close()
        # 每20个epoch保存一次模型
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            # 分别保存生成器和判别器
            torch.save({
                'state_dict': generator.state_dict(),
                'optimizer': g_optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(model_save_dir, f'generator_epoch_{epoch + 1}.pth'))
            print(f"\n模型已保存到: {model_save_dir}/generator_epoch_{epoch + 1}.pth")

    print("\n训练完成，保存最终模型...")
    torch.save({
        'state_dict': generator.state_dict(),
        'optimizer': g_optimizer.state_dict(),
    }, 'saved_models/final_model.pth')

    print("最终模型已保存到 saved_models/final_model.pth")