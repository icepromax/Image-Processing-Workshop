import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uuid
import uvicorn
from fastapi.staticfiles import StaticFiles
from CycleGan.style_transfer import MonetStyleTransfer  # 导入风格迁移类
from GAN_Unet.gan_unet import auto_infer_with_merge
from GAN_Unet.train import GeneratorUNet
from realesrgan import RealESRGANer  # 导入Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet  # 导入模型架构
import os

# 配置
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 这行代码会在目录不存在时自动创建


app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"]  # 关键头暴露
)

device='cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型
MODELS = {
    "cyclegan": MonetStyleTransfer("CycleGan/G_BA_19.pth"),
    "real-esrgan": None,  # 延迟加载
    "Gan_unet": None  # 新增修复模型
}


def init_inpainting_model(device=device):
    """初始化图像修复模型"""
    device = device
    generator = GeneratorUNet().to(device)
    checkpoint = torch.load("GAN_Unet/saved_model/60.pth", map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    return generator

def init_realesrgan(device=device):
    """初始化Real-ESRGAN模型

    功能：
        创建并配置Real-ESRGAN超分辨率模型的实例，用于图像放大和增强

    返回：
        RealESRGANer: 配置好的超分辨率处理器实例
    """
    # 1. 创建RRDBNet模型架构
    # 参数说明：
    # num_in_ch=3   : 输入图像的通道数（RGB为3通道）
    # num_out_ch=3  : 输出图像的通道数
    # num_feat=64   : 特征图的初始通道数
    # num_block=23  : 残差块的数量（影响模型深度和性能）
    # num_grow_ch=32: 特征图通道数的增长量
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    # 2. 创建RealESRGAN处理器
    upsampler = RealESRGANer(
        scale=4,  # 放大倍数（4倍超分辨率）
        model_path="realesrgan/weights/RealESRGAN_x4plus.pth",  # 预训练权重路径
        model=model,  # 使用的模型架构
        tile=256,  # 图像分块处理的尺寸（大图像分块处理避免内存溢出）
        tile_pad=10,  # 分块重叠像素（消除块间接缝）
        pre_pad=0,  # 预处理填充像素
        half=False, # 是否使用半精度浮点运算（False保持全精度）
    )

    return upsampler

@app.on_event("startup")
async def load_models():
    MODELS["real-esrgan"] = init_realesrgan()
    MODELS["Gan_unet"] = init_inpainting_model()


@app.post("/repair")
async def repair_image(
        image: UploadFile = File(..., description="上传的图片文件"),
        model: str = Form(..., description="选择的模型类型")
):
    """
    图像处理接口

    功能：
        接收上传的图片文件，根据选择的AI模型进行图像修复/增强处理

    参数:
        - image: 图片文件 (multipart/form-data格式)
        - model: 模型名称 (form-data格式)，当前支持：
            * cyclegan - 风格迁移处理
            * real-esrgan - 超分辨率增强
            * 其他 - 原图返回

    返回:
        JSON响应包含处理状态和结果图片URL

    异常:
        - HTTP 400: 文件类型错误/图片读取失败
        - HTTP 500: 处理过程中出现错误
    """
    try:
        # ==================== 1. 文件验证 ====================
        # 检查文件扩展名，仅允许jpg/jpeg/png格式
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(
                status_code=400,
                detail="只支持JPG/PNG格式图片"
            )

        # ==================== 2. 文件存储准备 ====================
        # 生成唯一ID避免文件名冲突
        file_id = str(uuid.uuid4())
        # 获取原始文件扩展名
        ext = os.path.splitext(image.filename)[1]
        # 构建输入/输出文件路径
        input_path = f"{UPLOAD_DIR}/{file_id}_input{ext}"  # 原始文件保存路径
        output_path = f"{UPLOAD_DIR}/{file_id}_output.jpg"  # 处理结果保存路径(统一转为jpg)

        # 保存上传的原始文件
        with open(input_path, "wb") as buffer:
            content = await image.read()  # 异步读取文件内容
            buffer.write(content)

        # ==================== 3. 模型处理逻辑 ====================
        # 根据选择的模型类型执行不同处理流程
        if model == "cyclegan":
            # ----- CycleGAN风格迁移处理 -----
            success = MODELS["cyclegan"].process_image(
                input_path,
                output_path
            )
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="风格迁移处理失败"
                )

        elif model == "real-esrgan":
            # ----- Real-ESRGAN超分辨率处理 -----
            # 延迟加载模型(节省内存)
            if MODELS["real-esrgan"] is None:
                MODELS["real-esrgan"] = init_realesrgan()

            # 使用OpenCV读取图片
            img = cv2.imread(input_path)
            if img is None:  # 读取失败检查
                raise HTTPException(
                    status_code=400,
                    detail="图片读取失败"
                )

            # 执行4倍超分辨率增强
            output, _ = MODELS["real-esrgan"].enhance(
                img,
                outscale=4  # 放大倍数
            )
            # 保存结果
            cv2.imwrite(output_path, output)
        elif model == "Gan_unet":
            if MODELS["Gan_unet"] is None:
                MODELS["Gan_unet"] = init_inpainting_model()

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            result = auto_infer_with_merge(
                generator=MODELS["Gan_unet"],
                image_path=input_path,
                device=device
            )
            result["final_result"].save(output_path, quality=95)

        else:
            # ----- 其他模型情况 -----
            # 直接返回原图(示例代码，实际可扩展其他模型)
            with open(output_path, "wb") as buffer:
                buffer.write(content)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ==================== 4. 返回处理结果 ====================
        return JSONResponse({
            "status": "success",
            "model_used": model,  # 实际使用的模型
            "repairedImage": f"/uploads/{file_id}_output.jpg"  # 结果文件URL
        })

    # ==================== 异常处理 ====================
    except HTTPException:  # 已知异常直接抛出
        raise
    except Exception as e:  # 未知异常捕获
        raise HTTPException(
            status_code=500,
            detail=f"图片处理失败: {str(e)}"
        )


@app.get("/result/{filename}")
async def get_result(filename: str):
    """获取处理后的图片"""
    file_path = f"{UPLOAD_DIR}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="文件不存在")
    return FileResponse(file_path)


@app.get("/")
def health_check():
    return {"status": "running", "message": "图像处理API服务已启动"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
