# ============================================
# 模型训练脚本 - train.py
# 功能：使用YOLOv11n预训练模型进行迁移学习
#       训练出能够识别学生行为的模型
# ============================================

# 导入所需库
from ultralytics import YOLO  # 导入YOLO模型类
import os                      # 文件操作
import torch                   # PyTorch深度学习框架


def train_model():
    """训练YOLOv11n模型"""
    
    # 打印标题信息
    print("=" * 60)
    print("YOLOv11n 训练脚本 (优化版)")
    print("=" * 60)

    # ========== 解决OpenMP冲突 ==========
    # Windows系统下，PyTorch和其他库可能存在OpenMP库冲突
    # 设置此环境变量可避免程序崩溃
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # ========== 设备选择 ==========
    # torch.cuda.is_available() 检查是否有可用的NVIDIA GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("GPU 不可用，使用 CPU 训练")

    print(f"设备: {device}")
    print(f"数据配置文件: data.yaml")
    print("-" * 60)

    # ========== 加载预训练权重 ==========
    # 使用仓库内的预训练权重文件（yolov8n.pt 或 yolo26n.pt），如使用其他权重请修改此路径
    weight_file = 'yolov8n.pt'
    
    # os.path.exists() 检查文件是否存在
    if not os.path.exists(weight_file):
        print(f"\n错误: 权重文件不存在: {weight_file}")
        print("请检查文件路径是否正确")
        return

    print(f"\n加载模型: {weight_file}")
    try:
        # YOLO(weight_file) 创建YOLO模型实例并加载预训练权重
        # 迁移学习：利用COCO数据集上训练的先验知识
        model = YOLO(weight_file)
        print("模型加载成功，开始训练...")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保权重文件已正确下载")
        return

    # 打印优化参数信息
    print("-" * 60)
    print("优化参数:")
    print("  - epochs: 200")
    print("  - batch: 8")
    print("  - imgsz: 640")
    print("  - cos_lr: True")
    print("  - dropout: 0.1")
    print("  - box: 7.5")
    print("  - cls: 0.5")
    print("-" * 60)

    # ========== 开始训练 ==========
    results = model.train(
        # ---------- 基础参数 ----------
        data='data.yaml',           # 数据配置文件路径
        epochs=200,                 # 训练轮数：把所有数据看200遍
        batch=8,                    # 每批处理8张图片
        imgsz=640,                  # 图片统一缩放到640×640像素
        
        # ---------- 学习率参数 ----------
        lr0=0.01,                   # 初始学习率：学习速度
        lrf=0.01,                   # 最终学习率：学习结束时的速度
        momentum=0.937,             # 动量：加速梯度下降
        weight_decay=0.0005,        # 权重衰减：防止过拟合
        warmup_epochs=5,            # 热身轮数：前5轮慢慢增加学习率
        warmup_momentum=0.8,        # 热身阶段动量
        warmup_bias_lr=0.1,         # 热身阶段偏置学习率
        
        # ---------- 损失函数权重 ----------
        box=7.5,                    # 边界框损失权重
        cls=0.5,                    # 分类损失权重
        dfl=1.5,                    # 分布焦点损失权重
        
        # ---------- 训练控制参数 ----------
        patience=20,                # 早停耐心值：20轮没进步就停止
        verbose=True,               # 打印详细训练信息
        save=True,                  # 保存模型权重
        save_period=10,             # 每10轮保存一次
        plots=True,                 # 生成训练曲线图
        augment=True,               # 启用数据增强
        cache=False,                # 是否缓存数据到内存
        device=device,              # 使用的设备（cuda/cpu）
        project='runs/train',       # 训练结果保存目录
        name='exp_opt',             # 实验名称
        workers=0,                  # 数据加载线程数（Windows设为0）
        
        # ---------- 高级优化参数 ----------
        cos_lr=True,                # 使用余弦学习率调度（先快后慢）
        dropout=0.1,                # Dropout正则化：随机丢弃10%神经元
        
        # ---------- 数据增强参数 ----------
        close_mosaic=15,            # 最后15轮关闭马赛克增强
        degrees=10.0,               # 随机旋转角度范围：±10°
        translate=0.1,              # 随机平移范围：±10%
        scale=0.5,                  # 随机缩放范围：0.5~1.5倍
        fliplr=0.5,                 # 水平翻转概率：50%
        mosaic=1.0,                 # 马赛克增强概率：100%
        mixup=0.1,                  # MixUp增强概率：10%
        copy_paste=0.1              # Copy-Paste增强概率：10%
    )

    # ========== 训练完成 ==========
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型保存路径: {results.save_dir}")
    print("=" * 60)


# ========== 主程序入口 ==========
if __name__ == '__main__':
    train_model()