# ============================================
# 模型评估脚本 - eval.py
# 功能：评估训练好的模型性能，计算mAP等指标
# ============================================

# 导入所需库
from ultralytics import YOLO
import os
import torch


def evaluate_model(model_path, data_yaml='data.yaml'):
    """
    评估模型性能
    
    参数：
        model_path: 训练好的模型权重路径
        data_yaml: 数据配置文件路径
    """
    
    # 解决OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 加载训练好的模型
    model = YOLO(model_path)

    # 选择设备（GPU优先）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 开始评估 ==========
    results = model.val(
        data=data_yaml,
        imgsz=640,                  # 输入图片尺寸
        conf=0.25,                  # 置信度阈值：低于25%的检测不计入
        iou=0.5,                    # IoU阈值：重叠超过50%算正确匹配
        device=device,              # 使用的设备
        verbose=True,               # 打印详细信息
        save_json=True,             # 保存评估结果为JSON
        project='runs/val',         # 评估结果保存目录
        name='exp'                  # 实验名称
    )

    # ========== 打印评估结果 ==========
    print("\nEvaluation Results:")
    print(f"mAP@0.5: {results.box.map50:.4f}")       # IoU=0.5时的平均精度
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")    # IoU从0.5到0.95的平均精度
    print(f"Precision: {results.box.mp:.4f}")        # 精确率
    print(f"Recall: {results.box.mr:.4f}")           # 召回率

    # 打印每个类别的详细指标
    for i, cls in enumerate(results.names):
        print(f"\nClass {i} ({cls}):")
        print(f"  AP@0.5: {results.box.ap50[i]:.4f}")     # 该类别AP@0.5
        print(f"  Precision: {results.box.p[i]:.4f}")     # 该类别精确率
        print(f"  Recall: {results.box.r[i]:.4f}")        # 该类别召回率


# ========== 主程序入口 ==========
if __name__ == '__main__':
    import sys  # 用于获取命令行参数
    
    # 默认模型路径（上一次训练的最佳模型）
    model_path = 'runs/detect/runs/train/exp-3/weights/best.pt'
    
    # 如果命令行有传入参数，则使用传入的模型路径
    # sys.argv[0] 是脚本名，sys.argv[1] 是第一个参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
    else:
        evaluate_model(model_path)