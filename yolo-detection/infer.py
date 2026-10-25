# ============================================
# 推理与提交生成脚本 - infer.py
# 功能：使用训练好的模型对测试图片进行预测
#       生成符合比赛要求的提交文件
# ============================================

# 导入所需库
import os
import zipfile                  # 用于打包zip文件
from ultralytics import YOLO
from PIL import Image           # 用于获取图片尺寸


def infer_and_generate_submit(model_path, test_dir, submit_dir):
    """
    推理并生成提交文件
    
    参数：
        model_path: 训练好的模型权重路径
        test_dir: 测试图片目录
        submit_dir: 提交结果保存目录
    """
    
    # 加载训练好的模型
    model = YOLO(model_path)

    # 创建提交目录（不存在则创建）
    os.makedirs(submit_dir, exist_ok=True)

    # ========== 获取测试图片列表 ==========
    # 列表推导式：遍历test_dir目录，筛选出所有.png文件，并按字母排序
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    # ========== 遍历测试图片进行预测 ==========
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)

        # 模型推理
        results = model(
            img_path,
            imgsz=640,          # 输入图片尺寸
            conf=0.15,          # 置信度阈值：较低，尽量多检测
            iou=0.5,            # IoU阈值：0.5
            max_det=20,         # 每张图最多检测20个目标
            augment=True        # 推理时使用数据增强（TTA）
        )

        # 生成输出txt文件名（去掉图片扩展名）
        base_name = os.path.splitext(img_file)[0]
        txt_path = os.path.join(submit_dir, f"{base_name}.txt")

        # ========== 解析预测结果 ==========
        detections = []
        for result in results:
            boxes = result.boxes  # 获取检测框
            
            for box in boxes:
                # 获取检测框信息
                cls_id = int(box.cls[0].item())             # 类别编号（0-3）
                conf = box.conf[0].item()                   # 置信度（0-1）
                x_center = box.xywh[0][0].item()            # 检测框中心x坐标
                y_center = box.xywh[0][1].item()            # 检测框中心y坐标
                width = box.xywh[0][2].item()               # 检测框宽度
                height = box.xywh[0][3].item()              # 检测框高度

                # 获取原始图片尺寸（用于坐标归一化）
                img = Image.open(img_path)
                img_width, img_height = img.size

                # ========== 坐标归一化 ==========
                # 将像素坐标转换为0-1之间的比例值
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # 边界裁剪：确保坐标在0-1范围内（防止超出图片边界）
                x_center_norm = max(0.0, min(1.0, x_center_norm))
                y_center_norm = max(0.0, min(1.0, y_center_norm))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))

                # 保存检测结果
                detections.append({
                    'cls_id': cls_id,
                    'conf': conf,
                    'x_center': x_center_norm,
                    'y_center': y_center_norm,
                    'width': width_norm,
                    'height': height_norm
                })

        # ========== 排序和截断 ==========
        # 按置信度从高到低排序
        detections.sort(key=lambda x: x['conf'], reverse=True)
        # 只保留前20个检测结果
        detections = detections[:20]

        # ========== 保存预测结果到txt文件 ==========
        # 输出格式：<class_id> <confidence> <x_center> <y_center> <width> <height>
        with open(txt_path, 'w') as f:
            for det in detections:
                f.write(f"{det['cls_id']} {det['conf']:.6f} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n")

        # 如果没有检测结果，创建空文件
        if not detections:
            with open(txt_path, 'w') as f:
                pass

        # 打印处理进度
        print(f"Processed: {img_file} -> {len(detections)} detections")

    # ========== 打包提交文件 ==========
    zip_path = 'submit.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for txt_file in os.listdir(submit_dir):
            txt_path = os.path.join(submit_dir, txt_file)
            zf.write(txt_path, os.path.join('submit', txt_file))

    # 打印完成信息
    print(f"\nSubmit zip generated: {zip_path}")
    print(f"Total test images processed: {len(test_images)}")


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # 模型路径（优化版训练的最佳模型）
    model_path = 'runs/train/exp_opt/weights/best.pt'
    # 测试图片目录
    test_dir = 'test/images'
    # 提交结果保存目录
    submit_dir = 'submit'

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
    else:
        infer_and_generate_submit(model_path, test_dir, submit_dir)