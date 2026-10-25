# ============================================
# 数据准备脚本 - prepare_data.py
# 功能：将原始数据整理为YOLO要求的目录结构
# ============================================

# 导入所需库
import os           # 操作系统接口，用于文件/文件夹操作
import shutil       # 高级文件操作，用于复制文件


def prepare_yolo_dataset(root_dir):
    """
    准备YOLO格式数据集
    
    参数：
        root_dir: 项目根目录，默认为当前目录 '.'
    """
    
    # ========== 定义路径 ==========
    # 原始数据路径
    train_img_src = os.path.join(root_dir, 'train', 'train')      # 原始训练图片位置
    train_label_src = os.path.join(root_dir, 'train', 'train_label')  # 原始训练标签位置
    test_img_src = os.path.join(root_dir, 'test', 'test')         # 原始测试图片位置
    
    # YOLO要求的目标路径
    train_img_dst = os.path.join(root_dir, 'train', 'images')     # 目标：训练图片目录
    train_label_dst = os.path.join(root_dir, 'train', 'labels')   # 目标：训练标签目录
    test_img_dst = os.path.join(root_dir, 'test', 'images')       # 目标：测试图片目录

    # ========== 创建目录 ==========
    # os.makedirs() 创建文件夹，exist_ok=True表示已存在则不报错
    os.makedirs(train_img_dst, exist_ok=True)
    os.makedirs(train_label_dst, exist_ok=True)
    os.makedirs(test_img_dst, exist_ok=True)

    # ========== 复制训练图片 ==========
    # os.listdir() 列出目录下所有文件
    # str.endswith('.png') 判断文件名是否以.png结尾
    for img_file in os.listdir(train_img_src):
        if img_file.endswith('.png'):
            src_path = os.path.join(train_img_src, img_file)   # 源文件完整路径
            dst_path = os.path.join(train_img_dst, img_file)   # 目标文件完整路径
            shutil.copy(src_path, dst_path)                     # shutil.copy() 复制文件

    # ========== 复制训练标签 ==========
    for label_file in os.listdir(train_label_src):
        if label_file.endswith('.txt'):
            src_path = os.path.join(train_label_src, label_file)
            dst_path = os.path.join(train_label_dst, label_file)
            shutil.copy(src_path, dst_path)

    # ========== 复制测试图片 ==========
    for img_file in os.listdir(test_img_src):
        if img_file.endswith('.png'):
            src_path = os.path.join(test_img_src, img_file)
            dst_path = os.path.join(test_img_dst, img_file)
            shutil.copy(src_path, dst_path)

    # ========== 打印统计信息 ==========
    # len(os.listdir()) 统计目录下文件数量
    print("Dataset prepared successfully!")
    print(f"Training images: {len(os.listdir(train_img_dst))}")
    print(f"Training labels: {len(os.listdir(train_label_dst))}")
    print(f"Test images: {len(os.listdir(test_img_dst))}")


# ========== 主程序入口 ==========
# if __name__ == '__main__': 判断是否直接运行此文件
# 当直接运行时，__name__ 等于 '__main__'，代码执行
# 如果被其他文件导入，则不执行
if __name__ == '__main__':
    root_dir = '.'  # 当前目录
    prepare_yolo_dataset(root_dir)