# 计算机视觉实战项目

计算机视觉（CV）方向实践项目集合，覆盖从 **OpenCV 传统图像处理** 到 **YOLO 深度学习目标检测** 的完整技术路线，逐步进阶。

## 📦 子项目

### 1. parts-detection — OpenCV 工业零件检测
基于 OpenCV 传统图像处理的工业零件检测，完成灰度化、中值滤波去噪、形态学、轮廓查找与可视化标注的完整流水线。

- 技术栈：Python + OpenCV
- 入口：[`parts-detection/part.ipynb`](parts-detection/part.ipynb)
- 输入 `part.png` → 输出 `result_part.png`

### 2. yolo-detection — YOLO 学生课堂行为识别
基于 Ultralytics YOLO 的目标检测比赛项目，识别课堂中学生 **举手 / 写字 / 听课 / 站立** 四种行为。

- 技术栈：Python + Ultralytics YOLO + PyTorch
- 流程：`prepare_data.py` 数据准备 → `train.py` 模型训练 → `infer.py` 推理并生成提交 → `eval.py` 评估（mAP）
- 4 类：`handup`（举手）、`write`（写字）、`listen`（听课）、`stand`（站立）

## 🛠 技术栈

Python | OpenCV | Ultralytics YOLO | PyTorch

## 📁 目录结构

```
cv-practice-projects/
├── parts-detection/          # OpenCV 传统图像处理
│   ├── part.ipynb
│   ├── data/part.png
│   └── result_part.png
├── yolo-detection/           # YOLO 目标检测
│   ├── prepare_data.py       # 数据准备
│   ├── train.py              # 模型训练
│   ├── infer.py              # 推理与提交生成
│   ├── eval.py               # 模型评估
│   ├── data.yaml             # 数据集配置（4类）
│   ├── yolo26n.pt            # 预训练权重
│   ├── yolov8n.pt            # 预训练权重
│   ├── samples/              # 演示样本图
│   └── submit_sample/        # 提交格式样例
├── requirements.txt
└── README.md
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. OpenCV 零件检测
jupyter notebook parts-detection/part.ipynb

# 3. YOLO 学生行为识别（需自行准备完整数据集）
cd yolo-detection
python train.py    # 训练
python infer.py    # 推理并生成提交
python eval.py     # 评估
```

## 📌 说明

- YOLO 项目为比赛实战，**完整训练/测试数据集较大且为比赛私有**，仓库仅提供代码、预训练权重与少量演示样本，完整数据请按比赛说明获取
- 各子项目均使用相对路径，克隆仓库后可运行（YOLO 需先准备数据）
