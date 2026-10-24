# 工业零件检测（OpenCV 传统图像处理 Demo）

基于 **OpenCV 传统图像处理**的工业零件检测小项目，从一张工业零件图片 `part.png` 出发，完成灰度化、去噪、形态学、轮廓筛选与可视化标注的完整流水线。

## ✨ 处理流水线

1. **读取图像并灰度化**：`cv.imread` → `cv.cvtColor` 转灰度
2. **中值滤波去噪**：`cv.medianBlur`，对椒盐噪声效果最佳
3. **形态学操作**：
   - 自适应阈值二值化（高斯加权）
   - 闭运算：填充零件内部细小孔洞
   - 开运算：去除微小噪点
4. **轮廓查找与筛选**：`cv.findContours`（RETR_EXTERNAL），按面积过滤有效零件（保留大于图像 1.2% 的轮廓）
5. **可视化标注**：在原图绘制绿色轮廓、红色外接矩形、蓝色中心坐标、零件标签
6. **保存结果**：输出 `result_part.png`

## 🛠 技术栈

Python | OpenCV (cv2)

## 📁 目录结构

```
opencv-parts-detection/
├── part.ipynb           # 完整处理流程 Notebook
├── data/
│   └── part.png         # 输入：工业零件原图
├── result_part.png      # 输出：可视化结果图
├── requirements.txt
└── README.md
```

## 🚀 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 用 Jupyter 打开并运行
jupyter notebook part.ipynb
```

运行后会在当前目录生成 `result_part.png`（检测结果可视化）。

## 📌 说明

- 本项目为传统计算机视觉（非深度学习）图像处理演示，体现 OpenCV 基础能力
- 输入图像与 notebook 使用相对路径，克隆仓库后可直接运行
