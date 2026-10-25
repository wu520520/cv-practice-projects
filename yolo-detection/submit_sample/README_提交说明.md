# 提交样例说明

本压缩包是提交格式样例包，包含：

- `submit/`：预测 TXT 样例目录
- `README_提交说明.md`：本说明文件

## 正式提交要求

正式提交时不要带本 README，也不要带训练集、测试集或其他说明文件。正式提交的 `submit.zip` 只保留 `submit/` 目录及其中的预测 TXT 文件。

评分脚本实际校验规则：

1. `submit/` 内必须包含完整 200 个测试图对应的 TXT 预测文件：`test_001.txt` 到 `test_200.txt`。
2. 每个 TXT 文件名必须与测试图片同名，扩展名为 `.txt`。
3. 若某张图没有检测结果，也必须提交对应的空 TXT 文件。
4. 不允许缺少预测 TXT，也不允许多提交测试集外的 TXT。
5. 每行预测格式必须为 6 列：`<class_id> <confidence> <x_center> <y_center> <width> <height>`。
6. `class_id` 取值：`0=handup`，`1=write`，`2=listen`，`3=stand`。
7. `confidence`、`x_center`、`y_center`、`width`、`height` 都必须在 `[0, 1]` 范围内。
8. `width` 和 `height` 必须大于 0。
9. 每张图最多评估 20 个框；若超过 20 个，评分脚本会按 `confidence` 从高到低保留前 20 个。

评分指标为 mAP@0.5，分数越高越好。当前 `submit_sample.zip` 内 200 个 TXT 都为空，只用于说明提交结构，得分为 0。
