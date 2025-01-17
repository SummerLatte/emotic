# EMOTIC 数据格式说明

本文档描述了数据预处理脚本生成的数据格式。数据主要由两个脚本生成：`mat2py.py` 和 `check_faces.py`。

## 1. mat2py.py 生成的数据

该脚本将原始的 EMOTIC MAT 文件转换为 numpy 数组格式。

### 1.1 生成文件列表

对于每个数据集（train/val/test），生成以下文件：

- `{dataset}_context_arr.npy`: 场景图像数组
- `{dataset}_body_arr.npy`: 人物图像数组
- `{dataset}_cat_arr.npy`: 离散情绪标签数组
- `{dataset}_cont_arr.npy`: 连续情绪值数组
- `{dataset}_body_original_sizes.npy`: 人物图像原始尺寸数组

### 1.2 数据格式说明

1. 图像数组
   - `context_arr`: 形状为 (N, 224, 224, 3)，场景图像
   - `body_arr`: 形状为 (N, 128, 128, 3)，人物图像
   - 数据类型：uint8，值范围 [0, 255]
   - RGB 格式

2. 标签数组
   - `cat_arr`: 形状为 (N, 26)，离散情绪标签，数据类型 int32
   - `cont_arr`: 形状为 (N, 3)，连续情绪值，数据类型 float32

3. 原始尺寸数组
   - `body_original_sizes`: 形状为 (N, 2)，格式为 [height, width]
   - 数据类型：int32

## 2. check_faces.py 生成的数据

该脚本对人物图像进行人脸检测，生成人脸相关的数据。

### 2.1 生成文件列表

对于每个数据集（train/val/test），生成以下文件：

- `{dataset}_face_arr.npy`: 人脸图像数组
- `{dataset}_has_face.npy`: 人脸存在标记数组

### 2.2 数据格式说明

1. 人脸图像数组 (`face_arr`)
   - 形状：(N, 128, 128, 3)
   - 数据类型：uint8，值范围 [0, 255]
   - RGB 格式
   - 对于检测到人脸的图像：保存最大的人脸区域
   - 对于未检测到人脸的图像：保存调整大小后的原始图像

2. 人脸标记数组 (`has_face`)
   - 形状：(N,)
   - 数据类型：int32
   - 值含义：
     - 1：检测到人脸
     - 0：未检测到人脸

### 2.3 辅助文件

脚本同时生成以下辅助文件：

1. `detection_results.json`：详细的检测结果，包含：
   - 总图像数
   - 检测到人脸的图像数
   - 每个图像的具体检测信息（边界框、检测方法等）

2. `results.txt`：简要的统计信息，包含：
   - 数据集名称
   - 处理的图像总数
   - 检测到人脸的图像数量
   - 未检测到人脸的图像索引列表

## 3. 数据使用说明

1. 加载数据示例：
```python
import numpy as np

# 加载图像和标签
context_images = np.load('train_context_arr.npy')
body_images = np.load('train_body_arr.npy')
cat_labels = np.load('train_cat_arr.npy')
cont_labels = np.load('train_cont_arr.npy')

# 加载人脸数据
face_images = np.load('train_face_arr.npy')
has_face = np.load('train_has_face.npy')
```

2. 注意事项：
   - 所有图像数组都是 RGB 格式
   - 使用 OpenCV 显示时需要转换为 BGR 格式
   - 图像数组的值范围为 [0, 255]，使用前建议归一化到 [0, 1]
   - 人脸检测结果可用于筛选数据或作为额外的特征 