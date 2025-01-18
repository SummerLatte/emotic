import numpy as np
import os
from typing import Dict, List, Tuple
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
except:
    font = None
    print("警告: 无法加载中文字体，将使用默认字体")

def visualize_sample(arrays: Dict[str, np.ndarray], idx: int = None) -> None:
    """可视化一个随机样本的所有数据"""
    if not arrays:
        print("没有数据可供可视化")
        return
    
    # 如果没有指定索引，随机选择一个
    if idx is None:
        idx = random.randint(0, next(iter(arrays.values())).shape[0] - 1)
    
    print(f"\n可视化索引 {idx} 的样本:")
    
    # 创建一个大图
    plt.figure(figsize=(15, 8))
    
    # 获取数据集前缀（train_/val_/test_）
    prefix = next(iter(arrays.keys())).split('_')[0] + '_'
    
    # 显示图像数据
    image_data = {
        "Context": arrays.get(f"{prefix}context_arr.npy"),
        "Body": arrays.get(f"{prefix}body_arr.npy"),
        "Face": arrays.get(f"{prefix}face_arr.npy")
    }
    
    for i, (title, data) in enumerate(image_data.items()):
        if data is not None:
            plt.subplot(2, 3, i+1)
            plt.imshow(data[idx])
            plt.title(title)
            plt.axis('off')
    
    # 显示标签数据
    plt.subplot(2, 3, 4)
    cat_arr = arrays.get(f"{prefix}cat_arr.npy")
    if cat_arr is not None:
        cat_data = cat_arr[idx]
        plt.bar(range(len(cat_data)), cat_data)
        plt.title("Category Labels")
        plt.xticks([])  # 隐藏x轴标签
    
    plt.subplot(2, 3, 5)
    cont_arr = arrays.get(f"{prefix}cont_arr.npy")
    if cont_arr is not None:
        cont_data = cont_arr[idx]
        plt.bar(['V', 'A', 'D'], cont_data)
        plt.title("Continuous Labels (VAD)")
    
    # 显示其他信息
    plt.subplot(2, 3, 6)
    info_text = "Sample Info:\n"
    
    sizes_arr = arrays.get(f"{prefix}body_original_sizes.npy")
    if sizes_arr is not None:
        h, w = sizes_arr[idx]
        info_text += f"Original Size: {w}x{h}\n"
    
    face_arr = arrays.get(f"{prefix}has_face.npy")
    if face_arr is not None:
        has_face = "Yes" if face_arr[idx] == 1 else "No"
        info_text += f"Has Face: {has_face}"
    
    plt.text(0.1, 0.5, info_text, fontsize=10)
    plt.axis('off')
    plt.title("Other Info")
    
    plt.tight_layout()
    plt.show()

def load_dataset_arrays(base_dir: str, prefix: str) -> Dict[str, np.ndarray]:
    """加载指定前缀的所有数组文件"""
    arrays = {}
    expected_files = [
        f"{prefix}_context_arr.npy",
        f"{prefix}_body_arr.npy",
        f"{prefix}_cat_arr.npy",
        f"{prefix}_cont_arr.npy",
        f"{prefix}_body_original_sizes.npy",
        f"{prefix}_face_arr.npy",
        f"{prefix}_has_face.npy"
    ]
    
    for filename in expected_files:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            arrays[filename] = np.load(path)
        else:
            print(f"警告: 文件不存在 - {filename}")
    
    return arrays

def check_array_shapes(arrays: Dict[str, np.ndarray]) -> List[str]:
    """检查数组的形状是否符合预期"""
    errors = []
    if not arrays:
        return ["没有找到任何数组文件"]
    
    # 获取样本数量
    n_samples = None
    for name, arr in arrays.items():
        current_samples = arr.shape[0]
        if n_samples is None:
            n_samples = current_samples
        elif current_samples != n_samples:
            errors.append(f"样本数量不匹配: {name} 有 {current_samples} 个样本，期望 {n_samples} 个样本")
    
    # 检查具体的形状
    expected_shapes = {
        "context_arr.npy": (None, 224, 224, 3),
        "body_arr.npy": (None, 128, 128, 3),
        "cat_arr.npy": (None, 26),
        "cont_arr.npy": (None, 3),
        "body_original_sizes.npy": (None, 2),
        "face_arr.npy": (None, 128, 128, 3),
        "has_face.npy": (None,)
    }
    
    for name, arr in arrays.items():
        suffix = name.split("_", 1)[1]  # 移除前缀（train_/val_/test_）
        if suffix in expected_shapes:
            expected = list(expected_shapes[suffix])
            expected[0] = arr.shape[0]  # 替换None为实际的样本数量
            if arr.shape != tuple(expected):
                errors.append(f"形状不匹配: {name} 的形状为 {arr.shape}，期望 {tuple(expected)}")
    
    return errors

def check_data_types(arrays: Dict[str, np.ndarray]) -> List[str]:
    """检查数组的数据类型是否符合预期"""
    errors = []
    expected_types = {
        "context_arr.npy": np.uint8,
        "body_arr.npy": np.uint8,
        "cat_arr.npy": np.int32,
        "cont_arr.npy": np.float32,
        "body_original_sizes.npy": np.int32,
        "face_arr.npy": np.uint8,
        "has_face.npy": np.int32
    }
    
    for name, arr in arrays.items():
        suffix = name.split("_", 1)[1]  # 移除前缀
        if suffix in expected_types:
            expected_type = expected_types[suffix]
            if arr.dtype != expected_type:
                errors.append(f"数据类型不匹配: {name} 的类型为 {arr.dtype}，期望 {expected_type}")
    
    return errors

def check_value_ranges(arrays: Dict[str, np.ndarray]) -> List[str]:
    """检查数组的值范围是否合理"""
    errors = []
    
    for name, arr in arrays.items():
        if "context_arr.npy" in name or "body_arr.npy" in name or "face_arr.npy" in name:
            if arr.min() < 0 or arr.max() > 255:
                errors.append(f"值范围异常: {name} 的范围为 [{arr.min()}, {arr.max()}]，期望 [0, 255]")
        
        elif "has_face.npy" in name:
            unique_values = np.unique(arr)
            if not np.all(np.isin(unique_values, [0, 1])):
                errors.append(f"值范围异常: {name} 包含非法值 {unique_values}，期望只包含 0 和 1")
    
    return errors

def main():
    base_dir = "emotic_pre"
    datasets = ["train", "val", "test"]
    
    for dataset in datasets:
        print(f"\n检查 {dataset} 数据集:")
        print("-" * 50)
        
        # 加载数组
        arrays = load_dataset_arrays(base_dir, dataset)
        if not arrays:
            print("未找到任何数组文件")
            continue
        
        # 执行检查
        shape_errors = check_array_shapes(arrays)
        type_errors = check_data_types(arrays)
        range_errors = check_value_ranges(arrays)
        
        # 报告结果
        all_errors = shape_errors + type_errors + range_errors
        if all_errors:
            print("发现以下问题:")
            for error in all_errors:
                print(f"- {error}")
        else:
            print("所有检查通过！")
            print(f"样本数量: {next(iter(arrays.values())).shape[0]}")
            
            # 为每个数据集随机可视化一个样本
            visualize_sample(arrays)

if __name__ == "__main__":
    main() 