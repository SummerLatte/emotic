import numpy as np
import cv2
import argparse
import mediapipe as mp
import os
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/developer/code/emoticFirst/Emotic/emotic_pre', help='Directory containing the preprocessed data')
    parser.add_argument('--max_images', type=int, default=1000, help='Maximum number of images to process')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'val', 'test'], help='Which dataset to check')
    parser.add_argument('--save_dir', type=str, default='face_check_results', help='Directory to save results')
    parser.add_argument('--target_size', type=int, default=128, help='Target size for face images')
    return parser.parse_args()

class ImagePreprocessor:
    @staticmethod
    def enhance_contrast(image):
        """增强对比度"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def adjust_gamma(image, gamma=1.2):
        """调整gamma值"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    @staticmethod
    def denoise(image):
        """降噪"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    @staticmethod
    def sharpen(image):
        """锐化"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

class MultiFaceDetector:
    def __init__(self):
        # 初始化MediaPipe检测器
        mp_face_detection = mp.solutions.face_detection
        self.mp_detector = mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.3
        )
        
        # 初始化Haar Cascade检测器
        self.haar_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 初始化图像预处理器
        self.preprocessor = ImagePreprocessor()

    def detect_with_mediapipe(self, image):
        """使用MediaPipe检测人脸，返回检测结果和边界框列表"""
        results = self.mp_detector.process(image)
        if results.detections:
            boxes = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
                w, h = int(bbox.width * iw), int(bbox.height * ih)
                boxes.append((x, y, w, h))
            return True, boxes
        return False, []

    def detect_with_haar(self, image):
        """使用Haar Cascade检测人脸，返回检测结果和边界框列表"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.haar_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(20, 20)
        )
        if len(faces) > 0:
            boxes = [(x, y, w, h) for (x, y, w, h) in faces]
            return True, boxes
        return False, []

    def detect(self, image):
        """组合所有检测器的结果，返回检测结果和边界框列表"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        try:
            # 准备不同的图像版本
            image_versions = [
                ("original", image_rgb),
                ("enhanced", self.preprocessor.enhance_contrast(image_rgb)),
                ("gamma", self.preprocessor.adjust_gamma(image_rgb)),
                ("denoised", self.preprocessor.denoise(image_rgb)),
                ("sharpened", self.preprocessor.sharpen(image_rgb))
            ]
            
            all_boxes = []
            # 对每个图像版本尝试所有检测器
            for version_name, img in image_versions:
                # 尝试MediaPipe
                has_face, boxes = self.detect_with_mediapipe(img)
                if has_face:
                    all_boxes.extend([(box, version_name, "mediapipe") for box in boxes])
                    continue
                
                # 尝试Haar Cascade
                has_face, boxes = self.detect_with_haar(img)
                if has_face:
                    all_boxes.extend([(box, version_name, "haar") for box in boxes])
            
            if all_boxes:
                return True, all_boxes
            return False, []
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return False, []

def get_largest_face(boxes_with_info):
    """选择最大的人脸区域"""
    max_area = 0
    largest_face = None
    largest_method_info = None
    
    for box, *method_info in boxes_with_info:
        x, y, w, h = box
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = box
            largest_method_info = method_info
    
    return largest_face, largest_method_info

def process_face_region(image, box, target_size):
    """处理人脸区域：裁剪、调整大小并返回"""
    if box is None:
        # 如果没有检测到人脸，直接调整整个图像大小
        return cv2.resize(image, (target_size, target_size))
    
    x, y, w, h = box
    
    # 确保坐标不越界
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # 提取人脸区域
    face_region = image[y:y+h, x:x+w]
    
    # 调整大小
    face_resized = cv2.resize(face_region, (target_size, target_size))
    
    return face_resized

def main():
    args = parse_args()
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # 初始化多重人脸检测器
    detector = MultiFaceDetector()
    
    # 加载数据
    print(f"Loading {args.dataset} data...")
    data_dir = args.data_dir
    print(f"Looking for files in: {data_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
        
    body_arr_path = os.path.join(data_dir, f'{args.dataset}_body_arr.npy')
    original_sizes_path = os.path.join(data_dir, f'{args.dataset}_body_original_sizes.npy')
    
    print(f"Loading body array from: {body_arr_path}")
    body_images = np.load(body_arr_path)
    print(f"Body array shape: {body_images.shape}")
    
    try:
        print(f"Loading original sizes from: {original_sizes_path}")
        original_sizes = np.load(original_sizes_path)
        print(f"Original sizes array shape: {original_sizes.shape}")
        print("Successfully loaded original size information")
        use_original = True
    except Exception as e:
        print(f"Error loading original sizes: {str(e)}")
        print("Original size information not found")
        use_original = False
    
    # 限制处理图像数量
    num_images = min(args.max_images, len(body_images))
    body_images = body_images[:num_images]
    if use_original:
        original_sizes = original_sizes[:num_images]
    
    # 准备数组存储所有图像的处理结果
    face_images = []  # 存储所有图像（包括没有人脸的）
    has_face_list = []  # 记录每张图像是否包含人脸
    face_boxes = []  # 记录人脸框位置（没有人脸的记录为None）
    
    # 统计结果
    results = {
        'total_images': num_images,
        'images_with_faces': 0,
        'no_face_indices': [],
        'face_detections': []  # 存储所有检测到的人脸信息
    }
    
    # 处理每张图像
    print("Processing images...")
    for idx in tqdm(range(num_images)):
        image = body_images[idx]
        
        # 如果有原始尺寸信息，先将图像恢复到原始尺寸
        if use_original:
            h, w = original_sizes[idx]
            image_original = cv2.resize(image, (w, h))
            has_face, boxes_with_info = detector.detect(image_original)
            
            # 如果原始尺寸检测失败，尝试128x128版本
            if not has_face:
                has_face, boxes_with_info = detector.detect(image)
                if has_face:
                    image_original = image  # 如果在128x128版本中检测到，使用该版本
        else:
            has_face, boxes_with_info = detector.detect(image)
            image_original = image
        
        if has_face:
            results['images_with_faces'] += 1
            
            # 选择最大的人脸
            largest_face, method_info = get_largest_face(boxes_with_info)
            
            # 处理人脸区域
            face_image = process_face_region(image_original, largest_face, args.target_size)
            
            # 记录信息
            face_images.append(face_image)
            has_face_list.append(1)  # 1表示有人脸
            face_boxes.append(largest_face)
            
            # 记录这张图像的人脸信息
            results['face_detections'].append({
                'image_index': idx,
                'face': {
                    'box': [int(x) for x in largest_face],
                    'method': method_info[1],
                    'image_version': method_info[0]
                }
            })
        else:
            results['no_face_indices'].append(idx)
            
            # 处理没有人脸的图像
            face_image = process_face_region(image_original, None, args.target_size)
            
            # 记录信息
            face_images.append(face_image)
            has_face_list.append(0)  # 0表示没有人脸
            face_boxes.append(None)
            
            # 保存没有检测到人脸的图像
            save_dir = os.path.join(args.save_dir, f'no_face_{idx}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, 'original.jpg'),
                       cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR))
    
    # 转换为numpy数组并保存
    face_images = np.array(face_images)
    has_face_arr = np.array(has_face_list)
    
    print(f"\nSaving arrays...")
    print(f"Face images shape: {face_images.shape}")
    print(f"Has face array shape: {has_face_arr.shape}")
    
    # 保存为与mat2py相同的格式
    np.save(os.path.join(args.save_dir, f'{args.dataset}_face_arr.npy'), face_images)
    np.save(os.path.join(args.save_dir, f'{args.dataset}_has_face.npy'), has_face_arr)
    
    # 打印统计结果
    print("\nResults:")
    print(f"Using original size information: {use_original}")
    print(f"Total images processed: {results['total_images']}")
    print(f"Images with faces: {results['images_with_faces']} ({results['images_with_faces']/results['total_images']*100:.2f}%)")
    print(f"Images without faces: {len(results['no_face_indices'])}")
    
    # 保存详细结果到JSON文件
    with open(os.path.join(args.save_dir, 'detection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存简要结果到文本文件
    with open(os.path.join(args.save_dir, 'results.txt'), 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Using original size information: {use_original}\n")
        f.write(f"Total images processed: {results['total_images']}\n")
        f.write(f"Images with faces: {results['images_with_faces']}\n")
        f.write(f"Images without faces: {len(results['no_face_indices'])}\n")
        f.write("\nIndices of images without faces:\n")
        f.write(str(results['no_face_indices']))

if __name__ == '__main__':
    main() 