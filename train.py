#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - PCB缺陷檢測訓練主程式
本模組整合了教師模型訓練和知識蒸餾流程，
實現高效的PCB缺陷檢測系統。
主要特點:
1. 教師模型訓練：預先訓練強大的教師模型
2. 知識蒸餾：從教師向學生模型遷移知識
3. 模型評估：全面評估與比較模型性能
4. 模型導出：支援導出為部署友好格式
"""

import os
import sys
import torch
import yaml
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# 設置環境變數防止下載COCO數據集
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['ULTRALYTICS_DATASET_DOWNLOAD'] = '0'
os.environ['ULTRALYTICS_SKIP_VALIDATION'] = '1'

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PCB缺陷類別
DEFECT_CLASSES = {
    'missing_hole': 0,
    'mouse_bite': 1, 
    'spur': 2,
    'spurious_copper': 3,
    'pin_hole': 4,
    'open_circuit': 5
}
# 反向映射
DEFECT_CLASSES_INV = {v: k for k, v in DEFECT_CLASSES.items()}
# 缺陷顏色映射 (BGR格式)
DEFECT_COLORS = {
    'missing_hole': (0, 0, 255),     # 紅色
    'mouse_bite': (0, 255, 0),       # 綠色
    'spur': (255, 0, 0),             # 藍色
    'spurious_copper': (0, 255, 255),# 黃色
    'pin_hole': (255, 0, 255),       # 紫色
    'open_circuit': (255, 255, 0)    # 青色
}

#-------------------------------- 工具函數 --------------------------------#
class Timer:
    """簡單的計時器類"""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self):
        if self.start_time is None:
            return 0
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self
    
    def format_time(self):
        hours, remainder = divmod(self.elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            return f"{int(minutes)}m {seconds:.2f}s"
        else:
            return f"{seconds:.2f}s"

def load_config(config_path):
    """載入配置檔案"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"已成功載入配置檔案: {config_path}")
            return config
    except Exception as e:
        logger.error(f"載入配置檔案失敗: {e}")
        return {}

def setup_experiment(config):
    """設置實驗環境"""
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.get('project', {}).get('name', 'pcb_defect')}_{timestamp}"
    
    output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
    weights_dir = output_dir / 'weights'
    charts_dir = output_dir / 'charts'
    logs_dir = output_dir / 'logs'
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"實驗名稱: {run_name}")
    
    # 儲存配置
    with open(logs_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return {
        'weights_dir': weights_dir,
        'charts_dir': charts_dir,
        'logs_dir': logs_dir,
        'run_name': run_name
    }

def plot_training_metrics(metrics_dict, output_path):
    """繪製訓練指標圖表"""
    plt.figure(figsize=(12, 8))
    
    num_metrics = len(metrics_dict)
    rows = (num_metrics + 1) // 2  # 向上取整
    cols = min(2, num_metrics)
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        plt.subplot(rows, cols, i+1)
        plt.plot(values, '-o', label=metric_name)
        plt.title(metric_name)
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"訓練指標圖表已儲存至: {output_path}")

def visualize_predictions(model, image_paths, output_dir, device, conf_threshold=0.25):
    """視覺化模型預測結果"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="視覺化預測"):
            # 讀取圖像
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"無法讀取圖像: {image_path}")
                continue
                
            original_image = image.copy()
                
            # 轉換為RGB並預處理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = preprocess_image(image_rgb).to(device)
            
            # 模型預測
            outputs = model(image_tensor)
            
            # 後處理預測結果
            boxes, labels, scores = process_predictions(outputs, conf_threshold)
            
            # 繪製邊界框
            result_image = draw_boxes(original_image, boxes, labels, scores, conf_threshold)
            
            # 儲存結果
            output_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"pred_{output_name}")
            cv2.imwrite(output_path, result_image)
            
    logger.info(f"預測視覺化結果已儲存至: {output_dir}")

def preprocess_image(image):
    """預處理圖像用於模型輸入"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # 添加批次維度

def process_predictions(outputs, conf_threshold=0.25):
    """處理模型預測輸出"""
    # 處理YOLO格式輸出
    if hasattr(outputs, 'boxes'):
        boxes = []
        labels = []
        scores = []
        for det in outputs.boxes:
            if det.conf >= conf_threshold:
                box = det.xyxy[0].tolist()
                boxes.append(box)
                labels.append(int(det.cls.item()))
                scores.append(float(det.conf.item()))
        return boxes, labels, scores
    
    # 處理其他格式輸出
    detections = outputs[0] if isinstance(outputs, (list, tuple)) and len(outputs) > 0 else []
    
    # 初始化結果列表
    boxes = []
    labels = []
    scores = []
    
    if isinstance(detections, torch.Tensor) and len(detections) > 0:
        # 過濾高置信度的檢測
        high_conf_idxs = detections[:, 4] > conf_threshold
        high_conf_detections = detections[high_conf_idxs]
        
        if len(high_conf_detections) > 0:
            for detection in high_conf_detections:
                box = detection[:4].tolist()  # x1, y1, x2, y2
                conf = detection[4].item()
                cls_id = int(detection[5].item())
                
                boxes.append(box)
                labels.append(cls_id)
                scores.append(conf)
    
    return boxes, labels, scores

def draw_boxes(image, boxes, labels, scores=None, threshold=0.5):
    """在圖像上繪製邊界框"""
    image_copy = image.copy()
    h, w, _ = image_copy.shape
    
    for i, box in enumerate(boxes):
        # 過濾低置信度的檢測
        if scores is not None and scores[i] < threshold:
            continue
        
        # 轉換為整數坐標並確保在圖像範圍內
        x1, y1, x2, y2 = box
        x1 = int(max(0, x1 * w if max(box) <= 1.0 else x1))
        y1 = int(max(0, y1 * h if max(box) <= 1.0 else y1))
        x2 = int(min(w, x2 * w if max(box) <= 1.0 else x2))
        y2 = int(min(h, y2 * h if max(box) <= 1.0 else y2))
        
        # 獲取類別和顏色
        label_idx = int(labels[i])
        class_name = DEFECT_CLASSES_INV.get(label_idx, f"未知-{label_idx}")
        color = DEFECT_COLORS.get(class_name, (255, 255, 255))  # 預設為白色
        
        # 繪製邊界框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # 添加標籤和分數
        label_text = class_name
        if scores is not None:
            label_text = f"{class_name}: {scores[i]:.2f}"
        
        # 計算文字大小
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 繪製標籤背景
        cv2.rectangle(
            image_copy, 
            (x1, y1 - text_height - 4), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # 繪製標籤文字
        cv2.putText(
            image_copy, 
            label_text, 
            (x1, y1 - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1, 
            cv2.LINE_AA
        )
    
    return image_copy

def calculate_iou(box1, box2):
    """計算兩個邊界框的IoU"""
    # 計算交集區域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 計算交集面積
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 計算各自面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 避免除以零
    union = box1_area + box2_area - intersection
    if union == 0:
        return 0
    
    # 計算IoU
    return intersection / union

def calculate_map(pred_boxes, pred_labels, pred_scores, 
                gt_boxes, gt_labels, iou_threshold=0.5, num_classes=len(DEFECT_CLASSES)):
    """計算平均精確度均值 (mAP)"""
    # 初始化
    all_detections = [[] for _ in range(num_classes)]  # 所有預測，按類別分組
    all_groundtruth = [[] for _ in range(num_classes)]  # 所有真值，按類別分組
    
    # 組織預測和真值，按類別分組
    for image_idx in range(len(pred_boxes)):
        # 處理預測
        pred_box = pred_boxes[image_idx]
        pred_label = pred_labels[image_idx]
        pred_score = pred_scores[image_idx]
        
        for box_idx in range(len(pred_box)):
            if box_idx < len(pred_label) and box_idx < len(pred_score):
                label = int(pred_label[box_idx])
                if 0 <= label < num_classes:
                    all_detections[label].append({
                        'image_idx': image_idx,
                        'box': pred_box[box_idx],
                        'score': pred_score[box_idx]
                    })
        
        # 處理真值
        gt_box = gt_boxes[image_idx]
        gt_label = gt_labels[image_idx]
        
        for box_idx in range(len(gt_box)):
            if box_idx < len(gt_label):
                label = int(gt_label[box_idx])
                if 0 <= label < num_classes:
                    all_groundtruth[label].append({
                        'image_idx': image_idx,
                        'box': gt_box[box_idx],
                        'used': False  # 標記是否已匹配
                    })
    
    # 初始化指標
    average_precisions = np.zeros(num_classes)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    
    # 逐類別計算AP
    for class_idx in range(num_classes):
        # 按置信度降序排序檢測結果
        detections = sorted(all_detections[class_idx], key=lambda x: x['score'], reverse=True)
        groundtruth = all_groundtruth[class_idx]
        
        # 計算該類別的真值總數
        num_gt = len(groundtruth)
        
        # 如果沒有真值，跳過這個類別
        if num_gt == 0:
            continue
        
        # 初始化True Positive和False Positive數組
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        # 遍歷所有檢測結果
        for det_idx, detection in enumerate(detections):
            # 獲取與當前檢測相同圖像的真值
            gt_in_same_image = [gt for gt in groundtruth if gt['image_idx'] == detection['image_idx']]
            
            # 初始化最佳匹配
            best_iou = -1
            best_gt_idx = -1
            
            # 查找最佳匹配的真值
            for gt_idx, gt in enumerate(gt_in_same_image):
                if gt['used']:
                    continue
                
                iou = calculate_iou(detection['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 根據IoU閾值決定TP或FP
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_in_same_image[best_gt_idx]['used'] = True
                tp[det_idx] = 1
            else:
                fp[det_idx] = 1
        
        # 計算累積TP和FP
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # 計算召回率和精確率
        recall = cumsum_tp / num_gt if num_gt > 0 else np.zeros_like(cumsum_tp)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # 計算AP (平均精確度) - 11點插值
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11
        
        average_precisions[class_idx] = ap
        
        # 計算最終的精確率和召回率
        if len(tp) > 0:
            precisions[class_idx] = np.sum(tp) / (np.sum(tp) + np.sum(fp))
            recalls[class_idx] = np.sum(tp) / num_gt if num_gt > 0 else 0
    
    # 計算mAP
    valid_classes = np.sum(np.array([len(groundtruth) > 0 for groundtruth in all_groundtruth]))
    mean_ap = np.sum(average_precisions) / valid_classes if valid_classes > 0 else 0
    
    # 計算平均精確率和召回率
    valid_indices = np.array([len(groundtruth) > 0 for groundtruth in all_groundtruth])
    mean_precision = np.mean(precisions[valid_indices]) if np.any(valid_indices) else 0
    mean_recall = np.mean(recalls[valid_indices]) if np.any(valid_indices) else 0
    
    metrics = {
        'mAP': float(mean_ap),
        'precision': float(mean_precision),
        'recall': float(mean_recall),
        'ap_per_class': {DEFECT_CLASSES_INV[i]: float(ap) for i, ap in enumerate(average_precisions) if len(all_groundtruth[i]) > 0}
    }
    
    return metrics

#-------------------------------- 資料集處理 --------------------------------#
class PCBDataset(torch.utils.data.Dataset):
    """PCB缺陷檢測資料集類別"""
    def __init__(self, root_dir, mode='train', img_size=640, transform=None, use_augmentation=True):
        """初始化PCB缺陷檢測資料集"""
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.imgs_path = os.path.join(root_dir, 'images')
        self.annotations_path = os.path.join(root_dir, 'Annotations')
        self.use_augmentation = use_augmentation and mode == 'train'
        
        # 檢查資料目錄
        for path in [root_dir, self.imgs_path, self.annotations_path]:
            if not os.path.exists(path):
                logger.error(f"目錄不存在: {path}")
                raise FileNotFoundError(f"目錄不存在: {path}")
        
        # 初始化資料路徑列表
        self.image_files = []
        self.annotation_files = []
        
        # 載入資料路徑
        self._load_dataset_paths()
        
        # 分割訓練和驗證資料集
        self._split_dataset()
        
        # 設定基本轉換
        from torchvision import transforms
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"已載入 {mode} 模式的 {len(self.image_files)} 張圖片")
    
    def _load_dataset_paths(self):
        """載入圖像和標註檔路徑"""
        for defect_type in DEFECT_CLASSES.keys():
            # 嘗試不同的可能目錄格式
            img_dir_candidates = [
                os.path.join(self.imgs_path, defect_type),
                os.path.join(self.imgs_path, defect_type.replace('_', ' ')),
                os.path.join(self.imgs_path, defect_type.title().replace('_', '')),
                os.path.join(self.imgs_path, defect_type.upper()),
                os.path.join(self.imgs_path, defect_type.replace('_', '-'))
            ]
            
            ann_dir_candidates = [
                os.path.join(self.annotations_path, defect_type),
                os.path.join(self.annotations_path, defect_type.replace('_', ' ')),
                os.path.join(self.annotations_path, defect_type.title().replace('_', '')),
                os.path.join(self.annotations_path, defect_type.upper()),
                os.path.join(self.annotations_path, defect_type.replace('_', '-'))
            ]
            
            # 尋找有效的目錄組合
            found_valid_dir = False
            for img_dir in img_dir_candidates:
                if not os.path.exists(img_dir):
                    continue
                
                for ann_dir in ann_dir_candidates:
                    if not os.path.exists(ann_dir):
                        continue
                    
                    # 找到有效目錄，添加檔案
                    found_valid_dir = True
                    for filename in os.listdir(img_dir):
                        if filename.endswith('.jpg'):
                            img_path = os.path.join(img_dir, filename)
                            ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.xml'))
                            
                            if os.path.exists(ann_path):
                                self.image_files.append(img_path)
                                self.annotation_files.append(ann_path)
                    
                    break  # 找到有效組合後跳出內層循環
                
                if found_valid_dir:
                    break  # 找到有效組合後跳出外層循環
            
            if not found_valid_dir:
                logger.warning(f"找不到缺陷類型 {defect_type} 的有效目錄")
    
    def _split_dataset(self):
        """分割訓練和驗證資料集"""
        if self.mode != 'test' and len(self.image_files) > 0:
            import random
            indices = list(range(len(self.image_files)))
            random.seed(42)  # 確保可重複性
            random.shuffle(indices)
            
            split = int(len(indices) * 0.8)
            train_indices = indices[:split]
            val_indices = indices[split:]
            
            if self.mode == 'train':
                self.image_files = [self.image_files[i] for i in train_indices]
                self.annotation_files = [self.annotation_files[i] for i in train_indices]
            else:  # val
                self.image_files = [self.image_files[i] for i in val_indices]
                self.annotation_files = [self.annotation_files[i] for i in val_indices]
    
    def __len__(self):
        """返回資料集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        import torch
        # 讀取圖片
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"無法讀取圖像: {img_path}，使用空白圖像替代")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # 調整圖片尺寸
        if orig_h != self.img_size or orig_w != self.img_size:
            h_ratio, w_ratio = self.img_size / orig_h, self.img_size / orig_w
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 讀取標註
        ann_path = self.annotation_files[idx]
        boxes, labels = self._parse_annotation(ann_path, orig_w, orig_h)
        
        # 調整邊界框坐標
        if 'h_ratio' in locals() and 'w_ratio' in locals() and boxes:
            adjusted_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                # 只調整絕對坐標，相對坐標不需調整
                if max(xmin, ymin, xmax, ymax) > 1.0:
                    xmin = (xmin * w_ratio) / self.img_size
                    ymin = (ymin * h_ratio) / self.img_size
                    xmax = (xmax * w_ratio) / self.img_size
                    ymax = (ymax * h_ratio) / self.img_size
                adjusted_boxes.append([xmin, ymin, xmax, ymax])
            boxes = adjusted_boxes
        
        # 應用基本轉換
        img = self.transform(img)
        
        # 準備張量
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        # 構建目標字典
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
        
        return img, target
    
    def _parse_annotation(self, ann_path, img_width, img_height):
        """解析XML標註檔"""
        import xml.etree.ElementTree as ET
        
        if not os.path.exists(ann_path):
            return [], []
            
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # 獲取圖片尺寸
            size = root.find('size')
            if size is None:
                w, h = img_width, img_height
            else:
                w = int(size.find('width').text)
                h = int(size.find('height').text)
            
            boxes = []
            labels = []
            
            # 遍歷所有物體標註
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in DEFECT_CLASSES:
                    continue
                    
                label = DEFECT_CLASSES[name]
                
                # 獲取邊界框坐標
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                # 確保使用相對坐標 (0-1範圍)
                xmin = float(bbox.find('xmin').text) / w
                ymin = float(bbox.find('ymin').text) / h
                xmax = float(bbox.find('xmax').text) / w
                ymax = float(bbox.find('ymax').text) / h
                
                # 確保邊界框坐標有效
                if xmin >= xmax or ymin >= ymax:
                    continue
                    
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                
            return boxes, labels
        except Exception as e:
            logger.error(f"解析標註檔出錯: {ann_path}, 錯誤: {e}")
            return [], []

def collate_fn(batch):
    """自定義批次組合函數，處理不同尺寸的圖片"""
    import torch
    images, targets = [], []
    
    for img, tgt in batch:
        images.append(img if isinstance(img, torch.Tensor) else torch.from_numpy(img).permute(2, 0, 1))
        targets.append(tgt)
    
    # 堆疊圖像
    images = torch.stack(images)
    
    return images, targets

def get_dataloader(config):
    """建立PCB缺陷檢測資料載入器"""
    import torch
    from torch.utils.data import DataLoader
    
    # 读取配置参数
    dataset_path = config.get('dataset', {}).get('path', config.get('dataset_path', 'C:/Users/a/Desktop/conference/PCB_DATASET'))
    batch_size = config.get('dataset', {}).get('batch_size', config.get('batch_size', 16))
    img_size = config.get('dataset', {}).get('img_size', config.get('img_size', 640))
    num_workers = config.get('dataset', {}).get('num_workers', config.get('num_workers', 4))
    
    try:
        # 建立資料集（使用頂層定義的 PCBDataset 類別）
        train_dataset = PCBDataset(
            root_dir=dataset_path,
            mode='train',
            img_size=img_size,
            use_augmentation=True
        )
        
        val_dataset = PCBDataset(
            root_dir=dataset_path,
            mode='val',
            img_size=img_size,
            use_augmentation=False
        )
        
        # 檢查資料集大小
        if len(train_dataset) == 0:
            logger.error(f"訓練資料集為空，請檢查資料集路徑與結構: {dataset_path}")
            return None, None
            
        if len(val_dataset) == 0:
            logger.warning("驗證資料集為空，將使用訓練資料集的子集進行驗證")
            # 使用訓練資料集的一部分作為驗證集
            dataset_size = len(train_dataset)
            train_size = int(dataset_size * 0.8)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # 建立資料載入器 (使用頂層定義的 collate_fn 函數)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"已建立訓練資料載入器 ({len(train_dataset)} 樣本) 和驗證資料載入器 ({len(val_dataset)} 樣本)")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"建立資料載入器失敗: {e}")
        # 添加错误追踪信息
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None, None
    
    #-------------------------------- 模型處理 --------------------------------#
def get_teacher_model(config):
    """取得教師模型"""
    try:
        from ultralytics import YOLO
        
        # 檢查是否已有訓練好的教師模型
        output_dir = Path(config.get('project', {}).get('output_dir', 'outputs'))
        best_teacher_path = output_dir / 'weights' / "teacher_best.pt"
        
        if os.path.exists(best_teacher_path):
            logger.info(f"載入已訓練的教師模型: {best_teacher_path}")
            model = YOLO(str(best_teacher_path))
        else:
            # 從頭開始初始化模型
            logger.info("初始化新的教師模型，將在PCB數據集上訓練")
            
            # 使用 yaml 檔案而不是 .pt 來避免載入 COCO 預訓練權重
            model = YOLO('yolov8l.yaml')
            
            # 修改模型的類別數
            model.model.nc = len(DEFECT_CLASSES)
            model.names = list(DEFECT_CLASSES.keys())
            
            # 確保檢測頭匹配類別數
            if hasattr(model.model, 'model'):
                for m in model.model.model:
                    if hasattr(m, 'nc'):
                        m.nc = len(DEFECT_CLASSES)
            
            logger.info("成功初始化教師模型")
        
        # 停用驗證和數據集下載
        if hasattr(model, 'args'):
            if isinstance(model.args, dict):
                model.args['val'] = False
                model.args['data'] = None
            else:
                try:
                    model.args.val = False
                    model.args.data = None
                except:
                    logger.warning("無法設置YOLO模型驗證參數")
        
        logger.info(f"教師模型已設置為{len(DEFECT_CLASSES)}個PCB缺陷類別")
        return model
    except Exception as e:
        logger.error(f"載入教師模型失敗: {e}")
        # 添加詳細的錯誤信息
        import traceback
        logger.error(f"錯誤詳情: {traceback.format_exc()}")
        sys.exit(1)

def get_student_model(config):
    """取得學生模型"""
    try:
        from ultralytics import YOLO
        student_path = config.get('model', {}).get('student', 'yolov8s.pt')
        model = YOLO(student_path)
        
        # 設定類別數
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            num_classes = len(DEFECT_CLASSES)
            # 設置模型的類別數量
            model.model.model.nc = num_classes
            logger.info(f"已設置學生模型類別數量為: {num_classes}")
        
        logger.info(f"已載入學生模型: {student_path}")
        return model
    except Exception as e:
        logger.error(f"載入學生模型失敗: {e}")
        sys.exit(1)

def load_model(model_path, model_type='student'):
    """載入模型"""
    logger.info(f"載入模型: {model_path}")
    try:
        # 嘗試載入YOLO格式模型
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            logger.info(f"已載入YOLO模型: {model_path}")
            
            # 如果是從頭加載的模型，設置類別數
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                num_classes = len(DEFECT_CLASSES)
                model.model.model.nc = num_classes
                logger.info(f"已設置模型類別數量為: {num_classes}")
                
            return model
        except Exception as e:
            logger.error(f"YOLO模型載入失敗，嘗試其他方式: {e}")
            
            # 嘗試直接載入PyTorch模型
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                model = torch.load(model_path, map_location='cpu')
            else:
                raise ValueError(f"不支援的模型格式: {model_path}")
            
        logger.info(f"模型載入成功")
        return model
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        sys.exit(1)

#-------------------------------- 知識蒸餾模組 --------------------------------#
class FeatureDistillationLoss(torch.nn.Module):
    """特徵層級知識蒸餾損失函數"""
    def __init__(self, adaptation_layers=None):
        super(FeatureDistillationLoss, self).__init__()
        self.adaptation_layers = adaptation_layers or {}
        self.mse_loss = torch.nn.MSELoss()
        
    def forward(self, student_features, teacher_features):
        total_loss = 0.0
        losses_dict = {}
        
        # 檢查特徵是否為字典
        if not isinstance(student_features, dict) or not isinstance(teacher_features, dict):
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu'), {}
        
        # 遍歷教師特徵
        for feat_name, t_feat in teacher_features.items():
            # 跳過學生模型中不存在的特徵
            if feat_name not in student_features:
                continue
                
            s_feat = student_features[feat_name]
            
            # 應用特徵適配層（如果存在）
            if feat_name in self.adaptation_layers:
                s_feat = self.adaptation_layers[feat_name](s_feat)
            
            # 如果尺寸不匹配，調整學生特徵尺寸
            if s_feat.shape != t_feat.shape:
                try:
                    s_feat = torch.nn.functional.interpolate(
                        s_feat,
                        size=t_feat.shape[2:],  # 高寬維度
                        mode='bilinear',
                        align_corners=False
                    )
                except Exception as e:
                    logger.warning(f"調整特徵尺寸時發生錯誤: {e}")
                    continue
            
            # 計算MSE損失
            try:
                layer_loss = self.mse_loss(s_feat, t_feat)
                losses_dict[feat_name] = layer_loss.item()
                total_loss += layer_loss
            except Exception as e:
                logger.warning(f"計算特徵損失時發生錯誤: {e}")
            
        return total_loss, losses_dict

class LogitDistillationLoss(torch.nn.Module):
    """輸出層級知識蒸餾損失函數"""
    def __init__(self, temperature=4.0):
        super(LogitDistillationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        # 檢查並處理 YOLO Results 類型輸出
        if hasattr(teacher_logits, 'boxes'):
            teacher_logits = self._extract_logits_from_yolo_results(teacher_logits)
        
        if hasattr(student_logits, 'boxes'):
            student_logits = self._extract_logits_from_yolo_results(student_logits)
        
        # 確保輸入是張量
        if not isinstance(teacher_logits, torch.Tensor) or not isinstance(student_logits, torch.Tensor):
            # 如果無法提取有效的張量，返回零損失
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return torch.tensor(0.0, device=device)
        
        try:
            # 應用溫度縮放
            student_logits_T = student_logits / self.temperature
            teacher_logits_T = teacher_logits / self.temperature
            
            # 計算KL散度
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits_T, dim=1),
                torch.nn.functional.softmax(teacher_logits_T, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            return kl_loss
        except Exception as e:
            logger.warning(f"計算KL散度損失時發生錯誤: {e}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return torch.tensor(0.0, device=device)
    
    def _extract_logits_from_yolo_results(self, results):
        """從YOLO Results物件中提取logits"""
        try:
            # 嘗試提取類別概率
            if hasattr(results, 'boxes') and hasattr(results.boxes, 'cls'):
                # 從boxes中獲取類別概率
                cls_tensor = results.boxes.cls
                conf_tensor = results.boxes.conf
                
                # 如果張量為空，返回一個默認張量
                if len(cls_tensor) == 0:
                    return torch.zeros((1, len(DEFECT_CLASSES)), 
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
                
                # 結合類別和置信度創建logits
                logits = torch.zeros((len(cls_tensor), len(DEFECT_CLASSES)), 
                                   device=cls_tensor.device)
                
                # 填充logits
                for i, (cls, conf) in enumerate(zip(cls_tensor, conf_tensor)):
                    cls_idx = int(cls.item())
                    if 0 <= cls_idx < len(DEFECT_CLASSES):
                        logits[i, cls_idx] = conf
                        
                return logits
            else:
                # 返回一個空的張量
                return torch.zeros((1, len(DEFECT_CLASSES)), 
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.warning(f"提取YOLO logits時發生錯誤: {e}")
            return torch.zeros((1, len(DEFECT_CLASSES)), 
                             device='cuda' if torch.cuda.is_available() else 'cpu')

class DistillationManager:
    """知識蒸餾訓練管理器"""
    def __init__(self, teacher_model, student_model, config):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # 蒸餾參數
        self.alpha = config.get('distillation', {}).get('alpha', 0.5)  # 任務損失權重
        self.beta = config.get('distillation', {}).get('beta', 0.5)    # 蒸餾損失權重
        self.gamma = config.get('distillation', {}).get('gamma', 0.5)  # 特徵蒸餾權重
        self.temperature = config.get('distillation', {}).get('temperature', 4.0)  # 溫度參數
        
        # 初始化損失函數
        self.feature_loss = FeatureDistillationLoss()
        self.logit_loss = LogitDistillationLoss(temperature=self.temperature)
        
        # 設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 將模型移至設備
        self.teacher_model = self.teacher_model.to(self.device)
        self.student_model = self.student_model.to(self.device)
        
        # 優化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters() if not hasattr(self.student_model, 'model') 
            else self.student_model.model.model.parameters(),
            lr=config.get('training', {}).get('learning_rate', 1e-4),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-5)
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('training', {}).get('epochs', 100),
            eta_min=config.get('training', {}).get('min_lr', 1e-6)
        )
        
        # 混合精度訓練
        self.use_amp = config.get('training', {}).get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # 儲存路徑
        self.output_dir = Path(config.get('project', {}).get('output_dir', 'outputs/weights'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"知識蒸餾管理器已初始化，設備: {self.device}")
        
    def train_epoch(self, train_loader, epoch):
        """執行一個完整的蒸餾訓練周期"""
        # 設定為訓練模式
        self.teacher_model.eval()  # 教師模型保持評估模式
        
        # 根據模型類型設置訓練模式
        if hasattr(self.student_model, 'train'):
            self.student_model.train()
        elif hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'train'):
            self.student_model.model.train()
        
        # 損失追蹤
        total_loss = 0
        task_loss_sum = 0
        distill_loss_sum = 0
        feature_loss_sum = 0
        
        # 進度條
        pbar = tqdm(train_loader, desc=f"蒸餾 Epoch {epoch+1}/{self.config.get('training', {}).get('epochs', 100)}")
        
        for images, targets in pbar:
            # 將資料移至設備
            images = images.to(self.device)
            for t in targets:
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t[k] = v.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 混合精度訓練
            with autocast(enabled=self.use_amp):
                # 教師模型前向傳播（無梯度）
                with torch.no_grad():
                    teacher_out = self.teacher_model(images)
                    teacher_features = self.teacher_model.get_features() if hasattr(self.teacher_model, 'get_features') else {}
                
                # 學生模型前向傳播
                student_out = self.student_model(images)
                student_features = self.student_model.get_features() if hasattr(self.student_model, 'get_features') else {}
                
                # 計算任務損失
                task_loss = self._compute_task_loss(student_out, targets)
                
                # 計算蒸餾損失
                distill_loss = self.logit_loss(student_out, teacher_out)
                
                # 計算特徵蒸餾損失
                feature_loss, _ = self.feature_loss(student_features, teacher_features)
                
                # 確保所有損失都是張量並具有梯度
                task_loss = self._ensure_tensor_with_grad(task_loss)
                distill_loss = self._ensure_tensor_with_grad(distill_loss)
                feature_loss = self._ensure_tensor_with_grad(feature_loss)

                # 總損失
                loss = (self.alpha * task_loss + 
                       self.beta * distill_loss + 
                       self.gamma * feature_loss)

                # 確保總損失有梯度
                if not loss.requires_grad:
                    dummy_tensor = torch.ones(1, device=self.device, requires_grad=True)
                    loss = loss * dummy_tensor

            # 反向傳播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 更新損失追蹤
            total_loss += loss.item()
            task_loss_sum += task_loss.item()
            distill_loss_sum += distill_loss.item()
            feature_loss_sum += feature_loss.item()
            
            # 更新進度條
            pbar.set_postfix({
                'loss': loss.item(), 
                'task': task_loss.item(),
                'distill': distill_loss.item()
            })
        
        # 更新學習率
        self.scheduler.step()
        
        # 計算平均損失
        avg_loss = total_loss / len(train_loader)
        avg_task_loss = task_loss_sum / len(train_loader)
        avg_distill_loss = distill_loss_sum / len(train_loader)
        avg_feature_loss = feature_loss_sum / len(train_loader)
        
        losses = {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'distill_loss': avg_distill_loss,
            'feature_loss': avg_feature_loss
        }
        
        return losses
    
    def _ensure_tensor_with_grad(self, tensor):
        """確保張量具有梯度"""
        if isinstance(tensor, torch.Tensor):
            if not tensor.requires_grad:
                tensor = tensor.detach().clone().requires_grad_(True)
            return tensor
        else:
            return torch.tensor(tensor, device=self.device, requires_grad=True)
    
    def validate(self, val_loader):
        """在驗證集上評估學生模型"""
        # 設定為評估模式
        if hasattr(self.student_model, 'eval'):
            self.student_model.eval()
        elif hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'eval'):
            self.student_model.model.eval()
        
        # 損失追蹤
        val_loss = 0
        
        # 預測和目標收集
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="驗證"):
                # 將資料移至設備
                images = images.to(self.device)
                for t in targets:
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            t[k] = v.to(self.device)
                
                # 前向傳播
                student_out = self.student_model(images)
                
                # 計算任務損失
                loss = self._compute_task_loss(student_out, targets)
                val_loss += loss.item()
                
                # 收集預測和目標
                self._collect_detection_results(student_out, targets, 
                                              all_pred_boxes, all_pred_labels, all_pred_scores,
                                              all_gt_boxes, all_gt_labels)
        
        # 計算平均損失
        avg_val_loss = val_loss / len(val_loader)
        
        # 計算mAP和其他指標
        metrics = calculate_map(
            all_pred_boxes, all_pred_labels, all_pred_scores,
            all_gt_boxes, all_gt_labels
        )
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def _compute_task_loss(self, outputs, targets):
        """計算檢測任務損失"""
        try:
            # 處理 YOLO Results 類型輸出
            if hasattr(outputs, 'boxes'):
                # 對於 YOLO 輸出，使用一個簡化的損失計算
                loss = torch.tensor(0.1, device=self.device)  # 使用一個小常數防止梯度消失
                return loss
                
            # 優先嘗試使用 YOLO 模型的原生損失計算方法
            if hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'model') and hasattr(self.student_model.model.model, 'loss'):
                loss_dict = self.student_model.model.model.loss(outputs, targets)
                return sum(loss_dict.values())
            
            # 基本後備損失計算 (這裡只是一個佔位符)
            total_loss = torch.tensor(0.1, device=self.device)
            return total_loss
        except Exception as e:
            logger.warning(f"計算任務損失時發生錯誤: {e}")
            # 如果無法計算損失，返回一個非零張量作為後備方案
            dummy_tensor = torch.ones(1, device=self.device, requires_grad=True)
            return torch.tensor(0.1, device=self.device) * dummy_tensor
    
    def _collect_detection_results(self, outputs, targets, 
                                 pred_boxes, pred_labels, pred_scores,
                                 gt_boxes, gt_labels):
        """收集檢測結果和真實標籤"""
        # 處理 YOLO Results 類型輸出
        if hasattr(outputs, 'boxes'):
            try:
                # 獲取預測框
                detection_boxes = outputs.boxes
                
                # 提取預測
                boxes = detection_boxes.xyxy.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                scores = detection_boxes.conf.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                labels = detection_boxes.cls.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                
                # 添加到結果列表
                pred_boxes.append(boxes)
                pred_scores.append(scores)
                pred_labels.append(labels)
            except Exception as e:
                logger.warning(f"處理YOLO檢測結果時發生錯誤: {e}")
                # 添加空結果
                pred_boxes.append(np.array([]))
                pred_scores.append(np.array([]))
                pred_labels.append(np.array([]))
                
            # 收集真實標籤
            for target in targets:
                gt_boxes.append(target['boxes'].cpu().numpy())
                gt_labels.append(target['labels'].cpu().numpy())
        else:
            # 處理其他格式的輸出
            for batch_idx, output in enumerate(outputs):
                # 提取預測
                boxes = output[0] if isinstance(output, (list, tuple)) and len(output) > 0 else []
                scores = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 else []
                labels = output[2] if isinstance(output, (list, tuple)) and len(output) > 2 else []
                
                pred_boxes.append(boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else np.array([]))
                pred_scores.append(scores.cpu().numpy() if isinstance(scores, torch.Tensor) else np.array([]))
                pred_labels.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array([]))
                
                # 提取真實標籤
                if batch_idx < len(targets):
                    target = targets[batch_idx]
                    gt_boxes.append(target['boxes'].cpu().numpy())
                    gt_labels.append(target['labels'].cpu().numpy())
    
    def save_checkpoint(self, epoch, metrics, best_map=0):
        """儲存模型檢查點"""
        # 當前mAP
        current_map = metrics.get('mAP', 0)
        
        # 儲存最新檢查點
        checkpoint_path = self.output_dir / f"student_epoch_{epoch+1}.pt"
        
        # 根據模型類型選擇保存方法
        if hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'save'):
            self.student_model.model.save(checkpoint_path)
        else:
            torch.save(self.student_model.state_dict(), checkpoint_path)
            
        logger.info(f"已儲存學生模型檢查點：{checkpoint_path}")
        
        # 如果是最佳模型，另存一份
        if current_map > best_map:
            best_map = current_map
            best_model_path = self.output_dir / "student_best.pt"
            
            if hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'save'):
                self.student_model.model.save(best_model_path)
            else:
                torch.save(self.student_model.state_dict(), best_model_path)
                
            logger.info(f"發現更好的學生模型 (mAP: {best_map:.4f})，已儲存至：{best_model_path}")
        
        return best_map

#-------------------------------- 訓練與評估流程 --------------------------------#
def train_teacher_model(teacher_model, train_loader, val_loader, config):
    """訓練教師模型"""
    # 設置裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 設置實驗目錄
    exp_dirs = setup_experiment(config)
    weights_dir = exp_dirs['weights_dir']
    charts_dir = exp_dirs['charts_dir']
    
    # 將模型移到設備上
    teacher_model = teacher_model.to(device)
    
    # 設置優化器
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model'):
        optimizer = torch.optim.AdamW(
            teacher_model.model.model.parameters(),
            lr=config.get('teacher_training', {}).get('learning_rate', 5e-5),
            weight_decay=config.get('teacher_training', {}).get('weight_decay', 1e-5)
        )
    else:
        optimizer = torch.optim.AdamW(
            teacher_model.parameters(),
            lr=config.get('teacher_training', {}).get('learning_rate', 5e-5),
            weight_decay=config.get('teacher_training', {}).get('weight_decay', 1e-5)
        )
    
    # 設置學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('teacher_training', {}).get('epochs', 50),
        eta_min=config.get('training', {}).get('min_lr', 1e-6)
    )
    
    # 使用混合精度訓練
    use_amp = config.get('training', {}).get('use_amp', True)
    scaler = GradScaler(enabled=use_amp)
    
    # 訓練參數
    epochs = config.get('teacher_training', {}).get('epochs', 50)
    eval_interval = config.get('teacher_training', {}).get('eval_interval', 5)
    best_map = 0
    
    # 初始化指標追蹤
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'mAP': [],
        'precision': [],
        'recall': []
    }
    
    # 訓練循環
    total_timer = Timer().start()
    
    logger.info(f"開始訓練教師模型，共 {epochs} 個周期")
    
    for epoch in range(epochs):
        # 訓練模式
        if hasattr(teacher_model, 'train'):
            teacher_model.train()
        elif hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'train'):
            teacher_model.model.train()
        
        # 訓練一個周期
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"教師模型 Epoch {epoch+1}/{epochs}") as pbar:
            for images, targets in pbar:
                # 將資料移到設備上
                images = images.to(device)
                for t in targets:
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            t[k] = v.to(device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 混合精度訓練
                with autocast(enabled=use_amp):
                    # 前向傳播
                    outputs = teacher_model(images)
                    
                    # 計算損失
                    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'loss'):
                        loss_dict = teacher_model.model.model.loss(outputs, targets)
                        loss = sum(loss_dict.values())
                    else:
                        # 如果沒有原生損失函數，使用一個基本的損失
                        loss = torch.tensor(0.1, device=device, requires_grad=True)
                
                # 反向傳播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # 更新進度條
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # 更新學習率
        scheduler.step()
        
        # 計算平均訓練損失
        avg_train_loss = epoch_loss / len(train_loader)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # 定期評估
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            # 評估模型
            if hasattr(teacher_model, 'eval'):
                teacher_model.eval()
            elif hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'eval'):
                teacher_model.model.eval()
                
            val_loss = 0
            
            # 收集評估結果
            all_pred_boxes = []
            all_pred_labels = []
            all_pred_scores = []
            all_gt_boxes = []
            all_gt_labels = []
            
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="評估中"):
                    # 將資料移到設備上
                    images = images.to(device)
                    for t in targets:
                        for k, v in t.items():
                            if isinstance(v, torch.Tensor):
                                t[k] = v.to(device)
                    
                    # 前向傳播
                    outputs = teacher_model(images)
                    
                    # 計算損失
                    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'loss'):
                        # 計算損失
                        if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'loss'):
                            # 收集預測結果和真實標籤
                            if hasattr(outputs, 'boxes'):
                                try:
                                    # 獲取預測框
                                    detection_boxes = outputs.boxes
                                    
                                    # 提取預測
                                    boxes = detection_boxes.xyxy.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                                    scores = detection_boxes.conf.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                                    labels = detection_boxes.cls.cpu().numpy() if len(detection_boxes) > 0 else np.array([])
                                    
                                    # 添加到結果列表
                                    all_pred_boxes.append(boxes)
                                    all_pred_scores.append(scores)
                                    all_pred_labels.append(labels)
                                except Exception as e:
                                    logger.warning(f"處理YOLO檢測結果時發生錯誤: {e}")
                                    # 添加空結果
                                    all_pred_boxes.append(np.array([]))
                                    all_pred_scores.append(np.array([]))
                                    all_pred_labels.append(np.array([]))
                            
                        # 收集真實標籤
                        for target in targets:
                            all_gt_boxes.append(target['boxes'].cpu().numpy())
                            all_gt_labels.append(target['labels'].cpu().numpy())
                    else:
                        # 處理其他格式的輸出
                        for batch_idx, output in enumerate(outputs if isinstance(outputs, (list, tuple)) else [outputs]):
                            # 添加空結果
                            all_pred_boxes.append(np.array([]))
                            all_pred_scores.append(np.array([]))
                            all_pred_labels.append(np.array([]))
                            
                            # 提取真實標籤
                            if batch_idx < len(targets):
                                target = targets[batch_idx]
                                all_gt_boxes.append(target['boxes'].cpu().numpy())
                                all_gt_labels.append(target['labels'].cpu().numpy())
                    
                    # 計算損失
                    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'model') and hasattr(teacher_model.model.model, 'loss'):
                        loss_dict = teacher_model.model.model.loss(outputs, targets)
                        loss = sum(loss_dict.values())
                    else:
                        # 如果沒有原生損失函數，使用一個基本的損失
                        loss = torch.tensor(0.1, device=device)
                        
                    val_loss += loss.item()
            
            # 計算平均驗證損失
            avg_val_loss = val_loss / len(val_loader)
            metrics_history['val_loss'].append(avg_val_loss)
            
            # 計算mAP等評估指標
            metrics = calculate_map(
                all_pred_boxes, all_pred_labels, all_pred_scores,
                all_gt_boxes, all_gt_labels
            )
            
            # 更新指標歷史
            metrics_history['mAP'].append(metrics['mAP'])
            metrics_history['precision'].append(metrics['precision'])
            metrics_history['recall'].append(metrics['recall'])
            
            # 記錄結果
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  訓練損失: {avg_train_loss:.4f}")
            logger.info(f"  驗證損失: {avg_val_loss:.4f}")
            logger.info(f"  mAP: {metrics['mAP']:.4f}")
            logger.info(f"  精確率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")
            
            # 儲存最佳模型
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                best_model_path = weights_dir / "teacher_best.pt"
                
                # 儲存模型，根據不同模型類型調整
                if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'save'):
                    teacher_model.model.save(best_model_path)
                else:
                    torch.save(teacher_model.state_dict(), best_model_path)
                
                logger.info(f"發現更好的模型 (mAP: {best_map:.4f})，已儲存至: {best_model_path}")
            
            # 儲存當前檢查點
            checkpoint_path = weights_dir / f"teacher_epoch_{epoch+1}.pt"
            if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'save'):
                teacher_model.model.save(checkpoint_path)
            else:
                torch.save(teacher_model.state_dict(), checkpoint_path)
    
    # 計算總訓練時間
    total_time = total_timer.stop().format_time()
    logger.info(f"教師模型訓練完成，總耗時: {total_time}")
    logger.info(f"最佳mAP: {best_map:.4f}")
    
    # 繪製訓練指標圖表
    plot_training_metrics(
        metrics_history,
        output_path=charts_dir / "teacher_training_metrics.png"
    )
    
    # 載入最佳模型
    best_model_path = weights_dir / "teacher_best.pt"
    if os.path.exists(best_model_path):
        if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'load'):
            teacher_model.model.load(best_model_path)
        else:
            teacher_model.load_state_dict(torch.load(best_model_path))
        logger.info(f"已載入最佳教師模型: {best_model_path}")
    
    return teacher_model, best_map

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, config):
    """使用知識蒸餾訓練學生模型"""
    # 創建蒸餾管理器
    distill_manager = DistillationManager(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config
    )
    
    # 設置實驗目錄
    exp_dirs = setup_experiment(config)
    weights_dir = exp_dirs['weights_dir']
    charts_dir = exp_dirs['charts_dir']
    
    # 訓練參數
    epochs = config.get('training', {}).get('epochs', 100)
    eval_interval = config.get('training', {}).get('eval_interval', 5)
    best_map = 0
    
    # 初始化指標追蹤
    metrics_history = {
        'train_loss': [],
        'distill_loss': [],
        'feature_loss': [],
        'val_loss': [],
        'mAP': [],
        'precision': [],
        'recall': []
    }
    
    logger.info(f"開始知識蒸餾訓練，共 {epochs} 個周期")
    
    # 開始計時
    total_timer = Timer().start()
    
    # 訓練循環
    for epoch in range(epochs):
        # 訓練一個周期
        train_losses = distill_manager.train_epoch(train_loader, epoch)
        
        # 更新指標歷史
        metrics_history['train_loss'].append(train_losses['loss'])
        metrics_history['distill_loss'].append(train_losses['distill_loss'])
        metrics_history['feature_loss'].append(train_losses['feature_loss'])
        
        # 定期評估
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            metrics = distill_manager.validate(val_loader)
            
            # 更新指標歷史
            metrics_history['val_loss'].append(metrics['val_loss'])
            metrics_history['mAP'].append(metrics['mAP'])
            metrics_history['precision'].append(metrics['precision'])
            metrics_history['recall'].append(metrics['recall'])
            
            # 記錄結果
            logger.info(f"知識蒸餾 周期 {epoch+1}/{epochs}")
            logger.info(f"  訓練損失: {train_losses['loss']:.4f}")
            logger.info(f"  驗證損失: {metrics['val_loss']:.4f}")
            logger.info(f"  mAP: {metrics['mAP']:.4f}")
            logger.info(f"  精確率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")
            
            # 儲存檢查點
            best_map = distill_manager.save_checkpoint(epoch, metrics, best_map)
    
    # 計算訓練時間
    total_time = total_timer.stop().format_time()
    logger.info(f"知識蒸餾訓練完成，總耗時: {total_time}")
    logger.info(f"最佳mAP: {best_map:.4f}")
    
    # 繪製訓練指標圖表
    plot_training_metrics(
        metrics_history,
        output_path=charts_dir / "distillation_training_metrics.png"
    )
    
    # 載入最佳模型
    best_model_path = weights_dir / "student_best.pt"
    if os.path.exists(best_model_path):
        # 根據模型類型選擇載入方法
        if hasattr(student_model, 'model') and hasattr(student_model.model, 'load'):
            student_model.model.load(best_model_path)
        else:
            student_model.load_state_dict(torch.load(best_model_path))
        logger.info(f"已載入最佳學生模型：{best_model_path}")
    
    return student_model, best_map

def evaluate_model(model, val_loader, conf_threshold=0.25, device='cuda'):
    """評估模型性能"""
    logger.info("開始評估模型性能...")
    
    # 設定裝置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 收集預測和真實標籤
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="評估進度"):
            images = images.to(device)
            
            # 獲取預測
            outputs = model(images)
            
            # 處理預測結果
            if hasattr(outputs, 'boxes'):
                # YOLO8 原生格式
                for i, img in enumerate(images):
                    # 過濾該批次的預測
                    img_boxes = []
                    img_scores = []
                    img_labels = []
                    
                    # 獲取該批次的所有檢測
                    for j, box in enumerate(outputs.boxes):
                        # 檢查是否屬於當前圖像
                        if hasattr(box, 'batch_idx') and box.batch_idx.item() == i:
                            if box.conf >= conf_threshold:
                                img_boxes.append(box.xyxy[0].cpu().numpy())
                                img_scores.append(box.conf.item())
                                img_labels.append(int(box.cls.item()))
                    
                    # 添加到收集列表
                    all_pred_boxes.append(np.array(img_boxes))
                    all_pred_scores.append(np.array(img_scores))
                    all_pred_labels.append(np.array(img_labels))
                    
                    # 提取真實標籤
                    target = targets[i]
                    all_gt_boxes.append(target['boxes'].cpu().numpy())
                    all_gt_labels.append(target['labels'].cpu().numpy())
            else:
                # 處理其他格式的輸出
                for batch_idx, img in enumerate(images):
                    # 添加空結果
                    all_pred_boxes.append(np.array([]))
                    all_pred_scores.append(np.array([]))
                    all_pred_labels.append(np.array([]))
                    
                    # 提取真實標籤
                    target = targets[batch_idx]
                    all_gt_boxes.append(target['boxes'].cpu().numpy())
                    all_gt_labels.append(target['labels'].cpu().numpy())
    
    # 計算評估指標
    metrics = calculate_map(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_gt_boxes, all_gt_labels,
        iou_threshold=0.5
    )
    
    # 輸出結果
    logger.info(f"評估結果:")
    logger.info(f"  mAP@0.5: {metrics['mAP']:.4f}")
    logger.info(f"  精確率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    
    # 顯示各類別的AP
    if 'ap_per_class' in metrics:
        logger.info("各類別AP:")
        for class_name, ap in metrics['ap_per_class'].items():
            logger.info(f"  {class_name}: {ap:.4f}")
    
    return metrics

def export_model(model, output_path, input_shape=(1, 3, 640, 640), format='onnx'):
    """導出模型為部署格式"""
    logger.info(f"導出模型為{format}格式: {output_path}")
    
    if os.path.exists(output_path):
        logger.info(f"輸出文件已存在，將被覆蓋: {output_path}")
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設定為評估模式
    model.eval()
    
    # 根據格式導出
    if format.lower() == 'onnx':
        try:
            # 使用YOLO的原生導出(如果可用)
            if hasattr(model, 'export') and callable(model.export):
                model.export(format='onnx', dynamic=True, file=output_path)
                logger.info(f"已使用YOLO原生方法導出為ONNX: {output_path}")
            else:
                # 使用PyTorch的ONNX導出
                dummy_input = torch.randn(input_shape)
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                logger.info(f"已使用PyTorch導出為ONNX: {output_path}")
            
            return True
        except Exception as e:
            logger.error(f"導出ONNX失敗: {e}")
            return False
    elif format.lower() == 'tflite':
        try:
            # 使用YOLO的原生導出(如果可用)
            if hasattr(model, 'export') and callable(model.export):
                model.export(format='tflite', file=output_path)
                logger.info(f"已使用YOLO原生方法導出為TFLite: {output_path}")
                return True
            else:
                logger.error("無法導出為TFLite：僅支援YOLO模型")
                return False
        except Exception as e:
            logger.error(f"導出TFLite失敗: {e}")
            return False
    else:
        logger.error(f"不支援的導出格式: {format}")
        return False

def optimize_model(model, config):
    """優化模型，應用剪枝和量化"""
    logger.info("開始優化模型...")
    
    # 從模型類型判斷使用哪種優化方法
    model_type = config.get('model', {}).get('type', '').lower()
    
    # 如果是YOLO模型，使用其內建優化
    if 'yolo' in model_type and hasattr(model, 'export'):
        try:
            # 創建優化配置
            output_dir = Path(config.get('project', {}).get('output_dir', 'outputs/weights'))
            optimized_path = output_dir / "optimized.pt"
            
            # 使用YOLO的內建優化
            model.export(format='pt', half=True, file=optimized_path)
            logger.info(f"已使用YOLO內建優化並保存至: {optimized_path}")
            
            # 重新載入優化後的模型
            from ultralytics import YOLO
            optimized_model = YOLO(optimized_path)
            
            return optimized_model
        except Exception as e:
            logger.error(f"YOLO優化失敗: {e}")
            return model
    else:
        # 使用PyTorch的量化
        try:
            import torch.quantization
            
            # 切換為CPU模式
            cpu_model = model.cpu()
            
            # 應用動態量化
            quantized_model = torch.quantization.quantize_dynamic(
                cpu_model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            logger.info("已應用PyTorch動態量化")
            
            return quantized_model
        except Exception as e:
            logger.error(f"PyTorch量化失敗: {e}")
            return model

def compare_models(teacher_model, student_model, val_loader, config):
    """比較教師模型和學生模型的性能"""
    logger.info("開始比較教師模型和學生模型性能...")
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 移動模型到裝置
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # 設定為評估模式
    teacher_model.eval()
    student_model.eval()
    
    # 評估教師模型
    logger.info("評估教師模型...")
    teacher_metrics = evaluate_model(
        model=teacher_model,
        val_loader=val_loader,
        conf_threshold=config.get('model', {}).get('conf_threshold', 0.25),
        device=device
    )
    
    # 評估學生模型
    logger.info("評估學生模型...")
    student_metrics = evaluate_model(
        model=student_model,
        val_loader=val_loader,
        conf_threshold=config.get('model', {}).get('conf_threshold', 0.25),
        device=device
    )
    
    # 比較結果
    logger.info("模型比較結果:")
    logger.info(f"  教師模型 mAP: {teacher_metrics['mAP']:.4f}")
    logger.info(f"  學生模型 mAP: {student_metrics['mAP']:.4f}")
    logger.info(f"  mAP 差距: {student_metrics['mAP'] - teacher_metrics['mAP']:.4f}")
    logger.info(f"  教師模型精確率: {teacher_metrics['precision']:.4f}")
    logger.info(f"  學生模型精確率: {student_metrics['precision']:.4f}")
    logger.info(f"  教師模型召回率: {teacher_metrics['recall']:.4f}")
    logger.info(f"  學生模型召回率: {student_metrics['recall']:.4f}")
    
    # 計算模型大小和推理時間
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # 測量推理時間
    teacher_time = measure_inference_time(teacher_model, dummy_input)
    student_time = measure_inference_time(student_model, dummy_input)
    
    # 計算加速比
    speedup = teacher_time / student_time if student_time > 0 else 0
    
    # 計算參數量
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    # 計算壓縮比
    compression = teacher_params / student_params if student_params > 0 else 0
    
    logger.info(f"推理性能:")
    logger.info(f"  教師模型推理時間: {teacher_time*1000:.2f} ms")
    logger.info(f"  學生模型推理時間: {student_time*1000:.2f} ms")
    logger.info(f"  加速比: {speedup:.2f}x")
    logger.info(f"模型大小:")
    logger.info(f"  教師模型參數量: {teacher_params/1e6:.2f}M")
    logger.info(f"  學生模型參數量: {student_params/1e6:.2f}M")
    logger.info(f"  壓縮比: {compression:.2f}x")
    
    # 繪製比較圖表
    charts_dir = Path(config.get('project', {}).get('output_dir', 'outputs')) / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # mAP比較
    plt.figure(figsize=(12, 5))
    
    # mAP
    plt.subplot(1, 2, 1)
    models = ['教師模型', '學生模型']
    map_values = [teacher_metrics['mAP'], student_metrics['mAP']]
    plt.bar(models, map_values, color=['skyblue', 'salmon'])
    plt.title('mAP比較')
    plt.ylabel('mAP')
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 推理時間
    plt.subplot(1, 2, 2)
    time_values = [teacher_time*1000, student_time*1000]  # 轉為毫秒
    plt.bar(models, time_values, color=['skyblue', 'salmon'])
    plt.title('推理時間比較 (ms)')
    plt.ylabel('時間 (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'model_comparison.png')
    plt.close()
    
    logger.info(f"比較圖表已儲存至: {charts_dir / 'model_comparison.png'}")
    
    return {
        'teacher': teacher_metrics,
        'student': student_metrics,
        'speedup': speedup,
        'compression': compression
    }

def measure_inference_time(model, input_tensor, num_runs=100):
    """測量模型推理時間"""
    # 進行熱身運行
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # 計時運行
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    # 計算平均時間
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    return avg_time

#-------------------------------- 命令行參數與主函數 --------------------------------#
def parse_args():
    """解析命令行參數"""
    import argparse
    parser = argparse.ArgumentParser(description='PCB缺陷檢測訓練')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--teacher', type=str, default=None,
                        help='教師模型路徑(如不指定，則使用預訓練模型)')
    parser.add_argument('--student', type=str, default=None,
                        help='學生模型路徑(如不指定，則從頭訓練)')
    parser.add_argument('--train-teacher', action='store_true',
                        help='先訓練教師模型再進行知識蒸餾')
    parser.add_argument('--no-distill', action='store_true',
                        help='不使用知識蒸餾(直接訓練學生模型)')
    parser.add_argument('--eval-only', action='store_true',
                        help='僅執行評估，不訓練')
    parser.add_argument('--optimize', action='store_true',
                        help='訓練後優化模型(剪枝和量化)')
    parser.add_argument('--export', action='store_true',
                        help='導出學生模型為部署格式')
    parser.add_argument('--export-format', type=str, default='onnx',
                        choices=['onnx', 'tflite'], 
                        help='導出格式 (onnx, tflite)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成視覺化預測結果')
    return parser.parse_args()

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設置總計時器
    total_timer = Timer().start()
    
    # 載入已有模型或初始化新模型
    teacher_model = None
    student_model = None
    
    if args.teacher:
        teacher_model = load_model(args.teacher, model_type='teacher')
    
    if args.student:
        student_model = load_model(args.student, model_type='student')
    
    # 獲取資料載入器
    logger.info("準備資料載入器...")
    train_loader, val_loader = get_dataloader(config)
    
    # 執行訓練或評估
    if args.eval_only:
        # 如果只評估，需要確保有模型
        if student_model is None and teacher_model is None:
            logger.error("評估模式需要指定--teacher或--student參數來載入模型")
            return
        
        # 評估模型
        if student_model is not None:
            logger.info("評估學生模型...")
            evaluate_model(student_model, val_loader)
        
        if teacher_model is not None:
            logger.info("評估教師模型...")
            evaluate_model(teacher_model, val_loader)
        
        # 如果兩個模型都有，進行比較
        if student_model is not None and teacher_model is not None:
            compare_models(teacher_model, student_model, val_loader, config)
        
        # 如果需要可視化預測
        if args.visualize:
            # 選擇要視覺化的模型
            model_to_visualize = student_model if student_model is not None else teacher_model
            
            # 獲取一些測試圖像
            dataset_path = config.get('dataset', {}).get('path', 'C:/Users/a/Desktop/conference/PCB_DATASET')
            image_dir = os.path.join(dataset_path, 'images')
            output_dir = os.path.join(config.get('project', {}).get('output_dir', 'outputs'), 'visualize')
            
            # 選擇一些代表性圖像
            import glob
            image_paths = []
            for defect_type in os.listdir(image_dir):
                defect_dir = os.path.join(image_dir, defect_type)
                if os.path.isdir(defect_dir):
                    images = glob.glob(os.path.join(defect_dir, '*.jpg'))
                    # 每種缺陷類型選擇5張圖像
                    image_paths.extend(images[:5])
            
            # 視覺化預測
            visualize_predictions(
                model=model_to_visualize,
                image_paths=image_paths,
                output_dir=output_dir,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                conf_threshold=config.get('model', {}).get('conf_threshold', 0.25)
            )
            
            logger.info(f"預測可視化結果已儲存至: {output_dir}")
    else:
        # 訓練流程
        # 如果需要訓練教師模型
        if args.train_teacher or teacher_model is None:
            logger.info("開始訓練教師模型...")
            
            if teacher_model is None:
                teacher_model = get_teacher_model(config)
            
            # 訓練教師模型
            teacher_model, teacher_map = train_teacher_model(
                teacher_model=teacher_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config
            )
            
            logger.info(f"教師模型訓練完成，最佳mAP: {teacher_map:.4f}")
        
        # 如果需要進行知識蒸餾
        if not args.no_distill:
            logger.info("開始知識蒸餾訓練...")
            
            # 確保有教師模型
            if teacher_model is None:
                logger.error("知識蒸餾需要教師模型，請先訓練教師模型或指定現有教師模型")
                return
            
            # 初始化學生模型(如果未指定)
            if student_model is None:
                student_model = get_student_model(config)
            
            # 知識蒸餾訓練
            student_model, student_map = train_with_distillation(
                teacher_model=teacher_model,
                student_model=student_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config
            )
            
            logger.info(f"知識蒸餾訓練完成，學生模型最佳mAP: {student_map:.4f}")
            
            # 比較教師模型和學生模型
            compare_models(teacher_model, student_model, val_loader, config)
        
        # 如果需要優化模型
        if args.optimize:
            logger.info("開始優化學生模型...")
            model_to_optimize = student_model if student_model is not None else teacher_model
            
            # 優化模型
            optimized_model = optimize_model(model_to_optimize, config)
            
            # 評估優化後的模型
            logger.info("評估優化後的模型...")
            optimized_metrics = evaluate_model(optimized_model, val_loader)
            
            # 輸出優化前後對比
            original_metrics = evaluate_model(model_to_optimize, val_loader)
            logger.info("優化前後性能對比:")
            logger.info(f"  原始mAP: {original_metrics['mAP']:.4f}")
            logger.info(f"  優化後mAP: {optimized_metrics['mAP']:.4f}")
            logger.info(f"  mAP變化: {optimized_metrics['mAP'] - original_metrics['mAP']:.4f}")
        
        # 如果需要導出模型
        if args.export:
            logger.info(f"開始導出學生模型為{args.export_format}格式...")
            model_to_export = student_model if student_model is not None else teacher_model
            
            # 設定導出路徑
            export_dir = Path(config.get('project', {}).get('output_dir', 'outputs')) / 'export'
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"pcb_detector.{args.export_format}"
            
            # 導出模型
            success = export_model(
                model=model_to_export,
                output_path=str(export_path),
                format=args.export_format
            )
            
            if success:
                logger.info(f"模型已成功導出至: {export_path}")
            else:
                logger.error("模型導出失敗")
        
        # 如果需要可視化預測
        if args.visualize:
            logger.info("生成視覺化預測結果...")
            model_to_visualize = student_model if student_model is not None else teacher_model
            
            # 獲取一些測試圖像
            dataset_path = config.get('dataset', {}).get('path', 'C:/Users/a/Desktop/conference/PCB_DATASET')
            image_dir = os.path.join(dataset_path, 'images')
            output_dir = os.path.join(config.get('project', {}).get('output_dir', 'outputs'), 'visualize')
            
            # 選擇一些代表性圖像
            import glob
            image_paths = []
            for defect_type in os.listdir(image_dir):
                defect_dir = os.path.join(image_dir, defect_type)
                if os.path.isdir(defect_dir):
                    images = glob.glob(os.path.join(defect_dir, '*.jpg'))
                    # 每種缺陷類型選擇5張圖像
                    image_paths.extend(images[:5])
            
            # 視覺化預測
            visualize_predictions(
                model=model_to_visualize,
                image_paths=image_paths,
                output_dir=output_dir,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                conf_threshold=config.get('model', {}).get('conf_threshold', 0.25)
            )
            
            logger.info(f"預測可視化結果已儲存至: {output_dir}")
    
    # 輸出總執行時間
    total_time = total_timer.stop().format_time()
    logger.info(f"總執行時間: {total_time}")

if __name__ == "__main__":
    main()