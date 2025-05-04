#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
eval.py - PCB缺陷檢測評估主程式
本模組整合了PCB缺陷檢測模型評估功能，提供全面的性能評估工具。
主要特點:
1. 整合評估指標計算：mAP、精確率、召回率等綜合評估
2. 支援多模型比較：可同時評估與比較多個模型性能
3. 提供視覺化輸出：檢測結果視覺化與性能對比圖表
4. 支援批次評估與單圖評估：滿足不同評估需求
"""

import os
import argparse
import torch
import yaml
import logging
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='PCB缺陷檢測評估')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路徑')
    parser.add_argument('--model-type', type=str, default='student',
                        help='模型類型 (teacher/student)')
    parser.add_argument('--compare', type=str, default=None,
                        help='比較模型路徑，逗號分隔多個模型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='資料集路徑，覆蓋配置文件中的設定')
    parser.add_argument('--image', type=str, default=None,
                        help='單張圖像評估模式，指定圖像路徑')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='輸出目錄，覆蓋配置文件中的設定')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='檢測置信度閾值')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='NMS和mAP計算的IoU閾值')
    parser.add_argument('--show-image', action='store_true',
                        help='顯示檢測結果圖像')
    parser.add_argument('--save-image', action='store_true',
                        help='儲存檢測結果圖像')
    parser.add_argument('--benchmark', action='store_true',
                        help='執行性能基準測試')
    return parser.parse_args()

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

def load_model(model_path, model_type='student'):
    """載入模型"""
    logger.info(f"載入模型: {model_path}")
    try:
        # 嘗試直接載入PyTorch模型
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            model = torch.load(model_path, map_location='cpu')
        else:
            # 嘗試載入YOLO格式模型
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                logger.info(f"已載入YOLO模型: {model_path}")
                return model
            except:
                logger.error(f"無法載入YOLO模型，嘗試載入自定義模型")
                # 導入自定義模型載入功能
                from models.model import load_model as custom_load_model
                model = custom_load_model(model_path, model_type)
        
        logger.info(f"模型載入成功")
        return model
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        return None

def get_dataloader(config):
    """獲取資料載入器"""
    try:
        # 先嘗試從data.dataset導入
        from data.dataset import get_dataloader as get_data
        return get_data(config)
    except ImportError:
        logger.error("無法導入資料模組，嘗試內嵌實現")
        
        # 內嵌實現簡化版資料載入器
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        import os
        import cv2
        import xml.etree.ElementTree as ET
        import random
        
        class SimplePCBDataset(Dataset):
            """簡化版PCB缺陷檢測資料集"""
            def __init__(self, root_dir, mode='val', img_size=640):
                self.root_dir = root_dir
                self.img_size = img_size
                self.mode = mode
                self.imgs_path = os.path.join(root_dir, 'images')
                self.annotations_path = os.path.join(root_dir, 'Annotations')
                
                # 讀取資料路徑
                self.image_files = []
                self.annotation_files = []
                
                # 從每個缺陷類型的資料夾載入圖片和標註
                for defect_type in DEFECT_CLASSES.keys():
                    img_dir = os.path.join(self.imgs_path, defect_type)
                    ann_dir = os.path.join(self.annotations_path, defect_type)
                    
                    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                        logger.warning(f"找不到路徑 {img_dir} 或 {ann_dir}")
                        continue
                        
                    for filename in os.listdir(img_dir):
                        if filename.endswith('.jpg'):
                            img_path = os.path.join(img_dir, filename)
                            ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.xml'))
                            
                            if os.path.exists(ann_path):
                                self.image_files.append(img_path)
                                self.annotation_files.append(ann_path)
                
                # 分割訓練和驗證資料集
                indices = list(range(len(self.image_files)))
                random.seed(42)  # 確保可重複性
                random.shuffle(indices)
                
                split = int(len(indices) * 0.8)
                train_indices = indices[:split]
                val_indices = indices[split:]
                
                if mode == 'train':
                    self.image_files = [self.image_files[i] for i in train_indices]
                    self.annotation_files = [self.annotation_files[i] for i in train_indices]
                else:  # val
                    self.image_files = [self.image_files[i] for i in val_indices]
                    self.annotation_files = [self.annotation_files[i] for i in val_indices]
                
                # 設定轉換
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.image_files)
            
            def __getitem__(self, idx):
                img_path = self.image_files[idx]
                ann_path = self.annotation_files[idx]
                
                # 讀取圖片
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 獲取原始尺寸
                h, w, _ = img.shape
                
                # 讀取標註
                boxes, labels = self._parse_annotation(ann_path, w, h)
                
                # 應用轉換
                img = self.transform(img)
                
                # 將邊界框和標籤轉換為張量
                if len(boxes) > 0:
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.as_tensor(labels, dtype=torch.int64)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros(0, dtype=torch.int64)
                
                # 構建目標字典
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'image_id': torch.tensor([idx])
                }
                
                return img, target
            
            def _parse_annotation(self, ann_path, img_width, img_height):
                """解析XML標註檔"""
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
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
                    xmin = float(bbox.find('xmin').text) / img_width
                    ymin = float(bbox.find('ymin').text) / img_height
                    xmax = float(bbox.find('xmax').text) / img_width
                    ymax = float(bbox.find('ymax').text) / img_height
                    
                    # 確保邊界框坐標有效
                    if xmin >= xmax or ymin >= ymax:
                        continue
                        
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                    
                return boxes, labels
        
        def collate_fn(batch):
            """自定義批次組合函數"""
            images, targets = list(zip(*batch))
            return torch.stack(images), targets
        
        # 從配置中讀取參數
        root_dir = config.get('dataset_path', 'C:/Users/a/Desktop/研討會/PCB_DATASET')
        batch_size = config.get('batch_size', 16)
        img_size = config.get('img_size', 640)
        num_workers = config.get('num_workers', 4)
        
        # 創建資料集和資料載入器
        val_dataset = SimplePCBDataset(root_dir=root_dir, mode='val', img_size=img_size)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"已建立資料載入器 ({len(val_dataset)} 樣本)")
        
        return None, val_loader

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
    iou = intersection / union
    
    return iou

def calculate_map(pred_boxes, pred_labels, pred_scores, 
                gt_boxes, gt_labels, iou_threshold=0.5, num_classes=len(DEFECT_CLASSES)):
    """計算平均精確度均值 (mAP)"""
    # 初始化
    all_detections = []  # 所有預測，按類別分組
    all_groundtruth = []  # 所有真值，按類別分組
    
    # 為每個類別初始化列表
    for _ in range(num_classes):
        all_detections.append([])
        all_groundtruth.append([])
    
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
        precision = np.divide(cumsum_tp, (cumsum_tp + cumsum_fp), out=np.zeros_like(cumsum_tp), where=(cumsum_tp + cumsum_fp) > 0)
        
        # 計算AP (平均精確度)
        # 11點插值平均精確度計算
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
    mean_precision = np.mean(precisions[np.array([len(groundtruth) > 0 for groundtruth in all_groundtruth])])
    mean_recall = np.mean(recalls[np.array([len(groundtruth) > 0 for groundtruth in all_groundtruth])])
    
    metrics = {
        'mAP': float(mean_ap),
        'precision': float(mean_precision),
        'recall': float(mean_recall),
        'ap_per_class': {DEFECT_CLASSES_INV[i]: float(ap) for i, ap in enumerate(average_precisions) if len(all_groundtruth[i]) > 0}
    }
    
    return metrics

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
    try:
        # 嘗試處理YOLO格式輸出
        if hasattr(outputs, 'boxes'):
            # YOLO8 原生格式
            boxes = []
            labels = []
            scores = []
            
            for i, det in enumerate(outputs.boxes):
                if det.conf >= conf_threshold:
                    x1, y1, x2, y2 = det.xyxy[0].tolist()
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(det.cls))
                    scores.append(float(det.conf))
                    
            return boxes, labels, scores
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # 自定義格式
            detections = outputs[0]
            
            # 初始化結果列表
            boxes = []
            labels = []
            scores = []
            
            if len(detections) > 0:
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
        else:
            # 未知格式
            logger.warning("無法識別的輸出格式，返回空結果")
            return [], [], []
    except Exception as e:
        logger.error(f"處理預測時發生錯誤: {e}")
        return [], [], []

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
        
        # 檢查坐標是否已經是絕對坐標(大於1)
        if max(x1, y1, x2, y2) <= 1.0:
            # 相對坐標轉絕對坐標
            x1 = int(max(0, x1 * w))
            y1 = int(max(0, y1 * h))
            x2 = int(min(w, x2 * w))
            y2 = int(min(h, y2 * h))
        else:
            # 已經是絕對坐標，進行取整和邊界檢查
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w, x2))
            y2 = int(min(h, y2))
        
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

def evaluate_model(model, val_loader, conf_threshold=0.25, iou_threshold=0.5, device='cuda'):
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
            
            # 處理每張圖像的預測結果
            for batch_idx, _ in enumerate(images):
                # 獲取該批次的預測
                if hasattr(outputs, 'boxes'):
                    # YOLO8 原生格式
                    batch_boxes = []
                    batch_labels = []
                    batch_scores = []
                    
                    # 篩選屬於該批次的預測
                    for i, det in enumerate(outputs.boxes):
                        if det.conf >= conf_threshold:
                            x1, y1, x2, y2 = det.xyxy[0].tolist()
                            batch_boxes.append([x1/images.shape[3], y1/images.shape[2], 
                                              x2/images.shape[3], y2/images.shape[2]])  # 轉為相對坐標
                            batch_labels.append(int(det.cls))
                            batch_scores.append(float(det.conf))
                            
                    all_pred_boxes.append(batch_boxes)
                    all_pred_labels.append(batch_labels)
                    all_pred_scores.append(batch_scores)
                else:
                    # 自定義格式
                    boxes, labels, scores = process_predictions(outputs, conf_threshold)
                    all_pred_boxes.append(boxes)
                    all_pred_labels.append(labels)
                    all_pred_scores.append(scores)
                
                # 獲取真實標籤
                target = targets[batch_idx]
                all_gt_boxes.append(target['boxes'].cpu().numpy())
                all_gt_labels.append(target['labels'].cpu().numpy())
    
    # 計算評估指標
    metrics = calculate_map(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_gt_boxes, all_gt_labels,
        iou_threshold=iou_threshold
    )
    
    # 輸出結果
    logger.info(f"評估結果:")
    logger.info(f"  mAP@{iou_threshold}: {metrics['mAP']:.4f}")
    logger.info(f"  精確率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    
    # 顯示各類別的AP
    if 'ap_per_class' in metrics:
        logger.info("各類別AP:")
        for class_name, ap in metrics['ap_per_class'].items():
            logger.info(f"  {class_name}: {ap:.4f}")
    
    return metrics

def benchmark_model(model, input_shape=(1, 3, 640, 640), num_runs=100, device='cuda'):
    """執行模型性能基準測試"""
    logger.info(f"開始性能基準測試，運行次數: {num_runs}")
    
    # 設定裝置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 創建示例輸入
    dummy_input = torch.randn(input_shape).to(device)
    
    # 熱身運行
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # 計時運行
    inference_times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="性能測試"):
            # 開始計時
            start_time = time.time()
            
            # 前向傳播
            _ = model(dummy_input)
            
            # 同步GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            # 記錄時間
            # 記錄時間
            inference_times.append(time.time() - start_time)
    
    # 計算統計數據
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # 計算推理速度(FPS)
    fps = 1.0 / avg_time
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / 1e6  # 轉換為百萬
    
    # 輸出結果
    logger.info(f"性能基準測試結果:")
    logger.info(f"  平均推理時間: {avg_time*1000:.2f} ms")
    logger.info(f"  最小推理時間: {min_time*1000:.2f} ms")
    logger.info(f"  最大推理時間: {max_time*1000:.2f} ms")
    logger.info(f"  標準差: {std_time*1000:.2f} ms")
    logger.info(f"  推理速度: {fps:.2f} FPS")
    logger.info(f"  模型參數量: {total_params_m:.2f} M")
    
    results = {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'fps': fps,
        'total_params': total_params,
        'total_params_m': total_params_m
    }
    
    return results

def plot_class_ap(ap_dict, output_path=None):
    """繪製各類別AP圖表"""
    # 設定圖表
    plt.figure(figsize=(10, 6))
    
    # 獲取類別和AP值
    classes = list(ap_dict.keys())
    ap_values = list(ap_dict.values())
    
    # 繪製條形圖
    bars = plt.bar(classes, ap_values, color='skyblue')
    
    # 設定圖表屬性
    plt.title('各類別平均精確度(AP)')
    plt.xlabel('缺陷類別')
    plt.ylabel('AP值')
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # 在每個柱子上添加數值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 儲存或顯示圖表
    if output_path:
        plt.savefig(output_path)
        logger.info(f"類別AP圖表已儲存至: {output_path}")
    else:
        plt.show()
    
    plt.close()

def evaluate_single_image(model, image_path, conf_threshold=0.25, device='cuda', show=False, save_path=None):
    """評估單張圖像"""
    logger.info(f"評估圖像: {image_path}")
    
    # 設定裝置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"無法讀取圖像: {image_path}")
        return None
    
    # 獲取原始尺寸
    h, w, _ = image.shape
    
    # 預處理圖像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_image(image_rgb).to(device)
    
    # 模型預測
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # 處理預測結果
    boxes, labels, scores = process_predictions(outputs, conf_threshold)
    
    # 將相對坐標轉為絕對坐標
    abs_boxes = []
    for box in boxes:
        # 檢查是否已經是絕對坐標
        if max(box) <= 1.0:
            x1, y1, x2, y2 = box
            abs_boxes.append([
                int(x1 * w), int(y1 * h), 
                int(x2 * w), int(y2 * h)
            ])
        else:
            abs_boxes.append(box)
    
    # 繪製預測結果
    result_image = draw_boxes(image, boxes, labels, scores, conf_threshold)
    
    # 顯示圖像
    if show:
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 儲存圖像
    if save_path:
        cv2.imwrite(save_path, result_image)
        logger.info(f"預測結果已儲存至: {save_path}")
    
    # 返回預測結果
    detection_results = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'class_names': [DEFECT_CLASSES_INV.get(label, f"未知-{label}") for label in labels],
        'result_image': result_image
    }
    
    return detection_results

def compare_models(models_dict, val_loader, conf_threshold=0.25, iou_threshold=0.5, device='cuda', output_dir=None):
    """比較多個模型的性能"""
    logger.info(f"開始比較 {len(models_dict)} 個模型")
    
    # 設定裝置
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 儲存所有模型的評估結果
    results = {}
    
    # 評估每個模型
    for model_name, model in models_dict.items():
        logger.info(f"評估模型: {model_name}")
        
        # 將模型移到裝置
        model.to(device)
        model.eval()
        
        # 執行評估
        metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        # 儲存結果
        results[model_name] = metrics
        
        # 如果指定了輸出目錄，繪製各類別AP圖表
        if output_dir and 'ap_per_class' in metrics:
            os.makedirs(output_dir, exist_ok=True)
            plot_class_ap(
                ap_dict=metrics['ap_per_class'],
                output_path=os.path.join(output_dir, f"{model_name}_class_ap.png")
            )
    
    # 如果指定了輸出目錄，繪製模型比較圖表
    if output_dir and len(results) > 1:
        # 繪製mAP比較圖表
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        map_values = [results[name]['mAP'] for name in model_names]
        
        bars = plt.bar(model_names, map_values, color='skyblue')
        
        plt.title('模型mAP比較')
        plt.xlabel('模型')
        plt.ylabel('mAP值')
        plt.ylim([0, 1.0])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # 在每個柱子上添加數值標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "models_map_comparison.png"))
        plt.close()
        
        # 繪製精確率和召回率比較圖表
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        precision_values = [results[name]['precision'] for name in model_names]
        recall_values = [results[name]['recall'] for name in model_names]
        
        plt.bar(x - width/2, precision_values, width, label='精確率', color='skyblue')
        plt.bar(x + width/2, recall_values, width, label='召回率', color='salmon')
        
        plt.title('模型精確率與召回率比較')
        plt.xlabel('模型')
        plt.ylabel('數值')
        plt.ylim([0, 1.0])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "models_pr_comparison.png"))
        plt.close()
        
        logger.info(f"模型比較圖表已儲存至: {output_dir}")
    
    # 輸出比較結果
    logger.info("模型比較結果:")
    for model_name, metrics in results.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    mAP: {metrics['mAP']:.4f}")
        logger.info(f"    精確率: {metrics['precision']:.4f}")
        logger.info(f"    召回率: {metrics['recall']:.4f}")
    
    return results

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 更新配置
    if args.dataset:
        config['dataset_path'] = args.dataset
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.conf_threshold:
        config['conf_threshold'] = args.conf_threshold
    
    # 設定輸出目錄
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入主模型
    model = load_model(args.model, args.model_type)
    if model is None:
        logger.error("載入模型失敗，程式退出")
        return
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 如果是單張圖像評估模式
    if args.image:
        # 設定輸出圖像路徑
        if args.save_image:
            save_path = os.path.join(output_dir, 'prediction_result.jpg')
        else:
            save_path = None
            
        # 評估單張圖像
        results = evaluate_single_image(
            model=model,
            image_path=args.image,
            conf_threshold=args.conf_threshold,
            device=device,
            show=args.show_image,
            save_path=save_path
        )
        
        if results:
            logger.info(f"檢測到 {len(results['boxes'])} 個缺陷")
            for i, (cls_name, score) in enumerate(zip(results['class_names'], results['scores'])):
                logger.info(f"  {i+1}. {cls_name}: {score:.4f}")
    else:
        # 獲取驗證資料載入器
        _, val_loader = get_dataloader(config)
        
        # 如果需要比較多個模型
        if args.compare:
            # 解析比較模型路徑
            compare_paths = args.compare.split(',')
            models_dict = {'主模型': model}
            
            # 載入比較模型
            for i, path in enumerate(compare_paths):
                compare_model = load_model(path.strip(), args.model_type)
                if compare_model:
                    # 獲取模型名稱
                    model_name = f"比較模型-{i+1}"
                    if '/' in path or '\\' in path:
                        model_name = os.path.basename(path).split('.')[0]
                    
                    models_dict[model_name] = compare_model
            
            # 設定比較輸出目錄
            compare_output_dir = os.path.join(output_dir, 'model_comparison')
            os.makedirs(compare_output_dir, exist_ok=True)
            
            # 執行模型比較
            compare_results = compare_models(
                models_dict=models_dict,
                val_loader=val_loader,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                device=device,
                output_dir=compare_output_dir
            )
        else:
            # 如果需要進行基準測試
            if args.benchmark:
                benchmark_results = benchmark_model(
                    model=model,
                    input_shape=(1, 3, config.get('img_size', 640), config.get('img_size', 640)),
                    num_runs=100,
                    device=device
                )
            
            # 評估主模型
            metrics = evaluate_model(
                model=model,
                val_loader=val_loader,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                device=device
            )
            
            # 繪製類別AP圖表
            if 'ap_per_class' in metrics:
                plot_class_ap(
                    ap_dict=metrics['ap_per_class'],
                    output_path=os.path.join(output_dir, "class_ap.png")
                )

if __name__ == "__main__":
    # 計時整個流程
    start_time = time.time()
    
    # 執行主函數
    main()
    
    # 計算並輸出總耗時
    elapsed_time = time.time() - start_time
    logger.info(f"評估完成，總耗時: {elapsed_time:.2f} 秒")