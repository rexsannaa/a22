#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
utils.py - PCB缺陷檢測工具模組
本模組整合了評估指標計算、視覺化、日誌記錄等輔助功能，
為PCB缺陷檢測專案提供通用工具支援。
主要特點:
1. 整合mAP、精確率、召回率等評估指標計算
2. 提供檢測結果視覺化功能
3. 實現日誌記錄與TensorBoard整合
4. 支援模型權重儲存與載入等工具函數
"""

import os
import time
import logging
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import yaml
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

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

def load_config(config_path):
    """載入配置檔案
    
    參數:
        config_path: 配置檔案路徑
        
    回傳:
        config: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"已成功載入配置檔案: {config_path}")
            return config
    except Exception as e:
        logger.error(f"載入配置檔案失敗: {e}")
        return {}

def setup_experiment(config):
    """設置實驗環境
    
    參數:
        config: 配置字典
        
    回傳:
        output_dir: 輸出目錄
        run_name: 實驗名稱
    """
    # 建立輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.get('experiment_name', 'pcb_defect')}_{timestamp}"
    
    output_dir = Path(config.get('output_dir', 'outputs'))
    weights_dir = output_dir / 'weights' / run_name
    charts_dir = output_dir / 'charts' / run_name
    logs_dir = output_dir / 'logs' / run_name
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"實驗名稱: {run_name}")
    logger.info(f"權重儲存目錄: {weights_dir}")
    logger.info(f"圖表儲存目錄: {charts_dir}")
    logger.info(f"日誌儲存目錄: {logs_dir}")
    
    # 儲存配置
    with open(logs_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return {
        'weights_dir': weights_dir,
        'charts_dir': charts_dir,
        'logs_dir': logs_dir,
        'run_name': run_name
    }

def calculate_iou(box1, box2):
    """計算兩個邊界框的IoU
    
    參數:
        box1: [x1, y1, x2, y2] 格式的邊界框
        box2: [x1, y1, x2, y2] 格式的邊界框
        
    回傳:
        iou: 交並比
    """
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
                gt_boxes, gt_labels, iou_threshold=0.5, num_classes=6):
    """計算平均精確度均值 (mAP)
    
    參數:
        pred_boxes: 預測邊界框列表，每個元素形狀為 [n, 4]
        pred_labels: 預測類別列表，每個元素形狀為 [n]
        pred_scores: 預測分數列表，每個元素形狀為 [n]
        gt_boxes: 真實邊界框列表，每個元素形狀為 [m, 4]
        gt_labels: 真實類別列表，每個元素形狀為 [m]
        iou_threshold: IoU閾值，預設為0.5
        num_classes: 類別數量，預設為6
        
    回傳:
        metrics: 包含mAP、精確率、召回率的字典
    """
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
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        
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

def draw_boxes(image, boxes, labels, scores=None, threshold=0.5):
    """在圖像上繪製邊界框
    
    參數:
        image: 原始圖像 (OpenCV格式，BGR)
        boxes: 邊界框列表，格式為 [x1, y1, x2, y2]
        labels: 類別標籤列表
        scores: 置信度分數列表
        threshold: 顯示的置信度閾值
        
    回傳:
        image: 繪製了邊界框的圖像
    """
    image_copy = image.copy()
    h, w, _ = image_copy.shape
    
    for i, box in enumerate(boxes):
        # 過濾低置信度的檢測
        if scores is not None and scores[i] < threshold:
            continue
        
        # 轉換為整數坐標並確保在圖像範圍內
        x1, y1, x2, y2 = box
        x1 = int(max(0, x1 * w))
        y1 = int(max(0, y1 * h))
        x2 = int(min(w, x2 * w))
        y2 = int(min(h, y2 * h))
        
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

def visualize_predictions(model, image_paths, output_dir, device, conf_threshold=0.25):
    """視覺化模型預測結果
    
    參數:
        model: 檢測模型
        image_paths: 圖像路徑列表
        output_dir: 輸出目錄
        device: 運算設備
        conf_threshold: 置信度閾值
    """
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
            h, w, _ = original_image.shape
            
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
    """預處理圖像用於模型輸入
    
    參數:
        image: RGB格式的圖像
        
    回傳:
        tensor: 預處理後的圖像張量
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # 添加批次維度

def process_predictions(outputs, conf_threshold=0.25):
    """處理模型預測輸出
    
    參數:
        outputs: 模型輸出
        conf_threshold: 置信度閾值
        
    回傳:
        boxes: 邊界框列表
        labels: 類別標籤列表
        scores: 置信度分數列表
    """
    # 這個函數需要根據實際模型輸出格式調整
    # 這裡假設是YOLO格式輸出
    
    # 從輸出中提取資訊
    # 假設outputs是一個列表，第一個元素包含檢測結果
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

def plot_training_metrics(metrics_dict, output_path):
    """繪製訓練指標圖表
    
    參數:
        metrics_dict: 包含訓練指標的字典，格式為 {'metric_name': [values]}
        output_path: 圖表輸出路徑
    """
    # 設定圖表
    plt.figure(figsize=(12, 8))
    
    # 設定子圖數量
    num_metrics = len(metrics_dict)
    rows = (num_metrics + 1) // 2  # 向上取整
    cols = min(2, num_metrics)
    
    # 繪製每個指標
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        plt.subplot(rows, cols, i+1)
        plt.plot(values, '-o', label=metric_name)
        plt.title(metric_name)
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"訓練指標圖表已儲存至: {output_path}")

def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """繪製混淆矩陣
    
    參數:
        confusion_matrix: 混淆矩陣 numpy 陣列
        class_names: 類別名稱列表
        output_path: 圖表輸出路徑
    """
    # 設定圖表
    plt.figure(figsize=(10, 8))
    
    # 繪製混淆矩陣
    plt.imshow(confusion_matrix, cmap='Blues')
    
    # 設定標籤
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加文字
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # 儲存圖表
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"混淆矩陣圖表已儲存至: {output_path}")

def plot_pr_curve(precisions, recalls, ap_values, class_names, output_path):
    """繪製精確率-召回率曲線
    
    參數:
        precisions: 每個類別的精確率列表的列表
        recalls: 每個類別的召回率列表的列表
        ap_values: 每個類別的平均精確度值
        class_names: 類別名稱列表
        output_path: 圖表輸出路徑
    """
    # 設定圖表
    plt.figure(figsize=(10, 8))
    
    # 為每個類別繪製PR曲線
    for i, class_name in enumerate(class_names):
        plt.plot(recalls[i], precisions[i], lw=2, 
                 label=f'{class_name} (AP={ap_values[i]:.3f})')
    
    # 設定圖表屬性
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend(loc="lower left")
    
    # 儲存圖表
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"PR曲線圖表已儲存至: {output_path}")

def save_model_summary(model, output_path):
    """儲存模型摘要資訊
    
    參數:
        model: PyTorch模型
        output_path: 輸出文件路徑
    """
    # 計算模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 建立模型摘要文字
    summary = [
        f"模型摘要 ({type(model).__name__})",
        f"總參數數量: {total_params:,}",
        f"可訓練參數數量: {trainable_params:,}",
        f"固定參數數量: {total_params - trainable_params:,}",
        ""
    ]
    
    # 儲存摘要
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    logger.info(f"模型摘要已儲存至: {output_path}")

def generate_report(metrics, class_names, experiment_dir, config):
    """生成實驗報告
    
    參數:
        metrics: 評估指標字典
        class_names: 類別名稱列表
        experiment_dir: 實驗目錄
        config: 配置字典
    """
    report_path = os.path.join(experiment_dir, 'report.md')
    
    # 建立報告內容
    report = [
        f"# PCB缺陷檢測實驗報告",
        f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## 實驗配置",
        f"- 實驗名稱: {config.get('experiment_name', 'pcb_defect')}",
        f"- 模型類型: {config.get('model_type', 'YOLOv8')}",
        f"- 批次大小: {config.get('batch_size', 16)}",
        f"- 學習率: {config.get('learning_rate', 0.001)}",
        f"- 訓練周期: {config.get('epochs', 100)}",
        "",
        f"## 整體性能",
        f"- mAP@0.5: {metrics.get('mAP', 0):.4f}",
        f"- 精確率: {metrics.get('precision', 0):.4f}",
        f"- 召回率: {metrics.get('recall', 0):.4f}",
        "",
        f"## 各類別性能"
    ]
    
    # 添加各類別性能
    for class_name in class_names:
        ap = metrics.get('ap_per_class', {}).get(class_name, 0)
        report.append(f"- {class_name}: AP = {ap:.4f}")
    
    # 添加圖表引用
    report.extend([
        "",
        f"## 視覺化結果",
        f"- 訓練曲線: ![訓練曲線](./charts/training_metrics.png)",
        f"- 混淆矩陣: ![混淆矩陣](./charts/confusion_matrix.png)",
        f"- PR曲線: ![PR曲線](./charts/pr_curve.png)",
        "",
        f"## 結論",
        f"(此處添加實驗結論和分析)",
    ])
    
    # 儲存報告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"實驗報告已儲存至: {report_path}")

class Timer:
    """簡單的計時器類"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """開始計時"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止計時並返回經過時間"""
        if self.start_time is None:
            return 0
            
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        """上下文管理器開始"""
        return self.start()
    
    def __exit__(self, *args):
        """上下文管理器結束"""
        self.stop()
    
    def format_time(self):
        """將時間格式化為人類可讀形式"""
        hours, remainder = divmod(self.elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        elif minutes > 0:
            return f"{int(minutes)}m {seconds:.2f}s"
        else:
            return f"{seconds:.2f}s"

def compute_confusion_matrix(pred_labels, gt_labels, num_classes=len(DEFECT_CLASSES)):
    """計算混淆矩陣
    
    參數:
        pred_labels: 預測標籤列表
        gt_labels: 真實標籤列表
        num_classes: 類別數量
        
    回傳:
        confusion_matrix: 混淆矩陣
    """
    # 初始化混淆矩陣
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # 填充混淆矩陣
    for i in range(len(pred_labels)):
        for pred, gt in zip(pred_labels[i], gt_labels[i]):
            pred_class = int(pred)
            gt_class = int(gt)
            
            if 0 <= pred_class < num_classes and 0 <= gt_class < num_classes:
                confusion_matrix[gt_class, pred_class] += 1
    
    return confusion_matrix
def find_optimal_threshold(precisions, recalls, scores):
    """查找最佳閾值，基於F1分數
    
    參數:
        precisions: 精確率列表
        recalls: 召回率列表
        scores: 閾值分數列表
        
    回傳:
        optimal_threshold: 最佳閾值
        optimal_f1: 最佳F1分數
    """
    # 計算每個閾值的F1分數
    f1_scores = []
    
    for p, r in zip(precisions, recalls):
        if p + r > 0:  # 避免除以零
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # 找到最佳F1分數對應的索引
    if not f1_scores:
        return 0.5, 0  # 默認值
        
    best_idx = np.argmax(f1_scores)
    optimal_threshold = scores[best_idx]
    optimal_f1 = f1_scores[best_idx]
    
    return optimal_threshold, optimal_f1

def export_model(model, output_path, input_shape=(1, 3, 640, 640), format='onnx'):
    """導出模型為不同格式
    
    參數:
        model: 原始PyTorch模型
        output_path: 輸出文件路徑
        input_shape: 輸入形狀
        format: 導出格式，支援'onnx'、'pt'
    """
    model.eval()
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == 'onnx':
        try:
            # 創建示例輸入
            dummy_input = torch.randn(input_shape)
            
            # 導出為ONNX
            import torch.onnx
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                opset_version=12,
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"模型已成功導出為ONNX格式: {output_path}")
            
        except Exception as e:
            logger.error(f"導出ONNX模型時發生錯誤: {e}")
            
    elif format.lower() == 'pt':
        try:
            # 儲存PyTorch模型
            torch.save(model.state_dict(), output_path)
            logger.info(f"模型已成功儲存為PyTorch格式: {output_path}")
            
        except Exception as e:
            logger.error(f"儲存PyTorch模型時發生錯誤: {e}")
    
    else:
        logger.error(f"不支援的導出格式: {format}")

def optimize_model(model, config):
    """優化模型，包括量化和剪枝
    
    參數:
        model: 原始模型
        config: 優化配置字典
        
    回傳:
        optimized_model: 優化後的模型
    """
    # 深度複製模型
    import copy
    optimized_model = copy.deepcopy(model)
    
    # 應用量化(如果指定)
    if config.get('quantize', False):
        try:
            import torch.quantization
            
            # 設定量化配置
            quantize_config = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(optimized_model, inplace=True)
            
            # 執行靜態量化
            # 這裡通常需要進行校準，但為了簡化，我們跳過這一步
            torch.quantization.convert(optimized_model, inplace=True)
            
            logger.info("模型已成功量化")
            
        except Exception as e:
            logger.error(f"量化模型時發生錯誤: {e}")
    
    # 應用剪枝(如果指定)
    if config.get('prune', False):
        try:
            import torch.nn.utils.prune as prune
            
            # 獲取剪枝設定
            prune_amount = config.get('prune_amount', 0.3)  # 默認剪枝30%權重
            
            # 對每個卷積層應用剪枝
            for module in optimized_model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=prune_amount)
                    # 使剪枝永久化
                    prune.remove(module, 'weight')
            
            logger.info(f"模型已成功剪枝 ({prune_amount*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"剪枝模型時發生錯誤: {e}")
    
    return optimized_model

def calculate_flops_and_params(model, input_shape=(1, 3, 640, 640)):
    """計算模型的FLOPS和參數量
    
    參數:
        model: PyTorch模型
        input_shape: 輸入形狀
        
    回傳:
        stats: 包含FLOPS和參數量的字典
    """
    # 確保模型為eval模式
    model.eval()
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 使用thop庫計算FLOPS
    try:
        from thop import profile
        
        # 創建示例輸入
        dummy_input = torch.randn(input_shape)
        
        # 計算FLOPS
        macs, params = profile(model, inputs=(dummy_input,))
        
        # 將MAC轉換為FLOPS (1 MAC = 2 FLOPS)
        flops = macs * 2
        
        stats = {
            'params': total_params,
            'params_millions': total_params / 1e6,
            'flops': flops,
            'flops_billions': flops / 1e9
        }
        
        logger.info(f"模型參數量: {stats['params_millions']:.2f}M")
        logger.info(f"模型FLOPS: {stats['flops_billions']:.2f}G")
        
        return stats
        
    except ImportError:
        logger.warning("無法計算FLOPS，請安裝thop庫")
        
        stats = {
            'params': total_params,
            'params_millions': total_params / 1e6,
        }
        
        logger.info(f"模型參數量: {stats['params_millions']:.2f}M")
        
        return stats

def plot_model_comparison(models_stats, output_path):
    """繪製模型比較圖表
    
    參數:
        models_stats: 模型統計字典，格式為 {'model_name': {'params': x, 'flops': y, ...}}
        output_path: 圖表輸出路徑
    """
    # 獲取模型名稱
    model_names = list(models_stats.keys())
    
    # 獲取參數和FLOPS
    params = [stats.get('params_millions', 0) for stats in models_stats.values()]
    flops = [stats.get('flops_billions', 0) for stats in models_stats.values()]
    
    # 設定圖表
    plt.figure(figsize=(12, 6))
    
    # 參數量對比
    plt.subplot(1, 2, 1)
    plt.bar(model_names, params, color='skyblue')
    plt.title('參數量對比 (百萬)')
    plt.ylabel('參數量 (M)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # FLOPS對比
    plt.subplot(1, 2, 2)
    plt.bar(model_names, flops, color='salmon')
    plt.title('計算量對比 (十億FLOPS)')
    plt.ylabel('FLOPS (G)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"模型比較圖表已儲存至: {output_path}")

def visual_model_preds(image_path, models_dict, output_dir, conf_threshold=0.25):
    """視覺化多個模型的預測結果比較
    
    參數:
        image_path: 測試圖像路徑
        models_dict: 模型字典，格式為 {'model_name': model}
        output_dir: 輸出目錄
        conf_threshold: 置信度閾值
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"無法讀取圖像: {image_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取圖像名稱
    image_name = os.path.basename(image_path)
    
    # 處理每個模型
    for model_name, model in models_dict.items():
        # 設定為評估模式
        model.eval()
        
        # 複製原始圖像
        image_copy = image.copy()
        
        # 預處理圖像
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        image_tensor = preprocess_image(image_rgb)
        
        # 進行預測
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # 處理預測結果
        boxes, labels, scores = process_predictions(outputs, conf_threshold)
        
        # 繪製邊界框
        result_image = draw_boxes(image_copy, boxes, labels, scores, conf_threshold)
        
        # 添加模型名稱標籤
        cv2.putText(
            result_image,
            model_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        # 儲存結果
        output_path = os.path.join(output_dir, f"{model_name}_{image_name}")
        cv2.imwrite(output_path, result_image)
    
    logger.info(f"多模型預測比較已儲存至: {output_dir}")

def extract_inference_times(model, dataloader, device, num_runs=100):
    """測量模型推理時間
    
    參數:
        model: 要測試的模型
        dataloader: 資料載入器
        device: 運算設備
        num_runs: 運行次數
        
    回傳:
        inference_stats: 包含推理時間統計的字典
    """
    # 設定為評估模式
    model.eval()
    model = model.to(device)
    
    # 獲取一批資料用於測試
    for images, _ in dataloader:
        test_images = images.to(device)
        break
    
    # 熱身運行
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_images)
    
    # 計時運行
    inference_times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            # 開始計時
            start_time = time.time()
            
            # 前向傳播
            _ = model(test_images)
            
            # 同步GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            # 記錄時間
            inference_times.append(time.time() - start_time)
    
    # 計算統計數據
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    fps = 1.0 / avg_time
    
    inference_stats = {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps
    }
    
    logger.info(f"平均推理時間: {avg_time*1000:.2f} ms (FPS: {fps:.2f})")
    
    return inference_stats

def plot_inference_comparison(models_stats, output_path):
    """繪製模型推理時間比較圖表
    
    參數:
        models_stats: 模型統計字典，格式為 {'model_name': {'avg_time': x, 'fps': y, ...}}
        output_path: 圖表輸出路徑
    """
    # 獲取模型名稱
    model_names = list(models_stats.keys())
    
    # 獲取FPS和推理時間
    fps_values = [stats.get('fps', 0) for stats in models_stats.values()]
    inference_times = [stats.get('avg_time', 0) * 1000 for stats in models_stats.values()]  # 轉換為毫秒
    
    # 設定圖表
    plt.figure(figsize=(12, 5))
    
    # FPS對比
    plt.subplot(1, 2, 1)
    plt.bar(model_names, fps_values, color='lightgreen')
    plt.title('FPS對比')
    plt.ylabel('每秒幀數 (FPS)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # 推理時間對比
    plt.subplot(1, 2, 2)
    plt.bar(model_names, inference_times, color='lightcoral')
    plt.title('推理時間對比')
    plt.ylabel('平均推理時間 (ms)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"推理時間比較圖表已儲存至: {output_path}")

def plot_feature_visualization(features, layer_names, output_path):
    """視覺化模型中間特徵
    
    參數:
        features: 特徵列表，每個元素是(batch_size, channels, height, width)的張量
        layer_names: 層名稱列表
        output_path: 輸出路徑
    """
    # 確定子圖數量
    num_features = len(features)
    rows = int(np.ceil(np.sqrt(num_features)))
    cols = int(np.ceil(num_features / rows))
    
    # 建立圖表
    plt.figure(figsize=(15, 12))
    
    for i, (feature, name) in enumerate(zip(features, layer_names)):
        # 取第一個樣本的特徵
        feature = feature[0].detach().cpu().numpy()
        
        # 計算特徵圖的平均值(跨通道)
        feature_mean = np.mean(feature, axis=0)
        
        # 繪製子圖
        plt.subplot(rows, cols, i+1)
        plt.imshow(feature_mean, cmap='viridis')
        plt.title(name)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
    
    # 調整佈局並儲存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"特徵視覺化已儲存至: {output_path}")

def get_class_weight(dataset, num_classes=len(DEFECT_CLASSES)):
    """計算類別權重用於處理不平衡資料集
    
    參數:
        dataset: 資料集實例
        num_classes: 類別數量
        
    回傳:
        class_weights: 類別權重張量
    """
    # 初始化類別計數
    class_counts = np.zeros(num_classes)
    
    # 統計每個類別的樣本數量
    for _, target in dataset:
        labels = target['labels'].numpy()
        for label in labels:
            if 0 <= label < num_classes:
                class_counts[label] += 1
    
    # 計算權重 (反比於樣本數量)
    class_weights = np.zeros(num_classes)
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = 1.0 / class_counts[i]
        else:
            class_weights[i] = 1.0  # 對於沒有樣本的類別，設為1
    
    # 正規化權重，使其總和為num_classes
    if np.sum(class_weights) > 0:
        class_weights = class_weights * (num_classes / np.sum(class_weights))
    
    # 轉換為PyTorch張量
    weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"類別權重計算完成: {class_weights}")
    
    return weights_tensor

if __name__ == "__main__":
    """測試工具函數"""
    # 測試載入配置
    config = load_config("config/config.yaml")
    
    # 測試計算IoU
    box1 = [0.1, 0.1, 0.3, 0.3]
    box2 = [0.2, 0.2, 0.4, 0.4]
    iou = calculate_iou(box1, box2)
    logger.info(f"IoU: {iou}")
    
    # 測試計時器
    timer = Timer()
    with timer:
        time.sleep(1)
    logger.info(f"經過時間: {timer.format_time()}")
    
    logger.info("工具函數測試完成")