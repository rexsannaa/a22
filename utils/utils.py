#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
utils.py - PCB缺陷檢測實用工具模組
整合評估指標、視覺化、日誌和模型工具功能。
主要功能:
1. 檢測性能評估指標計算
2. 結果視覺化與繪圖
3. 訓練監控與日誌
4. 模型優化與推理輔助
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import cv2
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from torchvision.ops import box_iou
import yaml
from datetime import datetime
import shutil
from PIL import Image
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCBEvaluator:
    """PCB缺陷檢測評估器，提供準確率評估和性能分析"""
    
    def __init__(self, config):
        """
        初始化評估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.class_names = ["background"] + config["dataset"]["defect_classes"]
        self.num_classes = len(self.class_names)
        self.conf_threshold = config["evaluation"]["conf_threshold"]
        self.iou_threshold = config["evaluation"]["iou_threshold"]
        self.reset()
    
    def reset(self):
        """重置評估統計"""
        self.stats = {
            'TP': defaultdict(list),      # 真陽性 (按類別)
            'FP': defaultdict(list),      # 假陽性 (按類別)
            'FN': defaultdict(list),      # 假陰性 (按類別)
            'scores': defaultdict(list),  # 檢測分數 (按類別)
            'gt_counter': defaultdict(int) # 地面真值計數器 (按類別)
        }
    
    def process_batch(self, predictions, targets):
        """
        處理一個批次的預測和目標
        
        Args:
            predictions: 模型預測結果
            targets: 目標標註
        """
        # 遍歷每個圖像的預測和目標
        for pred, target in zip(predictions, targets):
            # 獲取預測的邊界框、分數和標籤
            pred_boxes = pred["boxes"].detach().cpu()
            pred_scores = pred["scores"].detach().cpu()
            pred_labels = pred["labels"].detach().cpu()
            
            # 應用置信度閾值
            keep = pred_scores > self.conf_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            
            # 獲取目標邊界框和標籤
            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()
            
            # 更新地面真值計數器
            for gt_label in gt_labels:
                self.stats['gt_counter'][gt_label.item()] += 1
            
            # 處理每個類別
            for c in range(1, self.num_classes):  # 跳過背景類別
                # 獲取當前類別的預測和目標
                c_pred_boxes = pred_boxes[pred_labels == c]
                c_pred_scores = pred_scores[pred_labels == c]
                c_gt_boxes = gt_boxes[gt_labels == c]
                
                # 如果沒有目標，則所有預測都是假陽性
                if len(c_gt_boxes) == 0:
                    for score in c_pred_scores:
                        self.stats['FP'][c].append(1)
                        self.stats['scores'][c].append(score.item())
                    continue
                
                # 如果沒有預測，則所有目標都是假陰性
                if len(c_pred_boxes) == 0:
                    for _ in range(len(c_gt_boxes)):
                        self.stats['FN'][c].append(1)
                    continue
                
                # 計算IoU矩陣
                iou_matrix = box_iou(c_pred_boxes, c_gt_boxes)
                
                # 根據分數排序預測
                score_indices = c_pred_scores.argsort(descending=True)
                
                # 追蹤已匹配的目標
                gt_matched = torch.zeros(len(c_gt_boxes), dtype=torch.bool)
                
                # 遍歷排序後的預測
                for pred_idx in score_indices:
                    # 獲取當前預測的IoU
                    ious = iou_matrix[pred_idx]
                    
                    # 找到最佳匹配
                    best_gt_idx = torch.argmax(ious)
                    best_iou = ious[best_gt_idx]
                    
                    # 如果IoU超過閾值且目標未匹配，則為真陽性
                    if best_iou >= self.iou_threshold and not gt_matched[best_gt_idx]:
                        self.stats['TP'][c].append(1)
                        self.stats['FP'][c].append(0)
                        gt_matched[best_gt_idx] = True
                    else:
                        # 否則為假陽性
                        self.stats['TP'][c].append(0)
                        self.stats['FP'][c].append(1)
                    
                    # 保存分數
                    self.stats['scores'][c].append(c_pred_scores[pred_idx].item())
                
                # 計算假陰性 (未匹配的目標)
                for gt_idx in range(len(c_gt_boxes)):
                    if not gt_matched[gt_idx]:
                        self.stats['FN'][c].append(1)
    
    def compute_ap(self, recall, precision):
        """
        計算平均精度 (Average Precision)
        使用所有點法 (all-point interpolation)
        
        Args:
            recall: 召回率數組
            precision: 精確率數組
        
        Returns:
            平均精度
        """
        # 確保開始於零召回率
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # 計算精確率包絡線 (precision envelope)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
        # 尋找召回率變化點
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 計算面積 (採用梯形法則)
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    def compute_metrics(self):
        """
        計算評估指標
        
        Returns:
            包含各項指標的字典
        """
        metrics = {
            'AP': {},    # 每類平均精度
            'precision': {}, # 精確率
            'recall': {}, # 召回率
            'f1': {},    # F1分數
        }
        
        # 計算整體平均精度
        aps = []
        
        # 對每個類別計算指標
        for c in range(1, self.num_classes):
            # 獲取類別名稱
            class_name = self.class_names[c]
            
            # 如果沒有此類別的地面真值或預測，設置為0
            if c not in self.stats['gt_counter'] or self.stats['gt_counter'][c] == 0:
                metrics['AP'][class_name] = 0.0
                metrics['precision'][class_name] = 0.0
                metrics['recall'][class_name] = 0.0
                metrics['f1'][class_name] = 0.0
                continue
            
            # 獲取類別統計數據
            tp = np.array(self.stats['TP'][c]) if c in self.stats['TP'] else np.array([])
            fp = np.array(self.stats['FP'][c]) if c in self.stats['FP'] else np.array([])
            scores = np.array(self.stats['scores'][c]) if c in self.stats['scores'] else np.array([])
            
            # 根據分數排序
            indices = np.argsort(-scores)
            tp = tp[indices]
            fp = fp[indices]
            
            # 計算累積值
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            
            # 計算精確率和召回率
            total_gt = self.stats['gt_counter'][c]
            recall = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
            precision = tp_cum / (tp_cum + fp_cum + 1e-10)
            
            # 計算AP
            ap = self.compute_ap(recall, precision)
            
            # 存儲結果
            metrics['AP'][class_name] = ap
            aps.append(ap)
            
            # 計算整體精確率、召回率和F1
            if len(tp) > 0:
                # 使用累積最大值
                overall_precision = np.sum(tp) / (np.sum(tp) + np.sum(fp) + 1e-10)
                overall_recall = np.sum(tp) / total_gt if total_gt > 0 else 0
                
                # 計算F1分數
                f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-10)
                
                metrics['precision'][class_name] = overall_precision
                metrics['recall'][class_name] = overall_recall
                metrics['f1'][class_name] = f1
            else:
                metrics['precision'][class_name] = 0.0
                metrics['recall'][class_name] = 0.0
                metrics['f1'][class_name] = 0.0
        
        # 計算mAP (所有類別AP的平均)
        metrics['mAP'] = np.mean(aps) if aps else 0.0
        
        # 計算整體指標
        metrics['overall'] = {
            'precision': np.mean([metrics['precision'][self.class_names[c]] for c in range(1, self.num_classes)]),
            'recall': np.mean([metrics['recall'][self.class_names[c]] for c in range(1, self.num_classes)]),
            'f1': np.mean([metrics['f1'][self.class_names[c]] for c in range(1, self.num_classes)])
        }
        
        return metrics


class VisualizationUtils:
    """可視化工具類，用於繪製檢測結果和訓練統計"""
    
    @staticmethod
    def visualize_detection(image, boxes, labels, scores=None, class_names=None, threshold=0.5, figsize=(10, 10)):
        """
        可視化目標檢測結果
        
        Args:
            image: PIL圖像或張量
            boxes: 邊界框張量 [x1, y1, x2, y2]
            labels: 標籤張量
            scores: 置信度分數
            class_names: 類別名稱列表
            threshold: 顯示的置信度閾值
            figsize: 圖像尺寸
        
        Returns:
            matplotlib圖像
        """
        # 轉換圖像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = image.astype(np.uint8)
        elif isinstance(image, np.ndarray) and image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 創建圖
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        
        # 繪製邊界框
        if boxes is not None:
            boxes = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
            labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            scores = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
            
            for i, (box, label) in enumerate(zip(boxes, labels)):
                # 檢查置信度
                if scores is not None and scores[i] < threshold:
                    continue
                
                # 獲取坐標
                x1, y1, x2, y2 = box
                
                # 創建矩形
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # 添加標籤
                label_text = class_names[label] if class_names else f"Class {label}"
                if scores is not None:
                    label_text += f" {scores[i]:.2f}"
                    
                ax.text(x1, y1 - 5, label_text, color='white', fontsize=10, 
                        bbox=dict(facecolor='red', alpha=0.5))
        
        # 移除坐標軸
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(precisions, recalls, class_names, save_path=None):
        """
        繪製精確率-召回率曲線
        
        Args:
            precisions: 每個類別的精確率字典
            recalls: 每個類別的召回率字典
            class_names: 類別名稱列表
            save_path: 保存路徑
        
        Returns:
            matplotlib圖像
        """
        plt.figure(figsize=(10, 8))
        
        # 設置顏色循環
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names) - 1))
        
        # 為每個類別繪製曲線
        for i, class_name in enumerate(class_names[1:], 1):  # 跳過背景類別
            if class_name in precisions and class_name in recalls:
                precision = precisions[class_name]
                recall = recalls[class_name]
                
                # 排序點以確保曲線正確
                sort_idx = np.argsort(recall)
                plt.plot(recall[sort_idx], precision[sort_idx], 
                         label=f"{class_name}", color=colors[i-1], linewidth=2)
        
        # 設置圖表屬性
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # 保存圖表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_training_metrics(metrics_history, save_path=None):
        """
        繪製訓練指標
        
        Args:
            metrics_history: 訓練指標歷史記錄
            save_path: 保存路徑
        
        Returns:
            matplotlib圖像
        """
        # 提取指標
        epochs = list(range(1, len(metrics_history['train_loss']) + 1))
        train_loss = metrics_history['train_loss']
        val_loss = metrics_history['val_loss']
        mAP = metrics_history['mAP']
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 繪製損失
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training and Validation Loss', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 繪製mAP
        ax2.plot(epochs, mAP, 'g-', label='mAP')
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('mAP', fontsize=14)
        ax2.set_title('Mean Average Precision', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存圖表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
        """
        繪製混淆矩陣
        
        Args:
            confusion_matrix: 混淆矩陣
            class_names: 類別名稱列表
            save_path: 保存路徑
        
        Returns:
            matplotlib圖像
        """
        # 創建圖表
        plt.figure(figsize=(12, 10))
        
        # 使用seaborn繪製熱力圖
        ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                          xticklabels=class_names[1:], yticklabels=class_names[1:])
        
        # 設置標籤
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        
        # 調整標籤大小
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12, rotation=45)
        
        # 保存圖表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class MetricsLogger:
    """指標記錄器，用於追蹤和保存訓練指標"""
    
    def __init__(self, config):
        """
        初始化指標記錄器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.log_dir = config["paths"]["logs"]
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 建立記錄檔名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_log_{timestamp}.txt")
        
        # 初始化TensorBoard寫入器
        self.use_tensorboard = config["logging"]["tensorboard"]
        if self.use_tensorboard:
            self.tensorboard_dir = os.path.join(self.log_dir, f"runs_{timestamp}")
            self.writer = SummaryWriter(self.tensorboard_dir)
        
        # 初始化指標歷史記錄
        self.metrics_history = defaultdict(list)
        
        # 設置日誌格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # 紀錄配置
        with open(os.path.join(self.log_dir, f"config_{timestamp}.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"指標記錄器初始化完成，日誌保存至: {self.log_file}")
    
    def update(self, epoch, metrics):
        """
        更新指標
        
        Args:
            epoch: 當前輪次
            metrics: 指標字典
        """
        # 更新歷史記錄
        for k, v in metrics.items():
            # 僅存儲純量值
            if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.metrics_history[k].append(v)
        
        # 寫入日誌
        self._log_metrics(epoch, metrics)
        
        # 更新TensorBoard
        if self.use_tensorboard:
            self._update_tensorboard(epoch, metrics)
    
    def _log_metrics(self, epoch, metrics):
        """
        將指標寫入日誌
        
        Args:
            epoch: 當前輪次
            metrics: 指標字典
        """
        # 生成日誌消息
        log_message = f"Epoch {epoch} - "
        log_message += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items() if isinstance(v, (int, float))])
        
        # 記錄日誌
        logger.info(log_message)
    
    def _update_tensorboard(self, epoch, metrics):
        """
        更新TensorBoard
        
        Args:
            epoch: 當前輪次
            metrics: 指標字典
        """
        for k, v in metrics.items():
            # 只處理純量值
            if isinstance(v, (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.writer.add_scalar(k, v, epoch)
            # 處理類別指標
            elif isinstance(v, dict):
                # 遍歷子指標
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float, np.int32, np.int64, np.float32, np.float64)):
                        self.writer.add_scalar(f"{k}/{sub_k}", sub_v, epoch)
    
    def save_metrics_plot(self, save_path=None):
        """
        保存指標圖表
        
        Args:
            save_path: 保存路徑
        
        Returns:
            圖表路徑
        """
        # 默認保存路徑
        if save_path is None:
            save_dir = os.path.join(self.config["paths"]["output"], "charts")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "training_metrics.png")
        
        # 繪製圖表
        VisualizationUtils.plot_training_metrics(self.metrics_history, save_path)
        
        logger.info(f"訓練指標圖表已保存至: {save_path}")
        return save_path
    
    def close(self):
        """關閉TensorBoard寫入器"""
        if self.use_tensorboard:
            self.writer.close()


class ModelUtils:
    """模型工具類，提供模型優化、權重管理和推理輔助功能"""
    
    @staticmethod
    def calculate_flops(model, input_size=(416, 416), batch_size=1):
        """
        估算模型的FLOPS (浮點運算數)
        
        Args:
            model: 模型實例
            input_size: 輸入尺寸 (高度, 寬度)
            batch_size: 批次大小
        
        Returns:
            估算的FLOPS (G)
        """
        try:
            # 嘗試使用torch.profile (需要PyTorch 1.9+)
            from torch.autograd.profiler import profile
            
            # 創建測試輸入
            input_tensor = torch.randn(batch_size, 3, input_size[0], input_size[1])
            
            # 切換到評估模式
            model.eval()
            
            # 使用profile進行基本剖析
            with torch.no_grad(), profile(use_cuda=torch.cuda.is_available()) as prof:
                model([input_tensor])
            
            # 估算FLOPS (極其粗略)
            total_operations = 0
            for p in prof.key_averages():
                total_operations += p.flops
            
            # 返回GFLOPS
            return total_operations / 1e9
        except:
            # 如果profile不可用，使用參數數量粗略估計
            total_params = sum(p.numel() for p in model.parameters())
            # 假設每個參數平均參與2-4次運算
            ops_per_param = 3
            
            # 估計每次前向計算的運算次數
            estimated_flops = total_params * ops_per_param * input_size[0] * input_size[1] / 224 / 224
            
            # 返回GFLOPS
            return estimated_flops / 1e9
    
    @staticmethod
    def measure_inference_time(model, input_size=(416, 416), iterations=100, warmup=10, device="cuda"):
        """
        測量模型推理時間
        
        Args:
            model: 模型實例
            input_size: 輸入尺寸 (高度, 寬度)
            iterations: 測量迭代次數
            warmup: 預熱迭代次數
            device: 運行設備
        
        Returns:
            平均推理時間 (毫秒)
        """
        # 切換到評估模式
        model.eval()
        
        # 創建測試輸入
        input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # 預熱
        with torch.no_grad():
            for _ in range(warmup):
                _ = model([input_tensor])
        
        # 計時
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model([input_tensor])
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        # 計算平均時間 (毫秒)
        avg_time = (end_time - start_time) * 1000 / iterations
        
        return avg_time
    
    @staticmethod
    def save_model(model, optimizer=None, epoch=None, metrics=None, save_path=None, config=None):
        """
        保存模型
        
        Args:
            model: 模型實例
            optimizer: 優化器實例
            epoch: 當前輪次
            metrics: 評估指標
            save_path: 保存路徑
            config: 配置字典
        
        Returns:
            保存路徑
        """
        # 設置默認保存路徑
        if save_path is None and config is not None:
            weights_dir = config["paths"]["weights"]
            os.makedirs(weights_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(weights_dir, f"model_epoch{epoch}_{timestamp}.pth")
        
        # 創建保存字典
        save_dict = {
            'model_state_dict': model.state_dict()
        }
        
        # 添加優化器狀態
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        # 添加輪次和指標
        if epoch is not None:
            save_dict['epoch'] = epoch
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        # 添加配置
        if config is not None:
            save_dict['config'] = config
        
        # 保存模型
        torch.save(save_dict, save_path)
        logger.info(f"模型已保存至: {save_path}")
        
        return save_path