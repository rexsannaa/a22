#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
eval.py - PCB缺陷檢測知識蒸餾增強型混合模型評估程式
提供模型評估、性能分析與結果可視化功能。
主要功能:
1. 模型載入與推論
2. 評估指標計算
3. 檢測結果可視化
4. 詳細性能分析報告生成
"""

import os
import torch
import yaml
import argparse
import logging
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
from torchvision.ops import box_iou

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcb_evaluator")

# 導入必要模組
from data.dataset import create_dataloaders
from models.model import create_model, load_model
from utils.utils import PCBEvaluator, VisualizationUtils

class PCBEvaluation:
    """PCB缺陷檢測評估類，整合評估相關功能"""
    
    def __init__(self, config_path):
        """
        初始化評估器
        
        Args:
            config_path: 配置文件路徑
        """
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 建立輸出目錄
        eval_dir = os.path.join(self.config['paths']['output'], 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用設備: {self.device}")
        
        # 載入資料
        self.dataloaders = create_dataloaders(self.config)
        logger.info(f"資料載入完成: 測試集 {len(self.dataloaders['test'])} 批次")
        
        # 創建模型
        self.model = create_model(self.config, "student").to(self.device)
        
        # 創建評估器
        self.evaluator = PCBEvaluator(self.config)
        
        # 設置類別名稱
        self.class_names = ["background"] + self.config["dataset"]["defect_classes"]
    
    def load_model(self, model_path):
        """
        載入模型權重
        
        Args:
            model_path: 模型權重路徑
        """
        self.model = load_model(self.model, model_path, self.device)
        logger.info(f"模型權重載入完成: {model_path}")
    
    def evaluate(self, model_path, output_dir=None):
        """
        評估模型性能
        
        Args:
            model_path: 模型權重路徑
            output_dir: 評估結果輸出目錄
        
        Returns:
            評估指標字典
        """
        # 載入模型
        self.load_model(model_path)
        
        # 設置輸出目錄
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.config['paths']['output'], f'evaluation_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取測試資料加載器
        test_loader = self.dataloaders['test']
        
        # 切換到評估模式
        self.model.eval()
        
        # 重置評估器
        self.evaluator.reset()
        
        # 初始化統計
        total_loss = 0.0
        inference_times = []
        all_predictions = []
        all_targets = []
        all_image_paths = []
        
        # 評估
        logger.info("開始評估模型...")
        with torch.no_grad():
            for images, targets, image_paths in tqdm(test_loader, desc="評估中"):
                # 將數據移到設備上
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 計算損失
                try:
                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    total_loss += loss.item()
                except:
                    logger.warning("計算損失時出錯，可能是模型不支援損失計算")
                
                # 測量推理時間
                start_time = time.time()
                predictions = self.model(images)
                end_time = time.time()
                batch_time = (end_time - start_time) * 1000  # 毫秒
                inference_times.append(batch_time / len(images))  # 每張圖像的平均時間
                
                # 收集預測和目標
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_image_paths.extend(image_paths)
                
                # 處理批次
                self.evaluator.process_batch(predictions, targets)
        
        # 計算評估指標
        metrics = self.evaluator.compute_metrics()
        
        # 計算平均損失和推理時間
        avg_loss = total_loss / len(test_loader) if 'total_loss' in locals() else float('nan')
        avg_inference_time = np.mean(inference_times)
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        # 添加額外指標
        metrics['avg_inference_time_ms'] = avg_inference_time
        metrics['fps'] = fps
        metrics['test_loss'] = avg_loss
        
        # 輸出評估結果
        self._print_evaluation_results(metrics)
        
        # 生成評估報告
        self._generate_evaluation_report(metrics, all_predictions, all_targets, all_image_paths, output_dir)
        
        return metrics
    
    def _print_evaluation_results(self, metrics):
        """
        輸出評估結果
        
        Args:
            metrics: 評估指標字典
        """
        logger.info("=" * 50)
        logger.info("模型評估結果")
        logger.info("=" * 50)
        
        # 輸出mAP
        logger.info(f"mAP: {metrics['mAP']:.4f}")
        
        # 輸出各類別的AP
        logger.info("\n各類別的平均精度 (AP):")
        for class_name, ap in metrics['AP'].items():
            logger.info(f"  - {class_name}: {ap:.4f}")
        
        # 輸出整體指標
        logger.info("\n整體性能指標:")
        logger.info(f"  - 精確率: {metrics['overall']['precision']:.4f}")
        logger.info(f"  - 召回率: {metrics['overall']['recall']:.4f}")
        logger.info(f"  - F1分數: {metrics['overall']['f1']:.4f}")
        
        # 輸出推理效能
        logger.info("\n推理效能:")
        logger.info(f"  - 平均推理時間: {metrics['avg_inference_time_ms']:.2f} ms/張")
        logger.info(f"  - FPS: {metrics['fps']:.2f}")
        
        # 輸出測試集損失
        if not np.isnan(metrics['test_loss']):
            logger.info(f"  - 測試集損失: {metrics['test_loss']:.4f}")
        
        logger.info("=" * 50)
    
    def _generate_evaluation_report(self, metrics, predictions, targets, image_paths, output_dir):
        """
        生成評估報告
        
        Args:
            metrics: 評估指標字典
            predictions: 預測結果列表
            targets: 目標標註列表
            image_paths: 圖像路徑列表
            output_dir: 輸出目錄
        """
        # 1. 保存指標結果為文本
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("PCB缺陷檢測評估報告\n")
            f.write("=" * 50 + "\n")
            f.write(f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"mAP: {metrics['mAP']:.4f}\n\n")
            
            f.write("各類別的平均精度 (AP):\n")
            for class_name, ap in metrics['AP'].items():
                f.write(f"  - {class_name}: {ap:.4f}\n")
            
            f.write("\n整體性能指標:\n")
            f.write(f"  - 精確率: {metrics['overall']['precision']:.4f}\n")
            f.write(f"  - 召回率: {metrics['overall']['recall']:.4f}\n")
            f.write(f"  - F1分數: {metrics['overall']['f1']:.4f}\n")
            
            f.write("\n推理效能:\n")
            f.write(f"  - 平均推理時間: {metrics['avg_inference_time_ms']:.2f} ms/張\n")
            f.write(f"  - FPS: {metrics['fps']:.2f}\n")
            
            if not np.isnan(metrics['test_loss']):
                f.write(f"  - 測試集損失: {metrics['test_loss']:.4f}\n")
        
        logger.info(f"評估指標已保存至: {metrics_path}")
        
        # 2. 繪製精確率-召回率曲線
        self._plot_precision_recall_curve(metrics, output_dir)
        
        # 3. 可視化檢測結果
        self._visualize_detection_results(predictions, targets, image_paths, output_dir)
    
    def _plot_precision_recall_curve(self, metrics, output_dir):
        """
        繪製精確率-召回率曲線
        
        Args:
            metrics: 評估指標字典
            output_dir: 輸出目錄
        """
        # 提取精確率和召回率
        precisions = {}
        recalls = {}
        
        # 轉換格式以適應繪圖函數
        for class_name in self.class_names[1:]:  # 跳過背景
            if class_name in metrics['precision'] and class_name in metrics['recall']:
                precisions[class_name] = np.array([metrics['precision'][class_name]])
                recalls[class_name] = np.array([metrics['recall'][class_name]])
        
        # 繪製曲線
        pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
        VisualizationUtils.plot_precision_recall_curve(
            precisions, recalls, self.class_names, pr_curve_path
        )
        
        logger.info(f"精確率-召回率曲線已保存至: {pr_curve_path}")
    
    def _visualize_detection_results(self, predictions, targets, image_paths, output_dir):
        """
        可視化檢測結果
        
        Args:
            predictions: 預測結果列表
            targets: 目標標註列表
            image_paths: 圖像路徑列表
            output_dir: 輸出目錄
        """
        # 創建可視化目錄
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 設置閾值
        threshold = self.config["evaluation"]["conf_threshold"]
        
        # 隨機選擇樣本進行可視化（最多10個）
        indices = np.random.choice(len(predictions), min(10, len(predictions)), replace=False)
        
        # 載入原始圖像
        for i in indices:
            # 獲取圖像路徑和檔名
            img_path = image_paths[i]
            filename = os.path.basename(img_path)
            
            # 載入圖像
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 獲取預測和目標
            pred = predictions[i]
            target = targets[i]
            
            # 可視化預測結果
            pred_boxes = pred["boxes"].cpu()
            pred_labels = pred["labels"].cpu()
            pred_scores = pred["scores"].cpu()
            
            # 篩選高置信度預測
            keep = pred_scores > threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            # 繪製預測結果
            pred_img = image.copy()
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                x1, y1, x2, y2 = box.int().numpy()
                class_name = self.class_names[label]
                color = (0, 255, 0)  # 綠色
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(pred_img, f"{class_name} {score:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存預測結果
            pred_save_path = os.path.join(vis_dir, f"{filename}_pred.jpg")
            cv2.imwrite(pred_save_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            
            # 繪製目標標註
            gt_img = image.copy()
            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()
            
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.int().numpy()
                class_name = self.class_names[label]
                color = (255, 0, 0)  # 紅色
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(gt_img, class_name, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存目標標註
            gt_save_path = os.path.join(vis_dir, f"{filename}_gt.jpg")
            cv2.imwrite(gt_save_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        
        logger.info(f"已生成 {len(indices)} 個檢測結果可視化")
    
    def evaluate_speed(self, model_path, batch_sizes=[1, 2, 4, 8], iterations=100):
        """
        評估模型在不同批次大小下的推理速度
        
        Args:
            model_path: 模型權重路徑
            batch_sizes: 要測試的批次大小列表
            iterations: 每個批次大小的測試迭代次數
        
        Returns:
            速度評估結果字典
        """
        # 載入模型
        self.load_model(model_path)
        
        # 切換到評估模式
        self.model.eval()
        
        # 初始化結果字典
        speed_results = {
            'batch_size': batch_sizes,
            'time_per_batch': [],
            'time_per_image': [],
            'fps': []
        }
        
        logger.info("開始速度評估...")
        for batch_size in batch_sizes:
            logger.info(f"測試批次大小: {batch_size}")
            
            # 創建測試輸入
            dummy_input = [torch.randn(batch_size, 3, 416, 416).to(self.device)]
            
            # 預熱
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # 計時
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            end_time = time.time()
            
            # 計算時間
            total_time = end_time - start_time
            time_per_batch = total_time * 1000 / iterations  # 毫秒
            time_per_image = time_per_batch / batch_size
            fps = 1000 / time_per_image
            
            # 添加到結果
            speed_results['time_per_batch'].append(time_per_batch)
            speed_results['time_per_image'].append(time_per_image)
            speed_results['fps'].append(fps)
            
            logger.info(f"  批次時間: {time_per_batch:.2f} ms")
            logger.info(f"  每張圖像時間: {time_per_image:.2f} ms")
            logger.info(f"  FPS: {fps:.2f}")
        
        return speed_results
    
    def plot_speed_results(self, speed_results, output_path=None):
        """
        繪製速度評估結果
        
        Args:
            speed_results: 速度評估結果字典
            output_path: 輸出路徑
        
        Returns:
            matplotlib圖像
        """
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 繪製每張圖像時間
        ax1.plot(speed_results['batch_size'], speed_results['time_per_image'], 'b-o')
        ax1.set_xlabel('批次大小')
        ax1.set_ylabel('每張圖像時間 (ms)')
        ax1.set_title('批次大小 vs 推理時間')
        ax1.grid(True)
        
        # 繪製FPS
        ax2.plot(speed_results['batch_size'], speed_results['fps'], 'r-o')
        ax2.set_xlabel('批次大小')
        ax2.set_ylabel('FPS')
        ax2.set_title('批次大小 vs FPS')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存圖表
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"速度評估圖表已保存至: {output_path}")
        
        return fig


def main():
    """主函數"""
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="PCB缺陷檢測模型評估程式")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路徑")
    parser.add_argument("--model", type=str, required=True, help="模型權重路徑")
    parser.add_argument("--output", type=str, default=None, help="評估結果輸出目錄")
    parser.add_argument("--speed", action="store_true", help="是否執行速度評估")
    args = parser.parse_args()
    
    # 創建評估器
    evaluator = PCBEvaluation(args.config)
    
    # 執行評估
    metrics = evaluator.evaluate(args.model, args.output)
    
    # 如果需要，執行速度評估
    if args.speed:
        logger.info("開始執行速度評估...")
        speed_results = evaluator.evaluate_speed(args.model)
        
        # 繪製並保存速度評估結果
        if args.output:
            speed_plot_path = os.path.join(args.output, "speed_evaluation.png")
            evaluator.plot_speed_results(speed_results, speed_plot_path)
        else:
            evaluator.plot_speed_results(speed_results)

if __name__ == "__main__":
    main()