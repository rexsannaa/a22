#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - PCB缺陷檢測知識蒸餾增強型混合模型訓練程式
整合模型訓練、評估和知識蒸餾功能，提供統一的訓練入口。
主要功能:
1. 資料載入與預處理
2. 教師與學生模型建立
3. 知識蒸餾訓練流程
4. 模型評估與結果視覺化
5. 進度記錄與權重存儲
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml
import argparse
import logging
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
from torchvision.ops import box_iou
import random

# 設置隨機種子以確保可重現性
def set_seed(seed=42):
    """設置隨機種子以確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcb_detector")

# 從各模組導入所需功能
from data.dataset import create_dataloaders, PCBDefectDataset, create_transforms
from models.model import create_model, load_model, get_model_info
from models.distillation import create_distillation_trainer, DistillationLoss, DistillationTrainer
from utils.utils import PCBEvaluator, VisualizationUtils, MetricsLogger, ModelUtils

class PCBTrainer:
    """PCB缺陷檢測訓練器，整合所有訓練相關功能"""
    
    def __init__(self, config_path, resume_path=None):
        """
        初始化訓練器
        
        Args:
            config_path: 配置文件路徑
            resume_path: 恢復訓練的檢查點路徑
        """
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 創建輸出目錄
        for path_name in ['output', 'weights', 'logs']:
            os.makedirs(self.config['paths'][path_name], exist_ok=True)
        
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用設備: {self.device}")
        
        # 載入資料
        self.dataloaders = create_dataloaders(self.config)
        logger.info(f"資料載入完成: 訓練集 {len(self.dataloaders['train'])} 批次，"
                   f"驗證集 {len(self.dataloaders['val'])} 批次，"
                   f"測試集 {len(self.dataloaders['test'])} 批次")
        
        # 建立模型
        self._build_models()
        
        # 設置優化器
        self.optimizer = self._create_optimizer()
        
        # 設置學習率調度器
        self.scheduler = self._create_scheduler()
        
        # 創建蒸餾訓練器
        self.trainer = create_distillation_trainer(
            self.config, 
            self.teacher_model, 
            self.student_model, 
            self.optimizer, 
            self.device
        )
        
        # 創建評估器
        self.evaluator = PCBEvaluator(self.config)
        
        # 創建度量記錄器
        self.metrics_logger = MetricsLogger(self.config)
        
        # 訓練狀態
        self.start_epoch = 0
        self.best_map = 0.0
        self.best_epoch = 0
        self.metrics_history = defaultdict(list)
        
        # 如果提供了恢復路徑，載入檢查點
        if resume_path:
            self._load_checkpoint(resume_path)
        
        logger.info("訓練器初始化完成")
    
    def _build_models(self):
        """建立教師和學生模型"""
        # 創建教師模型
        self.teacher_model = create_model(self.config, "teacher").to(self.device)
        
        # 載入教師模型權重（如果有）
        teacher_weights = os.path.join(self.config["paths"]["weights"], "teacher_model.pth")
        if os.path.exists(teacher_weights):
            self.teacher_model = load_model(self.teacher_model, teacher_weights, self.device)
            logger.info(f"教師模型權重載入完成: {teacher_weights}")
        else:
            logger.info("未找到預訓練的教師模型權重，使用隨機初始化")
        
        # 創建學生模型
        self.student_model = create_model(self.config, "student").to(self.device)
        
        # 顯示模型信息
        teacher_info = get_model_info(self.teacher_model)
        student_info = get_model_info(self.student_model)
        
        logger.info(f"教師模型: {teacher_info['total_params']:,} 參數，"
                   f"{teacher_info['model_size_mb']:.2f} MB")
        logger.info(f"學生模型: {student_info['total_params']:,} 參數，"
                   f"{student_info['model_size_mb']:.2f} MB")
        logger.info(f"模型壓縮率: {(1 - student_info['total_params'] / teacher_info['total_params']) * 100:.2f}%")
    
    def _create_optimizer(self):
        """創建優化器"""
        # 取得配置
        optim_cfg = self.config["training"]
        optim_type = optim_cfg["optimizer"]
        lr = optim_cfg["learning_rate"]
        weight_decay = optim_cfg["weight_decay"]
        momentum = optim_cfg["momentum"]
        
        # 選擇優化器
        if optim_type == "sgd":
            return optim.SGD(
                self.student_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optim_type == "adam":
            return optim.Adam(
                self.student_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optim_type == "adamw":
            return optim.AdamW(
                self.student_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"未知的優化器類型: {optim_type}，使用AdamW")
            return optim.AdamW(
                self.student_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    
    def _create_scheduler(self):
        """創建學習率調度器"""
        # 取得配置
        sched_cfg = self.config["training"]
        sched_type = sched_cfg["lr_scheduler"]
        epochs = sched_cfg["epochs"]
        
        # 選擇調度器
        if sched_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=sched_cfg["scheduler_params"]["min_lr"]
            )
        elif sched_type == "step":
            return StepLR(
                self.optimizer,
                step_size=sched_cfg["scheduler_params"]["step_size"],
                gamma=sched_cfg["scheduler_params"]["gamma"]
            )
        elif sched_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",  # 監控mAP
                factor=sched_cfg["scheduler_params"]["gamma"],
                patience=5,
                min_lr=sched_cfg["scheduler_params"]["min_lr"]
            )
        else:
            logger.warning(f"未知的調度器類型: {sched_type}，使用CosineAnnealingLR")
            return CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-5
            )
    
    def _load_checkpoint(self, checkpoint_path):
        """載入訓練檢查點"""
        # 載入檢查點
        logger.info(f"恢復訓練自檢查點: {checkpoint_path}")
        
        # 使用蒸餾訓練器的方法載入
        self.start_epoch, metrics = self.trainer.load_checkpoint(checkpoint_path)
        
        # 如果有最佳mAP，恢復它
        if metrics and "mAP" in metrics:
            self.best_map = metrics["mAP"]
            logger.info(f"恢復最佳mAP: {self.best_map:.4f}")
    
    def _save_checkpoint(self, epoch, metrics=None, is_best=False):
        """保存訓練檢查點"""
        # 生成檢查點名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(self.config["paths"]["weights"])
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 常規檢查點
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch{epoch}_{timestamp}.pth"
        )
        
        # 保存檢查點
        self.trainer.save_checkpoint(epoch, checkpoint_path, metrics)
        
        # 如果是最佳模型，另存一份
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            self.trainer.save_checkpoint(epoch, best_path, metrics)
            logger.info(f"最佳模型已保存: {best_path}")
    
    def train(self):
        """執行完整訓練流程"""
        # 取得訓練參數
        epochs = self.config["training"]["epochs"]
        start_epoch = self.start_epoch
        
        # 初始早停計數器
        early_stop_patience = self.config["training"]["early_stop_patience"]
        early_stop_counter = 0
        
        # 訓練循環
        logger.info(f"開始訓練，共 {epochs} 輪次")
        for epoch in range(start_epoch, epochs):
            # 訓練一個輪次
            train_metrics = self._train_epoch(epoch)
            
            # 驗證
            val_metrics = self._validate(epoch)
            
            # 更新學習率調度器
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["mAP"])
            else:
                self.scheduler.step()
            
            # 合併指標
            metrics = {**train_metrics, **val_metrics}
            
            # 更新記錄器
            self.metrics_logger.update(epoch, metrics)
            
            # 檢查是否為最佳模型
            is_best = val_metrics["mAP"] > self.best_map
            if is_best:
                self.best_map = val_metrics["mAP"]
                self.best_epoch = epoch
                early_stop_counter = 0
                logger.info(f"新的最佳mAP: {self.best_map:.4f}")
            else:
                early_stop_counter += 1
            
            # 根據保存間隔保存模型
            save_interval = self.config["logging"]["save_interval"]
            if epoch % save_interval == 0 or is_best:
                self._save_checkpoint(epoch, metrics, is_best)
            
            # 檢查早停
            if early_stop_counter >= early_stop_patience:
                logger.info(f"早停: {early_stop_patience} 輪次無改善")
                break
        
        # 訓練結束後的評估
        logger.info(f"訓練完成，最佳mAP: {self.best_map:.4f} 在輪次 {self.best_epoch}")
        
        # 生成訓練指標圖表
        self.metrics_logger.save_metrics_plot()
        
        # 關閉記錄器
        self.metrics_logger.close()
        
        return self.best_map, self.best_epoch
    
    def _train_epoch(self, epoch):
        """訓練一個輪次"""
        # 切換到訓練模式
        self.student_model.train()
        self.teacher_model.eval()  # 教師始終處於評估模式
        
        # 初始化統計
        train_loss = 0.0
        batch_loss = 0.0
        batch_count = 0
        log_interval = self.config["logging"]["log_interval"]
        
        # 取得資料加載器
        train_loader = self.dataloaders["train"]
        
        # 輪次開始時間
        start_time = time.time()
        
        # 使用tqdm顯示進度
        with tqdm(train_loader, desc=f"輪次 {epoch+1}/{self.config['training']['epochs']}") as t:
            for batch_idx, (images, targets, _) in enumerate(t):
                try:
                    # 檢查每個目標是否有邊界框
                    for i, target in enumerate(targets):
                        if target['boxes'].numel() == 0:
                            # 創建一個簡單的虛擬框
                            height, width = images[i].shape[1:3]
                            targets[i]['boxes'] = torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=self.device)
                            targets[i]['labels'] = torch.tensor([0], device=self.device)  # 背景類別
                    
                    # 將數據移到設備上
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # 訓練步驟
                    loss_dict = self.trainer.train_step(images, targets, epoch)
                    
                    # 累積損失
                    batch_loss = loss_dict["total_loss"]
                    train_loss += batch_loss
                    batch_count += 1
                    
                    # 更新進度條
                    t.set_postfix(loss=batch_loss)
                    
                    # 記錄間隔
                    if batch_idx % log_interval == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(f"輪次 {epoch+1} [{batch_idx}/{len(train_loader)}] "
                                f"Loss: {batch_loss:.4f} LR: {lr:.6f}")
                except Exception as e:
                    logger.error(f"批次 {batch_idx} 處理失敗: {str(e)}")
                    continue
        
        # 計算平均損失
        avg_loss = train_loss / max(batch_count, 1)
        
        # 計算輪次時間
        epoch_time = time.time() - start_time
        
        # 輸出輪次統計
        logger.info(f"輪次 {epoch+1} 訓練完成 - "
                f"平均損失: {avg_loss:.4f}, "
                f"時間: {epoch_time:.1f}s")
        
        # 返回訓練指標
        return {
            "train_loss": avg_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time
        }
        
    def _validate(self, epoch):
        """驗證模型性能"""
        # 獲取設定
        eval_interval = self.config["evaluation"]["eval_interval"]
        
        # 如果不是評估間隔，跳過
        if epoch % eval_interval != 0:
            return {"val_loss": 0.0, "mAP": 0.0}
        
        logger.info(f"開始評估 輪次 {epoch+1}")
        
        # 使用蒸餾訓練器的驗證方法
        val_results = self.trainer.validate(self.dataloaders["val"])
        
        # 重置評估器
        self.evaluator.reset()
        
        # 處理預測結果
        self.evaluator.process_batch(val_results["predictions"], val_results["targets"])
        
        # 計算評估指標
        metrics = self.evaluator.compute_metrics()
        
        # 取得mAP和驗證損失
        mAP = metrics["mAP"]
        val_loss = val_results["val_loss"]
        
        # 輸出評估結果
        logger.info(f"輪次 {epoch+1} 評估結果 - "
                   f"Val Loss: {val_loss:.4f}, "
                   f"mAP: {mAP:.4f}")
        
        # 輸出各類別的AP
        for class_name, ap in metrics["AP"].items():
            logger.info(f"AP - {class_name}: {ap:.4f}")
        
        # 可視化樣本
        if self.config["logging"]["visualize_examples"] > 0:
            self._visualize_results(epoch)
        
        # 合併指標
        val_metrics = {
            "val_loss": val_loss,
            "mAP": mAP,
        }
        
        # 添加各類別AP
        for class_name, ap in metrics["AP"].items():
            val_metrics[f"AP_{class_name}"] = ap
        
        return val_metrics
    
    def _visualize_results(self, epoch):
        """可視化檢測結果"""
        # 獲取設定
        num_examples = self.config["logging"]["visualize_examples"]
        
        # 獲取測試資料加載器
        test_loader = self.dataloaders["test"]
        
        # 切換到評估模式
        self.student_model.eval()
        
        # 創建輸出目錄
        vis_dir = os.path.join(self.config["paths"]["output"], "visualizations", f"epoch_{epoch+1}")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 獲取類別名稱
        class_names = ["background"] + self.config["dataset"]["defect_classes"]
        
        # 設置閾值
        threshold = self.config["evaluation"]["conf_threshold"]
        
        # 選擇要可視化的樣本
        with torch.no_grad():
            for i, (images, targets, img_paths) in enumerate(test_loader):
                if i >= num_examples:
                    break
                
                # 將圖像移動到設備
                images = [img.to(self.device) for img in images]
                
                # 獲取預測
                predictions = self.student_model(images)
                
                # 可視化每個圖像
                for j, (image, pred, target, img_path) in enumerate(zip(images, predictions, targets, img_paths)):
                    # 限制樣本數量
                    if i * len(images) + j >= num_examples:
                        break
                    
                    # 獲取文件名
                    filename = os.path.basename(img_path)
                    
                    # 可視化
                    fig = VisualizationUtils.visualize_detection(
                        image, pred["boxes"], pred["labels"], pred["scores"],
                        class_names, threshold
                    )
                    
                    # 保存圖像
                    save_path = os.path.join(vis_dir, f"{filename}_pred.png")
                    fig.savefig(save_path)
                    plt.close(fig)
                    
                    # 可視化真實標註
                    fig = VisualizationUtils.visualize_detection(
                        image, target["boxes"], target["labels"], None,
                        class_names
                    )
                    
                    # 保存圖像
                    save_path = os.path.join(vis_dir, f"{filename}_gt.png")
                    fig.savefig(save_path)
                    plt.close(fig)
        
        logger.info(f"已生成 {min(num_examples, i+1)} 個預測可視化結果")

    def evaluate(self, model_path=None):
        """評估模型性能"""
        # 如果提供了模型路徑，載入權重
        if model_path:
            # 載入檢查點
            self.trainer.load_checkpoint(model_path)
            logger.info(f"已載入模型權重: {model_path}")
        
        # 獲取測試資料加載器
        test_loader = self.dataloaders["test"]
        
        # 切換到評估模式
        self.student_model.eval()
        
        # 重置評估器
        self.evaluator.reset()
        
        # 初始化統計
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 評估
        logger.info("開始評估模型...")
        with torch.no_grad():
            for images, targets, _ in tqdm(test_loader, desc="評估中"):
                # 將數據移到設備上
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向傳播
                loss_dict = self.student_model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_loss += loss.item()
                
                # 獲取預測
                predictions = self.student_model(images)
                
                # 收集預測和目標
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # 處理批次
                self.evaluator.process_batch(predictions, targets)
        
        # 計算評估指標
        metrics = self.evaluator.compute_metrics()
        
        # 計算平均損失
        avg_loss = total_loss / len(test_loader)
        
        # 輸出評估結果
        logger.info(f"評估結果 - Test Loss: {avg_loss:.4f}, mAP: {metrics['mAP']:.4f}")
        
        # 輸出各類別的AP
        for class_name, ap in metrics["AP"].items():
            logger.info(f"AP - {class_name}: {ap:.4f}")
        
        # 輸出整體指標
        logger.info(f"整體精確率: {metrics['overall']['precision']:.4f}")
        logger.info(f"整體召回率: {metrics['overall']['recall']:.4f}")
        logger.info(f"整體F1分數: {metrics['overall']['f1']:.4f}")
        
        # 可視化結果
        self._generate_evaluation_visualizations(all_predictions, all_targets, metrics)
        
        return metrics
    
    def _generate_evaluation_visualizations(self, predictions, targets, metrics):
        """生成評估可視化結果"""
        # 創建輸出目錄
        vis_dir = os.path.join(self.config["paths"]["output"], "evaluation")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 獲取類別名稱
        class_names = ["background"] + self.config["dataset"]["defect_classes"]
        
        # 繪製精確率-召回率曲線
        precisions = {}
        recalls = {}
        
        # 提取每個類別的精確率和召回率
        for class_name in class_names[1:]:  # 跳過背景
            if class_name in metrics["precision"] and class_name in metrics["recall"]:
                precisions[class_name] = np.array([metrics["precision"][class_name]])
                recalls[class_name] = np.array([metrics["recall"][class_name]])
        
        # 繪製曲線
        pr_curve_path = os.path.join(vis_dir, "precision_recall_curve.png")
        VisualizationUtils.plot_precision_recall_curve(
            precisions, recalls, class_names, pr_curve_path
        )
        
        logger.info(f"精確率-召回率曲線已保存至: {pr_curve_path}")
        
        # 可視化樣本預測 (最多10個)
        test_loader = self.dataloaders["test"]
        sample_images, sample_targets, sample_paths = next(iter(test_loader))
        
        # 限制樣本數量
        sample_count = min(10, len(sample_images))
        sample_images = sample_images[:sample_count]
        sample_targets = sample_targets[:sample_count]
        sample_paths = sample_paths[:sample_count]
        
        # 獲取預測
        self.student_model.eval()
        with torch.no_grad():
            sample_images_device = [img.to(self.device) for img in sample_images]
            sample_predictions = self.student_model(sample_images_device)
        
        # 可視化
        for i, (image, pred, target, img_path) in enumerate(zip(sample_images, sample_predictions, sample_targets, sample_paths)):
            # 獲取文件名
            filename = os.path.basename(img_path)
            
            # 預測可視化
            fig = VisualizationUtils.visualize_detection(
                image, pred["boxes"], pred["labels"], pred["scores"],
                class_names, self.config["evaluation"]["conf_threshold"]
            )
            
            # 保存圖像
            save_path = os.path.join(vis_dir, f"{filename}_pred.png")
            fig.savefig(save_path)
            plt.close(fig)
            
            # 目標可視化
            fig = VisualizationUtils.visualize_detection(
                image, target["boxes"], target["labels"], None,
                class_names
            )
            
            # 保存圖像
            save_path = os.path.join(vis_dir, f"{filename}_gt.png")
            fig.savefig(save_path)
            plt.close(fig)
        
        logger.info(f"已生成 {sample_count} 個評估可視化結果")

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="PCB缺陷檢測知識蒸餾訓練程式")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路徑")
    parser.add_argument("--resume", type=str, default=None, help="恢復訓練的檢查點路徑")
    parser.add_argument("--eval", type=str, default=None, help="評估模型的檢查點路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建訓練器
    trainer = PCBTrainer(args.config, args.resume)
    
    # 如果是評估模式
    if args.eval:
        logger.info(f"正在評估模型: {args.eval}")
        metrics = trainer.evaluate(args.eval)
        logger.info(f"評估完成，mAP: {metrics['mAP']:.4f}")
    else:
        # 訓練模型
        logger.info("開始訓練")
        best_map, best_epoch = trainer.train()
        logger.info(f"訓練完成，最佳mAP: {best_map:.4f}，輪次: {best_epoch}")

if __name__ == "__main__":
    main()