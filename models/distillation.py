#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
distillation.py - PCB缺陷檢測知識蒸餾模組
本模組整合了知識蒸餾與注意力機制相關功能，
用於從教師模型向學生模型遷移知識，提高PCB缺陷檢測效能。
主要特點:
1. 知識蒸餾訓練循環：管理教師-學生模型的訓練過程
2. 多層次特徵蒸餾：從淺層到深層的全面知識遷移
3. 整合缺陷特定注意力機制：優化不同類型PCB缺陷的特徵表示
4. 提供評估與推理支援：簡化蒸餾模型的評估與部署
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PCB缺陷類型
DEFECT_CLASSES = {
    'missing_hole': 0,
    'mouse_bite': 1, 
    'spur': 2,
    'spurious_copper': 3,
    'pin_hole': 4,
    'open_circuit': 5
}

class FeatureDistillationLoss(nn.Module):
    """特徵層級知識蒸餾損失函數"""
    
    def __init__(self, adaptation_layers=None):
        """
        初始化特徵蒸餾損失
        
        參數:
            adaptation_layers: 特徵適配層字典，用於調整學生特徵以匹配教師特徵維度
        """
        super(FeatureDistillationLoss, self).__init__()
        self.adaptation_layers = adaptation_layers or {}
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_features, teacher_features):
        """
        計算特徵蒸餾損失
        
        參數:
            student_features: 學生模型的特徵字典 {layer_name: tensor}
            teacher_features: 教師模型的特徵字典 {layer_name: tensor}
            
        回傳:
            total_loss: 特徵蒸餾總損失
            losses_dict: 各層損失明細字典
        """
        total_loss = 0.0
        losses_dict = {}
        
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
                s_feat = F.interpolate(
                    s_feat,
                    size=t_feat.shape[2:],  # 高寬維度
                    mode='bilinear',
                    align_corners=False
                )
            
            # 計算MSE損失
            layer_loss = self.mse_loss(s_feat, t_feat)
            losses_dict[feat_name] = layer_loss.item()
            total_loss += layer_loss
            
        return total_loss, losses_dict

class LogitDistillationLoss(nn.Module):
    """輸出層級知識蒸餾損失函數"""
    
    def __init__(self, temperature=4.0):
        """
        初始化輸出層蒸餾損失
        
        參數:
            temperature: 軟標籤溫度參數
        """
        super(LogitDistillationLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, student_logits, teacher_logits):
        """
        計算輸出層蒸餾損失
        
        參數:
            student_logits: 學生模型的輸出 [batch_size, num_classes]
            teacher_logits: 教師模型的輸出 [batch_size, num_classes]
            
        回傳:
            kl_loss: KL散度損失
        """
        # 應用溫度縮放
        student_logits_T = student_logits / self.temperature
        teacher_logits_T = teacher_logits / self.temperature
        
        # 計算KL散度
        kl_loss = F.kl_div(
            F.log_softmax(student_logits_T, dim=1),
            F.softmax(teacher_logits_T, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss

class AttentionTransferLoss(nn.Module):
    """注意力遷移損失函數"""
    
    def __init__(self):
        """初始化注意力遷移損失"""
        super(AttentionTransferLoss, self).__init__()
        
    def forward(self, student_attention, teacher_attention):
        """
        計算注意力遷移損失
        
        參數:
            student_attention: 學生模型的注意力圖 [batch_size, channels, height, width]
            teacher_attention: 教師模型的注意力圖 [batch_size, channels, height, width]
            
        回傳:
            at_loss: 注意力遷移損失
        """
        # 計算注意力圖的L2範數
        student_norm = self._normalize_attention(student_attention)
        teacher_norm = self._normalize_attention(teacher_attention)
        
        # 計算L2損失
        at_loss = torch.norm(student_norm - teacher_norm, p=2, dim=1).mean()
        
        return at_loss
        
    def _normalize_attention(self, attention):
        """對注意力圖進行L2正規化"""
        attention_power = attention.pow(2)
        attention_spatial = attention_power.sum(1, keepdim=True)
        attention_norm = torch.sqrt(attention_spatial + 1e-8)
        return attention_spatial / attention_norm

class DistillationManager:
    """知識蒸餾訓練管理器"""
    
    def __init__(self, 
                 teacher_model, 
                 student_model, 
                 config):
        """
        初始化知識蒸餾訓練管理器
        
        參數:
            teacher_model: 教師模型
            student_model: 學生模型
            config: 訓練配置字典
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # 蒸餾參數
        self.alpha = config.get('kd_alpha', 0.5)  # 任務損失權重
        self.beta = config.get('kd_beta', 0.5)    # 蒸餾損失權重
        self.gamma = config.get('kd_gamma', 0.5)  # 特徵蒸餾權重
        self.temperature = config.get('kd_temperature', 4.0)  # 溫度參數
        
        # 初始化損失函數
        self.feature_loss = FeatureDistillationLoss()
        self.logit_loss = LogitDistillationLoss(temperature=self.temperature)
        self.attention_loss = AttentionTransferLoss()
        
        # 設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 將模型移至設備
        self.teacher_model = self.teacher_model.to(self.device)
        self.student_model = self.student_model.to(self.device)
        
        # 優化器
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # 混合精度訓練
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # 儲存路徑
        self.output_dir = Path(config.get('output_dir', 'outputs/weights'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, train_loader, epoch):
        """
        執行一個完整的蒸餾訓練周期
        
        參數:
            train_loader: 訓練資料載入器
            epoch: 當前周期數
            
        回傳:
            losses: 損失字典
        """
        # 設定為訓練模式
        self.teacher_model.eval()  # 教師模型保持評估模式
        if hasattr(self.student_model, 'yolo_model'):
            # 使用PyTorch的標準train模式設置方法
            self.student_model.yolo_model.model.eval()  # 先設為eval模式確保一致性
            for module in self.student_model.yolo_model.model.modules():
                if isinstance(module, torch.nn.Module) and hasattr(module, 'training'):
                    module.training = True
        else:
            # 普通PyTorch模型
            self.student_model.train()
        
        # 損失追蹤
        total_loss = 0
        task_loss_sum = 0
        distill_loss_sum = 0
        feature_loss_sum = 0
        
        # 進度條
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.get('epochs', 100)}")
        
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
                    teacher_features = self.teacher_model.get_features()
                
                # 學生模型前向傳播
                student_out = self.student_model(images)
                student_features = self.student_model.get_features()
                
                # 計算任務損失
                task_loss = self._compute_task_loss(student_out, targets)
                
                # 計算蒸餾損失
                distill_loss = self.logit_loss(student_out[0], teacher_out[0])
                
                # 計算特徵蒸餾損失
                feature_loss, _ = self.feature_loss(student_features, teacher_features)
                
                # 總損失
                loss = (self.alpha * task_loss + 
                       self.beta * distill_loss + 
                       self.gamma * feature_loss)
            
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
    
    def validate(self, val_loader):
        """
        在驗證集上評估學生模型
        
        參數:
            val_loader: 驗證資料載入器
            
        回傳:
            metrics: 評估指標字典
        """
        # 設定為評估模式
        self.student_model.eval()
        
        # 損失追蹤
        val_loss = 0
        
        # 預測和目標收集
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
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
        metrics = self._compute_detection_metrics(
            all_pred_boxes, all_pred_labels, all_pred_scores,
            all_gt_boxes, all_gt_labels
        )
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, best_map=0):
        """
        儲存模型檢查點
        
        參數:
            epoch: 當前周期數
            metrics: 評估指標字典
            best_map: 迄今最佳mAP
            
        回傳:
            best_map: 更新的最佳mAP
        """
        # 當前mAP
        current_map = metrics.get('mAP', 0)
        
        # 儲存最新檢查點
        checkpoint_path = self.output_dir / f"student_epoch_{epoch+1}.pt"
        torch.save(self.student_model.state_dict(), checkpoint_path)
        logger.info(f"已儲存模型檢查點：{checkpoint_path}")
        
        # 如果是最佳模型，另存一份
        if current_map > best_map:
            best_map = current_map
            best_model_path = self.output_dir / "student_best.pt"
            torch.save(self.student_model.state_dict(), best_model_path)
            logger.info(f"發現更好的模型 (mAP: {best_map:.4f})，已儲存至：{best_model_path}")
        
        return best_map
    
    def _compute_task_loss(self, outputs, targets):
        """計算檢測任務損失"""
        # 這裡假設使用YOLO模型的原生損失計算
        # 實際實現可能需要根據模型的具體輸出格式調整
        loss_dict = self.student_model.yolo_model.model.loss(outputs, targets)
        return sum(loss_dict.values())
    
    def _collect_detection_results(self, outputs, targets, 
                                 pred_boxes, pred_labels, pred_scores,
                                 gt_boxes, gt_labels):
        """收集檢測結果和真實標籤"""
        # 假設outputs的格式是YOLO輸出格式
        # 實際實現需要根據模型輸出調整
        
        # 處理預測
        for batch_idx, output in enumerate(outputs):
            # 提取該批次的預測
            if isinstance(output, list) and len(output) > 0:
                boxes = output[0]  # [n, 4]
                scores = output[1]  # [n]
                labels = output[2]  # [n]
                
                pred_boxes.append(boxes.cpu().numpy())
                pred_scores.append(scores.cpu().numpy())
                pred_labels.append(labels.cpu().numpy())
            else:
                # 沒有檢測到任何物體
                pred_boxes.append(np.array([]))
                pred_scores.append(np.array([]))
                pred_labels.append(np.array([]))
                
            # 提取該批次的真實標籤
            target = targets[batch_idx]
            gt_boxes.append(target['boxes'].cpu().numpy())
            gt_labels.append(target['labels'].cpu().numpy())
    
    def _compute_detection_metrics(self, pred_boxes, pred_labels, pred_scores,
                                 gt_boxes, gt_labels, iou_threshold=0.5):
        """計算檢測評估指標 (mAP, 精確率, 召回率)"""
        # 這是一個簡化的mAP計算，實際實現可能需要更複雜的計算
        from utils.utils import calculate_map  # 假設有專門的評估指標計算函數
        
        metrics = {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        
        try:
            # 嘗試使用工具函數計算指標
            metrics = calculate_map(
                pred_boxes, pred_labels, pred_scores,
                gt_boxes, gt_labels, 
                iou_threshold=iou_threshold,
                num_classes=len(DEFECT_CLASSES)
            )
        except:
            logger.warning("無法使用utils.utils.calculate_map計算指標")
            # 這裡可以放一個簡化的備用實現
        
        return metrics

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, config):
    """
    使用知識蒸餾訓練學生模型
    
    參數:
        teacher_model: 教師模型
        student_model: 學生模型
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
        config: 訓練配置字典
        
    回傳:
        trained_student: 訓練完成的學生模型
        metrics: 最終評估指標
    """
    # 創建蒸餾管理器
    distill_manager = DistillationManager(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config
    )
    
    # 訓練參數
    epochs = config.get('epochs', 100)
    eval_interval = config.get('eval_interval', 5)
    best_map = 0
    
    logger.info(f"開始知識蒸餾訓練，共 {epochs} 個周期")
    
    # 訓練循環
    for epoch in range(epochs):
        # 訓練一個周期
        train_losses = distill_manager.train_epoch(train_loader, epoch)
        
        # 定期評估
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            metrics = distill_manager.validate(val_loader)
            
            # 記錄結果
            logger.info(f"周期 {epoch+1}/{epochs}")
            logger.info(f"  訓練損失: {train_losses['loss']:.4f}")
            logger.info(f"  驗證損失: {metrics['val_loss']:.4f}")
            logger.info(f"  mAP: {metrics['mAP']:.4f}")
            logger.info(f"  精確率: {metrics['precision']:.4f}")
            logger.info(f"  召回率: {metrics['recall']:.4f}")
            
            # 儲存檢查點
            best_map = distill_manager.save_checkpoint(epoch, metrics, best_map)
    
    logger.info("知識蒸餾訓練完成")
    logger.info(f"最佳mAP: {best_map:.4f}")
    
    # 載入最佳模型
    best_model_path = os.path.join(config.get('output_dir', 'outputs/weights'), 'student_best.pt')
    if os.path.exists(best_model_path):
        student_model.load_state_dict(torch.load(best_model_path))
        logger.info(f"已載入最佳模型：{best_model_path}")
    
    # 最終評估
    final_metrics = distill_manager.validate(val_loader)
    
    return student_model, final_metrics

def get_distillation_losses(config):
    """
    根據配置建立知識蒸餾損失函數
    
    參數:
        config: 配置字典
        
    回傳:
        losses_dict: 損失函數字典
    """
    temperature = config.get('kd_temperature', 4.0)
    
    losses = {
        'feature': FeatureDistillationLoss(),
        'logit': LogitDistillationLoss(temperature=temperature),
        'attention': AttentionTransferLoss()
    }
    
    return losses

if __name__ == "__main__":
    """測試知識蒸餾模組功能"""
    
    # 簡單測試
    from model import get_teacher_model, get_student_model
    
    # 載入配置
    config = {
        'kd_alpha': 0.5,
        'kd_beta': 0.5,
        'kd_gamma': 0.5,
        'kd_temperature': 4.0,
        'learning_rate': 1e-4,
        'epochs': 100,
        'batch_size': 16,
        'output_dir': 'outputs/weights'
    }
    
    # 創建模型
    teacher = get_teacher_model(config)
    student = get_student_model(config)
    
    # 測試特徵蒸餾損失
    feature_loss = FeatureDistillationLoss()
    
    # 假設特徵
    t_features = {
        'stage1': torch.randn(2, 128, 80, 80),
        'stage2': torch.randn(2, 256, 40, 40),
        'stage3': torch.randn(2, 512, 20, 20)
    }
    
    s_features = {
        'stage1': torch.randn(2, 64, 80, 80),
        'stage2': torch.randn(2, 128, 40, 40),
        'stage3': torch.randn(2, 256, 20, 20)
    }
    
    # 計算特徵損失
    loss, loss_dict = feature_loss(s_features, t_features)
    
    logger.info(f"特徵蒸餾損失: {loss.item()}")
    for k, v in loss_dict.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("知識蒸餾模組測試完成")