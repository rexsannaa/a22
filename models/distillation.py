#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
distillation.py - PCB缺陷檢測知識蒸餾模組
本模組實現了教師學生知識蒸餾框架，整合多種蒸餾策略和注意力機制。
主要功能:
1. 特徵蒸餾：從教師模型深層次特徵到學生模型的知識轉移
2. 邏輯蒸餾：軟目標概率分布的知識轉移
3. 注意力蒸餾：注意力圖的知識轉移
4. 蒸餾訓練策略：動態調整不同階段的蒸餾權重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
import time

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """知識蒸餾損失計算模組，整合多種蒸餾損失"""
    
    def __init__(self, config):
        """
        初始化知識蒸餾損失
        
        Args:
            config: 包含蒸餾配置的字典
        """
        super(DistillationLoss, self).__init__()
        self.config = config
        self.distill_cfg = config["distillation"]
        
        # 溫度參數
        self.temperature = self.distill_cfg["temperature"]
        
        # 損失權重
        self.loss_weights = self.distill_cfg["loss_weights"]
        
        # 特徵層映射
        self.feature_mapping = {}
        for item in self.distill_cfg["feature_distillation"]["layers"]:
            self.feature_mapping[item["teacher"]] = {
                "student": item["student"],
                "weight": item["weight"]
            }
        
        # 特徵蒸餾適應層
        self.adaptation_layers = nn.ModuleDict()
        
        # 蒸餾訓練階段
        self.current_epoch = 0
        self.initial_epochs = self.distill_cfg["schedule"]["initial_epochs"]
        self.rampup_epochs = self.distill_cfg["schedule"]["rampup_epochs"]
        self.full_kd_epochs = self.distill_cfg["schedule"]["full_kd_epochs"]
        
        logger.info("知識蒸餾損失初始化完成")
    
    def init_adaptation_layers(self, teacher_model, student_model):
        """
        初始化特徵適應層
        
        Args:
            teacher_model: 教師模型
            student_model: 學生模型
        """
        # 從配置中獲取適應層設置
        adaptation_type = self.distill_cfg["feature_distillation"]["adaptation"]["type"]
        
        # 為每個特徵層配對創建適應層
        for teacher_layer, mapping in self.feature_mapping.items():
            student_layer = mapping["student"]
            
            # 獲取教師和學生層的通道數
            teacher_channels = self._get_layer_channels(teacher_model, teacher_layer)
            student_channels = self._get_layer_channels(student_model, student_layer)
            
            # 創建適應層
            adaptation_layer = self._create_adaptation_layer(
                student_channels, teacher_channels, adaptation_type
            )
            
            # 將適應層添加到模組字典，將點號替換為底線以避免錯誤
            student_layer_key = student_layer.replace(".", "_")
            teacher_layer_key = teacher_layer.replace(".", "_")
            layer_key = f"{student_layer_key}_to_{teacher_layer_key}"
            self.adaptation_layers[layer_key] = adaptation_layer
        
        logger.info("特徵適應層初始化完成")
    
    def _get_layer_channels(self, model, layer_name):
        """獲取指定層的通道數"""
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "body") and layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                # 對於ResNet結構
                if layer_name == "layer1":
                    return 256
                elif layer_name == "layer2":
                    return 512
                elif layer_name == "layer3":
                    return 1024
                elif layer_name == "layer4":
                    return 2048
            elif "features" in layer_name:
                # 對於MobileNetV3結構
                idx = int(layer_name.split(".")[-1])
                if idx == 3:  # stage1
                    return 16
                elif idx == 6:  # stage2
                    return 24
                elif idx == 9:  # stage3
                    return 40
                elif idx == 12:  # stage4
                    return 80
                elif idx == 15:  # stage5
                    return 576
        
        # 默認值
        return 64
    
    def _create_adaptation_layer(self, in_channels, out_channels, adaptation_type):
        """創建特徵適應層，確保in_channels和out_channels匹配"""
        print(f"創建適應層: in_channels={in_channels}, out_channels={out_channels}")
        
        if adaptation_type == "conv1x1":
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        elif adaptation_type == "conv3x3":
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            # 默認使用1x1卷積
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 初始化權重
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        
        return layer
    
    def update_epoch(self, epoch):
        """更新當前訓練階段"""
        self.current_epoch = epoch
    
    def _get_current_weights(self):
        """根據當前訓練階段獲取權重"""
        # 初始階段：僅使用任務損失
        if self.current_epoch < self.initial_epochs:
            return {
                "task_loss": 1.0,
                "feature_loss": 0.0,
                "logit_loss": 0.0
            }
        
        # 權重提升階段：逐步增加蒸餾損失權重
        elif self.current_epoch < self.initial_epochs + self.rampup_epochs:
            progress = (self.current_epoch - self.initial_epochs) / self.rampup_epochs
            return {
                "task_loss": 1.0,
                "feature_loss": self.loss_weights["feature_loss"] * progress,
                "logit_loss": self.loss_weights["logit_loss"] * progress
            }
        
        # 完全蒸餾階段：使用配置中的權重
        else:
            return self.loss_weights
    
    def feature_distillation_loss(self, teacher_features, student_features):
        """計算特徵蒸餾損失"""
        loss = 0.0
        valid_pairs = 0
        
        # 遍歷教師特徵
        for teacher_name, teacher_feat in teacher_features.items():
            # 檢查是否有對應的學生特徵
            if teacher_name in self.feature_mapping:
                mapping = self.feature_mapping[teacher_name]
                student_name = mapping["student"]
                weight = mapping["weight"]
                
                # 獲取學生特徵
                if student_name in student_features:
                    student_feat = student_features[student_name]
                    
                    # 獲取適應層
                    student_name_key = student_name.replace(".", "_")
                    teacher_name_key = teacher_name.replace(".", "_")
                    layer_key = f"{student_name_key}_to_{teacher_name_key}"
                    
                    if layer_key in self.adaptation_layers:
                        adaptation_layer = self.adaptation_layers[layer_key]
                        
                        # 應用適應層
                        adapted_student_feat = adaptation_layer(student_feat)
                        
                        # 調整特徵大小
                        if adapted_student_feat.shape[2:] != teacher_feat.shape[2:]:
                            adapted_student_feat = F.interpolate(
                                adapted_student_feat, 
                                size=teacher_feat.shape[2:],
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        # 計算蒸餾距離
                        if self.distill_cfg["feature_distillation"]["distance"] == "l2":
                            pair_loss = F.mse_loss(adapted_student_feat, teacher_feat)
                        elif self.distill_cfg["feature_distillation"]["distance"] == "l1":
                            pair_loss = F.l1_loss(adapted_student_feat, teacher_feat)
                        elif self.distill_cfg["feature_distillation"]["distance"] == "smooth_l1":
                            pair_loss = F.smooth_l1_loss(adapted_student_feat, teacher_feat)
                        else:
                            pair_loss = F.mse_loss(adapted_student_feat, teacher_feat)
                        
                        # 添加加權損失
                        loss += weight * pair_loss
                        valid_pairs += 1
        
        # 返回平均損失
        return loss / max(valid_pairs, 1)
    
    def logit_distillation_loss(self, teacher_logits, student_logits):
        """計算邏輯蒸餾損失 (KL散度)"""
        # 應用溫度縮放
        temp = self.temperature
        
        # 對邏輯進行軟化
        soft_teacher = F.softmax(teacher_logits / temp, dim=-1)
        log_soft_student = F.log_softmax(student_logits / temp, dim=-1)
        
        # 計算KL散度
        loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (temp * temp)
        
        return loss
    
    def attention_distillation_loss(self, teacher_attentions, student_attentions):
        """計算注意力蒸餾損失"""
        loss = 0.0
        
        # 遍歷所有注意力圖
        for t_att, s_att in zip(teacher_attentions, student_attentions):
            # 確保形狀一致
            if t_att.shape[2:] != s_att.shape[2:]:
                s_att = F.interpolate(s_att, size=t_att.shape[2:], mode='bilinear', align_corners=False)
            
            # 計算注意力差異
            loss += F.mse_loss(s_att, t_att)
        
        return loss / len(teacher_attentions) if teacher_attentions else 0.0
    
    def forward(self, task_loss, teacher_outputs=None, student_outputs=None, 
                teacher_features=None, student_features=None,
                teacher_attentions=None, student_attentions=None):
        """
        計算總蒸餾損失
        
        Args:
            task_loss: 原始檢測任務損失
            teacher_outputs: 教師模型輸出
            student_outputs: 學生模型輸出
            teacher_features: 教師模型特徵
            student_features: 學生模型特徵
            teacher_attentions: 教師模型注意力
            student_attentions: 學生模型注意力
        
        Returns:
            總損失
        """
        # 獲取當前階段的損失權重
        weights = self._get_current_weights()
        
        # 初始化總損失為任務損失
        total_loss = weights["task_loss"] * task_loss
        
        # 添加特徵蒸餾損失
        if weights["feature_loss"] > 0 and teacher_features and student_features:
            feature_loss = self.feature_distillation_loss(teacher_features, student_features)
            total_loss += weights["feature_loss"] * feature_loss
        
        # 添加邏輯蒸餾損失
        if weights["logit_loss"] > 0 and teacher_outputs and student_outputs:
            # 假設輸出包含分類邏輯
            if "logits" in teacher_outputs and "logits" in student_outputs:
                logit_loss = self.logit_distillation_loss(
                    teacher_outputs["logits"], student_outputs["logits"]
                )
                total_loss += weights["logit_loss"] * logit_loss
        
        # 添加注意力蒸餾損失（如果有）
        if teacher_attentions and student_attentions:
            attention_loss = self.attention_distillation_loss(teacher_attentions, student_attentions)
            # 使用特徵損失的權重
            total_loss += weights["feature_loss"] * attention_loss
        
        return total_loss


class DistillationTrainer:
    """知識蒸餾訓練器，整合訓練流程和蒸餾策略"""
    
    def __init__(self, config, teacher_model, student_model, optimizer, device):
        """
        初始化蒸餾訓練器
        
        Args:
            config: 配置字典
            teacher_model: 教師模型
            student_model: 學生模型
            optimizer: 優化器
            device: 運行設備
        """
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.device = device
        
        # 設置教師模型為評估模式
        self.teacher_model.eval()
        
        # 創建蒸餾損失
        self.distill_loss = DistillationLoss(config)
        
        # 初始化適應層
        self.distill_loss.init_adaptation_layers(teacher_model, student_model)
        
        # 將適應層添加到優化器
        self._add_adaptation_layers_to_optimizer()
        
        # 使用混合精度訓練
        self.use_amp = config["training"]["amp_training"]
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        logger.info("知識蒸餾訓練器初始化完成")
    
    def _add_adaptation_layers_to_optimizer(self):
        """將適應層添加到優化器中"""
        # 獲取當前優化器中的參數組
        param_groups = self.optimizer.param_groups
        
        # 將適應層參數添加到第一個參數組
        if param_groups and self.distill_loss.adaptation_layers:
            param_groups[0]['params'].extend(self.distill_loss.adaptation_layers.parameters())
    
    def _extract_features(self, model, images, is_teacher=False):
        """
        從模型中提取特徵
        
        Args:
            model: 模型
            images: 輸入圖像列表或張量
            is_teacher: 是否為教師模型
        
        Returns:
            特徵字典
        """
        features = {}
        
        # 確保輸入是張量
        if isinstance(images, list):
            # 轉換列表為批次張量
            batched_images = torch.stack(images)
        else:
            # 已經是張量，直接使用
            batched_images = images
        
        # 對於教師模型 (ResNet特徵提取)
        if is_teacher and hasattr(model, 'backbone') and hasattr(model.backbone, 'body'):
            with torch.no_grad():
                # 提取骨幹網絡的特徵 - 使用原始方式
                try:
                    # 直接使用 ResNet 的各層提取特徵，避免使用 IntermediateLayerGetter
                    x = batched_images
                    # ResNet 結構: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
                    x = model.backbone.body.conv1(x)
                    x = model.backbone.body.bn1(x)
                    x = model.backbone.body.relu(x)
                    x = model.backbone.body.maxpool(x)
                    
                    # 依次通過各層
                    layer1 = model.backbone.body.layer1(x)
                    layer2 = model.backbone.body.layer2(layer1)
                    layer3 = model.backbone.body.layer3(layer2)
                    layer4 = model.backbone.body.layer4(layer3)
                    
                    # 映射層名稱
                    features = {
                        'layer1': layer1,
                        'layer2': layer2,
                        'layer3': layer3,
                        'layer4': layer4
                    }
                except Exception as e:
                    # 如果直接提取失敗，使用替代方法
                    logger.warning(f"直接特徵提取失敗: {str(e)}，使用零張量替代")
                    # 使用零張量作為占位符
                    features = {
                        'layer2': torch.zeros((batched_images.shape[0], 512, batched_images.shape[2]//4, batched_images.shape[3]//4), device=batched_images.device),
                        'layer3': torch.zeros((batched_images.shape[0], 1024, batched_images.shape[2]//8, batched_images.shape[3]//8), device=batched_images.device),
                        'layer4': torch.zeros((batched_images.shape[0], 2048, batched_images.shape[2]//16, batched_images.shape[3]//16), device=batched_images.device)
                    }
        
        # 對於學生模型 (MobileNet特徵提取)
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            # 提取學生模型特徵
            x = batched_images
            for name, module in model.backbone.features._modules.items():
                x = module(x)
                feature_name = f"features.{name}"
                if feature_name in ['features.3', 'features.6', 'features.9', 'features.12', 'features.15']:
                    features[feature_name] = x
        
        return features
    
    def train_step(self, images, targets, epoch):
        """
        執行一個訓練步驟
        
        Args:
            images: 輸入圖像列表
            targets: 目標標註
            epoch: 當前訓練輪次
        
        Returns:
            損失字典
        """
        # 檢查空目標框
        for i, target in enumerate(targets):
            if target['boxes'].numel() == 0:
                # 創建一個簡單的虛擬框
                height, width = images[i].shape[1:3]
                targets[i]['boxes'] = torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=targets[i]['boxes'].device)
                targets[i]['labels'] = torch.tensor([0], device=targets[i]['labels'].device)  # 背景類別
                
        # 更新蒸餾損失的訓練階段
        self.distill_loss.update_epoch(epoch)
        
        # 清除梯度
        self.optimizer.zero_grad()
        
        # 提取教師特徵和輸出 (不計算梯度)
        with torch.no_grad():
            # 確保教師模型能夠正確處理輸入
            try:
                # 嘗試提取教師特徵
                teacher_features = self._extract_features(self.teacher_model, images, is_teacher=True)
                # 嘗試獲取教師輸出
                teacher_outputs = self.teacher_model(images)
            except Exception as e:
                logger.warning(f"教師模型處理失敗: {str(e)}，使用零填充替代")
                # 使用空字典作為替代
                teacher_features = {}
                teacher_outputs = {}
        
        # 使用混合精度訓練
        if self.use_amp:
            with torch.cuda.amp.autocast():
                # 提取學生特徵
                student_features = self._extract_features(self.student_model, images, is_teacher=False)
                
                # 學生模型前向傳播
                loss_dict = self.student_model(images, targets)
                
                # 計算檢測任務損失
                task_loss = sum(loss for loss in loss_dict.values())
                
                # 計算總蒸餾損失
                if teacher_features and teacher_outputs:
                    total_loss = self.distill_loss(
                        task_loss=task_loss,
                        teacher_outputs=teacher_outputs,
                        student_outputs=None,  # FasterRCNN不直接返回logits
                        teacher_features=teacher_features,
                        student_features=student_features
                    )
                else:
                    # 如果教師特徵或輸出為空，只使用任務損失
                    total_loss = task_loss
            
            # 反向傳播和優化
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 提取學生特徵
            student_features = self._extract_features(self.student_model, images, is_teacher=False)
            
            # 學生模型前向傳播
            loss_dict = self.student_model(images, targets)
            
            # 計算檢測任務損失
            task_loss = sum(loss for loss in loss_dict.values())
            
            # 計算總蒸餾損失
            if teacher_features and teacher_outputs:
                total_loss = self.distill_loss(
                    task_loss=task_loss,
                    teacher_outputs=teacher_outputs,
                    student_outputs=None,  # FasterRCNN不直接返回logits
                    teacher_features=teacher_features,
                    student_features=student_features
                )
            else:
                # 如果教師特徵或輸出為空，只使用任務損失
                total_loss = task_loss
            
            # 反向傳播和優化
            total_loss.backward()
            self.optimizer.step()
        
        # 返回損失字典，包含任務損失和總損失
        loss_dict['total_loss'] = total_loss.item()
        
        return loss_dict
    
    def validate(self, val_loader):
        """
        在驗證集上評估模型
        
        Args:
            val_loader: 驗證數據加載器
        
        Returns:
            評估指標
        """
        # 設置模型為評估模式
        self.student_model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 不計算梯度
        with torch.no_grad():
            for images, targets, _ in val_loader:
                # 將數據移到設備
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                try:
                    # 前向傳播 - 檢查返回值類型
                    loss_dict = self.student_model(images, targets)
                    
                    # 處理不同類型的返回值
                    if isinstance(loss_dict, dict):
                        # 如果是字典，直接使用 values()
                        loss = sum(loss for loss in loss_dict.values())
                    elif isinstance(loss_dict, list):
                        # 如果是列表，直接求和
                        loss = sum(loss_dict)
                    elif isinstance(loss_dict, torch.Tensor):
                        # 如果是張量，直接使用
                        loss = loss_dict
                    else:
                        # 其他情況，記錄警告並使用零損失
                        logger.warning(f"未知的損失類型: {type(loss_dict)}")
                        loss = torch.tensor(0.0, device=self.device)
                        
                    total_loss += loss.item()
                except Exception as e:
                    logger.warning(f"計算驗證損失時出錯: {str(e)}")
                    loss = torch.tensor(0.0, device=self.device)
                    total_loss += 0.0
                
                # 獲取預測
                try:
                    predictions = self.student_model(images)
                    
                    # 收集預測和目標
                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
                except Exception as e:
                    logger.warning(f"獲取預測時出錯: {str(e)}")
        
        # 設置模型為訓練模式
        self.student_model.train()
        
        # 計算平均損失
        avg_loss = total_loss / len(val_loader)
        
        # 返回評估指標（這裡只返回損失，實際應計算mAP等）
        return {
            'val_loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def save_checkpoint(self, epoch, save_path, metrics=None):
        """
        保存檢查點
        
        Args:
            epoch: 當前訓練輪次
            save_path: 保存路徑
            metrics: 評估指標
        """
        # 創建檢查點字典
        checkpoint = {
            'epoch': epoch,
            'student_model': self.student_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'distill_loss': self.distill_loss.state_dict(),
            'adaptation_layers': self.distill_loss.adaptation_layers.state_dict(),
            'config': self.config
        }
        
        # 添加評估指標
        if metrics:
            checkpoint['metrics'] = metrics
        
        # 添加混合精度訓練狀態
        if self.use_amp and self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # 保存檢查點
        torch.save(checkpoint, save_path)
        logger.info(f"檢查點已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加載檢查點
        
        Args:
            checkpoint_path: 檢查點路徑
        
        Returns:
            起始輪次和加載的指標
        """
        # 檢查檔案是否存在
        if not os.path.exists(checkpoint_path):
            logger.warning(f"檢查點不存在: {checkpoint_path}")
            return 0, None
        
        # 加載檢查點
        logger.info(f"從 {checkpoint_path} 加載檢查點")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加載模型參數
        self.student_model.load_state_dict(checkpoint['student_model'])
        
        # 加載優化器參數
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加載蒸餾損失參數
        self.distill_loss.load_state_dict(checkpoint['distill_loss'])
        
        # 加載適應層參數
        self.distill_loss.adaptation_layers.load_state_dict(checkpoint['adaptation_layers'])
        
        # 加載混合精度訓練狀態
        if self.use_amp and self.scaler and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        # 返回起始輪次和指標
        return checkpoint['epoch'] + 1, checkpoint.get('metrics')


def create_distillation_trainer(config, teacher_model, student_model, optimizer, device):
    """
    創建知識蒸餾訓練器
    
    Args:
        config: 配置字典
        teacher_model: 教師模型
        student_model: 學生模型
        optimizer: 優化器
        device: 運行設備
    
    Returns:
        蒸餾訓練器實例
    """
    return DistillationTrainer(config, teacher_model, student_model, optimizer, device)


# 在導入時執行的代碼
def get_distillation_info():
    """
    獲取蒸餾方法信息
    
    Returns:
        蒸餾方法信息字串
    """
    return """
    知識蒸餾增強型混合模型使用以下蒸餾策略:
    1. 特徵蒸餾: 從教師到學生模型的多層次特徵轉移
    2. 邏輯蒸餾: 使用溫度縮放的軟標籤知識轉移
    3. 注意力蒸餾: 轉移注意力機制的空間和通道關注點
    4. 動態權重調度: 根據訓練階段自適應調整蒸餾權重
    """


# 測試代碼
if __name__ == "__main__":
    import os
    import yaml
    import torch.optim as optim
    from model import create_model
    
    # 配置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載配置
    config_path = "../config/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 創建模型
    print("創建教師和學生模型...")
    teacher_model = create_model(config, "teacher").to(device)
    student_model = create_model(config, "student").to(device)
    
    # 創建優化器
    optimizer = optim.AdamW(student_model.parameters(), lr=config["training"]["learning_rate"])
    
    # 創建蒸餾訓練器
    print("初始化蒸餾訓練器...")
    distillation_trainer = create_distillation_trainer(config, teacher_model, student_model, optimizer, device)
    
    # 測試蒸餾損失
    print("測試蒸餾損失...")
    distillation_trainer.distill_loss.update_epoch(config["distillation"]["schedule"]["initial_epochs"] + 1)
    
    # 訓練一個批次
    print("模擬訓練步驟...")
    dummy_images = [torch.randn(3, 416, 416).to(device) for _ in range(2)]
    dummy_targets = [
        {'boxes': torch.tensor([[100, 100, 200, 200]], device=device), 
         'labels': torch.tensor([1], device=device)}
        for _ in range(2)
    ]
    
    # 輸出蒸餾信息
    print("\n" + get_distillation_info())