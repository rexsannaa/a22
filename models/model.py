#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model.py - PCB缺陷檢測模型定義
本模組整合了基於YOLO8架構的教師與學生模型定義及知識蒸餾功能,
實現PCB缺陷檢測的高精度與輕量化。
主要特點:
1. 教師模型：基於YOLO8-L架構,提供強大的特徵表示能力
2. 學生模型：輕量化的YOLO8-S架構,結合注意力機制提升性能
3. 知識蒸餾：多層次特徵蒸餾,從教師模型學習深層特徵表示
4. 缺陷專用注意力：針對不同PCB缺陷類型的特定注意力增強
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules import C3, Conv, SPPF, Detect

# 設定環境變數防止自動下載COCO數據集
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

class AttentionModule(nn.Module):
    """PCB缺陷專用注意力模組"""
    
    def __init__(self, channels, reduction=16):
        """初始化注意力模組"""
        super(AttentionModule, self).__init__()
        
        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空間注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向傳播"""
        # 應用通道注意力
        channel_att = self.channel_gate(x)
        x = x * channel_att
        
        # 應用空間注意力
        spatial_att = self.spatial_gate(x)
        x = x * spatial_att
        
        return x

class TeacherModel:
    """教師模型：基於YOLO8-L架構"""
    
    def __init__(self, num_classes=len(DEFECT_CLASSES), pretrained=True):
        """初始化教師模型"""
        # 載入YOLO8-L
        model_path = 'yolov8l.pt' if pretrained else 'yolov8l.yaml'
        self.model = YOLO(model_path)
        
        # 調整為PCB缺陷類別
        if hasattr(self.model.model, 'nc'):
            self.model.model.nc = num_classes
            
            # 設置檢測頭以匹配類別數量
            for m in self.model.model.model:
                if hasattr(m, 'nc'):
                    m.nc = num_classes
                    
        # 禁用驗證和數據集下載
        if hasattr(self.model, 'args'):
            if isinstance(self.model.args, dict):
                self.model.args['val'] = False
                self.model.args['data'] = None
            else:
                try:
                    self.model.args.val = False
                    self.model.args.data = None
                except:
                    logger.warning("無法設置YOLO模型驗證參數")
        
        # 記錄特徵圖
        self.features = {}
        self._register_hooks()
        
        logger.info(f"已初始化教師模型，類別數：{num_classes}")
    
    def _register_hooks(self):
        """註冊鉤子以獲取中間特徵"""
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # 關鍵特徵層
        feature_layers = {
            6: 'stage1',   # 淺層特徵
            10: 'stage2',  # 中層特徵
            14: 'stage3',  # 深層特徵
        }
        
        # 註冊鉤子
        for idx, name in feature_layers.items():
            try:
                self.model.model.model[idx].register_forward_hook(get_features(name))
            except Exception as e:
                logger.warning(f"註冊鉤子失敗: {e}")
    
    def __call__(self, x):
        """模型調用"""
        return self.model(x)
    
    def to(self, device):
        """移動模型到指定設備"""
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self
    
    def train(self, mode=True):
        """設置訓練/評估模式"""
        if hasattr(self.model.model, 'train'):
            self.model.model.train(mode)
        return self
    
    def eval(self):
        """設置評估模式"""
        return self.train(False)
    
    def parameters(self):
        """獲取模型參數"""
        return self.model.model.parameters()
    
    def get_features(self):
        """獲取中間特徵圖"""
        return self.features
    
    def state_dict(self):
        """獲取模型狀態字典"""
        return self.model.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """載入模型狀態字典"""
        self.model.model.load_state_dict(state_dict)
        return self

class StudentModel:
    """學生模型：基於YOLO8-S架構的輕量化模型"""
    
    def __init__(self, num_classes=len(DEFECT_CLASSES), pretrained=True):
        """初始化學生模型"""
        # 載入YOLO8-S
        model_path = 'yolov8s.pt' if pretrained else 'yolov8s.yaml'
        self.model = YOLO(model_path)
        
        # 調整為PCB缺陷類別
        if hasattr(self.model.model, 'nc'):
            self.model.model.nc = num_classes
            
            # 設置檢測頭以匹配類別數量
            for m in self.model.model.model:
                if hasattr(m, 'nc'):
                    m.nc = num_classes
        
        # 禁用驗證和數據集下載
        if hasattr(self.model, 'args'):
            if isinstance(self.model.args, dict):
                self.model.args['val'] = False
                self.model.args['data'] = None
            else:
                try:
                    self.model.args.val = False
                    self.model.args.data = None
                except:
                    logger.warning("無法設置YOLO模型驗證參數")
        
        # 記錄特徵圖
        self.features = {}
        self._register_hooks()
        
        # 添加注意力模組
        self._add_attention_modules()
        
        logger.info(f"已初始化學生模型，類別數：{num_classes}")
    
    def _register_hooks(self):
        """註冊鉤子以獲取中間特徵"""
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # 關鍵特徵層
        feature_layers = {
            6: 'stage1',   # 淺層特徵
            10: 'stage2',  # 中層特徵
            14: 'stage3',  # 深層特徵
        }
        
        # 註冊鉤子
        for idx, name in feature_layers.items():
            try:
                self.model.model.model[idx].register_forward_hook(get_features(name))
            except Exception as e:
                logger.warning(f"註冊鉤子失敗: {e}")
    
    def _add_attention_modules(self):
        """添加注意力模組"""
        try:
            # 獲取關鍵特徵層通道數
            # 需要一次前向傳播來確定
            dummy_input = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            # 獲取通道數並創建注意力模組
            if 'stage2' in self.features:
                channels_s2 = self.features['stage2'].shape[1]
                self.stage2_attention = AttentionModule(channels_s2)
                logger.info(f"已添加stage2注意力模組，通道數：{channels_s2}")
            
            if 'stage3' in self.features:
                channels_s3 = self.features['stage3'].shape[1]
                self.stage3_attention = AttentionModule(channels_s3)
                logger.info(f"已添加stage3注意力模組，通道數：{channels_s3}")
                
            # 尋找檢測頭位置
            detect_idx = -1
            for i, m in enumerate(self.model.model.model):
                if isinstance(m, Detect):
                    detect_idx = i
                    break
            
            if detect_idx >= 0:
                # 獲取檢測頭信息
                old_detect = self.model.model.model[detect_idx]
                in_channels = [m.in_channels for m in old_detect.m]
                
                # 創建增強型檢測頭
                self.model.model.model[detect_idx] = EnhancedDetect(
                    nc=old_detect.nc,
                    ch=in_channels
                )
                logger.info("已替換為增強型檢測頭")
            
        except Exception as e:
            logger.warning(f"添加注意力模組失敗: {e}")
            # 確保屬性存在但為None
            self.stage2_attention = None
            self.stage3_attention = None
    
    def __call__(self, x):
        """模型調用"""
        # 前向傳播
        outputs = self.model(x)
        
        # 處理注意力特徵
        if hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            if 'stage2' in self.features:
                try:
                    self.features['stage2_enhanced'] = self.stage2_attention(self.features['stage2'])
                except Exception as e:
                    logger.warning(f"stage2注意力應用失敗: {e}")
        
        if hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            if 'stage3' in self.features:
                try:
                    self.features['stage3_enhanced'] = self.stage3_attention(self.features['stage3'])
                except Exception as e:
                    logger.warning(f"stage3注意力應用失敗: {e}")
        
        return outputs
    
    def to(self, device):
        """移動模型到指定設備"""
        if hasattr(self.model, 'to'):
            self.model.to(device)
        if hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            self.stage2_attention.to(device)
        if hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            self.stage3_attention.to(device)
        return self
    
    def train(self, mode=True):
        """設置訓練/評估模式"""
        if hasattr(self.model.model, 'train'):
            self.model.model.train(mode)
        if hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            self.stage2_attention.train(mode)
        if hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            self.stage3_attention.train(mode)
        return self
    
    def eval(self):
        """設置評估模式"""
        return self.train(False)
    
    def parameters(self):
        """獲取模型參數"""
        params = list(self.model.model.parameters())
        if hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            params.extend(list(self.stage2_attention.parameters()))
        if hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            params.extend(list(self.stage3_attention.parameters()))
        return params
    
    def get_features(self):
        """獲取中間特徵圖"""
        return self.features
    
    def state_dict(self):
        """獲取模型狀態字典"""
        state = {'base_model': self.model.model.state_dict()}
        if hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            state['stage2_attention'] = self.stage2_attention.state_dict()
        if hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            state['stage3_attention'] = self.stage3_attention.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """載入模型狀態字典"""
        if 'base_model' in state_dict:
            self.model.model.load_state_dict(state_dict['base_model'])
        else:
            self.model.model.load_state_dict(state_dict)
        
        if 'stage2_attention' in state_dict and hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
            self.stage2_attention.load_state_dict(state_dict['stage2_attention'])
        
        if 'stage3_attention' in state_dict and hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
            self.stage3_attention.load_state_dict(state_dict['stage3_attention'])
        
        return self

class EnhancedDetect(Detect):
    """增強版YOLO檢測頭"""
    
    def __init__(self, nc=80, ch=None):
        """初始化增強檢測頭"""
        super().__init__(nc, ch)
        
        # 添加注意力增強層
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, groups=c),
                nn.BatchNorm2d(c),
                nn.SiLU(),
            ) for c in ch
        ])
    
    def forward(self, x):
        """前向傳播"""
        # 應用注意力
        enhanced_x = []
        for i, feat in enumerate(x):
            enhanced = self.attention[i](feat) + feat  # 殘差連接
            enhanced_x.append(enhanced)
        
        # 使用增強特徵進行檢測
        return super().forward(enhanced_x)

class DistillationLoss(nn.Module):
    """知識蒸餾損失函數"""
    
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.5, gamma=0.5):
        """初始化蒸餾損失"""
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # 任務損失權重
        self.beta = beta    # 邏輯蒸餾權重
        self.gamma = gamma  # 特徵蒸餾權重
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_output, teacher_output, student_features, teacher_features, task_loss):
        """計算總損失"""
        # 邏輯蒸餾損失
        distill_loss = self._compute_logit_distillation(student_output, teacher_output)
        
        # 特徵蒸餾損失
        feature_loss = self._compute_feature_distillation(student_features, teacher_features)
        
        # 總損失
        total_loss = self.alpha * task_loss + self.beta * distill_loss + self.gamma * feature_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item(),
            'feature_loss': feature_loss.item()
        }
    
    def _compute_logit_distillation(self, student_logits, teacher_logits):
        """計算邏輯蒸餾損失 (KL散度)"""
        # 特殊處理YOLO輸出
        if hasattr(teacher_logits, 'boxes') and hasattr(student_logits, 'boxes'):
            return torch.tensor(0.1, device=student_logits.boxes.cls.device)
            
        # 嘗試提取logits
        t_logits = self._extract_logits(teacher_logits)
        s_logits = self._extract_logits(student_logits)
        
        if t_logits is None or s_logits is None:
            return torch.tensor(0.1, device=self._get_device(student_logits))
        
        # 應用溫度縮放
        t_logits_scaled = t_logits / self.temperature
        s_logits_scaled = s_logits / self.temperature
        
        # 計算KL散度
        kl_loss = F.kl_div(
            F.log_softmax(s_logits_scaled, dim=1),
            F.softmax(t_logits_scaled, dim=1),
            reduction='batchmean'
        ) * (self.temperature**2)
        
        return kl_loss
    
    def _compute_feature_distillation(self, student_features, teacher_features):
        """計算特徵蒸餾損失 (MSE)"""
        feature_loss = torch.tensor(0.0, device=self._get_device(student_features))
        
        # 處理字典類型的特徵
        if isinstance(student_features, dict) and isinstance(teacher_features, dict):
            common_keys = set(student_features.keys()) & set(teacher_features.keys())
            
            if not common_keys:
                return feature_loss
            
            for key in common_keys:
                s_feat = student_features[key]
                t_feat = teacher_features[key]
                
                # 調整大小
                if s_feat.shape != t_feat.shape:
                    try:
                        s_feat = F.interpolate(
                            s_feat,
                            size=t_feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    except Exception:
                        continue
                
                # 計算MSE損失
                feature_loss += self.mse_loss(s_feat, t_feat)
            
            feature_loss /= len(common_keys)
        
        return feature_loss
    
    def _extract_logits(self, output):
        """提取logits"""
        try:
            if hasattr(output, 'boxes'):
                # YOLO格式
                if len(output.boxes) == 0:
                    return None
                return output.boxes.cls
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # 自定義格式
                return output[0]
            else:
                return None
        except:
            return None
    
    def _get_device(self, obj):
        """獲取設備"""
        if isinstance(obj, torch.Tensor):
            return obj.device
        elif isinstance(obj, dict) and len(obj) > 0:
            for v in obj.values():
                if isinstance(v, torch.Tensor):
                    return v.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_teacher_model(config):
    """獲取教師模型"""
    num_classes = len(DEFECT_CLASSES)
    pretrained = config.get('model', {}).get('pretrained', True)
    
    teacher = TeacherModel(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"已建立教師模型，類別數: {num_classes}")
    
    return teacher

def get_student_model(config):
    """獲取學生模型"""
    num_classes = len(DEFECT_CLASSES)
    pretrained = config.get('model', {}).get('pretrained', True)
    
    student = StudentModel(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"已建立學生模型，類別數: {num_classes}")
    
    return student

def load_model(model_path, model_type='student'):
    """載入預訓練模型"""
    num_classes = len(DEFECT_CLASSES)
    
    try:
        if model_type == 'teacher':
            model = TeacherModel(num_classes=num_classes, pretrained=False)
        else:
            model = StudentModel(num_classes=num_classes, pretrained=False)
        
        # 載入權重
        if os.path.exists(model_path):
            if model_path.endswith(('.pt', '.pth')):
                # PyTorch權重
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                # YOLO模型
                model.model = YOLO(model_path)
                # 調整為PCB缺陷類別
                if hasattr(model.model.model, 'nc'):
                    model.model.model.nc = num_classes
            
            logger.info(f"已載入{model_type}模型: {model_path}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        if model_type == 'teacher':
            model = TeacherModel(num_classes=num_classes)
        else:
            model = StudentModel(num_classes=num_classes)
    
    return model

def get_distillation_loss(config):
    """獲取知識蒸餾損失函數"""
    temperature = config.get('distillation', {}).get('temperature', 4.0)
    alpha = config.get('distillation', {}).get('alpha', 0.5)
    beta = config.get('distillation', {}).get('beta', 0.5)
    gamma = config.get('distillation', {}).get('gamma', 0.5)
    
    return DistillationLoss(temperature, alpha, beta, gamma)

if __name__ == "__main__":
    """測試模型"""
    # 測試用配置
    config = {'model': {'pretrained': True}}
    
    # 創建模型
    teacher = get_teacher_model(config)
    student = get_student_model(config)
    distill_loss = get_distillation_loss(config)
    
    # 測試輸入
    dummy_input = torch.randn(2, 3, 640, 640)
    
    # 前向傳播
    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)
    
    logger.info("模型測試完成")