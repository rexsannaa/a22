#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model.py - PCB缺陷檢測模型定義
本模組整合了基於YOLO8架構的教師與學生模型定義，
實現知識蒸餾增強型混合模型用於PCB缺陷檢測。
主要特點:
1. 教師模型：基於YOLO8-L架構，提供強大的特徵表示能力
2. 學生模型：輕量化的YOLO8-S架構，結合雙分支特徵提取
3. 注意力機制：針對不同PCB缺陷類型的特定注意力模塊
4. 知識蒸餾：多層次特徵蒸餾，從淺層到深層的全面知識遷移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3, Conv, SPPF, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import logging

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

class AttentionModule(nn.Module):
    """針對PCB缺陷特性設計的注意力模組"""
    
    def __init__(self, channels, reduction_ratio=16):
        """
        初始化注意力模組
        
        參數:
            channels: 輸入特徵圖的通道數
            reduction_ratio: 通道降維比例
        """
        super(AttentionModule, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空間注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """前向傳播"""
        # 通道注意力
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        
        # 空間注意力
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        
        return x

class DefectSpecificAttention(nn.Module):
    """缺陷特定注意力機制"""
    
    def __init__(self, in_channels, num_classes=len(DEFECT_CLASSES)):
        """
        初始化缺陷特定注意力機制
        
        參數:
            in_channels: 輸入特徵圖的通道數
            num_classes: 缺陷類別數量
        """
        super(DefectSpecificAttention, self).__init__()
        
        # 對每個缺陷類別建立專門的注意力模組
        self.attention_modules = nn.ModuleList([
            AttentionModule(in_channels) for _ in range(num_classes)
        ])
        
        # 特徵加權模組
        self.feature_weights = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """前向傳播"""
        # 計算每個類別的權重
        weights = self.softmax(self.feature_weights(x))
        
        # 應用特定注意力並加權合併
        attended_features = []
        for i, attention in enumerate(self.attention_modules):
            # 提取對應類別的權重
            class_weight = weights[:, i:i+1]
            # 應用注意力
            attended_feature = attention(x) * class_weight
            attended_features.append(attended_feature)
            
        # 合併所有加權特徵
        output = sum(attended_features)
        
        return output

class TeacherModel(nn.Module):
    """教師模型：基於YOLO8-L架構"""
    
    def __init__(self, num_classes=len(DEFECT_CLASSES), pretrained=True):
        """
        初始化學生模型
        
        參數:
            num_classes: 缺陷類別數量
            pretrained: 是否使用預訓練權重
        """
        super(TeacherModel, self).__init__()
        
        # 載入YOLO8-L (禁止自動下載數據集)
        import os
        os.environ['YOLO_AUTOINSTALL'] = '0'
        os.environ['ULTRALYTICS_DATASET_DOWNLOAD'] = '0'
        os.environ['ULTRALYTICS_SKIP_VALIDATION'] = '1'  # 添加這行跳過驗證
        
        if pretrained:
            # 修改載入方式，確保不會使用COCO數據集的類別
            self.model = YOLO('yolov8l.pt')
            # 直接設置模型的類別數量
            if hasattr(self.model, 'model'):
                self.model.model.nc = num_classes
                logger.info(f"已設置教師模型類別數量為: {num_classes}")
                
                # 設置輸出層以匹配類別數量
                if hasattr(self.model.model, 'model'):
                    for m in self.model.model.model:
                        if hasattr(m, 'nc'):
                            m.nc = num_classes
                            logger.info(f"已設置檢測頭類別數量為: {num_classes}")
            
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
                        logger.warning("無法設置YOLO模型驗證參數，可能仍會嘗試下載COCO數據集")
            
            logger.info("已載入預訓練的YOLO8-L模型，並設置為PCB缺陷類別")
        else:
            # 從YAML配置初始化模型，不載入預訓練權重
            self.model = YOLO('yolov8l.yaml')
            # 調整為PCB缺陷類別數量
            if hasattr(self.model, 'model'):
                self.model.model.nc = num_classes
                logger.info(f"已初始化教師模型類別數量為: {num_classes}")
            
            # 同樣修改這裡的訪問方式
            if hasattr(self.model, 'args'):
                if isinstance(self.model.args, dict):
                    self.model.args['val'] = False
                    self.model.args['data'] = None
                else:
                    try:
                        self.model.args.val = False
                        self.model.args.data = None
                    except:
                        logger.warning("無法設置YOLO模型驗證參數，可能仍會嘗試下載COCO數據集")
            
            logger.info("已初始化YOLO8-L模型")
            
        # 獲取內部模型供直接操作
        self.yolo_model = self.model.model
            
        # 記錄中間特徵圖用於知識蒸餾
        self.features = {}
        
        # 註冊鉤子以擷取中間特徵
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
            
        # 需要蒸餾的關鍵特徵層
        feature_layers = {
            'backbone.6': 'stage1',  # 淺層特徵
            'backbone.10': 'stage2', # 中層特徵
            'backbone.14': 'stage3', # 深層特徵
        }
        
        # 註冊鉤子
        for layer_name, feature_name in feature_layers.items():
            self._get_layer(layer_name).register_forward_hook(
                get_features(feature_name)
            )
    def train(self, mode=True):
        """
        重寫train方法以適應YOLO模型
        
        參數:
            mode: 是否設為訓練模式
        """
        if mode:
            # 設置模型為訓練模式
            if hasattr(self.model, 'model'):
                self.model.model.train()
        else:
            # 設置模型為評估模式
            if hasattr(self.model, 'model'):
                self.model.model.eval()
        return self    
    def _get_layer(self, layer_name):
        """根據名稱獲取模型中的層"""
        parts = layer_name.split('.')
        curr_module = self.yolo_model
        
        # 處理第一個部分（例如 'backbone'）
        if parts[0] == 'backbone':
            # 如果是YOLO模型，直接獲取模型主體部分
            curr_module = self.yolo_model.model
            
            # 處理剩餘部分
            for part in parts[1:]:
                curr_module = curr_module[int(part)]
        else:
            # 原有的處理邏輯
            for part in parts:
                try:
                    curr_module = curr_module[int(part)]
                except ValueError:
                    # 嘗試將部分作為屬性名稱訪問
                    if hasattr(curr_module, part):
                        curr_module = getattr(curr_module, part)
                    else:
                        raise ValueError(f"無法找到層: {layer_name}, 部分: {part}")
                    
        return curr_module
        
    def forward(self, x):
        """前向傳播"""
        return self.model(x)
    
    def get_features(self):
        """獲取中間特徵圖用於知識蒸餾"""
        return self.features

class StudentModel(nn.Module):
    """學生模型：基於YOLO8-S架構的輕量化模型"""
    
    def __init__(self, num_classes=len(DEFECT_CLASSES), pretrained=True):
        """
        初始化學生模型
        
        參數:
            num_classes: 缺陷類別數量
            pretrained: 是否使用預訓練權重
        """
        super(StudentModel, self).__init__()
        
        # 載入YOLO8-S (禁止自動下載數據集)
        import os
        os.environ['YOLO_AUTOINSTALL'] = '0'
        os.environ['ULTRALYTICS_DATASET_DOWNLOAD'] = '0'
        os.environ['ULTRALYTICS_SKIP_VALIDATION'] = '1'  # 添加這行跳過驗證
        
        if pretrained:
            self.model = YOLO('yolov8s.pt')  # 移除 task='detect' 參數
            # 調整模型以匹配我們的類別數量
            self.model.model.nc = num_classes
            
            # 禁用驗證和數據集下載 - 修改這裡的訪問方式
            if hasattr(self.model, 'args'):
                if isinstance(self.model.args, dict):
                    # 如果args是字典，直接設置鍵值
                    self.model.args['val'] = False
                    self.model.args['data'] = None
                else:
                    # 如果args是對象，設置屬性
                    try:
                        self.model.args.val = False
                        self.model.args.data = None
                    except:
                        logger.warning("無法設置YOLO模型驗證參數，可能仍會嘗試下載COCO數據集")
            
            logger.info("已載入預訓練的YOLO8-S模型")
        else:
            self.model = YOLO('yolov8s.yaml')  # 移除 task='detect' 參數
            self.model.model.nc = num_classes
            
            # 同樣修改這裡的訪問方式
            if hasattr(self.model, 'args'):
                if isinstance(self.model.args, dict):
                    self.model.args['val'] = False
                    self.model.args['data'] = None
                else:
                    try:
                        self.model.args.val = False
                        self.model.args.data = None
                    except:
                        logger.warning("無法設置YOLO模型驗證參數，可能仍會嘗試下載COCO數據集")
            
            logger.info("已初始化YOLO8-S模型")
                
        # 獲取內部模型供直接操作
        self.yolo_model = self.model.model
        
        # 記錄中間特徵圖用於知識蒸餾
        self.features = {}
        
        # 修改並增強模型特徵提取能力
        self._print_model_structure()  # 診斷模型結構
        self._enhance_feature_extraction()
        
        # 註冊鉤子以擷取中間特徵
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
            
        # 需要蒸餾的關鍵特徵層
        feature_layers = {
            'backbone.6': 'stage1',  # 淺層特徵
            'backbone.10': 'stage2', # 中層特徵
            'backbone.14': 'stage3', # 深層特徵
        }
        
        # 註冊鉤子
        for layer_name, feature_name in feature_layers.items():
            try:
                self._get_layer(layer_name).register_forward_hook(
                    get_features(feature_name)
                )
            except Exception as e:
                logger.warning(f"註冊鉤子時發生錯誤: {e}")
                
    def _print_model_structure(self):
        """打印模型結構以診斷問題"""
        logger.info("打印模型結構:")
        for name, module in self.yolo_model.named_modules():
            logger.info(f"層名稱: {name}, 類型: {type(module).__name__}")
            
    def _enhance_feature_extraction(self):
        """增強模型特徵提取能力"""
        try:
            # 使用動態通道檢測
            self.features_info = {}
            
            # 註冊臨時鉤子來檢測特徵尺寸
            def get_feature_info(name):
                def hook(module, input, output):
                    self.features_info[name] = output.shape
                return hook
                
            # 記錄原始特徵尺寸的臨時鉤子
            temp_hooks = []
            for layer_name, feature_name in {
                'backbone.6': 'stage1',
                'backbone.10': 'stage2',
                'backbone.14': 'stage3',
            }.items():
                try:
                    h = self._get_layer(layer_name).register_forward_hook(
                        get_feature_info(feature_name)
                    )
                    temp_hooks.append(h)
                except Exception as e:
                    logger.warning(f"註冊臨時鉤子時發生錯誤: {e}")
            
            # 執行一次前向傳播以獲取特徵尺寸
            dummy_input = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                _ = self.yolo_model(dummy_input)
                
            # 移除臨時鉤子
            for h in temp_hooks:
                h.remove()
                
            # 根據實際特徵尺寸建立注意力模組
            if 'stage2' in self.features_info:
                channels_stage2 = self.features_info['stage2'][1]  # 獲取通道數
                self.stage2_attention = DefectSpecificAttention(channels_stage2)
                logger.info(f"已建立stage2注意力模組，通道數: {channels_stage2}")
                
            if 'stage3' in self.features_info:
                channels_stage3 = self.features_info['stage3'][1]  # 獲取通道數
                self.stage3_attention = DefectSpecificAttention(channels_stage3)
                logger.info(f"已建立stage3注意力模組，通道數: {channels_stage3}")
                
            # 獲取檢測頭的索引
            detect_index = -1
            for i, m in enumerate(self.yolo_model.model):
                if isinstance(m, Detect):
                    detect_index = i
                    break
            
            if detect_index >= 0:
                # 獲取原始檢測頭
                original_detect = self.yolo_model.model[detect_index]
                
                # 從實際前向傳播獲取通道數
                detect_in_channels = []
                for i in range(len(original_detect.cv2)):
                    try:
                        ch = original_detect.cv2[i].conv.in_channels
                        detect_in_channels.append(ch)
                    except (AttributeError, IndexError):
                        # 如果無法獲取，使用默認值或從特徵圖推斷
                        if i == 0 and 'stage1' in self.features_info:
                            detect_in_channels.append(self.features_info['stage1'][1])
                        elif i == 1 and 'stage2' in self.features_info:
                            detect_in_channels.append(self.features_info['stage2'][1])
                        elif i == 2 and 'stage3' in self.features_info:
                            detect_in_channels.append(self.features_info['stage3'][1])
                        else:
                            detect_in_channels.append(256 * (2 ** i))  # 估計通道數
                
                logger.info(f"檢測到的檢測頭通道數: {detect_in_channels}")
                
                # 替換檢測頭為自定義版本
                self.yolo_model.model[detect_index] = EnhancedDetect(
                    nc=original_detect.nc,
                    in_channels=detect_in_channels
                )
                logger.info("已成功替換檢測頭")
            else:
                logger.warning("未找到檢測頭層，無法替換為增強版本")
                    
        except Exception as e:
            logger.error(f"增強特徵提取時發生錯誤: {e}")
            logger.info("使用原始模型繼續")
            # 確保即使出錯也能繼續訓練，禁用注意力機制
            self.stage2_attention = None
            self.stage3_attention = None
    
    def _get_layer(self, layer_name):
        """根據名稱獲取模型中的層"""
        parts = layer_name.split('.')
        curr_module = self.yolo_model
        
        # 處理第一個部分（例如 'backbone'）
        if parts[0] == 'backbone':
            # 如果是YOLO模型，直接獲取模型主體部分
            curr_module = self.yolo_model.model
            
            # 處理剩餘部分
            for part in parts[1:]:
                curr_module = curr_module[int(part)]
        else:
            # 原有的處理邏輯
            for part in parts:
                try:
                    curr_module = curr_module[int(part)]
                except ValueError:
                    # 嘗試將部分作為屬性名稱訪問
                    if hasattr(curr_module, part):
                        curr_module = getattr(curr_module, part)
                    else:
                        raise ValueError(f"無法找到層: {layer_name}, 部分: {part}")
                
        return curr_module
        
    def forward(self, x):
        """前向傳播"""
        # 正常前向傳播，但處理YOLO模型特殊輸出
        if self.training:
            # 訓練模式下使用YOLO內部forward而非__call__
            if hasattr(self.yolo_model, 'forward') and callable(self.yolo_model.forward):
                with torch.no_grad():
                    _ = self.yolo_model.forward(x)  # 只是為了觸發特徵提取
                # 手動構建輸出格式，與教師模型匹配
                out = (self.yolo_model.model[-1].training_outputs 
                    if hasattr(self.yolo_model.model [-1], 'training_outputs') 
                    else [torch.zeros(x.shape[0], 1, 1), torch.zeros(x.shape[0], 1)])
            else:
                out = self.model(x)
        else:
            # 評估模式正常使用__call__方法
            out = self.model(x)
        
        # 處理中間特徵
        if self.training and self.features:
            # 應用注意力機制到中間特徵，安全地檢查注意力模組是否存在
            if 'stage2' in self.features and hasattr(self, 'stage2_attention') and self.stage2_attention is not None:
                try:
                    self.features['stage2_enhanced'] = self.stage2_attention(self.features['stage2'])
                except Exception as e:
                    logger.warning(f"應用stage2注意力時發生錯誤: {e}")
            
            if 'stage3' in self.features and hasattr(self, 'stage3_attention') and self.stage3_attention is not None:
                try:
                    self.features['stage3_enhanced'] = self.stage3_attention(self.features['stage3'])
                except Exception as e:
                    logger.warning(f"應用stage3注意力時發生錯誤: {e}")
        
        return out
    
    def get_features(self):
        """獲取中間特徵圖用於知識蒸餾"""
        return self.features

class EnhancedDetect(Detect):
    """增強版YOLO檢測頭，整合注意力特徵"""
    
    def __init__(self, nc=80, in_channels=None):
        """
        初始化增強版檢測頭
        
        參數:
            nc: 類別數量
            in_channels: 輸入通道數列表
        """
        super(EnhancedDetect, self).__init__(nc=nc, ch=in_channels)  # 使用正確的參數名
        
        # 獲取輸入通道數
        ch = self.cv2[0].conv.in_channels  # 從原始層提取通道數
        self.in_channels = [ch, ch*2, ch*4] if in_channels is None else in_channels
        
        # 添加額外處理層（權重共享以減少參數）
        self.enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels[i], self.in_channels[i], 
                        kernel_size=3, padding=1, groups=self.in_channels[i]),
                nn.BatchNorm2d(self.in_channels[i]),
                nn.SiLU()
            ) for i in range(len(self.in_channels))
        ])
        
    def forward(self, x):
        """前向傳播"""
        # 應用增強處理到每個特徵圖
        enhanced_x = []
        for i, feat in enumerate(x):
            enhanced_x.append(self.enhancers[i](feat) + feat)  # 殘差連接
            
        # 使用父類方法處理增強特徵
        return super().forward(enhanced_x)

def get_teacher_model(config):
    """獲取教師模型
    
    參數:
        config: 配置字典
        
    回傳:
        教師模型實例
    """
    num_classes = len(DEFECT_CLASSES)
    pretrained = config.get('pretrained', True)
    
    model = TeacherModel(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"已建立教師模型，類別數：{num_classes}")
    
    return model

def get_student_model(config):
    """獲取學生模型
    
    參數:
        config: 配置字典
        
    回傳:
        學生模型實例
    """
    num_classes = len(DEFECT_CLASSES)
    pretrained = config.get('pretrained', True)
    
    model = StudentModel(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"已建立學生模型，類別數：{num_classes}")
    
    return model

class DistillationLoss(nn.Module):
    """知識蒸餾損失函數"""
    
    def __init__(self, 
                 temperature=4.0, 
                 alpha=0.5, 
                 beta=0.5, 
                 gamma=0.5):
        """
        初始化知識蒸餾損失
        
        參數:
            temperature: 蒸餾溫度參數
            alpha: 蒸餾損失權重
            beta: 特徵蒸餾損失權重
            gamma: 原始任務損失權重
        """
        super(DistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 特徵匹配使用MSE損失
        self.feature_criterion = nn.MSELoss()
        
    def forward(self, student_outputs, teacher_outputs, 
                student_features, teacher_features, 
                targets, task_loss):
        """
        計算蒸餾損失
        
        參數:
            student_outputs: 學生模型輸出
            teacher_outputs: 教師模型輸出
            student_features: 學生模型中間特徵
            teacher_features: 教師模型中間特徵
            targets: 真實標籤
            task_loss: 原始任務損失
            
        回傳:
            total_loss: 總損失
        """
        # 原始任務損失
        original_loss = task_loss
        
        # 輸出蒸餾損失 (KL散度)
        distill_loss = 0
        if teacher_outputs is not None and student_outputs is not None:
            # 提取置信度分數
            teacher_scores = teacher_outputs[0]
            student_scores = student_outputs[0]
            
            # 應用溫度縮放
            teacher_scores = teacher_scores / self.temperature
            student_scores = student_scores / self.temperature
            
            # 計算KL散度
            distill_loss = F.kl_div(
                F.log_softmax(student_scores, dim=1),
                F.softmax(teacher_scores, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        # 特徵蒸餾損失
        feature_loss = 0
        if teacher_features and student_features:
            # 根據特徵名稱匹配特徵
            for name in teacher_features:
                if name in student_features:
                    t_feat = teacher_features[name]
                    s_feat = student_features[name]
                    
                    # 如果尺寸不一致，調整學生特徵尺寸
                    if s_feat.shape != t_feat.shape:
                        s_feat = F.interpolate(
                            s_feat, 
                            size=t_feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        
                    # 計算特徵匹配損失
                    feature_loss += self.feature_criterion(s_feat, t_feat)
        
        # 計算總損失
        total_loss = (self.alpha * distill_loss + 
                      self.beta * feature_loss + 
                      self.gamma * original_loss)
        
        return total_loss

def load_model(model_path, model_type='student'):
    """載入預訓練模型
    
    參數:
        model_path: 模型權重路徑
        model_type: 'teacher'或'student'
        
    回傳:
        載入的模型
    """
    num_classes = len(DEFECT_CLASSES)
    
    if model_type == 'teacher':
        model = TeacherModel(num_classes=num_classes, pretrained=False)
    else:
        model = StudentModel(num_classes=num_classes, pretrained=False)
    
    try:
        # 載入權重
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"已成功載入{model_type}模型：{model_path}")
    except Exception as e:
        logger.error(f"載入模型失敗：{e}")
        # 嘗試載入YOLO格式權重
        try:
            if model_type == 'teacher':
                model = YOLO(model_path)
                logger.info(f"已以YOLO格式載入教師模型：{model_path}")
            else:
                model = YOLO(model_path)
                logger.info(f"已以YOLO格式載入學生模型：{model_path}")
        except Exception as e2:
            logger.error(f"以YOLO格式載入模型失敗：{e2}")
    
    return model

if __name__ == "__main__":
    """測試模型定義"""
    # 簡單測試
    config = {'pretrained': True}
    
    # 創建模型
    teacher = get_teacher_model(config)
    student = get_student_model(config)
    
    # 測試輸入
    dummy_input = torch.randn(2, 3, 640, 640)
    
    # 前向傳播
    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)
    
    # 獲取特徵
    teacher_features = teacher.get_features()
    student_features = student.get_features()
    
    # 打印特徵形狀
    for name, feat in teacher_features.items():
        logger.info(f"教師特徵 {name}: {feat.shape}")
        
    for name, feat in student_features.items():
        logger.info(f"學生特徵 {name}: {feat.shape}")
    
    logger.info("模型測試完成")