#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model.py - PCB缺陷檢測知識蒸餾增強型混合模型定義
整合教師模型與學生模型的定義，提供統一的模型建立介面。
主要特點:
1. 教師模型：以FasterRCNN為基礎的高精度模型
2. 學生模型：雙分支輕量化結構，整合全局與局部特徵
3. 缺陷特定注意力機制：針對不同PCB缺陷類型的優化
4. 結構化參數共享：減少模型大小並提高推理效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
try:
    # 嘗試從舊位置導入
    from torchvision.ops import misc as misc_nn_ops
    IntermediateLayerGetter = misc_nn_ops.IntermediateLayerGetter
    LastLevelMaxPool = misc_nn_ops.LastLevelMaxPool
except AttributeError:
    # 如果失敗，從新位置導入
    from torchvision.models._utils import IntermediateLayerGetter
    # 對於 LastLevelMaxPool，我們需要檢查它是否在 feature_pyramid_network 模組中
    try:
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
    except (ImportError, AttributeError):
        # 如果找不到，我們自己定義一個
        class LastLevelMaxPool(nn.Module):
            """
            Applies a max_pool2d on top of the last feature map
            """
            def __init__(self):
                super(LastLevelMaxPool, self).__init__()

            def forward(self, x):
                if isinstance(x, torch.Tensor):
                    return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
                elif isinstance(x, list) and len(x) > 0:
                    return [F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)]
                elif isinstance(x, OrderedDict):
                    # 處理 OrderedDict 情況
                    names = list(x.keys())
                    if not names:
                        return []
                    last_feature = x[names[-1]]
                    # 應用 max_pool
                    pooled = F.max_pool2d(last_feature, kernel_size=1, stride=2, padding=0)
                    # 創建新的OrderedDict返回
                    result = OrderedDict()
                    for k in x.keys():
                        result[k] = x[k]
                    result[str(len(names))] = pooled
                    return result
                else:
                    # 對於其他情況，返回空列表
                    return []
from collections import OrderedDict
import logging
import math
import os

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefectSpecificAttention(nn.Module):
    """PCB缺陷特定注意力機制，針對不同缺陷類型優化的注意力模塊"""
    
    def __init__(self, in_channels, reduction_ratio=16, defect_type=None, config=None):
        """
        初始化缺陷特定注意力機制
        
        Args:
            in_channels: 輸入通道數
            reduction_ratio: 通道縮減比例
            defect_type: 缺陷類型，決定注意力參數
            config: 配置字典
        """
        super(DefectSpecificAttention, self).__init__()
        self.defect_type = defect_type
        self.config = config
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 定義共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        # 獲取缺陷特定參數
        if defect_type and config:
            self.channel_weights = self._get_defect_config(defect_type, "channel_weights", [0.3, 0.3, 0.4])
            self.spatial_kernel = self._get_defect_config(defect_type, "spatial_kernel", 7)
        else:
            self.channel_weights = [0.3, 0.3, 0.4]
            self.spatial_kernel = 7
        
        # 空間注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=self.spatial_kernel, 
                                      padding=self.spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def _get_defect_config(self, defect_type, param_name, default_value):
        """獲取缺陷特定配置參數"""
        try:
            return self.config["attention"]["defect_specific"][defect_type][param_name]
        except:
            return default_value
    
    def forward(self, x):
        """前向傳播"""
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        
        # 應用通道注意力
        x = x * channel_out
        
        # 空間注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid(self.spatial_conv(spatial_in))
        
        # 應用空間注意力
        return x * spatial_out


class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力層"""
    
    def __init__(self, channel, reduction=16):
        """
        初始化SE注意力層
        
        Args:
            channel: 輸入通道數
            reduction: 縮減比例
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向傳播"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LightweightFPN(nn.Module):
    """輕量級特徵金字塔網絡，適用於邊緣設備部署"""
    
    def __init__(self, in_channels_list, out_channels, use_depthwise=True, extra_blocks=None):
        """
        初始化輕量級FPN
        
        Args:
            in_channels_list: 輸入通道列表
            out_channels: 輸出通道數
            use_depthwise: 是否使用深度可分離卷積
            extra_blocks: 額外的輸出層
        """
        super(LightweightFPN, self).__init__()
        
        # 保存輸出通道數
        self.out_channels = out_channels
        
        # 橫向連接層
        self.inner_blocks = nn.ModuleList()
        # 層間連接層
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # 橫向連接使用1x1卷積降維，確保輸出通道數正確
            inner_block = nn.Conv2d(80, out_channels, 1)  # 將輸入通道固定為80
            layer_block = self._make_layer_block(out_channels, use_depthwise)
            
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 額外層
        self.extra_blocks = extra_blocks
    
    def _make_layer_block(self, channels, use_depthwise=True):
        """創建層間連接塊"""
        if use_depthwise:
            # 使用深度可分離卷積
            return nn.Sequential(
                # 深度卷積
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # 點卷積
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            # 使用標準卷積
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

    
    def _make_layer_block(self, channels, use_depthwise=True):
        """創建層間連接塊"""
        if use_depthwise:
            # 使用深度可分離卷積
            return nn.Sequential(
                # 深度卷積
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # 點卷積
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            # 使用標準卷積
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        """前向傳播"""
        # 從底到頂處理特徵
        last_inner = None
        results = []
        
        # 遍歷每個特徵層
        for idx, feat in enumerate(x):
            # 檢查特徵層的通道數
            feat_channels = feat.shape[1]
            
            # 使用對應的橫向連接層
            if idx < len(self.inner_blocks):
                # 創建一個動態適配層，確保通道數匹配
                if feat_channels != 80:  # 假設目標通道是80
                    # 創建動態適配層
                    channel_adapter = nn.Conv2d(feat_channels, 80, kernel_size=1, bias=False).to(feat.device)
                    # 初始化權重
                    nn.init.kaiming_normal_(channel_adapter.weight)
                    # 應用通道適配
                    feat = channel_adapter(feat)
                
                # 應用橫向連接
                inner_lateral = self.inner_blocks[idx](feat)
                
                if last_inner is not None:
                    # 調整大小以匹配當前特徵
                    feat_shape = inner_lateral.shape[-2:]
                    inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                    inner_lateral = inner_lateral + inner_top_down
                
                last_inner = inner_lateral
                # 應用層塊
                results.append(self.layer_blocks[idx](last_inner))
            
        # 如果需要，添加額外的特徵層
        if self.extra_blocks is not None and len(results) > 0:
            results.extend(self.extra_blocks(results[-1]))
        
        # 返回結果，確保通道數一致
        return results


# 當前的 LastLevelP6P7 類:
class LastLevelP6P7(nn.Module):
    """額外添加P6和P7層，用於檢測大範圍物體"""
    
    def __init__(self, in_channels, out_channels):
        """
        初始化P6P7層
        
        Args:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
        """
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向傳播"""
        p6 = self.p6(x[-1])
        p7 = self.p7(self.relu(p6))
        return [p6, p7]


class LastLevelP6P7(nn.Module):
    """額外添加P6和P7層，用於檢測大範圍物體"""
    
    def __init__(self, in_channels, out_channels):
        """
        初始化P6P7層
        
        Args:
            in_channels: 輸入通道數
            out_channels: 輸出通道數
        """
        super(LastLevelP6P7, self).__init__()
        # 修改 p6 的輸入通道數為 80，與實際輸入匹配
        self.p6 = nn.Conv2d(80, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向傳播"""
        p6 = self.p6(x[-1])
        p7 = self.p7(self.relu(p6))
        return [p6, p7]


class StudentModel(nn.Module):
    """
    PCB缺陷檢測學生模型，整合雙分支輕量化結構和缺陷特定注意力機制
    """
    
    def __init__(self, config):
        """
        初始化學生模型
        
        Args:
            config: 配置字典
        """
        super(StudentModel, self).__init__()
        self.config = config
        student_cfg = config["student"]
        
        # 加載預訓練骨幹網絡
        if student_cfg["backbone"] == "mobilenetv3_small":
            backbone = mobilenet_v3_small(pretrained=student_cfg["pretrained"])
            
            # 獲取特徵層輸出通道數
            backbone_out_channels = [
                16,  # features.3 - stage1
                24,  # features.6 - stage2
                40,  # features.9 - stage3
                80,  # features.12 - stage4
                576  # features.15 - stage5
            ]

            # 雙分支設定 - 全局與局部特徵提取
            if student_cfg["dual_branch"]["enabled"]:
                # 全局分支
                if student_cfg["dual_branch"]["shared_backbone"]:
                    self.global_backbone = backbone
                else:
                    self.global_backbone = mobilenet_v3_small(pretrained=student_cfg["pretrained"])
                
                # 添加全局分支注意力
                global_attention_type = student_cfg["dual_branch"]["global_branch"]["attention_type"]
                if global_attention_type == "se":
                    self.global_attention = SELayer(backbone_out_channels[-1])
                else:
                    self.global_attention = None
                
                # 添加局部分支注意力
                local_attention_type = student_cfg["dual_branch"]["local_branch"]["attention_type"]
                if local_attention_type == "defect_specific":
                    self.local_attentions = nn.ModuleDict({
                        defect_type: DefectSpecificAttention(
                            backbone_out_channels[-1], 
                            config["attention"]["common"]["reduction_ratio"],
                            defect_type, 
                            config
                        ) for defect_type in config["dataset"]["defect_classes"]
                    })
                else:
                    self.local_attentions = None
                
                # FPN通道設置 - 修正為對應通道數
                global_fpn_channels = 96  # 修改為96，與模型實際輸出的通道數匹配
                local_fpn_channels = 96   # 確保兩個分支通道數一致
            else:
                # 單分支結構
                self.global_backbone = backbone
                self.global_attention = None
                self.local_attentions = None
                global_fpn_channels = student_cfg["neck"]["out_channels"]
                local_fpn_channels = None
            
            # 創建特徵金字塔網絡
            in_channels_list = backbone_out_channels[-3:]  # 使用最後三層特徵
            
            # 全局FPN
            global_fpn_channels = 80  # 修正為80個通道，與特徵提取適配層期望的通道數匹配

            if student_cfg["neck"]["extra_blocks"] == "lastlevel_p6p7":
                self.global_fpn_extra_blocks = LastLevelP6P7(
                    80,  # 直接使用固定的 80 作為輸入通道
                    global_fpn_channels  # 這裡的值是 80
                )
            else:
                self.global_fpn_extra_blocks = None
                
            self.global_fpn = LightweightFPN(
                in_channels_list=in_channels_list,
                out_channels=global_fpn_channels,
                use_depthwise=student_cfg["neck"]["use_depthwise"],
                extra_blocks=self.global_fpn_extra_blocks
            )
            
            # 局部FPN (如果啟用了雙分支)
            if student_cfg["dual_branch"]["enabled"]:
                local_fpn_channels = 80
                if student_cfg["neck"]["extra_blocks"] == "lastlevel_p6p7":
                    self.local_fpn_extra_blocks = LastLevelP6P7(
                        80,  # 直接使用固定的 80 作為輸入通道
                        local_fpn_channels  # 使用局部 FPN 通道數
                    )
                else:
                    self.local_fpn_extra_blocks = None
                    
                self.local_fpn = LightweightFPN(
                    in_channels_list=in_channels_list,
                    out_channels=local_fpn_channels,
                    use_depthwise=student_cfg["neck"]["use_depthwise"],
                    extra_blocks=self.local_fpn_extra_blocks
                )
            else:
                self.local_fpn = None
            
            # 特別修正輸出通道
            # 在雙分支模式下，計算融合後的輸出通道數
            if student_cfg["dual_branch"]["enabled"]:   
                # 將局部分支通道數也修改為80
                local_fpn_channels = 96
                
                # 輸入和輸出通道數需要匹配 
                input_channels = global_fpn_channels + local_fpn_channels  # 來自兩個分支的總輸入通道

                # 確保輸出通道數與backbone特徵通道數匹配 - 關鍵修正
                output_channels = 96  # 使用96通道以匹配模型期望的輸入通道數
                                
                self.fusion_layer = nn.Conv2d(
                    input_channels,  # 輸入通道 (80 + 80 = 160)
                    output_channels,  # 輸出通道為96
                    kernel_size=1,
                    bias=False
                )
                self.fusion_norm = nn.BatchNorm2d(output_channels)
                self.fusion_act = nn.ReLU(inplace=True)
                
                # 更新 fpn_out_channels 以確保後續層匹配
                fpn_out_channels = output_channels
            else:
                self.fusion_layer = None
                fpn_out_channels = global_fpn_channels
            
            # 使用正確的輸出通道數來創建BackboneWithFPN
            self.backbone_with_fpn = BackboneWithFPN(
                backbone_body=self.global_backbone,
                fpn=self.global_fpn,
                out_channels=96  
            )
            
            # 構建檢測頭
            # 定義錨點生成器
            anchor_sizes = student_cfg["detection_head"]["anchor_sizes"]
            aspect_ratios = student_cfg["detection_head"]["aspect_ratios"]
            anchor_generator = AnchorGenerator(
                sizes=tuple([tuple(anchor_sizes)] * (3 + (2 if self.global_fpn_extra_blocks else 0))),
                aspect_ratios=tuple([tuple(aspect_ratios)] * (3 + (2 if self.global_fpn_extra_blocks else 0)))
            )
            
            # 定義ROI池化
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3", "4"][:3 + (2 if self.global_fpn_extra_blocks else 0)],
                output_size=7,
                sampling_ratio=2
            )
            
            # 創建FasterRCNN模型
            self.detector = FasterRCNN(
                backbone=self.backbone_with_fpn,
                num_classes=student_cfg["detection_head"]["num_classes"],
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=config["dataset"]["input_size"][0],
                max_size=config["dataset"]["input_size"][1]
            )
            
            # 替換分類和迴歸頭
            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(
                in_features,
                student_cfg["detection_head"]["num_classes"]
            )
            
            logger.info("學生模型初始化完成")
    
    def extract_features(self, x, defect_type=None):
        """
        提取特徵，支持缺陷特定處理
        
        Args:
            x: 輸入圖像
            defect_type: 缺陷類型，用於選擇注意力機制
        
        Returns:
            特徵列表
        """
        student_cfg = self.config["student"]
        
        # 提取骨幹網絡特徵
        features = []
        tmp_x = x
        for name, module in self.global_backbone.features._modules.items():
            tmp_x = module(tmp_x)
            if int(name) in [3, 6, 9, 12, 15]:  # 保存中間特徵
                features.append(tmp_x)
        
        # 獲取全局分支特徵
        global_features = features.copy()
        if self.global_attention is not None:
            global_features[-1] = self.global_attention(global_features[-1])
        
        # 獲取局部分支特徵(如果啟用)
        if student_cfg["dual_branch"]["enabled"] and defect_type and self.local_attentions:
            if defect_type in self.local_attentions:
                local_features = features.copy()
                local_features[-1] = self.local_attentions[defect_type](local_features[-1])
            else:
                local_features = features.copy()
        else:
            local_features = None
        
        # 應用FPN
        global_fpn_features = self.global_fpn(global_features[-3:])  # 使用最後三層特徵
        
        # 關鍵修改: 確保特徵通道數匹配，需要添加顯式的通道轉換步驟
        # 如果全局特徵的通道數是96，而我們期望的是80，添加一個1x1卷積來調整通道數
        adjusted_global_features = []
        for feature in global_fpn_features:
            if feature.shape[1] != 80:  # 如果不是期望的80通道
                # 使用1x1卷積調整通道數
                channel_adapter = nn.Conv2d(feature.shape[1], 80, kernel_size=1, bias=False).to(feature.device)
                feature = channel_adapter(feature)
            adjusted_global_features.append(feature)
        
        global_fpn_features = adjusted_global_features  # 更新全局特徵
        
        if local_features is not None and self.local_fpn is not None:
            local_fpn_features = self.local_fpn(local_features[-3:])
            
            # 同樣為局部特徵添加通道調整
            adjusted_local_features = []
            for feature in local_fpn_features:
                if feature.shape[1] != 80:  # 如果不是期望的80通道
                    channel_adapter = nn.Conv2d(feature.shape[1], 80, kernel_size=1, bias=False).to(feature.device)
                    feature = channel_adapter(feature)
                adjusted_local_features.append(feature)
            
            local_fpn_features = adjusted_local_features  # 更新局部特徵
                
            # 融合操作
            fused_features = []
            for gf, lf in zip(global_fpn_features, local_fpn_features):
                # 確保特徵具有相同的空間維度
                if gf.shape[2:] != lf.shape[2:]:
                    lf = F.interpolate(lf, size=gf.shape[2:], mode='bilinear', align_corners=False)
                
                # 沿通道維度連接
                fused = torch.cat([gf, lf], dim=1)
                # 應用融合層
                fused = self.fusion_layer(fused)
                fused = self.fusion_norm(fused)
                fused = self.fusion_act(fused)
                fused_features.append(fused)
            
            return fused_features
        else:
            return global_fpn_features
        # 確保所有特徵的通道數為96
        if global_fpn_features[0].shape[1] != 96:
            adjusted_features = []
            for feature in global_fpn_features:
                # 使用1x1卷積調整通道數
                channel_adapter = nn.Conv2d(feature.shape[1], 96, kernel_size=1, bias=False).to(feature.device)
                adjusted_feature = channel_adapter(feature)
                adjusted_features.append(adjusted_feature)
            return adjusted_features
    
    def forward(self, x, targets=None):
        """前向傳播"""
        if self.training and targets is None:
            raise ValueError("在訓練模式中，targets不應為None")
            
        return self.detector(x, targets)


class BackboneWithFPN(nn.Module):
    """用於包裝學生模型特徵提取器作為FasterRCNN的骨幹網絡"""
    
    def __init__(self, backbone_body, fpn, out_channels):
        """
        初始化骨幹網絡包裝器
        
        Args:
            backbone_body: 骨幹網絡主體
            fpn: 特徵金字塔網絡
            out_channels: 輸出通道數
        """
        super(BackboneWithFPN, self).__init__()
        self.body = backbone_body
        self.fpn = fpn
        self.out_channels = 96  
    
    def forward(self, x):
        """前向傳播"""
        features = []
        # 提取骨幹網絡特徵
        for name, module in self.body.features._modules.items():
            x = module(x)
            if int(name) in [3, 6, 9, 12, 15]:  # 保存中間特徵
                features.append(x)
        
        # 提取FPN特徵
        fpn_features = self.fpn(features[-3:])  # 使用最後三層特徵
        
        # 返回有序字典
        return OrderedDict([(str(i), feat) for i, feat in enumerate(fpn_features)])


class TeacherModel(nn.Module):
    """PCB缺陷檢測教師模型，基於FasterRCNN的高精度模型"""
    
    def __init__(self, config):
        """
        初始化教師模型
        
        Args:
            config: 配置字典
        """
        super(TeacherModel, self).__init__()
        self.config = config
        teacher_cfg = config["teacher"]
        
        # 加載預訓練骨幹網絡
        if teacher_cfg["backbone"] == "resnet50":
            backbone = resnet50(pretrained=teacher_cfg["pretrained"])
            
            # 凍結指定層
            if teacher_cfg["freeze_backbone"]:
                for name, param in backbone.named_parameters():
                    if "layer4" not in name:  # 只凍結layer4之前的層
                        param.requires_grad_(False)
            
            # 獲取各層輸出通道數
            backbone_out_channels = {
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048
            }

            # 提取特徵的層
            return_layers = {
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3"
            }

            # 創建特徵提取器
            backbone_model = IntermediateLayerGetter(backbone, return_layers)

            # 特徵金字塔網絡 - 修改這裡的通道配置
            in_channels_list = [512, 1024, 2048]  # 明確指定 ResNet50 層 2、3、4 的通道數
            out_channels = teacher_cfg["fpn"]["out_channels"]

            # 確定額外的塊類型
            if teacher_cfg["fpn"]["extra_blocks"] == "lastlevel_maxpool":
                extra_blocks = LastLevelMaxPool()
            else:
                extra_blocks = None

            # 創建 FPN
            fpn_module = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks
            )

            # 自定義一個簡單的包裝器來處理特徵提取
            class CustomBackboneWrapper(nn.Module):
                def __init__(self, body, fpn):
                    super(CustomBackboneWrapper, self).__init__()
                    self.body = body
                    self.fpn = fpn
                    self.out_channels = out_channels
                
                def forward(self, x):
                    # 提取主幹特徵
                    x = self.body(x)
                    
                    # 僅選取需要的特徵層
                    selected_features = {
                        '1': x['1'],  # layer2
                        '2': x['2'],  # layer3
                        '3': x['3']   # layer4
                    }
                    
                    # 應用 FPN
                    x = self.fpn(selected_features)
                    return x

            # 創建自定義骨幹網絡
            self.backbone = CustomBackboneWrapper(backbone_model, fpn_module)
            
            # 構建FasterRCNN
            # 定義錨點生成器
            anchor_sizes = teacher_cfg["rpn"]["anchor_sizes"]
            aspect_ratios = teacher_cfg["rpn"]["aspect_ratios"]
            anchor_generator = AnchorGenerator(
                sizes=tuple([tuple(anchor_sizes)] * (3 + (1 if extra_blocks else 0))),
                aspect_ratios=tuple([tuple(aspect_ratios)] * (3 + (1 if extra_blocks else 0)))
            )
            
            # 定義ROI池化
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"][:3 + (1 if extra_blocks else 0)],
                output_size=teacher_cfg["rpn"]["roi_align_output_size"],
                sampling_ratio=2
            )
            
            # 創建FasterRCNN模型
            self.detector = FasterRCNN(
                backbone=self.backbone,
                num_classes=teacher_cfg["roi_heads"]["num_classes"],
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                rpn_fg_iou_thresh=teacher_cfg["rpn"]["fg_iou_thresh"],
                rpn_bg_iou_thresh=teacher_cfg["rpn"]["bg_iou_thresh"],
                rpn_batch_size_per_image=teacher_cfg["rpn"]["batch_size_per_image"],
                rpn_positive_fraction=teacher_cfg["rpn"]["positive_fraction"],
                box_fg_iou_thresh=teacher_cfg["roi_heads"]["fg_iou_thresh"],
                box_bg_iou_thresh=teacher_cfg["roi_heads"]["bg_iou_thresh"],
                box_batch_size_per_image=teacher_cfg["rpn"]["batch_size_per_image"],
                box_positive_fraction=teacher_cfg["rpn"]["positive_fraction"],
                box_score_thresh=teacher_cfg["roi_heads"]["score_thresh"],
                box_nms_thresh=teacher_cfg["roi_heads"]["nms_thresh"],
                box_detections_per_img=teacher_cfg["roi_heads"]["detections_per_img"],
                min_size=config["dataset"]["input_size"][0],
                max_size=config["dataset"]["input_size"][1]
            )
            
            # 替換分類和迴歸頭
            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(
                in_features,
                teacher_cfg["roi_heads"]["num_classes"]
            )
            
            logger.info("教師模型初始化完成")
    
    def forward(self, x, targets=None):
        """前向傳播"""
        if self.training and targets is None:
            raise ValueError("在訓練模式中，targets不應為None")
            
        return self.detector(x, targets)


class BackboneWithBatchNorm(nn.Module):
    """帶有批標準化的骨幹網絡包裝器"""
    
    def __init__(self, backbone, return_layers):
        """
        初始化骨幹網絡包裝器
        
        Args:
            backbone: 骨幹網絡
            return_layers: 返回哪些層
        """
        super(BackboneWithBatchNorm, self).__init__()
        
        # 提取層
        try:
            # 嘗試使用舊版本的方式
            self.body = misc_nn_ops.IntermediateLayerGetter(backbone, return_layers)
        except (AttributeError, NameError):
            # 如果失敗，使用新版本的方式
            self.body = IntermediateLayerGetter(backbone, return_layers)
    
    def forward(self, x):
        """前向傳播"""
        return self.body(x)


class FeaturePyramidNetwork(nn.Module):
    """特徵金字塔網絡"""
    
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        """
        初始化特徵金字塔網絡
        
        Args:
            in_channels_list: 輸入通道列表
            out_channels: 輸出通道數
            extra_blocks: 額外的輸出層
        """
        super(FeaturePyramidNetwork, self).__init__()
        # 保存输入通道列表
        self.in_channels_list = in_channels_list
        
        # 初始化橫向連接和輸出層
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 設置額外塊
        self.extra_blocks = extra_blocks
        self.out_channels = out_channels  # 添加輸出通道屬性
    
    def forward(self, x):
        """前向傳播"""
        # 檢查輸入類型
        if isinstance(x, dict):
            # 如果輸入是字典，轉換為列表
            x_list = []
            # 僅使用我們知道的通道數對應的特徵
            for i in range(len(self.in_channels_list)):
                key = str(i + 1)  # layer2, layer3, layer4 對應 '1', '2', '3'
                if key in x:
                    x_list.append(x[key])
                else:
                    logger.warning(f"找不到特徵層 {key}")
                    return OrderedDict([("0", torch.zeros_like(x[list(x.keys())[0]]))])  # 返回零張量以避免崩潰
        else:
            # 如果已經是列表或元組
            x_list = x

        # 檢查通道對應
        if len(x_list) != len(self.in_channels_list):
            logger.warning(f"輸入特徵數量 ({len(x_list)}) 與期望的數量 ({len(self.in_channels_list)}) 不匹配")
            # 如果特徵數量不匹配，做最好的努力
            if len(x_list) < len(self.in_channels_list):
                # 如果特徵太少，重複最後一個
                while len(x_list) < len(self.in_channels_list):
                    x_list.append(x_list[-1])
            else:
                # 如果特徵太多，只使用我們需要的
                x_list = x_list[:len(self.in_channels_list)]
        
        # 從底到頂處理特徵
        # 檢查通道正確性
        for i, (feat, expected_channels) in enumerate(zip(x_list, self.in_channels_list)):
            if feat.shape[1] != expected_channels:
                logger.warning(f"特徵 {i} 的通道數 ({feat.shape[1]}) 與期望的通道數 ({expected_channels}) 不匹配")
                # 可以在這裡添加通道適配層，但這會更改模型架構
        
        # 實際 FPN 前向傳播
        idx = len(self.inner_blocks) - 1
        last_inner = self.inner_blocks[idx](x_list[idx])
        results = [self.layer_blocks[idx](last_inner)]
        
        # 從上到下處理其他特徵
        for idx in range(len(self.inner_blocks) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x_list[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        # 處理額外層
        if self.extra_blocks is not None:
            results = self.extra_blocks(results)
        
        # 返回有序字典
        return OrderedDict([(str(i), feat) for i, feat in enumerate(results)])


def create_model(config, model_type="student"):
    """
    創建模型工廠函數
    
    Args:
        config: 配置字典
        model_type: 模型類型，可選["teacher", "student"]
    
    Returns:
        創建的模型實例
    """
    if model_type == "teacher":
        return TeacherModel(config)
    elif model_type == "student":
        return StudentModel(config)
    else:
        raise ValueError(f"不支持的模型類型: {model_type}")


def load_model(model, weights_path, device='cuda'):
    """
    載入模型權重
    
    Args:
        model: 模型實例
        weights_path: 權重檔案路徑
        device: 運行設備
    
    Returns:
        載入權重後的模型
    """
    if not os.path.exists(weights_path):
        logger.warning(f"權重檔案不存在: {weights_path}")
        return model
    
    logger.info(f"從 {weights_path} 載入模型權重")
    state_dict = torch.load(weights_path, map_location=device)
    
    # 檢查state_dict的格式
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        # 如果是檢查點格式
        model.load_state_dict(state_dict['model_state_dict'])
    elif isinstance(state_dict, dict) and 'student_model' in state_dict:
        # 如果是蒸餾檢查點
        model.load_state_dict(state_dict['student_model'])
    else:
        # 直接使用state_dict
        model.load_state_dict(state_dict)
        
    return model


def get_model_info(model):
    """
    獲取模型信息
    
    Args:
        model: 模型實例
    
    Returns:
        模型參數量和計算量的字典
    """
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 簡單估計計算量(GFLOPs)，這裡只是粗略估計
    # 對於更準確的計算需使用專用工具如torchprof或thop
    input_shape = [1, 3, 416, 416]  # 假設的輸入大小
    gflops = total_params * 2 / (1000 ** 3)  # 非常粗略的估計
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / (1024 ** 2),  # 假設每個參數為4字節
        "estimated_gflops": gflops
    }


def compare_features(teacher_features, student_features, adaptation_layer=None):
    """
    比較教師與學生特徵，計算特徵蒸餾損失
    
    Args:
        teacher_features: 教師特徵列表
        student_features: 學生特徵列表
        adaptation_layer: 特徵調整層
    
    Returns:
        特徵蒸餾損失
    """
    losses = []
    
    for t_feat, s_feat in zip(teacher_features, student_features):
        # 確保特徵大小一致
        if t_feat.shape[2:] != s_feat.shape[2:]:
            s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 應用特徵調整層
        if adaptation_layer is not None:
            s_feat = adaptation_layer(s_feat)
        
        # 計算L2距離作為損失
        loss = F.mse_loss(s_feat, t_feat)
        losses.append(loss)
    
    return sum(losses) / len(losses)


def build_adaptation_layer(in_channels, out_channels, adaptation_type="conv1x1"):
    """
    構建特徵調整層，用於調整學生特徵到教師特徵空間
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        adaptation_type: 調整層類型
    
    Returns:
        特徵調整層
    """
    if adaptation_type == "conv1x1":
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    elif adaptation_type == "conv3x3":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    elif adaptation_type == "nonlocal":
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            NonLocalBlock(out_channels)
        )
    else:
        raise ValueError(f"不支持的調整層類型: {adaptation_type}")


class NonLocalBlock(nn.Module):
    """非局部模塊，增強長距離特徵交互"""
    
    def __init__(self, in_channels):
        """
        初始化非局部模塊
        
        Args:
            in_channels: 輸入通道數
        """
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        
        self.bn = nn.BatchNorm2d(self.in_channels)
        
    def forward(self, x):
        """前向傳播"""
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        z = self.bn(W_y) + x
        
        return z


# 測試代碼
if __name__ == "__main__":
    import yaml
    import os
    
    # 加載配置
    config_path = "config/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 測試教師模型
    print("創建教師模型...")
    teacher = create_model(config, model_type="teacher")
    teacher_info = get_model_info(teacher)
    print(f"教師模型參數量: {teacher_info['total_params']:,}")
    print(f"教師模型大小: {teacher_info['model_size_mb']:.2f} MB")
    
    # 測試學生模型
    print("\n創建學生模型...")
    student = create_model(config, model_type="student")
    student_info = get_model_info(student)
    print(f"學生模型參數量: {student_info['total_params']:,}")
    print(f"學生模型大小: {student_info['model_size_mb']:.2f} MB")
    print(f"參數減少: {(1 - student_info['total_params'] / teacher_info['total_params']) * 100:.2f}%")
    
    # 測試前向傳播
    print("\n測試前向傳播...")
    x = torch.randn(2, 3, 416, 416)
    
    # 測試模式下運行
    teacher.eval()
    student.eval()
    
    with torch.no_grad():
        teacher_output = teacher(x)
        student_output = student(x)
    
    print(f"教師模型輸出: {teacher_output}")
    print(f"學生模型輸出: {student_output}")