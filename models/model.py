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
from torchvision.ops import misc as misc_nn_ops
from collections import OrderedDict
import logging
import math

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
        
        # 橫向連接層
        self.inner_blocks = nn.ModuleList()
        # 層間連接層
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            # 橫向連接使用1x1卷積降維
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
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
    
    def forward(self, x):
        """前向傳播"""
        # 從底到頂處理特徵
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        # 從上到下處理其他特徵
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        # 處理額外層
        if self.extra_blocks is not None:
            results.extend(self.extra_blocks(x, results))
            
        return results


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
    
    def forward(self, x, y):
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
            self.backbone = mobilenet_v3_small(pretrained=student_cfg["pretrained"])
            
            # 獲取特徵層輸出通道數
            backbone_out_channels = [
                16,  # features.3 - stage1
                24,  # features.6 - stage2
                40,  # features.9 - stage3
                80,  # features.12 - stage4
                576  # features.15 - stage5
            ]
            
            # 實現雙分支結構
            if student_cfg["dual_branch"]["enabled"]:
                # 全局分支
                if student_cfg["dual_branch"]["shared_backbone"]:
                    self.global_backbone = self.backbone
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
                
                # FPN通道設置
                global_fpn_channels = student_cfg["dual_branch"]["global_branch"]["fpn_channels"]
                local_fpn_channels = student_cfg["dual_branch"]["local_branch"]["fpn_channels"]
            else:
                # 單分支結構
                self.global_backbone = self.backbone
                self.global_attention = None
                self.local_attentions = None
                global_fpn_channels = student_cfg["neck"]["out_channels"]
                local_fpn_channels = None
            
            # 創建特徵金字塔網絡
            in_channels_list = backbone_out_channels[-3:]  # 使用最後三層特徵
            
            # 全局FPN
            if student_cfg["neck"]["extra_blocks"] == "lastlevel_p6p7":
                self.global_fpn_extra_blocks = LastLevelP6P7(
                    backbone_out_channels[-1], 
                    global_fpn_channels
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
                if student_cfg["neck"]["extra_blocks"] == "lastlevel_p6p7":
                    self.local_fpn_extra_blocks = LastLevelP6P7(
                        backbone_out_channels[-1], 
                        local_fpn_channels
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
            
            # 合併全局和局部特徵的層
            if student_cfg["dual_branch"]["enabled"]:
                self.fusion_layer = nn.Conv2d(
                    global_fpn_channels + local_fpn_channels,
                    global_fpn_channels,
                    kernel_size=1,
                    bias=False
                )
                self.fusion_norm = nn.BatchNorm2d(global_fpn_channels)
                self.fusion_act = nn.ReLU(inplace=True)
            else:
                self.fusion_layer = None
            
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
                backbone=BackboneWithFPN(self),
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
        for name, module in self.backbone.features._modules.items():
            x = module(x)
            if int(name) in [3, 6, 9, 12, 15]:  # 保存中間特徵
                features.append(x)
        
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
        
        if local_features is not None and self.local_fpn is not None:
            local_fpn_features = self.local_fpn(local_features[-3:])
            
            # 融合全局和局部特徵
            fused_features = []
            for gf, lf in zip(global_fpn_features, local_fpn_features):
                # 確保特徵大小一致
                if gf.shape[2:] != lf.shape[2:]:
                    lf = F.interpolate(lf, size=gf.shape[2:], mode='bilinear', align_corners=False)
                
                # 拼接並融合
                fused = torch.cat([gf, lf], dim=1)
                fused = self.fusion_layer(fused)
                fused = self.fusion_norm(fused)
                fused = self.fusion_act(fused)
                fused_features.append(fused)
            
            return fused_features
        else:
            return global_fpn_features
    
    def forward(self, x, targets=None):
        """前向傳播"""
        if self.training and targets is None:
            raise ValueError("在訓練模式中，targets不應為None")
            
        return self.detector(x, targets)


class BackboneWithFPN(nn.Module):
    """用於包裝學生模型作為FasterRCNN的骨幹網絡"""
    
    def __init__(self, student_model):
        """
        初始化骨幹網絡包裝器
        
        Args:
            student_model: 學生模型實例
        """
        super(BackboneWithFPN, self).__init__()
        self.student_model = student_model
        self.out_channels = self.student_model.config["student"]["dual_branch"]["global_branch"]["fpn_channels"] \
            if self.student_model.config["student"]["dual_branch"]["enabled"] \
            else self.student_model.config["student"]["neck"]["out_channels"]
    
    def forward(self, x):
        """前向傳播"""
        features = self.student_model.extract_features(x)
        return {str(i): feat for i, feat in enumerate(features)}


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
            self.backbone = BackboneWithBatchNorm(backbone, return_layers)
            
            # 特徵金字塔網絡
            in_channels_list = [backbone_out_channels[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
            out_channels = teacher_cfg["fpn"]["out_channels"]
            
            if teacher_cfg["fpn"]["extra_blocks"] == "lastlevel_maxpool":
                extra_blocks = misc_nn_ops.LastLevelMaxPool()
            else:
                extra_blocks = None
            
            self.fpn = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "fpn",
                            FeaturePyramidNetwork(
                                in_channels_list=in_channels_list[-3:],  # 使用最後三層特徵
                                out_channels=out_channels,
                                extra_blocks=extra_blocks,
                            )
                        )
                    ]
                )
            )
            
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
                backbone=self,
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
        self.body = misc_nn_ops.IntermediateLayerGetter(backbone, return_layers)
    
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
    
    def forward(self, x):
        """前向傳播"""
        # 從底到頂處理特徵
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        # 從上到下處理其他特徵
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
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