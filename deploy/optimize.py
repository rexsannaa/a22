#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
optimize.py - PCB缺陷檢測模型優化模組
整合模型壓縮、量化、剪枝和部署優化功能，提供一站式優化解決方案。
主要功能:
1. 網絡量化 - 支援INT8/FP16精度轉換
2. 通道剪枝 - 移除低貢獻度通道
3. 知識蒸餾後微調 - 確保精度損失最小化
4. 模型圖優化 - 合併操作、消除冗餘節點
5. 跨平台部署支援 - 支援不同邊緣設備
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
import time
import copy
import onnx
import onnxruntime as ort
from collections import OrderedDict

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """PCB缺陷檢測模型優化器，提供多種模型壓縮與優化方法"""
    
    def __init__(self, config, model, device="cuda"):
        """
        初始化模型優化器
        
        Args:
            config: 配置字典
            model: 待優化模型
            device: 計算設備
        """
        self.config = config
        self.opt_config = config["optimization"]
        self.model = model.to(device)
        self.device = device
        
        logger.info("模型優化器初始化完成")
    
    def optimize(self, dataloader=None):
        """
        執行完整優化流程
        
        Args:
            dataloader: 校準資料加載器
            
        Returns:
            優化後模型
        """
        logger.info("開始模型優化流程...")
        model = self.model
        
        # 1. 先執行剪枝（如果啟用）
        if self.opt_config["pruning"]["enabled"]:
            logger.info("執行通道剪枝優化...")
            model = self.prune_channels(model, dataloader)
        
        # 2. 進行後訓練量化（如果啟用）
        if self.opt_config["quantization"]["enabled"]:
            logger.info("執行後訓練量化...")
            model = self.quantize_model(model, dataloader)
        
        # 3. 執行模型圖優化（如果啟用）
        if self.opt_config["graph_optimization"]["enabled"]:
            logger.info("執行計算圖優化...")
            model = self.optimize_graph(model)
        
        # 4. 如果啟用了知識蒸餾後微調
        if self.opt_config["post_distill"]["fine_tuning_epochs"] > 0 and dataloader:
            logger.info("執行知識蒸餾後微調...")
            model = self.fine_tune_model(model, dataloader)
        
        # 5. 最後轉換為目標設備優化模型
        target_device = self.opt_config["target_device"]
        logger.info(f"為目標設備 {target_device} 優化模型...")
        model = self.optimize_for_device(model, target_device)
        
        logger.info("模型優化流程完成")
        return model

    def prune_channels(self, model, dataloader=None):
        """
        執行通道剪枝優化
        
        Args:
            model: 待剪枝模型
            dataloader: 評估資料加載器
            
        Returns:
            剪枝後模型
        """
        # 獲取剪枝配置
        prune_config = self.opt_config["pruning"]
        target_sparsity = prune_config["target_sparsity"]
        method = prune_config["method"]
        granularity = prune_config["granularity"]
        exclude_layers = prune_config["exclude_layers"]
        
        # 複製模型防止對原模型修改
        pruned_model = copy.deepcopy(model)
        
        # 獲取所有卷積層
        convs_to_prune = []
        for name, module in pruned_model.named_modules():
            # 跳過被排除的層
            if any(exclude_name in name for exclude_name in exclude_layers):
                continue
                
            # 找出所有卷積層
            if isinstance(module, nn.Conv2d):
                convs_to_prune.append((name, module))
        
        # 按照不同方法執行剪枝
        if method == "magnitude":
            self._prune_by_magnitude(pruned_model, convs_to_prune, target_sparsity)
        elif method == "l1norm":
            self._prune_by_l1norm(pruned_model, convs_to_prune, target_sparsity)
        else:
            logger.warning(f"不支持的剪枝方法: {method}，使用預設的magnitude方法")
            self._prune_by_magnitude(pruned_model, convs_to_prune, target_sparsity)
        
        # 如果提供了dataloader，評估剪枝後的性能
        if dataloader:
            self._evaluate_pruned_model(pruned_model, dataloader)
        
        # 移除剪枝參數，創建新模型結構
        pruned_model = self._remove_pruning_params(pruned_model)
        
        logger.info(f"通道剪枝完成，目標稀疏度: {target_sparsity:.2f}")
        
        return pruned_model
    
    def _prune_by_magnitude(self, model, convs_to_prune, target_sparsity):
        """使用權重幅度剪枝方法"""
        # 收集所有權重的絕對值
        all_weights = []
        for _, module in convs_to_prune:
            weights = module.weight.data.abs().view(-1).cpu().numpy()
            all_weights.extend(weights)
        
        # 找出閾值
        all_weights = sorted(all_weights)
        threshold_idx = int(len(all_weights) * target_sparsity)
        if threshold_idx >= len(all_weights):
            threshold_idx = len(all_weights) - 1
        threshold = all_weights[threshold_idx]
        
        # 設置掩碼
        for name, module in convs_to_prune:
            mask = module.weight.data.abs() > threshold
            module.weight.data.mul_(mask)
            
            # 記錄剪枝率
            pruned = mask.numel() - mask.sum().item()
            logger.info(f"層 {name} 剪枝率: {pruned / mask.numel():.2f}")
    
    def _prune_by_l1norm(self, model, convs_to_prune, target_sparsity):
        """使用L1範數剪枝方法"""
        for name, module in convs_to_prune:
            # 計算每個輸出通道的L1範數
            l1_norm = module.weight.data.abs().sum(dim=(1, 2, 3))
            
            # 確定保留多少通道
            num_channels = l1_norm.size(0)
            num_prune = int(num_channels * target_sparsity)
            
            # 找出L1範數最小的通道
            _, indices = torch.topk(l1_norm, k=num_channels - num_prune, largest=True)
            mask = torch.zeros_like(l1_norm, dtype=torch.bool)
            mask[indices] = True
            
            # 應用掩碼到權重
            module.weight.data.mul_(mask.view(-1, 1, 1, 1))
            
            # 如果有偏置，也應用掩碼
            if module.bias is not None:
                module.bias.data.mul_(mask)
            
            # 記錄剪枝率
            logger.info(f"層 {name} 剪枝率: {num_prune / num_channels:.2f}")
    
    def _evaluate_pruned_model(self, model, dataloader):
        """評估剪枝後模型性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets, _ in dataloader:
                # 處理批次數據
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向傳播
                outputs = model(images)
                
                # 簡單評估（實際應用時應計算mAP等精確指標）
                for output, target in zip(outputs, targets):
                    # 假設IoU > 0.5為正確檢測
                    if len(output["boxes"]) > 0 and len(target["boxes"]) > 0:
                        # 計算IoU
                        iou = self._box_iou(output["boxes"], target["boxes"])
                        max_iou, _ = torch.max(iou, dim=1)
                        correct += (max_iou > 0.5).sum().item()
                        total += len(output["boxes"])
        
        accuracy = correct / max(total, 1)
        logger.info(f"剪枝後模型準確率評估: {accuracy:.4f}")
    
    def _box_iou(self, box1, box2):
        """計算兩組邊界框的IoU"""
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        
        area1 = box_area(box1)
        area2 = box_area(box2)
        
        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / union
        return iou
    
    def _remove_pruning_params(self, model):
        """移除被剪枝的參數，創建新結構"""
        # 此處簡化處理，實際應重建模型結構
        # 只移除零權重但保持通道數不變
        return model

    def quantize_model(self, model, dataloader=None):
        """
        執行模型量化
        
        Args:
            model: 待量化模型
            dataloader: 校準資料加載器
            
        Returns:
            量化後模型
        """
        # 獲取量化配置
        quant_config = self.opt_config["quantization"]
        method = quant_config["method"]
        precision = quant_config["precision"]
        per_channel = quant_config["per_channel"]
        calibration_samples = quant_config["calibration_samples"]
        
        # 使用PyTorch內置的量化工具
        if method == "post_training":
            # 使用靜態/後訓練量化
            quantized_model = self._post_training_quantize(
                model, dataloader, precision, per_channel, calibration_samples
            )
        elif method == "dynamic":
            # 使用動態量化
            quantized_model = self._dynamic_quantize(
                model, precision, per_channel
            )
        elif method == "qat":
            # 量化感知訓練（這需要更複雜的訓練循環，此處簡化）
            quantized_model = self._quantization_aware_training(
                model, dataloader, precision, per_channel
            )
        else:
            logger.warning(f"不支持的量化方法: {method}，使用預設的post_training方法")
            quantized_model = self._post_training_quantize(
                model, dataloader, precision, per_channel, calibration_samples
            )
        
        # 測量量化前後性能差異（如有校準資料）
        if dataloader:
            self._measure_quantization_impact(model, quantized_model, dataloader)
        
        logger.info(f"模型量化完成，使用 {precision} 精度")
        
        return quantized_model
    
    def _post_training_quantize(self, model, dataloader, precision, per_channel, calibration_samples):
        """使用靜態/後訓練量化"""
        # 準備量化配置
        qconfig = torch.quantization.get_default_qconfig('fbgemm' if precision == 'int8' else 'qnnpack')
        if not per_channel:
            qconfig = torch.quantization.default_per_tensor_qconfig
        
        # 創建量化模型
        quantized_model = copy.deepcopy(model)
        quantized_model.eval()
        
        # 融合操作（如conv+bn+relu）
        quantized_model = torch.quantization.fuse_modules(quantized_model, [
            ['conv', 'bn', 'relu'],
            ['conv', 'relu'],
        ], inplace=True)
        
        # 準備量化
        torch.quantization.prepare(quantized_model, inplace=True)
        
        # 校準（如有資料加載器）
        if dataloader:
            # 僅使用指定數量的樣本進行校準
            samples_processed = 0
            
            with torch.no_grad():
                for images, _, _ in dataloader:
                    if samples_processed >= calibration_samples:
                        break
                    
                    # 將數據移到設備上
                    images = [img.to(self.device) for img in images]
                    
                    # 前向傳播進行校準
                    _ = quantized_model(images)
                    
                    samples_processed += len(images)
        
        # 轉換為量化模型
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        
        return quantized_model
    
    def _dynamic_quantize(self, model, precision, per_channel):
        """使用動態量化"""
        # 簡單起見，使用torch的動態量化API
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}, 
            dtype=torch.qint8 if precision == 'int8' else torch.float16
        )
        
        return quantized_model
    
    def _quantization_aware_training(self, model, dataloader, precision, per_channel):
        """量化感知訓練"""
        # 量化感知訓練需要額外的訓練循環，此處簡化
        # 實際實現需要重新訓練模型
        logger.warning("量化感知訓練需要完整訓練循環，此處僅返回原模型")
        return model
    
    def _measure_quantization_impact(self, original_model, quantized_model, dataloader):
        """測量量化前後性能差異"""
        # 測量推理時間
        batch_size = dataloader.batch_size or 1
        input_size = (416, 416)  # 假設的輸入尺寸
        
        # 原始模型耗時
        orig_time = self._measure_inference_time(original_model, input_size)
        
        # 量化後模型耗時
        quant_time = self._measure_inference_time(quantized_model, input_size)
        
        # 計算加速比
        speedup = orig_time / max(quant_time, 1e-6)
        
        # 測量精度差異（簡化測量）
        orig_acc = self._quick_accuracy_test(original_model, dataloader)
        quant_acc = self._quick_accuracy_test(quantized_model, dataloader)
        
        # 輸出結果
        logger.info(f"量化前模型推理時間: {orig_time:.2f} ms")
        logger.info(f"量化後模型推理時間: {quant_time:.2f} ms")
        logger.info(f"加速比: {speedup:.2f}x")
        logger.info(f"量化前準確率: {orig_acc:.4f}")
        logger.info(f"量化後準確率: {quant_acc:.4f}")
        logger.info(f"準確率變化: {quant_acc - orig_acc:.4f}")
    
    def _measure_inference_time(self, model, input_size, iterations=50):
        """測量模型推理時間"""
        model.eval()
        
        # 創建測試輸入
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)
        
        # 預熱
        with torch.no_grad():
            for _ in range(10):
                _ = model([dummy_input])
        
        # 計時
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model([dummy_input])
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        # 計算平均時間 (毫秒)
        avg_time = (end_time - start_time) * 1000 / iterations
        
        return avg_time
    
    def _quick_accuracy_test(self, model, dataloader, max_samples=100):
        """快速評估模型準確率"""
        model.eval()
        correct = 0
        total = 0
        samples_processed = 0
        
        with torch.no_grad():
            for images, targets, _ in dataloader:
                if samples_processed >= max_samples:
                    break
                
                # 處理批次數據
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # 前向傳播
                outputs = model(images)
                
                # 簡單評估（實際應用時應計算mAP等精確指標）
                for output, target in zip(outputs, targets):
                    # 假設IoU > 0.5為正確檢測
                    if len(output["boxes"]) > 0 and len(target["boxes"]) > 0:
                        # 計算IoU
                        iou = self._box_iou(output["boxes"], target["boxes"])
                        max_iou, _ = torch.max(iou, dim=1)
                        correct += (max_iou > 0.5).sum().item()
                        total += len(output["boxes"])
                
                samples_processed += len(images)
        
        accuracy = correct / max(total, 1)
        return accuracy

    def optimize_graph(self, model):
        """
        執行計算圖優化
        
        Args:
            model: 待優化模型
            
        Returns:
            優化後模型
        """
        # 獲取圖優化配置
        graph_config = self.opt_config["graph_optimization"]
        fold_constants = graph_config["fold_constants"]
        eliminate_identity = graph_config["eliminate_identity"]
        
        # 先導出為ONNX格式
        dummy_input = torch.randn(1, 3, 416, 416).to(self.device)
        onnx_path = os.path.join(self.config["paths"]["output"], "temp_model.onnx")
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        # 導出模型
        with torch.no_grad():
            torch.onnx.export(
                model,
                [dummy_input],
                onnx_path,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
        
        # 使用ONNX Runtime進行優化
        optimized_onnx_path = os.path.join(self.config["paths"]["output"], "optimized_model.onnx")
        self._optimize_onnx_graph(onnx_path, optimized_onnx_path, fold_constants, eliminate_identity)
        
        # 導入優化後的模型
        ort_session = ort.InferenceSession(optimized_onnx_path)
        
        # 創建一個封裝ONNX Runtime的模型
        optimized_model = ONNXRuntimeModel(ort_session, model)
        
        # 清理臨時文件
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        logger.info("計算圖優化完成")
        
        return optimized_model
    
    def _optimize_onnx_graph(self, input_path, output_path, fold_constants=True, eliminate_identity=True):
        """使用ONNX優化器優化模型圖"""
        # 加載模型
        model = onnx.load(input_path)
        
        # 使用ONNX優化器
        passes = []
        if fold_constants:
            passes.append("fuse_bn_into_conv")
            passes.append("eliminate_unused_initializer")
        if eliminate_identity:
            passes.append("eliminate_identity")
            passes.append("eliminate_nop_dropout")
        
        # 確保有效的優化通道
        if passes:
            from onnxoptimizer import optimize
            optimized_model = optimize(model, passes)
            
            # 保存優化後的模型
            onnx.save(optimized_model, output_path)
            logger.info(f"ONNX圖已優化並保存至: {output_path}")
        else:
            # 如果沒有指定優化通道，只保存原模型
            onnx.save(model, output_path)
            logger.info(f"ONNX圖未優化，原模型已保存至: {output_path}")

    def fine_tune_model(self, model, dataloader):
        """
        知識蒸餾後微調
        
        Args:
            model: 待微調模型
            dataloader: 訓練資料加載器
            
        Returns:
            微調後模型
        """
        # 獲取微調配置
        ft_config = self.opt_config["post_distill"]
        epochs = ft_config["fine_tuning_epochs"]
        learning_rate = ft_config["learning_rate"]
        freeze_bn = ft_config["frozen_batch_norm"]
        
        # 切換到訓練模式
        model.train()
        
        # 如果設置了凍結BN，凍結所有BN層
        if freeze_bn:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        
        # 創建優化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )
        
        # 微調循環
        logger.info(f"開始知識蒸餾後微調，總輪次: {epochs}")
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            
            # 使用tqdm顯示進度
            with tqdm(dataloader, desc=f"微調輪次 {epoch+1}/{epochs}") as t:
                for images, targets, _ in t:
                    # 將數據移到設備上
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # 清除梯度
                    optimizer.zero_grad()
                    
                    # 前向傳播
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    
                    # 反向傳播和優化
                    loss.backward()
                    optimizer.step()
                    
                    # 更新統計
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # 更新進度條
                    t.set_postfix(loss=loss.item())
            
            # 輸出輪次結果
            avg_loss = total_loss / max(batch_count, 1)
            logger.info(f"微調輪次 {epoch+1}/{epochs} 完成，平均損失: {avg_loss:.4f}")
        
        # 切換回評估模式
        model.eval()
        
        logger.info("知識蒸餾後微調完成")
        
        return model

    def optimize_for_device(self, model, target_device):
        """
        針對特定設備優化模型
        
        Args:
            model: 待優化模型
            target_device: 目標設備
            
        Returns:
            優化後模型
        """
        # 根據目標設備應用特定優化
        if target_device == "jetson_nano":
            optimized_model = self._optimize_for_jetson(model)
        elif target_device == "raspberry_pi":
            optimized_model = self._optimize_for_raspberry_pi(model)
        else:
            logger.warning(f"未知目標設備: {target_device}，返回原模型")
            optimized_model = model
        
        logger.info(f"模型已為 {target_device} 優化")
        
        return optimized_model
    
    def _optimize_for_jetson(self, model):
        """針對Jetson Nano優化"""
        # 轉換為TensorRT引擎
        # 這裡只提供示例，實際應用可能需要額外的庫
        logger.info("將模型轉換為TensorRT格式")
        
        # 在實際應用中，這裡應該有TensorRT相關代碼
        # 因為依賴關係複雜，此處返回原模型
        return model
    
    def _optimize_for_raspberry_pi(self, model):
        """針對樹莓派優化"""
        # 針對ARM架構優化
        logger.info("優化模型以適應ARM處理器")
        
        # 在Raspberry Pi上，可能需要進一步量化或使用特殊加速庫
        # 此處簡化處理，返回原模型
        return model


class ONNXRuntimeModel(nn.Module):
    """封裝ONNX Runtime的模型類"""
    
    def __init__(self, ort_session, original_model):
        """
        初始化ONNX Runtime模型
        
        Args:
            ort_session: ONNX Runtime會話
            original_model: 原始PyTorch模型
        """
        super(ONNXRuntimeModel, self).__init__()
        self.ort_session = ort_session
        self.original_model = original_model
    
    def forward(self, x, targets=None):
        """
        前向傳播
        
        Args:
            x: 輸入張量
            targets: 目標標註（訓練時使用）
            
        Returns:
            如果有targets返回損失字典，否則返回預測結果
        """
        # 訓練模式使用原始模型
        if self.training and targets is not None:
            return self.original_model(x, targets)
        
        # 推理模式使用ONNX Runtime
        if not isinstance(x, list):
            x = [x]
        
        # 準備輸入
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: x[0].cpu().numpy()}
        
        # 執行推理
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        # 將輸出轉換為與原始模型相同的格式
        # 注意：這部分取決於模型輸出的具體結構
        # 實際應用中需要根據模型輸出定製
        # 此處為簡化示例
        output = {
            "boxes": torch.tensor(ort_outputs[0]),
            "labels": torch.tensor(ort_outputs[1]),
            "scores": torch.tensor(ort_outputs[2])
        }
        
        return [output]


def export_optimized_model(model, format_type, save_path, input_size=(416, 416)):
    """
    導出優化後的模型為部署格式
    
    Args:
        model: 優化後的模型
        format_type: 導出格式 (onnx, torchscript, tflite)
        save_path: 保存路徑
        input_size: 輸入尺寸
        
    Returns:
        導出文件路徑
    """
    # 確保路徑存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 切換模型為評估模式
    model.eval()
    
    # 創建示例輸入
    dummy_input = [torch.randn(1, 3, *input_size)]
    
    # 根據格式導出
    if format_type.lower() == "onnx":
        # 導出為ONNX
        logger.info(f"導出模型為ONNX格式: {save_path}")
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=12,
            input_names=["input"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "boxes": {0: "batch_size"},
                "labels": {0: "batch_size"},
                "scores": {0: "batch_size"}
            }
        )
        
        # 驗證導出的模型
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        
    elif format_type.lower() == "torchscript":
        # 導出為TorchScript
        logger.info(f"導出模型為TorchScript格式: {save_path}")
        script_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(script_model, save_path)
        
    elif format_type.lower() == "tflite":
        # 導出為TF Lite
        # 需要額外的轉換流程，通常需要先轉換為ONNX，然後使用tf-onnx轉換為TF
        logger.warning("TF Lite導出需要額外轉換，此處僅導出ONNX中間格式")
        temp_onnx = os.path.join(os.path.dirname(save_path), "temp.onnx")
        torch.onnx.export(model, dummy_input, temp_onnx)
        
        # 在實際應用中，這裡應該添加ONNX到TF Lite的轉換
        logger.info(f"需要另外將ONNX轉換為TF Lite: {temp_onnx} -> {save_path}")
        
    else:
        logger.error(f"不支持的導出格式: {format_type}")
        return None
    
    logger.info(f"模型成功導出為 {format_type} 格式")
    return save_path


def measure_model_performance(model, dataloader, device="cuda"):
    """
    測量模型性能指標
    
    Args:
        model: 待測量模型
        dataloader: 測試資料加載器
        device: 計算設備
        
    Returns:
        性能指標字典
    """
    # 切換到評估模式
    model.eval()
    
    # 初始化指標
    metrics = {
        "inference_time": 0.0,
        "fps": 0.0,
        "memory_usage": 0.0,
        "accuracy": 0.0
    }
    
    # 測量推理時間
    dummy_input = torch.randn(1, 3, 416, 416).to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    
    # 預熱
    with torch.no_grad():
        for _ in range(10):
            _ = model([dummy_input])
    
    # 計時
    iterations = 100
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model([dummy_input])
    
    torch.cuda.synchronize() if device == "cuda" else None
    end_time = time.time()
    
    # 計算平均時間和FPS
    avg_time = (end_time - start_time) * 1000 / iterations
    fps = 1000 / avg_time
    
    metrics["inference_time"] = avg_time
    metrics["fps"] = fps
    
    # 計算記憶體使用
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model([dummy_input])
        
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        metrics["memory_usage"] = memory_used
    
    # 計算準確率
    if dataloader:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets, _ in dataloader:
                # 處理批次數據
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # 前向傳播
                outputs = model(images)
                
                # 簡單評估（實際應用時應計算mAP等精確指標）
                for output, target in zip(outputs, targets):
                    if len(output["boxes"]) > 0 and len(target["boxes"]) > 0:
                        # 計算IoU
                        iou = box_iou(output["boxes"], target["boxes"])
                        max_iou, _ = torch.max(iou, dim=1)
                        correct += (max_iou > 0.5).sum().item()
                        total += len(output["boxes"])
        
        metrics["accuracy"] = correct / max(total, 1)
    
    return metrics


def box_iou(box1, box2):
    """計算兩組邊界框的IoU"""
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    
    area1 = box_area(box1)
    area2 = box_area(box2)
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def optimize_model_pipeline(config_path, model_path, output_dir=None, format_type="onnx"):
    """
    完整模型優化管道，從載入到導出
    
    Args:
        config_path: 配置文件路徑
        model_path: 模型權重路徑
        output_dir: 輸出目錄
        format_type: 導出格式
        
    Returns:
        優化後模型路徑
    """
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 設置輸出目錄
    if output_dir is None:
        output_dir = config["paths"]["output"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入數據集（用於校準和評估）
    from data.dataset import create_dataloaders
    dataloaders = create_dataloaders(config)
    calibration_loader = dataloaders["val"]
    
    # 載入模型
    from models.model import create_model, load_model
    model = create_model(config, model_type="student")
    model = load_model(model, model_path, device)
    
    # 創建優化器
    optimizer = ModelOptimizer(config, model, device)
    
    # 執行優化
    optimized_model = optimizer.optimize(calibration_loader)
    
    # 測量性能
    performance = measure_model_performance(optimized_model, dataloaders["test"], device)
    logger.info(f"優化後模型性能：\n" + 
                f"  推理時間: {performance['inference_time']:.2f} ms\n" +
                f"  FPS: {performance['fps']:.2f}\n" +
                f"  記憶體使用: {performance['memory_usage']:.2f} MB\n" +
                f"  準確率: {performance['accuracy']:.4f}")
    
    # 導出模型
    save_path = os.path.join(output_dir, f"optimized_model.{format_type}")
    export_path = export_optimized_model(optimized_model, format_type, save_path)
    
    return export_path


# 如果直接執行此腳本
if __name__ == "__main__":
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="PCB缺陷檢測模型優化工具")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路徑")
    parser.add_argument("--model", type=str, required=True, help="模型權重路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出目錄")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "tflite"], 
                        help="導出格式")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="運行設備")
    
    args = parser.parse_args()
    
    # 設置日誌級別
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 執行優化管道
    logger.info(f"開始優化模型: {args.model}")
    export_path = optimize_model_pipeline(args.config, args.model, args.output, args.format)
    
    logger.info(f"模型優化和導出完成: {export_path}")