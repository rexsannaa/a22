#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
optimize.py - PCB缺陷檢測模型優化模組
本模組整合了模型優化、部署相關功能，
提供一站式模型輕量化與效能提升解決方案。
主要特點:
1. 模型剪枝：通過結構化和非結構化剪枝減少模型參數
2. 模型量化：支援INT8/FP16量化以加速推理
3. 模型導出：支援ONNX、TorchScript等多種部署格式
4. 模型加速：針對邊緣設備的推理優化方案
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from pathlib import Path
import onnx
import copy
import time
from tqdm import tqdm

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """PCB缺陷檢測模型優化器"""
    
    def __init__(self, model, config=None):
        """
        初始化模型優化器
        
        參數:
            model: 原始模型
            config: 優化配置字典
        """
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimized_model = None
        
        # 優化相關參數
        self.prune_amount = self.config.get('prune_amount', 0.3)  # 剪枝比例(預設30%)
        self.prune_method = self.config.get('prune_method', 'l1')  # 剪枝方法
        self.quant_type = self.config.get('quant_type', 'static')  # 量化類型
        self.export_format = self.config.get('export_format', 'onnx')  # 導出格式
        
        # 初始化模型
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型優化器已初始化，設備: {self.device}")
    
    def apply_pruning(self):
        """應用模型剪枝"""
        logger.info(f"開始應用{self.prune_method}剪枝，剪枝比例: {self.prune_amount}")
        
        # 複製模型以保留原始版本
        pruned_model = copy.deepcopy(self.model)
        
        # 剪枝計數器
        pruned_params = 0
        total_params = 0
        
        # 遍歷模型層
        for name, module in pruned_model.named_modules():
            # 對卷積層和線性層應用剪枝
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 獲取參數數量
                param_size = module.weight.data.numel()
                total_params += param_size
                
                # 根據剪枝方法選擇剪枝函數
                if self.prune_method == 'l1':
                    prune.l1_unstructured(module, name='weight', amount=self.prune_amount)
                elif self.prune_method == 'random':
                    prune.random_unstructured(module, name='weight', amount=self.prune_amount)
                elif self.prune_method == 'structured':
                    # 對卷積層進行結構化剪枝(按輸出通道)
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(module, name='weight', amount=self.prune_amount, 
                                          n=2, dim=0)  # 按輸出通道剪枝
                
                # 將剪枝永久化
                prune.remove(module, 'weight')
                
                # 統計剪枝參數
                pruned_params += int(param_size * self.prune_amount)
                
        # 保存優化後的模型
        self.optimized_model = pruned_model
        
        logger.info(f"剪枝完成，總參數: {total_params:,}，剪枝參數: {pruned_params:,} ({pruned_params/total_params*100:.2f}%)")
        
        return self.optimized_model
    
    def apply_quantization(self, calibration_loader=None):
        """
        應用模型量化
        
        參數:
            calibration_loader: 校準資料載入器(用於靜態量化)
            
        回傳:
            quantized_model: 量化後的模型
        """
        logger.info(f"開始應用{self.quant_type}量化")
        
        # 使用已剪枝的模型(如果有)或原始模型
        model_to_quantize = self.optimized_model if self.optimized_model is not None else self.model
        model_to_quantize = copy.deepcopy(model_to_quantize)
        
        # 處理不同量化類型
        if self.quant_type == 'dynamic':
            # 動態量化(僅權重)
            try:
                # 將模型轉換為CPU (量化需要在CPU上進行)
                model_to_quantize.cpu()
                
                # 應用動態量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model_to_quantize,  # 模型
                    {nn.Linear, nn.Conv2d},  # 要量化的層類型
                    dtype=torch.qint8  # 量化數據類型
                )
                
                logger.info("動態量化完成")
                
            except Exception as e:
                logger.error(f"動態量化失敗: {e}")
                logger.info("使用替代方案：針對特定模塊進行量化")
                
                # 替代方案：手動量化特定模塊
                quantized_model = model_to_quantize
                
                # 為模型添加量化配置
                quantized_model.qconfig = torch.quantization.default_dynamic_qconfig
                torch.quantization.prepare_dynamic(quantized_model, inplace=True)
                torch.quantization.convert_dynamic(quantized_model, inplace=True)
                
        elif self.quant_type == 'static':
            # 靜態量化(權重和激活)
            if calibration_loader is None:
                logger.warning("靜態量化需要校準資料載入器，切換為動態量化")
                return self.apply_quantization()  # 回退到動態量化
            
            try:
                # 將模型轉換為CPU
                model_to_quantize.cpu()
                
                # 定義量化配置
                model_to_quantize.eval()
                model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                
                # 執行融合(例如Conv+BN+ReLU)
                model_to_quantize = torch.quantization.fuse_modules(model_to_quantize, 
                                                            [['conv', 'bn', 'relu']], 
                                                            inplace=False)
                
                # 準備量化
                model_prepared = torch.quantization.prepare(model_to_quantize)
                
                # 校準(使用一小部分校準資料)
                with torch.no_grad():
                    for i, (images, _) in enumerate(calibration_loader):
                        model_prepared(images)
                        # 使用少量批次進行校準
                        if i >= 10:
                            break
                            
                # 完成量化
                quantized_model = torch.quantization.convert(model_prepared)
                
                logger.info("靜態量化完成")
                
            except Exception as e:
                logger.error(f"靜態量化失敗: {e}")
                logger.info("回退到動態量化")
                return self.apply_quantization()  # 回退到動態量化
                
        elif self.quant_type == 'qat':
            # 量化感知訓練(需要重新訓練)
            logger.warning("量化感知訓練需要重新訓練模型，此功能尚未實現")
            logger.info("使用動態量化替代")
            return self.apply_quantization()  # 回退到動態量化
        
        else:
            logger.error(f"不支援的量化類型: {self.quant_type}")
            return model_to_quantize
        
        # 保存優化後的模型
        self.optimized_model = quantized_model
        
        return self.optimized_model
    
    def export_model(self, output_path=None, input_shape=(1, 3, 640, 640)):
        """
        將模型導出為不同格式
        
        參數:
            output_path: 輸出路徑
            input_shape: 輸入形狀
            
        回傳:
            output_path: 導出模型的路徑
        """
        # 使用已優化的模型(如果有)或原始模型
        model_to_export = self.optimized_model if self.optimized_model is not None else self.model
        model_to_export = copy.deepcopy(model_to_export)
        
        # 確定輸出路徑
        if output_path is None:
            output_dir = self.config.get('output_dir', 'outputs/weights')
            os.makedirs(output_dir, exist_ok=True)
            model_name = self.config.get('model_name', 'pcb_model')
            output_path = os.path.join(output_dir, f"{model_name}.{self.export_format}")
        
        # 根據不同格式進行導出
        if self.export_format == 'onnx':
            logger.info(f"導出模型為ONNX格式: {output_path}")
            
            try:
                # 創建示例輸入
                dummy_input = torch.randn(input_shape).to(self.device)
                
                # 確保模型處於評估模式
                model_to_export.eval()
                
                # 導出為ONNX
                torch.onnx.export(
                    model_to_export,               # 模型
                    dummy_input,                   # 示例輸入
                    output_path,                   # 輸出路徑
                    export_params=True,            # 存儲訓練好的參數權重
                    opset_version=12,              # ONNX版本
                    do_constant_folding=True,      # 是否執行常量折疊優化
                    input_names=['input'],         # 輸入節點名稱
                    output_names=['output'],       # 輸出節點名稱
                    dynamic_axes={                 # 動態軸(支援不同批次大小)
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 驗證ONNX模型
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                
                logger.info(f"ONNX模型已成功導出並驗證: {output_path}")
                
            except Exception as e:
                logger.error(f"ONNX導出失敗: {e}")
                
        elif self.export_format == 'torchscript':
            logger.info(f"導出模型為TorchScript格式: {output_path}")
            
            try:
                # 創建示例輸入
                dummy_input = torch.randn(input_shape).to(self.device)
                
                # 確保模型處於評估模式
                model_to_export.eval()
                
                # 追蹤並導出
                traced_model = torch.jit.trace(model_to_export, dummy_input)
                traced_model.save(output_path)
                
                logger.info(f"TorchScript模型已成功導出: {output_path}")
                
            except Exception as e:
                logger.error(f"TorchScript導出失敗: {e}")
                
        elif self.export_format == 'tensorrt':
            logger.warning("TensorRT導出需要額外的依賴，此功能尚未實現")
            logger.info("請先導出為ONNX，然後使用TensorRT工具轉換")
            
            # 修改為ONNX格式
            self.export_format = 'onnx'
            output_path = output_path.replace('.tensorrt', '.onnx')
            return self.export_model(output_path, input_shape)
            
        else:
            logger.error(f"不支援的導出格式: {self.export_format}")
            return None
        
        return output_path
    
    def profile_model(self, input_shape=(1, 3, 640, 640), num_runs=100):
        """
        分析模型性能
        
        參數:
            input_shape: 輸入形狀
            num_runs: 測試運行次數
            
        回傳:
            profile_results: 性能分析結果
        """
        logger.info(f"開始分析模型性能，運行次數: {num_runs}")
        
        # 使用已優化的模型(如果有)或原始模型
        model_to_profile = self.optimized_model if self.optimized_model is not None else self.model
        model_to_profile = model_to_profile.to(self.device)
        model_to_profile.eval()
        
        # 創建示例輸入
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 熱身運行
        with torch.no_grad():
            for _ in range(10):
                _ = model_to_profile(dummy_input)
                
        # 計時運行
        inference_times = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="性能分析"):
                # 開始計時
                start_time = time.time()
                
                # 前向傳播
                _ = model_to_profile(dummy_input)
                
                # 同步GPU
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                # 記錄時間
                inference_times.append(time.time() - start_time)
        
        # 計算統計數據
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        # 計算推理速度(FPS)
        fps = 1.0 / avg_time
        
        # 計算模型參數量
        total_params = sum(p.numel() for p in model_to_profile.parameters())
        total_params_m = total_params / 1e6  # 轉換為百萬
        
        # 嘗試計算MACs
        try:
            from thop import profile as thop_profile
            macs, _ = thop_profile(model_to_profile, inputs=(dummy_input,))
            macs_g = macs / 1e9  # 轉換為十億
            memory_mb = (macs * 4) / (1024 * 1024)  # 估算最大內存佔用(以MB為單位)
        except ImportError:
            logger.warning("無法計算MACs，請安裝thop套件")
            macs_g = "N/A"
            memory_mb = "N/A"
        
        # 彙總結果
        profile_results = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'fps': fps,
            'total_params': total_params,
            'total_params_m': total_params_m,
            'macs_g': macs_g,
            'memory_mb': memory_mb
        }
        
        # 輸出結果
        logger.info(f"性能分析結果:")
        logger.info(f"  平均推理時間: {avg_time*1000:.2f} ms")
        logger.info(f"  推理速度: {fps:.2f} FPS")
        logger.info(f"  模型參數量: {total_params_m:.2f} M")
        logger.info(f"  計算量: {macs_g if isinstance(macs_g, str) else f'{macs_g:.2f} G'}")
        logger.info(f"  估算記憶體需求: {memory_mb if isinstance(memory_mb, str) else f'{memory_mb:.2f} MB'}")
        
        return profile_results
    
    def optimize(self, calibration_loader=None, apply_pruning=True, apply_quantization=True):
        """
        一站式優化模型
        
        參數:
            calibration_loader: 校準資料載入器(用於靜態量化)
            apply_pruning: 是否應用剪枝
            apply_quantization: 是否應用量化
            
        回傳:
            optimized_model: 優化後的模型
        """
        logger.info("開始一站式模型優化")
        
        # 應用剪枝
        if apply_pruning:
            self.apply_pruning()
        
        # 應用量化
        if apply_quantization:
            self.apply_quantization(calibration_loader)
        
        return self.optimized_model

class YOLOOptimizer(ModelOptimizer):
    """YOLO模型特定的優化器"""
    
    def __init__(self, model, config=None):
        super(YOLOOptimizer, self).__init__(model, config)
        logger.info("初始化YOLO特定優化器")
        
    def apply_pruning(self):
        """針對YOLO架構的特定剪枝策略"""
        logger.info(f"應用YOLO特定剪枝策略")
        
        # 複製模型以保留原始版本
        pruned_model = copy.deepcopy(self.model)
        
        # 檢測頭通常需要保留完整性，所以我們對其進行特殊處理
        detection_heads = []
        backbone_modules = []
        
        # 分類模塊
        for name, module in pruned_model.named_modules():
            if 'detect' in name.lower() or 'pred' in name.lower():
                detection_heads.append((name, module))
            else:
                backbone_modules.append((name, module))
        
        # 對主幹網絡應用更激進的剪枝
        backbone_prune_amount = self.prune_amount
        head_prune_amount = self.prune_amount * 0.5  # 對檢測頭應用較溫和的剪枝
        
        # 剪枝計數器
        pruned_params = 0
        total_params = 0
        
        # 對主幹網絡應用剪枝
        for name, module in backbone_modules:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 獲取參數數量
                param_size = module.weight.data.numel()
                total_params += param_size
                
                # 應用剪枝
                if self.prune_method == 'l1':
                    prune.l1_unstructured(module, name='weight', amount=backbone_prune_amount)
                elif self.prune_method == 'random':
                    prune.random_unstructured(module, name='weight', amount=backbone_prune_amount)
                elif self.prune_method == 'structured':
                    # 對卷積層進行結構化剪枝
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(module, name='weight', amount=backbone_prune_amount, 
                                          n=2, dim=0)
                
                # 將剪枝永久化
                prune.remove(module, 'weight')
                
                # 統計剪枝參數
                pruned_params += int(param_size * backbone_prune_amount)
        
        # 對檢測頭應用溫和剪枝
        for name, module in detection_heads:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 獲取參數數量
                param_size = module.weight.data.numel()
                total_params += param_size
                
                # 應用溫和剪枝
                if self.prune_method == 'l1':
                    prune.l1_unstructured(module, name='weight', amount=head_prune_amount)
                elif self.prune_method == 'random':
                    prune.random_unstructured(module, name='weight', amount=head_prune_amount)
                
                # 將剪枝永久化
                prune.remove(module, 'weight')
                
                # 統計剪枝參數
                pruned_params += int(param_size * head_prune_amount)
        
        # 保存優化後的模型
        self.optimized_model = pruned_model
        
        logger.info(f"YOLO剪枝完成，總參數: {total_params:,}，剪枝參數: {pruned_params:,} ({pruned_params/total_params*100:.2f}%)")
        
        return self.optimized_model

def optimize_model(model, config):
    """
    優化模型的便捷函數
    
    參數:
        model: 原始模型
        config: 優化配置字典
        
    回傳:
        optimized_model: 優化後的模型
    """
    # 識別模型類型
    model_type = config.get('model_type', 'generic')
    
    # 根據模型類型選擇適當的優化器
    if 'yolo' in model_type.lower():
        optimizer = YOLOOptimizer(model, config)
    else:
        optimizer = ModelOptimizer(model, config)
    
    # 執行優化
    apply_pruning = config.get('apply_pruning', True)
    apply_quantization = config.get('apply_quantization', True)
    
    # 獲取校準載入器(如果有的話)
    calibration_loader = config.get('calibration_loader', None)
    
    # 一站式優化
    optimized_model = optimizer.optimize(
        calibration_loader=calibration_loader,
        apply_pruning=apply_pruning,
        apply_quantization=apply_quantization
    )
    
    # 導出模型(如果需要)
    if config.get('export_model', False):
        output_path = config.get('export_path', None)
        input_shape = config.get('input_shape', (1, 3, 640, 640))
        optimizer.export_model(output_path, input_shape)
    
    # 分析性能(如果需要)
    if config.get('profile_model', False):
        profile_results = optimizer.profile_model()
        
    return optimized_model

def create_deploy_package(model_path, config_path, output_dir=None):
    """
    創建部署包
    
    參數:
        model_path: 模型路徑
        config_path: 配置文件路徑
        output_dir: 輸出目錄
        
    回傳:
        package_path: 部署包路徑
    """
    import shutil
    import zipfile
    
    # 確定輸出目錄
    if output_dir is None:
        output_dir = 'outputs/deploy'
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建臨時目錄
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 複製模型和配置文件
        model_name = os.path.basename(model_path)
        config_name = os.path.basename(config_path)
        
        shutil.copy(model_path, os.path.join(temp_dir, model_name))
        shutil.copy(config_path, os.path.join(temp_dir, config_name))
        
        # 創建推理腳本
        inference_script = """#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import torch
import yaml
import argparse

def load_model(model_path):
    # 載入模型
    model = torch.jit.load(model_path) if model_path.endswith('.pt') else None
    return model

def preprocess_image(image_path, size=640):
    # 讀取和預處理圖像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 預處理邏輯
    return image

def main():
    parser = argparse.ArgumentParser(description='PCB缺陷檢測推理')
    parser.add_argument('--model', type=str, required=True, help='模型路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--image', type=str, required=True, help='圖像路徑')
    args = parser.parse_args()
    
    # 載入模型和配置
    model = load_model(args.model)
    
    # 讀取圖像
    image = preprocess_image(args.image)
    
    # 執行推理
    # 推理邏輯
    
    print('推理完成')

if __name__ == '__main__':
    main()
"""
        
        with open(os.path.join(temp_dir, 'infer.py'), 'w', encoding='utf-8') as f:
            f.write(inference_script)
        
        # 創建README文件
        readme_content = """# PCB缺陷檢測部署包

## 使用方法

1. 安裝依賴: `pip install -r requirements.txt`
2. 執行推理: `python infer.py --model model.pt --config config.yaml --image test.jpg`

## 文件說明

- `model.pt/onnx`: 優化後的模型文件
- `config.yaml`: 配置文件
- `infer.py`: 推理腳本
"""
        
        with open(os.path.join(temp_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 創建requirements.txt
        requirements_content = """torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
opencv-python>=4.5.0
pyyaml>=5.4.0
onnx>=1.9.0
onnxruntime>=1.8.0
"""
        
        with open(os.path.join(temp_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        # 創建ZIP檔案
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        package_name = f'pcb_defect_detection_deploy_{timestamp}.zip'
        package_path = os.path.join(output_dir, package_name)
        
        with zipfile.ZipFile(package_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"部署包已創建: {package_path}")
        
    except Exception as e:
        logger.error(f"創建部署包失敗: {e}")
        package_path = None
        
    finally:
        # 清理臨時目錄
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return package_path

if __name__ == "__main__":
    """測試模型優化模組"""
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='PCB缺陷檢測模型優化')
    parser.add_argument('--model', type=str, required=True, help='模型路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--output', type=str, default=None, help='輸出目錄')
    parser.add_argument('--prune', action='store_true', help='應用剪枝')
    parser.add_argument('--quantize', action='store_true', help='應用量化')
    parser.add_argument('--export', type=str, default=None, help='導出格式 (onnx, torchscript)')
    parser.add_argument('--profile', action='store_true', help='分析模型性能')
    args = parser.parse_args()
    
    # 載入配置
    from utils.utils import load_config
    config = load_config(args.config)
    
    # 增加命令行參數到配置
    config['apply_pruning'] = args.prune
    config['apply_quantization'] = args.quantize
    config['export_model'] = args.export is not None
    config['export_format'] = args.export
    config['profile_model'] = args.profile
    config['output_dir'] = args.output
    
    # 載入模型
    if args.model.endswith('.pt'):
        model = torch.load(args.model, map_location='cpu')
    else:
        # 如果是YOLO模型
        try:
            from models.model import load_model
            model = load_model(args.model)
        except ImportError:
            logger.error("無法載入模型，請確認模型路徑和格式")
            exit(1)
    
    # 優化模型
    optimizer = YOLOOptimizer(model, config) if 'yolo' in config.get('model_type', '').lower() else ModelOptimizer(model, config)
    
    if args.prune:
        optimizer.apply_pruning()
        logger.info("剪枝完成")
    
    if args.quantize:
        optimizer.apply_quantization()
        logger.info("量化完成")
    
    if args.export:
        output_path = optimizer.export_model()
        logger.info(f"模型已導出至 {output_path}")
    
    if args.profile:
        profile_results = optimizer.profile_model()
        
    logger.info("模型優化完成")