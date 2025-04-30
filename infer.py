#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
infer.py - PCB缺陷檢測知識蒸餾增強型混合模型推論程式
整合模型載入、推論、可視化與部署功能，提供簡潔的推論介面。
主要功能:
1. 單張/批次圖像推論
2. 檢測結果可視化
3. 模型性能監控
4. 靈活輸出格式支援
"""

import os
import time
import argparse
import logging
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from datetime import datetime

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcb_detector")

class PCBDetector:
    """PCB缺陷檢測器，整合模型載入和推論功能"""
    
    def __init__(self, config_path, model_path=None, device=None):
        """
        初始化檢測器
        
        Args:
            config_path: 配置文件路徑
            model_path: 模型權重路徑
            device: 推論設備 ('cuda'或'cpu')
        """
        # 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 設置設備
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {self.device}")
        
        # 載入類別
        self.class_names = ["background"] + self.config["dataset"]["defect_classes"]
        self.num_classes = len(self.class_names)
        
        # 設置閾值
        self.conf_threshold = self.config["evaluation"]["conf_threshold"]
        self.iou_threshold = self.config["evaluation"]["iou_threshold"]
        
        # 建立模型
        self.model = self._build_model()
        
        # 載入模型權重
        if model_path:
            self._load_model(model_path)
    
    def _build_model(self):
        """建立模型"""
        # 動態導入模型建立函數
        from models.model import create_model
        
        # 建立學生模型
        model = create_model(self.config, "student").to(self.device)
        model.eval()  # 設置為評估模式
        
        return model
    
    def _load_model(self, model_path):
        """載入模型權重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        logger.info(f"載入模型權重: {model_path}")
        
        try:
            # 載入權重
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 檢查是否為檢查點格式
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            elif 'student_model' in state_dict:
                self.model.load_state_dict(state_dict['student_model'])
            else:
                self.model.load_state_dict(state_dict)
            
            logger.info("模型權重載入成功")
        except Exception as e:
            logger.error(f"載入模型時出錯: {str(e)}")
            raise
    
    def preprocess_image(self, image_path=None, image=None):
        """
        預處理圖像
        
        Args:
            image_path: 圖像路徑
            image: 直接提供的圖像 (numpy數組)
            
        Returns:
            預處理後的圖像張量
        """
        # 獲取輸入尺寸
        input_size = self.config["dataset"]["input_size"]
        
        # 載入圖像
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法讀取圖像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image is None:
            raise ValueError("必須提供image_path或image")
        
        # 記錄原始尺寸
        original_size = image.shape[:2]
        
        # 調整尺寸
        image_resized = cv2.resize(image, (input_size[1], input_size[0]))
        
        # 標準化
        image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次維度
        
        # 應用標準化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor, original_size, image
    
    def detect(self, image_path=None, image=None):
        """
        對單張圖像進行缺陷檢測
        
        Args:
            image_path: 圖像文件路徑
            image: 直接提供的圖像 (numpy數組)
            
        Returns:
            檢測結果字典，包含檢測框、標籤和置信度
        """
        # 預處理圖像
        image_tensor, original_size, original_image = self.preprocess_image(image_path, image)
        
        # 推論
        start_time = time.time()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            detections = self.model([image_tensor])[0]
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        # 後處理結果
        return self._postprocess_detections(detections, original_size), inference_time, original_image
    
    def _postprocess_detections(self, detections, original_size):
        """後處理檢測結果，轉換為原始圖像座標系"""
        # 獲取輸入尺寸
        input_size = self.config["dataset"]["input_size"]
        
        # 獲取縮放比例
        scale_x = original_size[1] / input_size[1]
        scale_y = original_size[0] / input_size[0]
        
        # 獲取檢測結果
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        
        # 應用閾值
        keep = scores > self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # 轉換為原始圖像座標
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # 整理結果
        results = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "class_names": [self.class_names[label] for label in labels]
        }
        
        return results
    
    def batch_detect(self, image_dir, output_dir=None, visualize=True):
        """
        批次檢測目錄中的圖像
        
        Args:
            image_dir: 圖像目錄
            output_dir: 輸出目錄
            visualize: 是否保存可視化結果
            
        Returns:
            所有檢測結果的列表
        """
        # 確認輸出目錄
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if visualize:
                vis_dir = os.path.join(output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
        
        # 獲取所有圖像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            logger.warning(f"未在 {image_dir} 中找到支援的圖像文件")
            return []
        
        logger.info(f"開始處理 {len(image_files)} 個圖像文件")
        
        # 處理每個圖像
        all_results = []
        total_time = 0
        
        for image_file in tqdm(image_files, desc="檢測中"):
            # 執行檢測
            try:
                results, inference_time, original_image = self.detect(image_file)
                total_time += inference_time
                
                # 保存結果
                all_results.append({
                    "file": image_file,
                    "results": results,
                    "time": inference_time
                })
                
                # 視覺化結果
                if visualize and output_dir:
                    filename = os.path.basename(image_file)
                    vis_path = os.path.join(vis_dir, f"det_{filename}")
                    self.visualize_detections(original_image, results, vis_path)
                
                # 保存檢測信息
                if output_dir:
                    filename = os.path.basename(image_file)
                    result_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
                    self.save_detection_results(results, result_path)
                
            except Exception as e:
                logger.error(f"處理 {image_file} 時出錯: {str(e)}")
        
        # 輸出統計
        avg_time = total_time / len(image_files)
        logger.info(f"批次處理完成，平均推論時間: {avg_time:.2f} ms/張")
        
        return all_results
    
    def visualize_detections(self, image, detections, output_path=None):
        """
        可視化檢測結果
        
        Args:
            image: 原始圖像
            detections: 檢測結果
            output_path: 輸出路徑
            
        Returns:
            可視化圖像 (如果output_path為None)
        """
        # 複製圖像
        vis_image = image.copy()
        
        # 獲取檢測結果
        boxes = detections["boxes"]
        scores = detections["scores"]
        class_names = detections["class_names"]
        
        # 繪製檢測框
        for box, score, class_name in zip(boxes, scores, class_names):
            # 獲取座標
            x1, y1, x2, y2 = box.astype(int)
            
            # 選擇顏色 (根據類別)
            color_map = {
                "Missing_hole": (255, 0, 0),      # 紅色
                "Mouse_bite": (0, 255, 0),        # 綠色
                "Spur": (0, 0, 255),              # 藍色
                "Spurious_copper": (255, 255, 0), # 青色
                "Open_circuit": (255, 0, 255),    # 品紅色
                "Short": (0, 255, 255)            # 黃色
            }
            color = color_map.get(class_name, (255, 255, 255))  # 默認白色
            
            # 繪製邊界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 繪製類別和置信度
            label = f"{class_name} {score:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存或返回
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            return None
        else:
            return vis_image
    
    def save_detection_results(self, detections, output_path):
        """
        保存檢測結果為文本格式
        
        Args:
            detections: 檢測結果
            output_path: 輸出路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # 寫入檢測結果
            for box, score, class_name in zip(detections["boxes"], 
                                              detections["scores"], 
                                              detections["class_names"]):
                # 每行格式: class_name x1 y1 x2 y2 score
                f.write(f"{class_name} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f} {score:.4f}\n")
    
    def measure_performance(self, image_paths=None, iterations=100):
        """
        測量模型性能
        
        Args:
            image_paths: 測試圖像路徑列表
            iterations: 迭代次數
            
        Returns:
            性能指標字典
        """
        logger.info("開始測量模型性能...")
        
        # 如果沒有提供圖像路徑，創建隨機張量
        if not image_paths:
            input_size = self.config["dataset"]["input_size"]
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)
            
            # 預熱
            logger.info("進行預熱推論...")
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model([dummy_input])
            
            # 測量時間
            logger.info(f"進行 {iterations} 次計時推論...")
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = self.model([dummy_input])
            
            torch.cuda.synchronize() if self.device == "cuda" else None
            end_time = time.time()
            
            # 計算性能指標
            total_time = end_time - start_time
            avg_time = total_time * 1000 / iterations  # 毫秒
            fps = 1000 / avg_time
            
            # 測量內存使用
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = self.model([dummy_input])
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            else:
                memory_used = 0  # 在CPU上難以準確測量
        
        else:
            # 使用真實圖像測量
            times = []
            for path in tqdm(image_paths, desc="性能測試中"):
                _, inference_time, _ = self.detect(path)
                times.append(inference_time)
            
            avg_time = np.mean(times)
            fps = 1000 / avg_time
            memory_used = 0  # 使用真實圖像時不測量內存
        
        # 返回性能指標
        performance = {
            "avg_inference_time_ms": avg_time,
            "fps": fps,
            "memory_used_mb": memory_used
        }
        
        logger.info(f"性能測量結果:")
        logger.info(f"  - 平均推論時間: {avg_time:.2f} ms/張")
        logger.info(f"  - FPS: {fps:.2f}")
        if memory_used > 0:
            logger.info(f"  - 記憶體使用: {memory_used:.2f} MB")
        
        return performance
    
    def export_model(self, output_path, format_type="onnx"):
        """
        導出模型為部署格式
        
        Args:
            output_path: 輸出路徑
            format_type: 導出格式 ['onnx', 'torchscript']
            
        Returns:
            導出路徑
        """
        logger.info(f"開始導出模型為 {format_type} 格式...")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 確保模型在評估模式
        self.model.eval()
        
        # 創建示例輸入
        input_size = self.config["dataset"]["input_size"]
        dummy_input = [torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)]
        
        if format_type.lower() == "onnx":
            # 導出為ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
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
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
        elif format_type.lower() == "torchscript":
            # 導出為TorchScript
            script_model = torch.jit.trace(self.model, dummy_input)
            torch.jit.save(script_model, output_path)
            
        else:
            raise ValueError(f"不支持的導出格式: {format_type}")
        
        logger.info(f"模型已成功導出至: {output_path}")
        
        return output_path


def main():
    """主函數"""
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="PCB缺陷檢測推論程式")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路徑")
    parser.add_argument("--model", type=str, required=True, help="模型權重路徑")
    parser.add_argument("--device", type=str, default=None, help="推論設備 (cuda/cpu)")
    
    # 推論模式選項
    parser.add_argument("--image", type=str, default=None, help="單張圖像路徑")
    parser.add_argument("--dir", type=str, default=None, help="圖像目錄路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出目錄")
    parser.add_argument("--benchmark", action="store_true", help="執行性能基準測試")
    parser.add_argument("--export", type=str, default=None, help="導出模型路徑")
    parser.add_argument("--export-format", type=str, default="onnx", choices=["onnx", "torchscript"], 
                        help="導出格式")
    
    # 解析參數
    args = parser.parse_args()
    
    # 創建檢測器
    detector = PCBDetector(args.config, args.model, args.device)
    
    # 執行操作
    if args.benchmark:
        # 執行性能基準測試
        performance = detector.measure_performance()
        
    elif args.export:
        # 導出模型
        detector.export_model(args.export, args.export_format)
        
    elif args.image:
        # 處理單張圖像
        logger.info(f"處理圖像: {args.image}")
        results, inference_time, original_image = detector.detect(args.image)
        
        # 視覺化結果
        vis_image = detector.visualize_detections(original_image, results)
        
        # 顯示結果
        logger.info(f"檢測完成，推論時間: {inference_time:.2f} ms")
        for i, (box, score, name) in enumerate(zip(results["boxes"], results["scores"], results["class_names"])):
            logger.info(f"檢測 #{i+1}: {name} 置信度={score:.4f} 邊界框={box}")
        
        # 如果指定了輸出目錄，保存結果
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            basename = os.path.basename(args.image)
            vis_path = os.path.join(args.output, f"det_{basename}")
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"視覺化結果已保存至: {vis_path}")
            
            # 保存檢測結果文本
            result_path = os.path.join(args.output, f"{os.path.splitext(basename)[0]}.txt")
            detector.save_detection_results(results, result_path)
            logger.info(f"檢測結果已保存至: {result_path}")
        else:
            # 顯示圖像
            plt.figure(figsize=(10, 8))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
    elif args.dir:
        # 批次處理目錄
        all_results = detector.batch_detect(args.dir, args.output, visualize=True)
        logger.info(f"批次處理完成，共檢測 {len(all_results)} 個圖像")
        
    else:
        logger.error("請指定 --image 或 --dir 參數")
        parser.print_help()


if __name__ == "__main__":
    main()