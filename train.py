#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - PCB缺陷檢測訓練主程式
本模組實現了PCB缺陷檢測的知識蒸餾訓練流程，
整合教師模型和學生模型，實現高效訓練和評估。
主要特點:
1. 整合知識蒸餾訓練框架：從教師模型向學生模型遷移知識
2. 支援YOLO架構：基於YOLO8系列的目標檢測模型
3. 提供全面訓練監控：損失曲線、精度指標、視覺化等
4. 優化訓練策略：學習率調度、早停等機制
"""

import os
# 設置環境變數防止下載COCO數據集
os.environ['YOLO_AUTOINSTALL'] = '0'
os.environ['ULTRALYTICS_DATASET_DOWNLOAD'] = '0'
import argparse
import torch
import yaml
import logging
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 從專案模組導入功能
from data.dataset import get_dataloader
from models.model import get_teacher_model, get_student_model, load_model
from models.distillation import train_with_distillation, get_distillation_losses
from utils.utils import load_config, setup_experiment, Timer, plot_training_metrics, visualize_predictions
from deploy.optimize import optimize_model

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='PCB缺陷檢測訓練')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--teacher', type=str, default=None,
                        help='教師模型路徑(如不指定，則使用預訓練模型)')
    parser.add_argument('--student', type=str, default=None,
                        help='學生模型路徑(如不指定，則從頭訓練)')
    parser.add_argument('--no-distill', action='store_true',
                        help='不使用知識蒸餾(直接訓練學生模型)')
    parser.add_argument('--eval-only', action='store_true',
                        help='僅執行評估，不訓練')
    parser.add_argument('--optimize', action='store_true',
                        help='訓練後優化模型(剪枝和量化)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成視覺化預測結果')
    return parser.parse_args()

def train(config, teacher_model=None, student_model=None, use_distillation=True):
    """訓練PCB缺陷檢測模型
    
    參數:
        config: 配置字典
        teacher_model: 教師模型(如果使用知識蒸餾)
        student_model: 學生模型
        use_distillation: 是否使用知識蒸餾
        
    回傳:
        model: 訓練後的模型
        best_metrics: 最佳評估指標
    """
    # 設置隨機種子確保可重現性
    torch.manual_seed(config.get('seed', 42))
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 設置實驗目錄
    exp_dirs = setup_experiment(config)
    weights_dir = exp_dirs['weights_dir']
    charts_dir = exp_dirs['charts_dir']
    logs_dir = exp_dirs['logs_dir']
    
    # 獲取資料載入器
    logger.info("準備資料載入器...")
    train_loader, val_loader = get_dataloader(config)
    
    # 初始化模型
    if student_model is None:
        logger.info("初始化學生模型...")
        student_model = get_student_model(config)

    if use_distillation and teacher_model is None:
        logger.info("初始化教師模型...")
        teacher_model = get_teacher_model(config)
    
    # 將模型移到設備上
    student_model = student_model.to(device)
    if use_distillation:
        teacher_model = teacher_model.to(device)
    
    # 訓練參數
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    
    # 設置優化器
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 設置學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # 創建損失函數
    if use_distillation:
        logger.info("使用知識蒸餾訓練...")
        distill_losses = get_distillation_losses(config)
        student_model, best_metrics = train_with_distillation(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
    else:
        logger.info("使用標準訓練(無知識蒸餾)...")
        # 自定義訓練循環(無知識蒸餾)
        best_metrics = {'mAP': 0, 'precision': 0, 'recall': 0}
        metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'mAP': [],
            'precision': [],
            'recall': []
        }
        
        total_timer = Timer().start()
        best_epoch = 0
        
        for epoch in range(epochs):
            # 訓練一個周期
            student_model.train()
            epoch_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for images, targets in pbar:
                    # 將資料移到設備上
                    images = images.to(device)
                    for t in targets:
                        for k, v in t.items():
                            if isinstance(v, torch.Tensor):
                                t[k] = v.to(device)
                    
                    # 清除梯度
                    optimizer.zero_grad()
                    
                    # 前向傳播
                    outputs = student_model(images)
                    
                    # 計算損失
                    loss_dict = student_model.model.model.loss(outputs, targets)
                    loss = sum(loss_dict.values())
                    
                    # 反向傳播
                    loss.backward()
                    optimizer.step()
                    
                    # 更新進度條
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            # 更新學習率
            scheduler.step()
            
            # 計算平均訓練損失
            avg_train_loss = epoch_loss / len(train_loader)
            metrics_history['train_loss'].append(avg_train_loss)
            
            # 定期評估
            if (epoch + 1) % config.get('eval_interval', 5) == 0 or epoch == epochs - 1:
                # 評估模型
                student_model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc="Validation"):
                        # 將資料移到設備上
                        images = images.to(device)
                        for t in targets:
                            for k, v in t.items():
                                if isinstance(v, torch.Tensor):
                                    t[k] = v.to(device)
                        
                        # 前向傳播
                        outputs = student_model(images)
                        
                        # 計算損失
                        loss_dict = student_model.model.model.loss(outputs, targets)
                        loss = sum(loss_dict.values())
                        val_loss += loss.item()
                
                # 計算平均驗證損失
                avg_val_loss = val_loss / len(val_loader)
                metrics_history['val_loss'].append(avg_val_loss)
                
                # 計算mAP等評估指標
                from utils.utils import calculate_map
                
                # 收集預測和真實標籤
                all_pred_boxes = []
                all_pred_labels = []
                all_pred_scores = []
                all_gt_boxes = []
                all_gt_labels = []
                
                with torch.no_grad():
                    for images, targets in tqdm(val_loader, desc="Collecting predictions"):
                        images = images.to(device)
                        # 獲取預測
                        outputs = student_model(images)
                        
                        # 處理預測結果
                        for batch_idx, output in enumerate(outputs):
                            # 提取預測
                            boxes = output[0]  # [n, 4]
                            scores = output[1]  # [n]
                            labels = output[2]  # [n]
                            
                            all_pred_boxes.append(boxes.cpu().numpy())
                            all_pred_scores.append(scores.cpu().numpy())
                            all_pred_labels.append(labels.cpu().numpy())
                            
                            # 提取真實標籤
                            target = targets[batch_idx]
                            all_gt_boxes.append(target['boxes'].cpu().numpy())
                            all_gt_labels.append(target['labels'].cpu().numpy())
                
                # 計算評估指標
                metrics = calculate_map(
                    all_pred_boxes, all_pred_labels, all_pred_scores,
                    all_gt_boxes, all_gt_labels
                )
                
                # 更新指標歷史
                metrics_history['mAP'].append(metrics['mAP'])
                metrics_history['precision'].append(metrics['precision'])
                metrics_history['recall'].append(metrics['recall'])
                
                # 記錄結果
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  訓練損失: {avg_train_loss:.4f}")
                logger.info(f"  驗證損失: {avg_val_loss:.4f}")
                logger.info(f"  mAP: {metrics['mAP']:.4f}")
                logger.info(f"  精確率: {metrics['precision']:.4f}")
                logger.info(f"  召回率: {metrics['recall']:.4f}")
                
                # 儲存最佳模型
                if metrics['mAP'] > best_metrics['mAP']:
                    best_metrics = metrics
                    best_epoch = epoch + 1
                    best_model_path = weights_dir / "student_best.pt"
                    torch.save(student_model.state_dict(), best_model_path)
                    logger.info(f"發現更好的模型 (mAP: {best_metrics['mAP']:.4f})，已儲存至: {best_model_path}")
                
                # 儲存當前檢查點
                checkpoint_path = weights_dir / f"student_epoch_{epoch+1}.pt"
                torch.save(student_model.state_dict(), checkpoint_path)
        
        # 計算總訓練時間
        total_time = total_timer.stop().format_time()
        logger.info(f"訓練完成，總耗時: {total_time}")
        logger.info(f"最佳mAP: {best_metrics['mAP']:.4f} (Epoch {best_epoch})")
        
        # 繪製訓練指標圖表
        plot_training_metrics(
            metrics_history,
            output_path=charts_dir / "training_metrics.png"
        )
        
        # 載入最佳模型
        best_model_path = weights_dir / "student_best.pt"
        if os.path.exists(best_model_path):
            student_model.load_state_dict(torch.load(best_model_path))
            logger.info(f"已載入最佳模型: {best_model_path}")
    
    return student_model, best_metrics

def evaluate(model, config):
    """評估PCB缺陷檢測模型
    
    參數:
        model: 要評估的模型
        config: 配置字典
        
    回傳:
        metrics: 評估指標
    """
    logger.info("開始評估模型...")
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 獲取驗證資料載入器
    _, val_loader = get_dataloader(config)
    
    # 收集預測和真實標籤
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="評估中"):
            images = images.to(device)
            
            # 獲取預測
            outputs = model(images)
            
            # 處理預測結果
            for batch_idx, output in enumerate(outputs):
                # 提取預測
                boxes = output[0]  # [n, 4]
                scores = output[1]  # [n]
                labels = output[2]  # [n]
                
                all_pred_boxes.append(boxes.cpu().numpy())
                all_pred_scores.append(scores.cpu().numpy())
                all_pred_labels.append(labels.cpu().numpy())
                
                # 提取真實標籤
                target = targets[batch_idx]
                all_gt_boxes.append(target['boxes'].cpu().numpy())
                all_gt_labels.append(target['labels'].cpu().numpy())
    
    # 計算評估指標
    from utils.utils import calculate_map
    metrics = calculate_map(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_gt_boxes, all_gt_labels
    )
    
    # 輸出結果
    logger.info(f"評估結果:")
    logger.info(f"  mAP: {metrics['mAP']:.4f}")
    logger.info(f"  精確率: {metrics['precision']:.4f}")
    logger.info(f"  召回率: {metrics['recall']:.4f}")
    
    # 顯示各類別的AP
    if 'ap_per_class' in metrics:
        logger.info("各類別AP:")
        for class_name, ap in metrics['ap_per_class'].items():
            logger.info(f"  {class_name}: {ap:.4f}")
    
    return metrics

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設置總計時器
    total_timer = Timer().start()
    
    # 載入已有模型
    teacher_model = None
    student_model = None
    
    if args.teacher:
        teacher_model = load_model(args.teacher, model_type='teacher')
    
    if args.student:
        student_model = load_model(args.student, model_type='student')
    
    # 執行訓練或評估
    if args.eval_only:
        if student_model is None:
            logger.error("評估模式需要指定--student參數來載入模型")
            return
        
        metrics = evaluate(student_model, config)
        
        # 如果需要可視化預測
        if args.visualize:
            # 獲取一些測試圖像
            dataset_path = config.get('dataset_path', 'C:/Users/a/Desktop/研討會/PCB_DATASET')
            image_dir = os.path.join(dataset_path, 'images')
            output_dir = os.path.join(config.get('output_dir', 'outputs'), 'visualize')
            
            # 選擇一些代表性圖像
            import glob
            image_paths = []
            for defect_type in os.listdir(image_dir):
                defect_dir = os.path.join(image_dir, defect_type)
                if os.path.isdir(defect_dir):
                    images = glob.glob(os.path.join(defect_dir, '*.jpg'))
                    # 每種缺陷類型選擇5張圖像
                    image_paths.extend(images[:5])
            
            # 視覺化預測
            visualize_predictions(
                model=student_model,
                image_paths=image_paths,
                output_dir=output_dir,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                conf_threshold=config.get('conf_threshold', 0.25)
            )
            
            logger.info(f"預測可視化結果已儲存至: {output_dir}")
            
    else:
        # 訓練模型
        trained_model, best_metrics = train(
            config=config,
            teacher_model=teacher_model,
            student_model=student_model,
            use_distillation=not args.no_distill
        )
        
        # 如果需要優化模型
        if args.optimize:
            logger.info("開始優化模型...")
            
            # 優化配置
            optimize_config = {
                'model_type': 'yolo',
                'prune_amount': config.get('prune_amount', 0.3),
                'quant_type': config.get('quant_type', 'dynamic'),
                'export_format': config.get('export_format', 'onnx'),
                'output_dir': os.path.join(config.get('output_dir', 'outputs'), 'weights'),
                'model_name': config.get('model_name', 'pcb_defect_detector')
            }
            
            # 應用優化
            optimized_model = optimize_model(trained_model, optimize_config)
            
            # 評估優化後的模型
            logger.info("評估優化後的模型...")
            optimized_metrics = evaluate(optimized_model, config)
            
            # 輸出優化前後對比
            logger.info("優化前後性能對比:")
            logger.info(f"  原始mAP: {best_metrics['mAP']:.4f}")
            logger.info(f"  優化後mAP: {optimized_metrics['mAP']:.4f}")
            logger.info(f"  mAP變化: {optimized_metrics['mAP'] - best_metrics['mAP']:.4f}")
    
    # 輸出總執行時間
    total_time = total_timer.stop().format_time()
    logger.info(f"總執行時間: {total_time}")

if __name__ == "__main__":
    main()