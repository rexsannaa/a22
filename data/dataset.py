#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
dataset.py - PCB缺陷檢測資料處理模組
本模組整合了資料集讀取、預處理、增強和批次生成等功能，
用於準備PCB缺陷檢測模型的訓練和評估資料。
主要特點:
1. 支援多種PCB缺陷類型(Missing_hole, Mouse_bite, Spur, Spurious_copper等)
2. 整合資料增強策略(旋轉、翻轉、縮放等)
3. 自動解析XML標註檔案並轉換為模型所需格式
4. 實現高效的資料批次生成和預處理管道
"""

import os
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import logging
from pathlib import Path

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 缺陷類別映射
DEFECT_CLASSES = {
    'missing_hole': 0,
    'mouse_bite': 1, 
    'spur': 2,
    'spurious_copper': 3,
    'pin_hole': 4,
    'open_circuit': 5
}

class PCBDataset(Dataset):
    """PCB缺陷檢測資料集類別，支援從XML標註檔讀取邊界框資訊"""
    
    def __init__(self, 
                 root_dir, 
                 mode='train',
                 img_size=640,
                 transform=None,
                 use_augmentation=True,
                 mosaic_prob=0.5):
        """初始化PCB缺陷檢測資料集"""
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.imgs_path = os.path.join(root_dir, 'images')
        self.annotations_path = os.path.join(root_dir, 'Annotations')
        self.use_augmentation = use_augmentation and mode == 'train'
        self.mosaic_prob = mosaic_prob if self.use_augmentation else 0
        
        # 檢查資料目錄
        for path in [root_dir, self.imgs_path, self.annotations_path]:
            if not os.path.exists(path):
                logger.error(f"目錄不存在: {path}")
                raise FileNotFoundError(f"目錄不存在: {path}")
        
        # 初始化資料路徑列表
        self.image_files = []
        self.annotation_files = []
        
        # 載入資料路徑
        self._load_dataset_paths()
        
        # 分割訓練和驗證資料集
        self._split_dataset()
        
        # 設定基本轉換和資料增強
        self.transform = transform or self._get_default_transforms()
        if self.use_augmentation:
            self.augmentations = self._get_augmentations()
        
        # 載入旋轉角度資訊(如果有的話)
        self.rotation_angles = self._load_rotation_info()
        
        logger.info(f"已載入 {mode} 模式的 {len(self.image_files)} 張圖片")
    
    def _load_dataset_paths(self):
        """載入圖像和標註檔路徑"""
        for defect_type in DEFECT_CLASSES.keys():
            # 嘗試不同的可能目錄格式
            img_dir_candidates = [
                os.path.join(self.imgs_path, defect_type),
                os.path.join(self.imgs_path, defect_type.replace('_', ' ')),
                os.path.join(self.imgs_path, defect_type.title().replace('_', '')),
                os.path.join(self.imgs_path, defect_type.upper()),
                os.path.join(self.imgs_path, defect_type.replace('_', '-'))
            ]
            
            ann_dir_candidates = [
                os.path.join(self.annotations_path, defect_type),
                os.path.join(self.annotations_path, defect_type.replace('_', ' ')),
                os.path.join(self.annotations_path, defect_type.title().replace('_', '')),
                os.path.join(self.annotations_path, defect_type.upper()),
                os.path.join(self.annotations_path, defect_type.replace('_', '-'))
            ]
            
            # 尋找有效的目錄組合
            found_valid_dir = False
            for img_dir in img_dir_candidates:
                if not os.path.exists(img_dir):
                    continue
                
                for ann_dir in ann_dir_candidates:
                    if not os.path.exists(ann_dir):
                        continue
                    
                    # 找到有效目錄，添加檔案
                    found_valid_dir = True
                    for filename in os.listdir(img_dir):
                        if filename.endswith('.jpg'):
                            img_path = os.path.join(img_dir, filename)
                            ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.xml'))
                            
                            if os.path.exists(ann_path):
                                self.image_files.append(img_path)
                                self.annotation_files.append(ann_path)
                    
                    break  # 找到有效組合後跳出內層循環
                
                if found_valid_dir:
                    break  # 找到有效組合後跳出外層循環
            
            if not found_valid_dir:
                logger.warning(f"找不到缺陷類型 {defect_type} 的有效目錄")
        
        # 如果子目錄沒有找到檔案，嘗試在主目錄搜尋
        if len(self.image_files) == 0:
            logger.warning("未在子目錄中找到任何圖像，嘗試在主目錄中搜尋...")
            
            if os.path.exists(self.imgs_path):
                for filename in os.listdir(self.imgs_path):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(self.imgs_path, filename)
                        ann_path = os.path.join(self.annotations_path, filename.replace('.jpg', '.xml'))
                        
                        if os.path.exists(ann_path):
                            self.image_files.append(img_path)
                            self.annotation_files.append(ann_path)
    
    def _split_dataset(self):
        """分割訓練和驗證資料集"""
        if self.mode != 'test' and len(self.image_files) > 0:
            indices = list(range(len(self.image_files)))
            random.seed(42)  # 確保可重複性
            random.shuffle(indices)
            
            split = int(len(indices) * 0.8)
            train_indices = indices[:split]
            val_indices = indices[split:]
            
            if self.mode == 'train':
                self.image_files = [self.image_files[i] for i in train_indices]
                self.annotation_files = [self.annotation_files[i] for i in train_indices]
            else:  # val
                self.image_files = [self.image_files[i] for i in val_indices]
                self.annotation_files = [self.annotation_files[i] for i in val_indices]
    
    def _load_rotation_info(self):
        """載入旋轉角度資訊"""
        rotation_angles = {}
        for defect_type in DEFECT_CLASSES.keys():
            angles_file = os.path.join(self.root_dir, 'rotation', f"{defect_type}_angles.txt")
            if os.path.exists(angles_file):
                with open(angles_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            img_id, angle = parts
                            rotation_angles[img_id] = int(angle)
        return rotation_angles
    
    def __len__(self):
        """返回資料集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        # 是否使用Mosaic增強
        if random.random() < self.mosaic_prob and self.mode == 'train':
            return self._get_mosaic_sample(idx)
        
        # 一般情況下獲取單個樣本
        return self._get_single_sample(idx)
    
    def _get_single_sample(self, idx):
        """處理單個樣本"""
        # 讀取圖片
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"無法讀取圖像: {img_path}，使用空白圖像替代")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # 調整圖片尺寸
        if orig_h != self.img_size or orig_w != self.img_size:
            h_ratio, w_ratio = self.img_size / orig_h, self.img_size / orig_w
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 讀取標註
        ann_path = self.annotation_files[idx]
        boxes, labels = self._parse_annotation(ann_path, orig_w, orig_h)
        
        # 調整邊界框坐標
        if 'h_ratio' in locals() and 'w_ratio' in locals() and boxes:
            adjusted_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                # 只調整絕對坐標，相對坐標不需調整
                if max(xmin, ymin, xmax, ymax) > 1.0:
                    xmin = (xmin * w_ratio) / self.img_size
                    ymin = (ymin * h_ratio) / self.img_size
                    xmax = (xmax * w_ratio) / self.img_size
                    ymax = (ymax * h_ratio) / self.img_size
                adjusted_boxes.append([xmin, ymin, xmax, ymax])
            boxes = adjusted_boxes
        
        # 應用資料增強
        if self.use_augmentation and self.augmentations and boxes:
            transformed = self.augmentations(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # 應用基本轉換
        img = self.transform(img)
        
        # 準備張量
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        # 構建目標字典
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
        
        return img, target
    
    def _get_mosaic_sample(self, idx):
        """使用Mosaic增強生成樣本"""
        # 選擇三個額外的隨機索引
        indices = [idx] + [random.randint(0, len(self.image_files) - 1) for _ in range(3)]
        
        # 創建Mosaic畫布
        mosaic_img = np.zeros((self.img_size * 2, self.img_size * 2, 3), dtype=np.uint8)
        
        # 用於存儲合併後的邊界框和標籤
        combined_boxes = []
        combined_labels = []
        
        # 拼接四張圖片
        for i, idx in enumerate(indices):
            # 讀取圖片和標註
            img_path = self.image_files[idx]
            ann_path = self.annotation_files[idx]
            
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            # 確定位置(左上、右上、左下、右下)
            x_offset = (i % 2) * self.img_size
            y_offset = (i // 2) * self.img_size
            
            # 調整大小到指定尺寸
            resized_img = cv2.resize(img, (self.img_size, self.img_size))
            
            # 放入Mosaic畫布
            mosaic_img[y_offset:y_offset+self.img_size, x_offset:x_offset+self.img_size] = resized_img
            
            # 讀取標註
            boxes, labels = self._parse_annotation(ann_path, orig_w, orig_h)
            
            # 調整邊界框坐標
            if boxes:
                adjusted_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    # 調整邊界框到mosaic區域中的位置
                    adjusted_boxes.append([
                        (xmin + (i % 2)) * 0.5,  # 考慮x方向的位置
                        (ymin + (i // 2)) * 0.5,  # 考慮y方向的位置
                        (xmax + (i % 2)) * 0.5,
                        (ymax + (i // 2)) * 0.5
                    ])
                
                combined_boxes.extend(adjusted_boxes)
                combined_labels.extend(labels)
        
        # 裁剪到目標大小
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        # 調整邊界框到新大小
        ratio = 0.5  # 從2x尺寸到1x尺寸
        final_boxes = []
        for box in combined_boxes:
            xmin, ymin, xmax, ymax = box
            final_boxes.append([xmin * ratio, ymin * ratio, xmax * ratio, ymax * ratio])
        
        # 應用基本轉換
        mosaic_img = self.transform(mosaic_img)
        
        # 準備張量
        boxes_tensor = torch.as_tensor(final_boxes, dtype=torch.float32) if final_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(combined_labels, dtype=torch.int64) if combined_labels else torch.zeros(0, dtype=torch.int64)
        
        # 構建目標字典
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
        
        return mosaic_img, target
    
    def _parse_annotation(self, ann_path, img_width, img_height):
        """解析XML標註檔"""
        if not os.path.exists(ann_path):
            return [], []
            
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # 獲取圖片尺寸
            size = root.find('size')
            if size is None:
                w, h = img_width, img_height
            else:
                w = int(size.find('width').text)
                h = int(size.find('height').text)
            
            boxes = []
            labels = []
            
            # 遍歷所有物體標註
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in DEFECT_CLASSES:
                    continue
                    
                label = DEFECT_CLASSES[name]
                
                # 獲取邊界框坐標
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                    
                # 確保使用相對坐標 (0-1範圍)
                xmin = float(bbox.find('xmin').text) / w
                ymin = float(bbox.find('ymin').text) / h
                xmax = float(bbox.find('xmax').text) / w
                ymax = float(bbox.find('ymax').text) / h
                
                # 確保邊界框坐標有效
                if xmin >= xmax or ymin >= ymax:
                    continue
                    
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                
            return boxes, labels
        except Exception as e:
            logger.error(f"解析標註檔出錯: {ann_path}, 錯誤: {e}")
            return [], []
    
    def _get_default_transforms(self):
        """獲取默認的圖像轉換"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_augmentations(self):
        """獲取資料增強轉換"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def collate_fn(batch):
    """自定義批次組合函數，處理不同尺寸的圖片"""
    images, targets = [], []
    
    for img, tgt in batch:
        images.append(img if isinstance(img, torch.Tensor) else torch.from_numpy(img).permute(2, 0, 1))
        targets.append(tgt)
    
    # 堆疊圖像
    images = torch.stack(images)
    
    return images, targets

def get_dataloader(config):
    """建立資料載入器
    
    參數:
        config: 配置字典，包含資料集路徑和批次大小等
    
    回傳:
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
    """
    # 從配置中讀取參數
    root_dir = config.get('dataset', {}).get('path', config.get('dataset_path', 'C:/Users/a/Desktop/conference/PCB_DATASET'))
    batch_size = config.get('dataset', {}).get('batch_size', config.get('batch_size', 16))
    img_size = config.get('dataset', {}).get('img_size', config.get('img_size', 640))
    num_workers = config.get('dataset', {}).get('num_workers', config.get('num_workers', 4))
    
    # 檢查資料集根目錄
    if not os.path.exists(root_dir):
        logger.error(f"資料集根目錄不存在: {root_dir}")
        raise FileNotFoundError(f"資料集根目錄不存在: {root_dir}")
    
    try:
        # 建立資料集
        train_dataset = PCBDataset(
            root_dir=root_dir,
            mode='train',
            img_size=img_size,
            use_augmentation=True
        )
        
        val_dataset = PCBDataset(
            root_dir=root_dir,
            mode='val',
            img_size=img_size,
            use_augmentation=False
        )
        
        # 檢查資料集大小
        if len(train_dataset) == 0:
            logger.error(f"訓練資料集為空，請檢查資料集路徑與結構: {root_dir}")
            raise ValueError("訓練資料集為空")
            
        if len(val_dataset) == 0:
            logger.warning("驗證資料集為空，將使用訓練資料集的子集進行驗證")
            # 使用訓練資料集的一部分作為驗證集
            dataset_size = len(train_dataset)
            train_size = int(dataset_size * 0.8)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # 建立資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"已建立訓練資料載入器 ({len(train_dataset)} 樣本) 和驗證資料載入器 ({len(val_dataset)} 樣本)")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"建立資料載入器失敗: {e}")
        raise

if __name__ == "__main__":
    """測試資料集模組功能"""
    # 載入配置
    config = {
        'dataset_path': 'C:/Users/a/Desktop/conference/PCB_DATASET',
        'batch_size': 4,
        'img_size': 640
    }
    
    # 建立資料載入器
    try:
        train_loader, val_loader = get_dataloader(config)
        
        # 驗證資料載入
        for i, (images, targets) in enumerate(train_loader):
            if i > 0: break
                
            logger.info(f"批次 {i+1}:")
            logger.info(f"圖像形狀: {images.shape}")
            logger.info(f"目標數量: {len(targets)}")
            
            for j, target in enumerate(targets):
                logger.info(f"  第 {j+1} 個樣本: {len(target['boxes'])} 個邊界框")
                
        logger.info("資料集測試完成")
    except Exception as e:
        logger.error(f"資料集測試失敗: {e}")