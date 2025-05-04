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
from PIL import Image
import albumentations as A
from pathlib import Path
import yaml
import logging

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
        """
        初始化PCB缺陷檢測資料集
        
        參數:
            root_dir: 資料集根目錄
            mode: 'train', 'val', 或 'test'
            img_size: 模型輸入圖像大小
            transform: 圖像轉換操作
            use_augmentation: 是否使用資料增強
            mosaic_prob: 使用Mosaic增強的機率
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.imgs_path = os.path.join(root_dir, 'images')
        self.annotations_path = os.path.join(root_dir, 'Annotations')
        self.use_augmentation = use_augmentation and mode == 'train'
        self.mosaic_prob = mosaic_prob if self.use_augmentation else 0
        
        # 讀取資料路徑
        self.image_files = []
        self.annotation_files = []
        
        # 從每個缺陷類型的資料夾載入圖片和標註
        for defect_type in DEFECT_CLASSES.keys():
            img_dir = os.path.join(self.imgs_path, defect_type)
            ann_dir = os.path.join(self.annotations_path, defect_type)
            
            if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
                logger.warning(f"找不到路徑 {img_dir} 或 {ann_dir}")
                continue
                
            for filename in os.listdir(img_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(img_dir, filename)
                    ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.xml'))
                    
                    if os.path.exists(ann_path):
                        self.image_files.append(img_path)
                        self.annotation_files.append(ann_path)
        
        # 分割訓練和驗證資料集
        if mode != 'test':
            indices = list(range(len(self.image_files)))
            random.seed(42)  # 確保可重複性
            random.shuffle(indices)
            
            split = int(len(indices) * 0.8)
            train_indices = indices[:split]
            val_indices = indices[split:]
            
            if mode == 'train':
                self.image_files = [self.image_files[i] for i in train_indices]
                self.annotation_files = [self.annotation_files[i] for i in train_indices]
            else:  # val
                self.image_files = [self.image_files[i] for i in val_indices]
                self.annotation_files = [self.annotation_files[i] for i in val_indices]
        
        logger.info(f"已載入 {mode} 模式的 {len(self.image_files)} 張圖片")
        
        # 設定基本轉換
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_default_transforms()
            
        # 設定資料增強
        if self.use_augmentation:
            self.augmentations = self._get_augmentations()
            
        # 讀取旋轉角度資訊(如果有的話)
        self.rotation_angles = {}
        for defect_type in DEFECT_CLASSES.keys():
            angles_file = os.path.join(root_dir, 'rotation', f"{defect_type}_angles.txt")
            if os.path.exists(angles_file):
                with open(angles_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            img_id, angle = parts
                            self.rotation_angles[img_id] = int(angle)
    
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 讀取標註
        ann_path = self.annotation_files[idx]
        boxes, labels = self._parse_annotation(ann_path)
        
        # 應用資料增強
        if self.use_augmentation:
            transformed = self.augmentations(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # 應用基本轉換
        img = self.transform(img)
        
        # 將邊界框和標籤轉換為張量
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # 如果沒有邊界框，返回空張量
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        # 構建目標字典
        target = {
            'boxes': boxes,
            'labels': labels,
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, labels = self._parse_annotation(ann_path)
            
            # 確定位置(左上、右上、左下、右下)
            x_offset = (i % 2) * self.img_size
            y_offset = (i // 2) * self.img_size
            
            # 調整大小
            h, w = img.shape[:2]
            resize_ratio = min(self.img_size / w, self.img_size / h)
            new_w, new_h = int(w * resize_ratio), int(h * resize_ratio)
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # 放入Mosaic畫布
            mosaic_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            
            # 調整邊界框坐標
            if len(boxes) > 0:
                # 縮放邊界框
                scaled_boxes = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    scaled_boxes.append([
                        xmin * resize_ratio + x_offset,
                        ymin * resize_ratio + y_offset,
                        xmax * resize_ratio + x_offset,
                        ymax * resize_ratio + y_offset
                    ])
                
                combined_boxes.extend(scaled_boxes)
                combined_labels.extend(labels)
        
        # 裁剪到目標大小
        mosaic_img = cv2.resize(mosaic_img, (self.img_size, self.img_size))
        
        # 調整邊界框到新大小
        ratio = self.img_size / (self.img_size * 2)
        final_boxes = []
        for box in combined_boxes:
            xmin, ymin, xmax, ymax = box
            final_boxes.append([
                xmin * ratio, ymin * ratio, 
                xmax * ratio, ymax * ratio
            ])
        
        # 應用基本轉換
        mosaic_img = self.transform(mosaic_img)
        
        # 將邊界框和標籤轉換為張量
        if len(final_boxes) > 0:
            boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
            labels = torch.as_tensor(combined_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        # 構建目標字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return mosaic_img, target
    
    def _parse_annotation(self, ann_path):
        """解析XML標註檔"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # 獲取圖片尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
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
            xmin = float(bbox.find('xmin').text) / img_width
            ymin = float(bbox.find('ymin').text) / img_height
            xmax = float(bbox.find('xmax').text) / img_width
            ymax = float(bbox.find('ymax').text) / img_height
            
            # 確保邊界框坐標有效
            if xmin >= xmax or ymin >= ymax:
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
        return boxes, labels
    
    def _get_default_transforms(self):
        """獲取默認的圖像轉換"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
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


def get_dataloader(config):
    """建立資料載入器
    
    參數:
        config: 配置字典，包含資料集路徑和批次大小等
    
    回傳:
        train_loader: 訓練資料載入器
        val_loader: 驗證資料載入器
    """
    # 從配置中讀取參數
    if 'dataset' in config and 'path' in config['dataset']:
        root_dir = config['dataset']['path']
    else:
        root_dir = config.get('dataset_path', 'C:/Users/a/Desktop/研討會/PCB_DATASET')
    
    # 確認路徑存在
    if not os.path.exists(root_dir):
        logger.error(f"資料集根目錄不存在: {root_dir}")
        logger.info(f"當前工作目錄: {os.getcwd()}")
        raise FileNotFoundError(f"資料集根目錄不存在: {root_dir}")
        
    logger.info(f"使用資料集路徑: {root_dir}")
    batch_size = config.get('batch_size', 16)
    img_size = config.get('img_size', 640)
    num_workers = config.get('num_workers', 4)
    
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


def collate_fn(batch):
    """自定義批次組合函數，處理不同尺寸的邊界框"""
    images, targets = list(zip(*batch))
    return torch.stack(images), targets


if __name__ == "__main__":
    """測試資料集模組功能"""
    # 載入配置
    config = {
        'dataset_path': 'C:/Users/a/Desktop/研討會/PCB_DATASET',
        'batch_size': 4,
        'img_size': 640
    }
    
    # 建立資料載入器
    train_loader, val_loader = get_dataloader(config)
    
    # 驗證資料載入
    for i, (images, targets) in enumerate(train_loader):
        if i > 0:
            break
            
        logger.info(f"批次 {i+1}:")
        logger.info(f"圖像形狀: {images.shape}")
        logger.info(f"目標數量: {len(targets)}")
        
        for j, target in enumerate(targets):
            logger.info(f"  第 {j+1} 個樣本: {len(target['boxes'])} 個邊界框")
            
    logger.info("資料集測試完成")