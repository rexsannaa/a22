#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
dataset.py - PCB缺陷檢測資料處理模組
整合資料加載、預處理和增強功能，提供統一的資料處理界面。
主要功能:
1. 資料集切分與樣本解析
2. XML標註檔讀取與轉換
3. 圖像增強與預處理
4. 資料批次生成
"""

import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from glob import glob
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCBDefectDataset(Dataset):
    """PCB缺陷檢測資料集類，整合資料加載與處理"""
    
    def __init__(self, config, mode="train", transform=None):
        """
        初始化PCB缺陷檢測資料集
        
        Args:
            config: 包含資料集配置的字典
            mode: 資料集模式，可選["train", "val", "test"]
            transform: 資料增強轉換
        """
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # 設定路徑與類別
        self.images_dir = config["paths"]["images"]
        self.annotations_dir = config["paths"]["annotations"]
        self.rotation_dir = config["paths"]["rotation"]
        self.defect_classes = config["dataset"]["defect_classes"]
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(self.defect_classes)}
        self.class_to_idx['background'] = 0
        
        # 加載與切分資料
        self.samples = self._load_samples()
        
        logger.info(f"已加載 {len(self.samples)} 個樣本用於 {mode} 模式")
    
    def _load_samples(self):
        """加載並切分資料集樣本"""
        all_samples = []
        
        # 收集所有圖像文件
        for defect_class in self.defect_classes:
            img_pattern = os.path.join(self.images_dir, defect_class, "*.jpg")
            img_files = glob(img_pattern)
            
            for img_file in img_files:
                # 對應的標註檔
                basename = os.path.basename(img_file)
                xml_file = os.path.join(self.annotations_dir, defect_class, basename.replace(".jpg", ".xml"))
                
                if os.path.exists(xml_file):
                    all_samples.append((img_file, xml_file, defect_class))
        
        # 分割訓練/驗證/測試集
        train_ratio = self.config["dataset"]["train_ratio"]
        val_ratio = self.config["dataset"]["val_ratio"]
        test_ratio = self.config["dataset"]["test_ratio"]
        
        # 確保比例總和為1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        # 切分資料集
        train_val_samples, test_samples = train_test_split(
            all_samples, test_size=test_ratio, random_state=42
        )
        
        train_samples, val_samples = train_test_split(
            train_val_samples, test_size=val_ratio/(train_ratio+val_ratio), random_state=42
        )
        
        if self.mode == "train":
            return train_samples
        elif self.mode == "val":
            return val_samples
        elif self.mode == "test":
            return test_samples
        else:
            return all_samples
    
    def __len__(self):
        """返回資料集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """獲取指定索引的樣本"""
        img_path, xml_path, defect_class = self.samples[idx]
        
        # 讀取圖像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 讀取標註
        boxes, labels = self._parse_annotation(xml_path)
        
        # 準備增強輸入
        sample = {
            'image': image,
            'bboxes': boxes,
            'labels': labels
        }
        
        # 應用增強
        if self.transform:
            sample = self.transform(**sample)
        
        # 轉換為模型輸入格式
        target = {
            'boxes': torch.as_tensor(sample['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(sample['labels'], dtype=torch.int64)
        }
        
        return sample['image'], target, img_path
    
    def _parse_annotation(self, xml_path):
        """解析XML標註檔"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 獲取圖像大小
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        labels = []
        
        # 解析每個物體
        for obj in root.findall('object'):
            # 獲取類別
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
                
            label = self.class_to_idx[name]
            
            # 獲取邊界框
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 確保邊界框在圖像內
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            
            # 添加到列表
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return boxes, labels


def create_transforms(config, mode="train"):
    """
    創建資料轉換和增強管道
    
    Args:
        config: 配置字典
        mode: 轉換模式，可選["train", "val", "test"]
    
    Returns:
        albumentations轉換管道
    """
    input_size = config["dataset"]["input_size"]
    
    if mode == "train":
        # 訓練時使用更多增強
        return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.HorizontalFlip(p=0.5 if config["augmentation"]["use_flip"] else 0),
            A.VerticalFlip(p=0.3 if config["augmentation"]["use_flip"] else 0),
            A.RandomRotate90(p=0.5 if config["augmentation"]["use_rotation"] else 0),
            A.RandomBrightnessContrast(
                p=0.5 if config["augmentation"]["brightness_contrast"] else 0
            ),
            A.HueSaturationValue(
                p=0.3 if config["augmentation"]["hsv_aug"] else 0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # 驗證和測試時僅做基本轉換
        return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def create_dataloaders(config):
    """
    創建訓練、驗證和測試資料加載器
    
    Args:
        config: 配置字典
    
    Returns:
        字典包含訓練、驗證和測試資料加載器
    """
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    
    # 創建轉換
    train_transforms = create_transforms(config, mode="train")
    val_transforms = create_transforms(config, mode="val")
    test_transforms = create_transforms(config, mode="test")
    
    # 創建資料集
    train_dataset = PCBDefectDataset(config, mode="train", transform=train_transforms)
    val_dataset = PCBDefectDataset(config, mode="val", transform=val_transforms)
    test_dataset = PCBDefectDataset(config, mode="test", transform=test_transforms)
    
    # 創建資料加載器
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def collate_fn(batch):
    """
    自定義批次收集函數，處理不同大小的物體
    
    Args:
        batch: 批次數據
    
    Returns:
        處理後的批次
    """
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def get_rotation_angles(config):
    """
    讀取資料增強旋轉角度信息
    
    Args:
        config: 配置字典
    
    Returns:
        旋轉角度字典
    """
    rotation_dir = config["paths"]["rotation"]
    angles_dict = {}
    
    # 讀取旋轉角度文件
    for defect_class in config["dataset"]["defect_classes"]:
        angles_file = os.path.join(rotation_dir, f"{defect_class}_angles.txt")
        if os.path.exists(angles_file):
            with open(angles_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 2:
                            img_name, angle = parts
                            angles_dict[img_name] = int(angle)
    
    return angles_dict


def visualize_sample(image, target, class_names, figure_size=(10, 10)):
    """
    可視化資料集樣本和標註
    
    Args:
        image: 圖像張量
        target: 目標字典
        class_names: 類別名稱列表
        figure_size: 圖像大小
    """
    # 轉換為numpy數組
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        image = image.astype(np.uint8)
    
    # 繪製圖像
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    plt.figure(figsize=figure_size)
    plt.imshow(image)
    
    # 獲取標註
    boxes = target["boxes"].numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
    labels = target["labels"].numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"]
    
    # 繪製邊界框
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        # 顯示標籤
        class_name = class_names[label]
        plt.text(xmin, ymin - 5, class_name, color='r', fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def load_config(config_path):
    """
    加載配置文件
    
    Args:
        config_path: 配置文件路徑
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# 測試代碼
if __name__ == "__main__":
    # 加載配置
    config_path = "config/config.yaml"
    config = load_config(config_path)
    
    # 創建轉換
    transforms = create_transforms(config, mode="train")
    
    # 創建資料集
    dataset = PCBDefectDataset(config, mode="train", transform=transforms)
    
    # 顯示資料集信息
    print(f"資料集大小: {len(dataset)}")
    print(f"類別: {dataset.class_to_idx}")
    
    # 測試獲取一個樣本
    image, target, img_path = dataset[0]
    print(f"圖像形狀: {image.shape}")
    print(f"目標: {target}")
    print(f"圖像路徑: {img_path}")
    
    # 可視化樣本
    class_names = ["background"] + config["dataset"]["defect_classes"]
    visualize_sample(image, target, class_names)
    
    # 測試資料加載器
    dataloaders = create_dataloaders(config)
    print(f"訓練資料加載器批次數: {len(dataloaders['train'])}")
    print(f"驗證資料加載器批次數: {len(dataloaders['val'])}")
    print(f"測試資料加載器批次數: {len(dataloaders['test'])}")