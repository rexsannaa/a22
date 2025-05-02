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
import torchvision.transforms as T
from torchvision.transforms import functional as F
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
        
        # 檢查是否有邊界框，如果沒有則創建一個虛擬框
        if len(boxes) == 0:
            # 創建一個虛擬的小框，類別設為背景 (0)
            height, width = image.shape[:2]
            boxes = [[10, 10, 30, 30]]  # 一個小框在左上角
            labels = [0]  # 背景類別
            
            # 記錄警告日誌
            logging.warning(f"檔案 {img_path} 沒有標註框，已創建虛擬框")
        
        # 轉換為張量
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 應用轉換
        if self.transform:
            image_tensor, boxes, labels = self.transform(image_tensor, boxes, labels)
        
        # 標準化
        image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        
        # 轉換為模型輸入格式
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        return image_tensor, target, img_path
    
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
            # 獲取類別，處理大小寫不一致問題
            name = obj.find('name').text
            # 將標註名稱轉換為首字母大寫且包含下劃線格式（與目錄名稱一致）
            name_parts = name.split('_')
            name = '_'.join(part.capitalize() for part in name_parts)
            
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

class PCBTransform:
    """PCB資料轉換類，實現資料增強和預處理"""
    
    def __init__(self, input_size, mode="train", config=None):
        """
        初始化轉換
        
        Args:
            input_size: 輸入尺寸 [高, 寬]
            mode: 模式，可選["train", "val", "test"]
            config: 配置字典
        """
        self.input_size = input_size
        self.mode = mode
        self.config = config
        
        # 訓練時使用數據增強
        self.use_augmentation = mode == "train" and config is not None
    
    def __call__(self, image, boxes, labels):
        """
        應用轉換
        
        Args:
            image: 圖像張量 [C, H, W]
            boxes: 邊界框列表 [[x1, y1, x2, y2], ...]
            labels: 標籤列表
            
        Returns:
            轉換後的圖像和標註
        """
        # 確保輸入是張量
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float() / 255.0
        
        # 獲取原始尺寸
        c, h, w = image.shape
        
        # 調整圖像大小
        image = F.resize(image, self.input_size)
        
        # 調整邊界框尺寸
        scale_x = self.input_size[1] / w
        scale_y = self.input_size[0] / h
        
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            new_box = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]
            new_boxes.append(new_box)
        
        # 如果是訓練模式且啟用了增強
        if self.use_augmentation and self.mode == "train":
            # 水平翻轉
            if random.random() < 0.5 and self.config["augmentation"]["use_flip"]:
                image = F.hflip(image)
                for i, box in enumerate(new_boxes):
                    x1, y1, x2, y2 = box
                    new_boxes[i] = [self.input_size[1] - x2, y1, self.input_size[1] - x1, y2]
            
            # 垂直翻轉
            if random.random() < 0.3 and self.config["augmentation"]["use_flip"]:
                image = F.vflip(image)
                for i, box in enumerate(new_boxes):
                    x1, y1, x2, y2 = box
                    new_boxes[i] = [x1, self.input_size[0] - y2, x2, self.input_size[0] - y1]
            
            # 亮度和對比度調整
            if random.random() < 0.5 and self.config["augmentation"]["brightness_contrast"]:
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = F.adjust_brightness(image, brightness_factor)
                image = F.adjust_contrast(image, contrast_factor)
            
            # 色調和飽和度調整
            if random.random() < 0.3 and self.config["augmentation"]["hsv_aug"]:
                hue_factor = random.uniform(-0.1, 0.1)
                saturation_factor = random.uniform(0.8, 1.2)
                image = F.adjust_hue(image, hue_factor)
                image = F.adjust_saturation(image, saturation_factor)
        
        return image, new_boxes, labels


def create_transforms(config, mode="train"):
    """
    創建資料轉換
    
    Args:
        config: 配置字典
        mode: 轉換模式，可選["train", "val", "test"]
    
    Returns:
        轉換函數
    """
    input_size = config["dataset"]["input_size"]
    return PCBTransform(input_size, mode, config)


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