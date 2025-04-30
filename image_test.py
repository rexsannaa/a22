#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
image_test.py - PCB圖片讀取測試
測試PCB數據集中的圖片讀取情況，診斷潛在問題。
"""

import os
import cv2
import yaml
import glob
import logging
from tqdm import tqdm

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_image_reading(images_dir, defect_classes):
    """測試圖片讀取"""
    all_images = []
    readable_count = 0
    unreadable_count = 0
    
    # 收集所有圖片檔案路徑
    for defect_class in defect_classes:
        class_dir = os.path.join(images_dir, defect_class)
        if not os.path.exists(class_dir):
            logger.error(f"類別目錄不存在: {class_dir}")
            continue
            
        # 獲取該類別下所有jpg圖片
        img_pattern = os.path.join(class_dir, "*.jpg")
        img_files = glob.glob(img_pattern)
        logger.info(f"類別 {defect_class} 找到 {len(img_files)} 個圖片檔案")
        
        all_images.extend([(img_file, defect_class) for img_file in img_files])
    
    logger.info(f"總共找到 {len(all_images)} 個圖片檔案")
    
    # 測試每個圖片檔案
    for img_file, defect_class in tqdm(all_images, desc="測試圖片讀取"):
        # 檢查檔案是否存在
        if not os.path.exists(img_file):
            logger.error(f"圖片檔案不存在: {img_file}")
            unreadable_count += 1
            continue
            
        # 檢查檔案大小
        file_size = os.path.getsize(img_file)
        if file_size == 0:
            logger.error(f"圖片檔案大小為0: {img_file}")
            unreadable_count += 1
            continue
            
        # 嘗試讀取圖片
        try:
            image = cv2.imread(img_file)
            if image is None:
                logger.error(f"無法讀取圖片 (cv2.imread 返回 None): {img_file}")
                unreadable_count += 1
                continue
                
            # 檢查圖片尺寸和通道
            height, width, channels = image.shape
            logger.debug(f"圖片 {img_file}, 尺寸: {width}x{height}, 通道數: {channels}")
            readable_count += 1
            
        except Exception as e:
            logger.error(f"讀取圖片時發生錯誤 {img_file}: {str(e)}")
            unreadable_count += 1
    
    logger.info(f"測試結果: 共 {len(all_images)} 個檔案, {readable_count} 個可讀取, {unreadable_count} 個不可讀取")
    return readable_count, unreadable_count

def test_dataset_structure(config):
    """測試數據集結構"""
    # 檢查路徑配置
    images_dir = config["paths"]["images"]
    annotations_dir = config["paths"]["annotations"]
    
    if not os.path.exists(images_dir):
        logger.error(f"圖片目錄不存在: {images_dir}")
        return False
        
    if not os.path.exists(annotations_dir):
        logger.error(f"標註目錄不存在: {annotations_dir}")
        return False
    
    # 檢查PCB數據集結構
    logger.info(f"檢查PCB數據集結構...")
    logger.info(f"圖片目錄: {images_dir}")
    logger.info(f"標註目錄: {annotations_dir}")
    
    # 列出缺陷類別目錄
    defect_classes = config["dataset"]["defect_classes"]
    logger.info(f"配置中的缺陷類別: {defect_classes}")
    
    # 檢查實際目錄結構
    for defect_class in defect_classes:
        img_dir = os.path.join(images_dir, defect_class)
        ann_dir = os.path.join(annotations_dir, defect_class)
        
        if not os.path.exists(img_dir):
            logger.error(f"圖片類別目錄不存在: {img_dir}")
        else:
            img_count = len(glob.glob(os.path.join(img_dir, "*.jpg")))
            logger.info(f"類別 {defect_class} 的圖片數量: {img_count}")
            
        if not os.path.exists(ann_dir):
            logger.error(f"標註類別目錄不存在: {ann_dir}")
        else:
            xml_count = len(glob.glob(os.path.join(ann_dir, "*.xml")))
            logger.info(f"類別 {defect_class} 的標註數量: {xml_count}")
    
    return True

def test_single_image(image_path):
    """測試單個圖片讀取並顯示詳細信息"""
    logger.info(f"測試讀取單個圖片: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"圖片文件不存在: {image_path}")
        return
    
    # 檢查文件權限
    try:
        with open(image_path, 'rb') as f:
            header = f.read(16)  # 讀取文件頭
            logger.info(f"文件頭16字節: {header.hex()}")
    except Exception as e:
        logger.error(f"打開文件失敗: {str(e)}")
        return
        
    # 嘗試用OpenCV讀取
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"cv2.imread返回None，無法讀取圖片")
            return
            
        # 顯示圖片詳細信息
        height, width, channels = image.shape
        dtype = image.dtype
        min_val = image.min()
        max_val = image.max()
        
        logger.info(f"圖片讀取成功:")
        logger.info(f"  - 尺寸: {width}x{height}")
        logger.info(f"  - 通道數: {channels}")
        logger.info(f"  - 數據類型: {dtype}")
        logger.info(f"  - 像素值範圍: {min_val} ~ {max_val}")
        
        # 嘗試顏色轉換
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"  - BGR轉RGB成功")
        except Exception as e:
            logger.error(f"  - BGR轉RGB失敗: {str(e)}")
        
    except Exception as e:
        logger.error(f"讀取圖片時發生錯誤: {str(e)}")

def test_dataset_loading():
    """測試數據集加載邏輯"""
    from data.dataset import PCBDefectDataset, create_transforms
    
    # 載入配置
    config = load_config("config/config.yaml")
    
    # 創建轉換
    transforms = create_transforms(config, mode="train")
    
    try:
        # 創建數據集但不應用轉換
        dataset = PCBDefectDataset(config, mode="train", transform=None)
        logger.info(f"數據集初始化成功，大小: {len(dataset)}")
        
        # 嘗試讀取前10個樣本
        for i in range(min(10, len(dataset))):
            try:
                img_path = dataset.samples[i][0]
                logger.info(f"測試讀取樣本 {i}: {img_path}")
                
                # 直接使用OpenCV讀取
                image = cv2.imread(img_path)
                if image is None:
                    logger.error(f"  - 無法使用OpenCV讀取圖片")
                else:
                    logger.info(f"  - OpenCV讀取成功，尺寸: {image.shape}")
                    
                # 嘗試使用數據集的__getitem__方法
                try:
                    image_tensor, target, path = dataset[i]
                    logger.info(f"  - 數據集讀取成功，張量尺寸: {image_tensor.shape}")
                except Exception as e:
                    logger.error(f"  - 數據集讀取失敗: {str(e)}")
                    
            except Exception as e:
                logger.error(f"處理樣本 {i} 時出錯: {str(e)}")
                
    except Exception as e:
        logger.error(f"數據集初始化失敗: {str(e)}")

def main():
    """主函數"""
    logger.info("開始PCB圖片讀取測試")
    
    # 載入配置
    config_path = "config/config.yaml"
    config = load_config(config_path)
    
    # 測試數據集結構
    test_dataset_structure(config)
    
    # 提供更詳細的數據集加載測試
    logger.info("\n=== 測試數據集加載邏輯 ===")
    test_dataset_loading()
    
    # 測試所有圖片的讀取情況
    logger.info("\n=== 測試所有圖片讀取 ===")
    readable, unreadable = test_image_reading(
        config["paths"]["images"], 
        config["dataset"]["defect_classes"]
    )
    
    # 如果有不可讀取的圖片，嘗試測試一個具體的圖片
    if unreadable > 0 and readable > 0:
        logger.info("\n=== 測試單個可讀取的圖片 ===")
        # 找到第一個可讀取的圖片
        for defect_class in config["dataset"]["defect_classes"]:
            img_pattern = os.path.join(config["paths"]["images"], defect_class, "*.jpg")
            img_files = glob.glob(img_pattern)
            if img_files:
                test_image = img_files[0]
                test_single_image(test_image)
                break
    
    logger.info("測試完成")

if __name__ == "__main__":
    main()