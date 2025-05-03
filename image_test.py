#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
image_test.py - PCB圖片讀取和標註檢查工具
檢查PCB數據集中的圖片讀取和XML標註情況，診斷潛在問題。
主要功能:
1. 測試圖片讀取情況
2. 檢查XML標註完整性
3. 統計空標註情況
4. 提供數據集狀態報告
5. 檢查模型特徵對齊問題
"""

import os
import cv2
import yaml
import glob
import logging
import traceback
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

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
    image_sizes = {}
    image_channels = {}
    
    # 收集所有圖像文件
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
            
            # 統計圖片尺寸分佈
            size_key = f"{width}x{height}"
            image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
            
            # 統計通道數分佈
            image_channels[channels] = image_channels.get(channels, 0) + 1
            
            readable_count += 1
            
        except Exception as e:
            logger.error(f"讀取圖片時發生錯誤 {img_file}: {str(e)}")
            unreadable_count += 1
    
    # 輸出圖片尺寸統計
    logger.info("圖片尺寸分佈:")
    for size, count in sorted(image_sizes.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {size}: {count} 張圖片 ({count/len(all_images)*100:.2f}%)")
    
    # 輸出通道數統計
    logger.info("通道數分佈:")
    for channels, count in sorted(image_channels.items()):
        logger.info(f"  - {channels} 通道: {count} 張圖片 ({count/len(all_images)*100:.2f}%)")
    
    logger.info(f"測試結果: 共 {len(all_images)} 個檔案, {readable_count} 個可讀取, {unreadable_count} 個不可讀取")
    return readable_count, unreadable_count, image_sizes, image_channels

def check_xml_annotations(annotations_dir, defect_classes):
    """檢查XML標註文件的完整性"""
    all_xmls = []
    empty_annotations = []
    valid_annotations = []
    invalid_annotations = []
    annotation_stats = {}
    bndbox_sizes = {}
    
    # 收集所有XML文件
    for defect_class in defect_classes:
        class_dir = os.path.join(annotations_dir, defect_class)
        if not os.path.exists(class_dir):
            logger.error(f"標註類別目錄不存在: {class_dir}")
            continue
        
        # 獲取該類別下所有XML文件
        xml_pattern = os.path.join(class_dir, "*.xml")
        xml_files = glob.glob(xml_pattern)
        logger.info(f"類別 {defect_class} 找到 {len(xml_files)} 個XML標註檔案")
        
        # 初始化類別統計
        annotation_stats[defect_class] = {
            "total": len(xml_files),
            "empty": 0,
            "valid": 0,
            "invalid": 0,
            "box_count": 0
        }
        
        all_xmls.extend([(xml_file, defect_class) for xml_file in xml_files])
    
    logger.info(f"總共找到 {len(all_xmls)} 個XML標註檔案")
    
    # 檢查每個XML文件
    for xml_file, defect_class in tqdm(all_xmls, desc="檢查XML標註"):
        try:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 檢查圖像尺寸資訊
            size_elem = root.find('size')
            if size_elem is None:
                logger.warning(f"XML缺少尺寸資訊: {xml_file}")
                
            # 檢查是否有物體標註
            objects = root.findall('object')
            if len(objects) == 0:
                logger.warning(f"XML文件沒有標註物體: {xml_file}")
                empty_annotations.append(xml_file)
                annotation_stats[defect_class]["empty"] += 1
                continue
            
            # 檢查每個物體標註
            valid_file = True
            box_count = 0
            
            for obj in objects:
                # 檢查類別
                name_elem = obj.find('name')
                if name_elem is None:
                    logger.error(f"標註缺少類別名稱: {xml_file}")
                    valid_file = False
                    continue
                    
                name = name_elem.text
                if not any(cls.lower() in name.lower() for cls in defect_classes):
                    logger.warning(f"標註類別不匹配: {name} 不在已知類別列表中: {xml_file}")
                
                # 檢查邊界框
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    logger.error(f"標註缺失邊界框: {xml_file}")
                    valid_file = False
                    continue
                
                # 檢查邊界框座標
                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    # 簡單檢查邊界框是否有效
                    if xmin >= xmax or ymin >= ymax:
                        logger.error(f"邊界框座標無效 [{xmin}, {ymin}, {xmax}, {ymax}]: {xml_file}")
                        valid_file = False
                    else:
                        box_count += 1
                        
                        # 統計邊界框尺寸
                        box_width = xmax - xmin
                        box_height = ymax - ymin
                        box_area = box_width * box_height
                        
                        # 按面積分類
                        if box_area < 100:
                            area_category = "極小 (<100)"
                        elif box_area < 1000:
                            area_category = "小 (100-1000)"
                        elif box_area < 10000:
                            area_category = "中 (1000-10000)"
                        else:
                            area_category = "大 (>10000)"
                            
                        if area_category not in bndbox_sizes:
                            bndbox_sizes[area_category] = 0
                        bndbox_sizes[area_category] += 1
                        
                except (AttributeError, ValueError) as e:
                    logger.error(f"邊界框座標解析錯誤: {xml_file} - {str(e)}")
                    valid_file = False
            
            # 更新統計
            if valid_file:
                valid_annotations.append(xml_file)
                annotation_stats[defect_class]["valid"] += 1
                annotation_stats[defect_class]["box_count"] += box_count
            else:
                invalid_annotations.append(xml_file)
                annotation_stats[defect_class]["invalid"] += 1
                
        except Exception as e:
            logger.error(f"解析XML文件錯誤 {xml_file}: {str(e)}")
            invalid_annotations.append(xml_file)
            annotation_stats[defect_class]["invalid"] += 1
    
    # 統計結果
    total_xmls = len(all_xmls)
    total_empty = len(empty_annotations)
    total_valid = len(valid_annotations)
    total_invalid = len(invalid_annotations)
    
    logger.info("=" * 50)
    logger.info("XML標註檢查結果:")
    logger.info(f"總XML文件數: {total_xmls}")
    logger.info(f"空標註文件數: {total_empty} ({total_empty/total_xmls*100:.2f}%)")
    logger.info(f"有效標註文件數: {total_valid} ({total_valid/total_xmls*100:.2f}%)")
    logger.info(f"無效標註文件數: {total_invalid} ({total_invalid/total_xmls*100:.2f}%)")
    logger.info("=" * 50)
    logger.info("各類別標註情況:")
    
    for defect_class, stats in annotation_stats.items():
        logger.info(f"類別 {defect_class}:")
        logger.info(f"  - 總XML文件數: {stats['total']}")
        logger.info(f"  - 空標註文件數: {stats['empty']} ({stats['empty']/max(stats['total'], 1)*100:.2f}%)")
        logger.info(f"  - 有效標註文件數: {stats['valid']} ({stats['valid']/max(stats['total'], 1)*100:.2f}%)")
        logger.info(f"  - 總邊界框數: {stats['box_count']}")
        logger.info(f"  - 平均每文件邊界框數: {stats['box_count']/max(stats['valid'], 1):.2f}")
    
    # 輸出邊界框尺寸分佈
    logger.info("邊界框尺寸分佈:")
    total_boxes = sum(bndbox_sizes.values())
    for size_category, count in sorted(bndbox_sizes.items(), key=lambda x: x[0]):
        logger.info(f"  - {size_category}: {count} 個 ({count/total_boxes*100:.2f}%)")
    
    # 返回統計結果
    return {
        "total": total_xmls,
        "empty": total_empty,
        "valid": total_valid,
        "invalid": total_invalid,
        "empty_files": empty_annotations,
        "class_stats": annotation_stats,
        "box_sizes": bndbox_sizes
    }

def check_image_xml_pairs(config):
    """檢查圖片和標註的配對情況"""
    images_dir = config["paths"]["images"]
    annotations_dir = config["paths"]["annotations"]
    defect_classes = config["dataset"]["defect_classes"]
    
    missing_xml = []
    missing_image = []
    paired_files = []
    
    for defect_class in defect_classes:
        # 圖片目錄
        class_img_dir = os.path.join(images_dir, defect_class)
        # 標註目錄
        class_ann_dir = os.path.join(annotations_dir, defect_class)
        
        # 確保目錄存在
        if not os.path.exists(class_img_dir):
            logger.error(f"圖片類別目錄不存在: {class_img_dir}")
            continue
        if not os.path.exists(class_ann_dir):
            logger.error(f"標註類別目錄不存在: {class_ann_dir}")
            continue
        
        # 獲取所有圖片文件
        img_files = glob.glob(os.path.join(class_img_dir, "*.jpg"))
        img_basenames = [os.path.basename(f).replace(".jpg", "") for f in img_files]
        
        # 獲取所有標註文件
        xml_files = glob.glob(os.path.join(class_ann_dir, "*.xml"))
        xml_basenames = [os.path.basename(f).replace(".xml", "") for f in xml_files]
        
        # 檢查配對情況
        for img_file, img_basename in zip(img_files, img_basenames):
            xml_file = os.path.join(class_ann_dir, f"{img_basename}.xml")
            
            if not os.path.exists(xml_file):
                missing_xml.append((img_file, defect_class))
            else:
                paired_files.append((img_file, xml_file, defect_class))
        
        for xml_file, xml_basename in zip(xml_files, xml_basenames):
            img_file = os.path.join(class_img_dir, f"{xml_basename}.jpg")
            
            if not os.path.exists(img_file):
                missing_image.append((xml_file, defect_class))
    
    # 輸出統計結果
    logger.info("=" * 50)
    logger.info("圖片和標註配對檢查結果:")
    logger.info(f"配對成功的文件數: {len(paired_files)}")
    logger.info(f"缺少標註的圖片數: {len(missing_xml)}")
    logger.info(f"缺少圖片的標註數: {len(missing_image)}")
    
    if missing_xml:
        logger.warning("缺少標註的前10個圖片:")
        for i, (img_file, defect_class) in enumerate(missing_xml[:10]):
            logger.warning(f"  - {img_file}")
    
    if missing_image:
        logger.warning("缺少圖片的前10個標註:")
        for i, (xml_file, defect_class) in enumerate(missing_image[:10]):
            logger.warning(f"  - {xml_file}")
    
    return {
        "paired": len(paired_files),
        "missing_xml": len(missing_xml),
        "missing_image": len(missing_image),
        "paired_files": paired_files,
        "missing_xml_files": missing_xml,
        "missing_image_files": missing_image
    }

def test_lastlevelmaxpool():
    """測試 LastLevelMaxPool 是否正常運作"""
    logger.info("測試 LastLevelMaxPool 是否正常運作...")
    
    try:
        # 創建 LastLevelMaxPool 實例
        extra_block = LastLevelMaxPool()
        
        # 測試不同輸入類型
        # 1. 測試張量輸入
        x_tensor = torch.randn(2, 64, 16, 16)
        try:
            result = extra_block(x_tensor)
            logger.info("LastLevelMaxPool 處理張量輸入成功")
        except Exception as e:
            logger.error(f"LastLevelMaxPool 處理張量輸入失敗: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 2. 測試列表輸入
        x_list = [torch.randn(2, 64, 16, 16)]
        try:
            result = extra_block(x_list)
            logger.info("LastLevelMaxPool 處理列表輸入成功")
        except Exception as e:
            logger.error(f"LastLevelMaxPool 處理列表輸入失敗: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 3. 測試字典輸入
        x_dict = {"0": torch.randn(2, 64, 16, 16)}
        try:
            result = extra_block(x_dict)
            logger.info("LastLevelMaxPool 處理字典輸入成功")
        except Exception as e:
            logger.error(f"LastLevelMaxPool 處理字典輸入失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
        # 4. 測試正確的參數數量
        try:
            # 嘗試使用顯式參數調用
            signature = str(extra_block.forward.__code__.co_varnames)
            arg_count = extra_block.forward.__code__.co_argcount
            logger.info(f"LastLevelMaxPool.forward 方法簽名: {signature}")
            logger.info(f"LastLevelMaxPool.forward 方法需要 {arg_count} 個參數")
        except Exception as e:
            logger.error(f"獲取方法簽名失敗: {str(e)}")
        
    except Exception as e:
        logger.error(f"LastLevelMaxPool 測試失敗: {str(e)}")
        logger.error(traceback.format_exc())

def test_model_feature_alignment(config):
    """測試模型的特徵對齊問題，檢查教師和學生模型的特徵提取是否正常"""
    logger.info("測試模型特徵對齊...")

    try:
        # 嘗試導入必要的模塊
        from models.model import TeacherModel, StudentModel, create_model
        
        # 創建一個虛擬輸入
        dummy_input = torch.randn(2, 3, 416, 416)
        
        # 嘗試用簡化版的 LastLevelMaxPool 修復教師模型
        class FixedLastLevelMaxPool(nn.Module):
            def __init__(self):
                super(FixedLastLevelMaxPool, self).__init__()
            
            def forward(self, x, y=None, names=None):
                # 處理不同的輸入類型
                if isinstance(x, torch.Tensor):
                    return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
                elif isinstance(x, list) and len(x) > 0:
                    return [F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)]
                elif isinstance(x, dict) and len(x) > 0:
                    keys = list(x.keys())
                    last_key = keys[-1]
                    pooled = F.max_pool2d(x[last_key], kernel_size=1, stride=2, padding=0)
                    # 返回包含原始特徵和新特徵的字典
                    result = {k: v for k, v in x.items()}
                    new_key = str(int(last_key) + 1) if last_key.isdigit() else "pooled"
                    result[new_key] = pooled
                    return result
                else:
                    return []
        
        # 測試修復版的 LastLevelMaxPool
        fixed_extra_block = FixedLastLevelMaxPool()
        logger.info("嘗試使用修復版 LastLevelMaxPool...")
        
        # 測試字典輸入
        x_dict = {"0": torch.randn(2, 64, 16, 16)}
        try:
            result = fixed_extra_block(x_dict)
            logger.info("修復版 LastLevelMaxPool 處理字典輸入成功")
        except Exception as e:
            logger.error(f"修復版 LastLevelMaxPool 處理字典輸入失敗: {str(e)}")
        
        # 建議修復方法
        logger.info("=" * 50)
        logger.info("模型特徵對齊問題診斷結果:")
        logger.info("問題: LastLevelMaxPool.forward() 方法需要 'y' 和 'names' 參數")
        logger.info("建議修復方式:")
        logger.info("1. 修改 models/model.py 中的 LastLevelMaxPool 類，在 forward 方法中添加可選參數:")
        logger.info("   def forward(self, x, y=None, names=None):")
        logger.info("2. 或使用上面示範的修復版 FixedLastLevelMaxPool 類替換現有的 LastLevelMaxPool")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"模型特徵對齊測試失敗: {str(e)}")
        logger.error(traceback.format_exc())

def visualize_sample_annotations(config, count=5, with_empty=True):
    """可視化一些樣本標註，包括空標註"""
    images_dir = config["paths"]["images"]
    annotations_dir = config["paths"]["annotations"]
    defect_classes = config["dataset"]["defect_classes"]
    
    # 收集有效和空標註樣本
    valid_samples = []
    empty_samples = []
    
    for defect_class in defect_classes:
        xml_pattern = os.path.join(annotations_dir, defect_class, "*.xml")
        xml_files = glob.glob(xml_pattern)
        
        for xml_file in xml_files:
            # 獲取對應圖片
            img_basename = os.path.basename(xml_file).replace(".xml", ".jpg")
            img_file = os.path.join(images_dir, defect_class, img_basename)
            
            if not os.path.exists(img_file):
                continue
            
            # 解析XML
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                objects = root.findall('object')
                
                if len(objects) == 0:
                    empty_samples.append((img_file, xml_file, defect_class))
                else:
                    valid_samples.append((img_file, xml_file, defect_class))
            except Exception:
                continue
    
    # 確保有足夠的樣本
    valid_count = min(count, len(valid_samples))
    empty_count = min(count, len(empty_samples)) if with_empty else 0
    
    # 隨機選擇樣本
    if valid_count > 0:
        selected_valid = np.random.choice(len(valid_samples), valid_count, replace=False)
    else:
        selected_valid = []
        
    if empty_count > 0:
        selected_empty = np.random.choice(len(empty_samples), empty_count, replace=False)
    else:
        selected_empty = []
    
    # 可視化有效標註
    logger.info(f"可視化 {valid_count} 個有效標註樣本...")
    for i in selected_valid:
        img_file, xml_file, defect_class = valid_samples[i]
        visualize_annotation(img_file, xml_file, f"有效標註 - {defect_class}")
    
    # 可視化空標註
    if with_empty and empty_count > 0:
        logger.info(f"可視化 {empty_count} 個空標註樣本...")
        for i in selected_empty:
            img_file, xml_file, defect_class = empty_samples[i]
            visualize_annotation(img_file, xml_file, f"空標註 - {defect_class}")
    
    plt.show()

def visualize_annotation(img_file, xml_file, title):
    """可視化單個標註"""
    # 讀取圖片
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 解析XML標註
    boxes = []
    labels = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 獲取圖片尺寸信息
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            logger.info(f"標註中的圖片尺寸: {width}x{height}")
            
            # 檢查實際圖片尺寸與標註尺寸是否一致
            actual_height, actual_width = image.shape[:2]
            if width != actual_width or height != actual_height:
                logger.warning(f"圖片尺寸不一致! 標註: {width}x{height}, 實際: {actual_width}x{actual_height}")
        
        for obj in root.findall('object'):
            # 獲取類別
            name = obj.find('name').text
            
            # 獲取邊界框
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                try:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(name)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"解析XML時出錯: {str(e)}")
    
    # 繪製圖像和標註
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"{title} - {os.path.basename(img_file)}")
    
    # 繪製邊界框
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin - 5, label, color='r', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    if not boxes:
        plt.text(10, 30, "沒有標註框", color='red', fontsize=14,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"visualization_{os.path.basename(img_file)}")
    plt.close()
    
    logger.info(f"已可視化並保存: visualization_{os.path.basename(img_file)}")

def test_model_component_compatibility():
    """測試模型組件的相容性，特別是檢查 LastLevelMaxPool 類的問題"""
    logger.info("測試模型組件相容性...")
    
    # 檢查 torchvision 版本
    import torchvision
    logger.info(f"torchvision 版本: {torchvision.__version__}")
    
    # 測試 LastLevelMaxPool 類的相容性
    try:
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
        
        # 創建一個 LastLevelMaxPool 實例
        maxpool = LastLevelMaxPool()
        
        # 檢查 forward 方法的參數
        import inspect
        params = inspect.signature(maxpool.forward).parameters
        param_names = list(params.keys())
        
        logger.info(f"LastLevelMaxPool.forward 方法參數: {param_names}")
        
        # 測試 forward 方法
        test_input = {"0": torch.randn(1, 64, 32, 32)}
        
        try:
            output = maxpool(test_input)
            logger.info("LastLevelMaxPool 前向傳播成功")
        except Exception as e:
            logger.error(f"LastLevelMaxPool 前向傳播失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
        # 給出修復建議
        if len(param_names) > 1:
            logger.warning(f"LastLevelMaxPool.forward 方法需要 {len(param_names)} 個參數，可能導致模型調用失敗")
            logger.info("""
            修復建議:
            在 models/model.py 中修改 LastLevelMaxPool 類的使用，確保正確傳入所有需要的參數。
            可能的解決方案是重新定義一個自訂的 LastLevelMaxPool 類，如:
            
            class CustomLastLevelMaxPool(nn.Module):
                def __init__(self):
                    super(CustomLastLevelMaxPool, self).__init__()
                
                def forward(self, x):
                    if isinstance(x, torch.Tensor):
                        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
                    elif isinstance(x, list) and len(x) > 0:
                        return [F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)]
                    elif isinstance(x, dict) and len(x) > 0:
                        # 取最後一個特徵做池化
                        keys = list(x.keys())
                        last_feature = x[keys[-1]]
                        pooled = F.max_pool2d(last_feature, kernel_size=1, stride=2, padding=0)
                        # 返回包含新特徵的字典
                        result = dict(x)
                        new_key = str(int(keys[-1]) + 1) if keys[-1].isdigit() else "pooled"
                        result[new_key] = pooled
                        return result
                    else:
                        return []
            """)
        
    except ImportError:
        logger.error("無法導入 LastLevelMaxPool，可能是 torchvision 版本問題")
    except Exception as e:
        logger.error(f"測試 LastLevelMaxPool 時發生錯誤: {str(e)}")
        logger.error(traceback.format_exc())

def test_feature_extraction(config):
    """測試特徵提取過程，模擬知識蒸餾中的特徵處理"""
    logger.info("測試特徵提取過程...")
    
    try:
        # 創建一個簡單的 ResNet 模型作為教師模型
        teacher_model = resnet50(pretrained=False)
        
        # 創建一個虛擬的輸入張量
        dummy_input = torch.randn(2, 3, 416, 416)
        
        # 儲存中間特徵的字典
        features = {}
        
        # 獲取教師模型的層
        layers = [
            teacher_model.conv1,
            teacher_model.bn1,
            teacher_model.relu,
            teacher_model.maxpool,
            teacher_model.layer1,
            teacher_model.layer2,
            teacher_model.layer3,
            teacher_model.layer4
        ]
        
        # 逐層前向傳播並儲存特徵
        x = dummy_input
        for i, layer in enumerate(layers):
            try:
                x = layer(x)
                if i >= 4:  # 只儲存主要層的特徵
                    layer_name = f"layer{i - 3}"
                    features[layer_name] = x
                    logger.info(f"提取 {layer_name} 特徵成功，形狀: {x.shape}")
            except Exception as e:
                logger.error(f"層 {i} 特徵提取失敗: {str(e)}")
        
        # 測試 FPN 輸入
        if len(features) >= 3:
            logger.info("測試 FPN 輸入...")
            fpn_input = [features["layer2"], features["layer3"], features["layer4"]]
            
            try:
                # 嘗試創建 LastLevelMaxPool 並測試
                from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
                extra_blocks = LastLevelMaxPool()
                
                # 測試 extra_blocks 的輸入格式
                try:
                    result = extra_blocks(fpn_input[-1])
                    logger.info("LastLevelMaxPool 使用張量輸入成功")
                except Exception as e:
                    logger.error(f"LastLevelMaxPool 使用張量輸入失敗: {str(e)}")
                
                # 測試字典輸入
                fpn_input_dict = {str(i): feat for i, feat in enumerate(fpn_input)}
                try:
                    # 這裏可能會失敗，因為 LastLevelMaxPool.forward 可能需要額外參數
                    result = extra_blocks(fpn_input_dict)
                    logger.info("LastLevelMaxPool 使用字典輸入成功")
                except Exception as e:
                    logger.error(f"LastLevelMaxPool 使用字典輸入失敗: {str(e)}")
                    logger.error("這與訓練過程中出現的錯誤吻合。提供了解決方案:")
                    
                    # 提供解決方案
                    logger.info("=" * 50)
                    logger.info("問題診斷與解決方案:")
                    logger.info("LastLevelMaxPool.forward() 方法需要額外參數 'y' 和 'names'，但在調用時未提供這些參數。")
                    logger.info("解決方案:")
                    logger.info("1. 修改 models/model.py 中 LastLevelMaxPool 類的使用，可能需要重新定義一個自訂版本，如下:")
                    logger.info("""
class FixedLastLevelMaxPool(nn.Module):
    def __init__(self):
        super(FixedLastLevelMaxPool, self).__init__()
    
    def forward(self, x):
        # 處理不同的輸入類型
        if isinstance(x, torch.Tensor):
            return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
        elif isinstance(x, list) and len(x) > 0:
            return [F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)]
        elif isinstance(x, dict) and len(x) > 0:
            # 處理字典輸入
            keys = list(x.keys())
            last_key = keys[-1]
            last_feature = x[last_key]
            
            # 應用池化
            pooled = F.max_pool2d(last_feature, kernel_size=1, stride=2, padding=0)
            
            # 創建新的 key
            new_key = str(int(last_key) + 1) if last_key.isdigit() else "pooled"
            
            # 返回包含原始特徵和新特徵的字典
            result = {k: v for k, v in x.items()}
            result[new_key] = pooled
            
            return result
        else:
            return []
                    """)
                    logger.info("2. 替換 models/model.py 中的 LastLevelMaxPool 使用:")
                    logger.info("""
# 替換這行:
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

# 為:
# 自訂 LastLevelMaxPool 實現
class LastLevelMaxPool(nn.Module):
    # 上述實現...
                    """)
                    logger.info("=" * 50)
            except ImportError:
                logger.error("無法導入 LastLevelMaxPool")
    
    except Exception as e:
        logger.error(f"特徵提取測試失敗: {str(e)}")
        logger.error(traceback.format_exc())

def check_dataset_balance(config):
    """檢查數據集各類別的平衡性"""
    logger.info("檢查數據集類別平衡性...")
    
    images_dir = config["paths"]["images"]
    defect_classes = config["dataset"]["defect_classes"]
    
    # 各類別圖片計數
    class_counts = {}
    total_images = 0
    
    for defect_class in defect_classes:
        class_dir = os.path.join(images_dir, defect_class)
        if not os.path.exists(class_dir):
            logger.error(f"類別目錄不存在: {class_dir}")
            class_counts[defect_class] = 0
            continue
            
        # 獲取類別下的圖片數量
        img_files = glob.glob(os.path.join(class_dir, "*.jpg"))
        class_counts[defect_class] = len(img_files)
        total_images += len(img_files)
    
    # 輸出類別平衡情況
    logger.info("=" * 50)
    logger.info("數據集類別平衡情況:")
    logger.info(f"總圖片數量: {total_images}")
    
    if total_images > 0:
        for defect_class, count in class_counts.items():
            percentage = count / total_images * 100
            logger.info(f"類別 {defect_class}: {count} 張圖片 ({percentage:.2f}%)")
            
        # 評估類別平衡性
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info(f"類別不平衡比例 (最大/最小): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            logger.warning("數據集存在明顯的類別不平衡問題，可能影響模型訓練效果")
            logger.info("建議: 對少數類別進行上採樣或使用資料增強，或對多數類別進行下採樣")
    
    return class_counts

def main():
    """主函數"""
    logger.info("開始PCB數據集檢查")
    
    # 載入配置
    config_path = "config/config.yaml"
    config = load_config(config_path)
    
    # 檢查數據集結構
    logger.info("\n=== 檢查數據集結構 ===")
    test_dataset_structure(config)
    
    # 檢查圖片讀取
    logger.info("\n=== 檢查圖片讀取 ===")
    readable_count, unreadable_count, image_sizes, image_channels = test_image_reading(
        config["paths"]["images"], 
        config["dataset"]["defect_classes"]
    )
    
    # 檢查XML標註
    logger.info("\n=== 檢查XML標註 ===")
    xml_stats = check_xml_annotations(
        config["paths"]["annotations"],
        config["dataset"]["defect_classes"]
    )
    
    # 檢查圖片和標註的配對
    logger.info("\n=== 檢查圖片和標註配對 ===")
    pair_stats = check_image_xml_pairs(config)
    
    # 檢查數據集類別平衡性
    logger.info("\n=== 檢查數據集類別平衡性 ===")
    class_counts = check_dataset_balance(config)
    
    # 測試 LastLevelMaxPool 運作
    logger.info("\n=== 測試 LastLevelMaxPool 功能 ===")
    test_lastlevelmaxpool()
    
    # 測試模型組件相容性
    logger.info("\n=== 測試模型組件相容性 ===")
    test_model_component_compatibility()
    
    # 測試特徵提取過程
    logger.info("\n=== 測試特徵提取過程 ===")
    test_feature_extraction(config)
    
    # 可視化一些樣本
    logger.info("\n=== 可視化樣本 ===")
    visualize_sample_annotations(config, count=3, with_empty=True)
    
    # 提供解決方案建議
    logger.info("\n=== 問題診斷和解決方案 ===")
    
    # 檢查空標註比例
    empty_ratio = xml_stats["empty"] / max(xml_stats["total"], 1)
    if empty_ratio > 0.1:
        logger.warning(f"發現大量空標註檔案 ({empty_ratio*100:.2f}%)，這可能導致訓練中的警告信息")
        logger.info("建議修改 data/dataset.py 中的 _parse_annotation 函數，增加處理空標註的邏輯")
        
    # 檢查缺少配對的文件
    if pair_stats["missing_xml"] > 0:
        logger.warning(f"有 {pair_stats['missing_xml']} 個圖片缺少對應標註，這會影響數據加載")
        logger.info("建議確保所有圖片都有對應的XML標註檔案")
    
    # 關鍵問題：LastLevelMaxPool 錯誤
    logger.info("\n模型關鍵錯誤診斷:")
    logger.info("訓練日誌中顯示的錯誤 'LastLevelMaxPool.forward() missing 2 required positional arguments: 'y' and 'names''")
    logger.info("這是由於 torchvision 版本中 LastLevelMaxPool 實現的參數不匹配導致")
    logger.info("解決方案: 修改 models/model.py 中的 LastLevelMaxPool 類，使用自訂的實現:")
    
    logger.info("""
# 在 models/model.py 中添加以下代碼以替換原有的 LastLevelMaxPool 類
class CustomLastLevelMaxPool(nn.Module):
    def __init__(self):
        super(CustomLastLevelMaxPool, self).__init__()

    def forward(self, x):
        # 處理不同的輸入類型
        if isinstance(x, torch.Tensor):
            return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]
        elif isinstance(x, list) and len(x) > 0:
            return [F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)]
        elif isinstance(x, dict) and len(x) > 0:
            # 處理字典輸入
            keys = list(x.keys())
            last_key = keys[-1]
            last_feature = x[last_key]
            
            # 應用池化
            pooled = F.max_pool2d(last_feature, kernel_size=1, stride=2, padding=0)
            
            # 創建新的 key
            new_key = str(int(last_key) + 1) if last_key.isdigit() else "pooled"
            
            # 返回包含原始特徵和新特徵的字典
            result = {k: v for k, v in x.items()}
            result[new_key] = pooled
            
            return result
        else:
            return []
    """)
    
    logger.info("接著在 models/model.py 中替換使用方式:")
    logger.info("""
# 替換導入語句
try:
    # 嘗試從舊位置導入
    from torchvision.ops import misc as misc_nn_ops
    IntermediateLayerGetter = misc_nn_ops.IntermediateLayerGetter
    # 替換 LastLevelMaxPool 的導入
    # from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
    # 改為使用自訂版本
    LastLevelMaxPool = CustomLastLevelMaxPool
except AttributeError:
    # 如果失敗，從新位置導入
    from torchvision.models._utils import IntermediateLayerGetter
    # 同樣替換 LastLevelMaxPool
    LastLevelMaxPool = CustomLastLevelMaxPool
    """)
    
    logger.info("檢查完成")

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

if __name__ == "__main__":
    main()