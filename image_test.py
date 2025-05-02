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
"""

import os
import cv2
import yaml
import glob
import logging
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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
            readable_count += 1
            
        except Exception as e:
            logger.error(f"讀取圖片時發生錯誤 {img_file}: {str(e)}")
            unreadable_count += 1
    
    logger.info(f"測試結果: 共 {len(all_images)} 個檔案, {readable_count} 個可讀取, {unreadable_count} 個不可讀取")
    return readable_count, unreadable_count

def check_xml_annotations(annotations_dir, defect_classes):
    """檢查XML標註文件的完整性"""
    all_xmls = []
    empty_annotations = []
    valid_annotations = []
    invalid_annotations = []
    annotation_stats = {}
    
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
                name = obj.find('name').text
                if name not in defect_class:
                    logger.warning(f"標註類別不匹配: 檔案在 {defect_class} 目錄但標註為 {name}: {xml_file}")
                
                # 檢查邊界框
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    logger.error(f"標註缺失邊界框: {xml_file}")
                    valid_file = False
                    continue
                
                # 檢查邊界框座標
                try:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    # 簡單檢查邊界框是否有效
                    if xmin >= xmax or ymin >= ymax:
                        logger.error(f"邊界框座標無效 [{xmin}, {ymin}, {xmax}, {ymax}]: {xml_file}")
                        valid_file = False
                    else:
                        box_count += 1
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
    
    # 返回統計結果
    return {
        "total": total_xmls,
        "empty": total_empty,
        "valid": total_valid,
        "invalid": total_invalid,
        "empty_files": empty_annotations,
        "class_stats": annotation_stats
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
    selected_valid = np.random.choice(len(valid_samples), valid_count, replace=False)
    selected_empty = np.random.choice(len(empty_samples), empty_count, replace=False) if empty_count > 0 else []
    
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
    test_image_reading(
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
    
    # 提供模型錯誤診斷
    logger.info("\n模型通道錯誤診斷:")
    logger.info("日誌中顯示的錯誤 'Given groups=1, weight of size [96, 80, 1, 1], expected input[8, 96, 13, 13] to have 80 channels, but got 96 channels instead'")
    logger.info("這可能是 models/model.py 中特徵適應層或融合層的通道不匹配問題")
    logger.info("建議檢查 StudentModel 中全局和局部分支的通道數設置")
    
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