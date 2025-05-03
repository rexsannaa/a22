#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
visualization_test.py - 測試XML標註與圖像顯示
讀取XML標註文件並繪製邊界框以驗證標註是否正確。
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import argparse

def visualize_xml_annotation(xml_path, image_path, save_path=None, show=True):
    """
    讀取XML標註並在圖像上繪製邊界框
    
    Args:
        xml_path: XML標註檔案路徑
        image_path: 圖像檔案路徑
        save_path: 保存可視化結果的路徑
        show: 是否顯示圖像
    """
    # 讀取圖像
    if not os.path.exists(image_path):
        print(f"圖像檔案不存在: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖像: {image_path}")
        return
    
    # 轉換為RGB (matplotlib使用RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 獲取圖像尺寸
    img_height, img_width = image.shape[:2]
    print(f"圖像尺寸: {img_width}x{img_height}")
    
    # 讀取XML標註檔案
    if not os.path.exists(xml_path):
        print(f"XML檔案不存在: {xml_path}")
        return
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"解析XML時出錯: {str(e)}")
        return
    
    # 獲取XML中的圖像尺寸
    size_elem = root.find('size')
    if size_elem is not None:
        xml_width = int(size_elem.find('width').text)
        xml_height = int(size_elem.find('height').text)
        print(f"XML中的圖像尺寸: {xml_width}x{xml_height}")
        
        # 檢查尺寸是否一致
        if xml_width != img_width or xml_height != img_height:
            print(f"警告: 圖像尺寸不一致! XML: {xml_width}x{xml_height}, 實際: {img_width}x{img_height}")
            # 計算縮放比例
            scale_x = img_width / xml_width
            scale_y = img_height / xml_height
            print(f"縮放比例: x={scale_x}, y={scale_y}")
    
    # 創建繪圖
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    
    # 遍歷每個物體
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        if bndbox is not None:
            try:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                # 檢查是否需要縮放
                if 'scale_x' in locals():
                    xmin = int(xmin * scale_x)
                    xmax = int(xmax * scale_x)
                    ymin = int(ymin * scale_y)
                    ymax = int(ymax * scale_y)
                
                # 繪製邊界框
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # 添加標籤
                plt.text(xmin, ymin - 5, name, color='red', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.7))
                
                print(f"繪製邊界框: {name} [{xmin}, {ymin}, {xmax}, {ymax}]")
            except Exception as e:
                print(f"處理邊界框時出錯: {str(e)}")
    
    # 設置圖表屬性
    plt.title(f"標註可視化: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.tight_layout()
    
    # 保存結果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存可視化結果至: {save_path}")
    
    # 顯示結果
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="XML標註可視化工具")
    parser.add_argument("--xml", type=str, required=True, help="XML標註檔案路徑")
    parser.add_argument("--image", type=str, required=True, help="圖像檔案路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出檔案路徑")
    parser.add_argument("--no-show", action="store_true", help="不顯示圖像")
    
    args = parser.parse_args()
    
    visualize_xml_annotation(args.xml, args.image, args.output, not args.no_show)

if __name__ == "__main__":
    main()