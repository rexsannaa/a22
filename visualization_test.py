#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
visualize_bbox.py - 可視化XML標註框並保存結果
"""

import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import os

def visualize_and_save(xml_path, img_path, output_path):
    # 讀取圖像
    image = cv2.imread(img_path)
    if image is None:
        print(f"無法讀取圖像: {img_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 獲取圖像尺寸
    img_height, img_width = image.shape[:2]
    print(f"實際圖像尺寸: {img_width}x{img_height}")
    
    # 解析XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 獲取XML標註尺寸
        size_elem = root.find('size')
        xml_width = int(size_elem.find('width').text)
        xml_height = int(size_elem.find('height').text)
        print(f"XML中的尺寸: {xml_width}x{xml_height}")
        
        # 設置圖像顯示
        plt.figure(figsize=(12, 10))
        plt.imshow(image)
        
        # 繪製所有邊界框
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # 繪製框
            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # 添加標籤
            plt.text(xmin, ymin-5, name, color='red', fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            print(f"繪製框: {name} [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        plt.axis('off')
        
        # 直接保存到指定路徑
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存可視化結果至: {output_path}")
        
        # 顯示圖像
        plt.show()
        
    except Exception as e:
        print(f"處理XML出錯: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python visualize_bbox.py <xml_path> <img_path> <output_path>")
        print("例如: python visualize_bbox.py xml文件路徑 圖像路徑 output.png")
    else:
        xml_path = sys.argv[1]
        img_path = sys.argv[2]
        output_path = sys.argv[3]
        visualize_and_save(xml_path, img_path, output_path)