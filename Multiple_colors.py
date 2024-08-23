import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
import os

def calculate_color(distance, reference_distance=0.2685):
    # 定义颜色渐变的区间
    min_ratio, max_ratio = 0.7, 1.3  # 扩大渐变区间
    min_color = np.array([0, 25, 0])  # 与更近的距离关联的颜色
    mid_color = np.array([255, 50, 0])  # 渐变中间值
    max_color = np.array([0, 255, 0])  # 与更远的距离关联的颜色
    
    if distance < 0.2 * reference_distance or distance > 2.2 * reference_distance:
        return None  # 不需要画出
    elif distance >= min_ratio * reference_distance and distance <= max_ratio * reference_distance:
        # 根据距离计算颜色比例
        ratio = (distance - min_ratio * reference_distance) / ((max_ratio - min_ratio) * reference_distance)
        if ratio < 0.5:
            # 在更靠近的情况下，使用从min_color到mid_color的渐变
            color = min_color + 2 * ratio * (mid_color - min_color)
        else:
            # 在更远的情况下，使用从mid_color到max_color的渐变
            color = mid_color + 2 * (ratio - 0.5) * (max_color - mid_color)
        return np.clip(color, 0, 255)
    else:
        return None  # 不在需要画的范围内


def plot_all_lines_with_color_adjustment(file_path, original_image_path, save_path, reference_distance=0.2685):
    data = pd.read_csv(file_path)
    points = data[['Y', 'X']].values
    
    original_image = plt.imread(original_image_path)
    image_height, image_width = original_image.shape[:2]
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    ax.imshow(original_image, cmap='gray')
    ax.axis('off')
    
    lines = []
    colors = []
    
    for i, row in data.iterrows():
        start_coords = (row['X'], row['Y'])
        for j in range(1, 7):
            closest_point_id = row[f'Closest_Point_ID_{j}']
            closest_coords = data.loc[data['Point ID'] == closest_point_id, ['X', 'Y']].values[0]
            distance = row[f'Distance_{j}']
            color = calculate_color(distance, reference_distance)
            if color is not None:  # 只有在需要画出的情况下才添加
                lines.append([start_coords, closest_coords])
                colors.append(color / 255.0)  
    
    if lines:
        line_collection = LineCollection(lines, colors=colors, linewidths=2)
        ax.add_collection(line_collection)
    
    fig.set_size_inches(image_width / 100.0, image_height / 100.0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"结果图像已保存至{save_path}")

file_path = r"D:\zxk\project_over\8axis\AtomSegNet-master\tiaojie\19.49.34 CCD Acquire_0010\distances_angles.csv"
original_image_path = r"D:\zxk\project_over\8axis\AtomSegNet-master\tiaojie\19.49.34 CCD Acquire_0010\19.49.34 CCD Acquire_0010.png"
save_path = r"D:\zxk\project_over\8axis\AtomSegNet-master\tiaojie\19.49.34 CCD Acquire_0010\distance.png"

plot_all_lines_with_color_adjustment(file_path, original_image_path, save_path)
