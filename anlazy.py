import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
from matplotlib.collections import LineCollection

class draw_three_axis_zone:
    def __init__(self, main_directory, scale_txt_maker):
        self.params_110 = {
            'angle_4': 54.74,'angle_2': 70.52,
            'distance_4': 0.249,'distance_2': 0.288}
        self.params_111 = {'angle': 60,'distance': 0.167}
        self.params_100 = {'angle': 90,'distance': 0.204}
        self.loss = 5#角度范围°
        self.rate = 0.15#距离形变比例

        self.main_directory = main_directory
        self.scale_txt_maker = scale_txt_maker
        self.process_all_files()

    def process_all_files(self):
        # 创建scale文件未启用
        # self.create_scale_txt_in_subfolders()
        for subdir, _, files in os.walk(self.main_directory):
            txt_files = [f for f in files if f.endswith(".txt") and f != "scale.txt"]
            image_files = [f for f in files if not any(suffix in f for suffix in ["_origin_Gen1-noNoiseNoBackgroundSuperresolution"
                                                                                , "_origin_denoise&bgremoval&superres"]) and f != "result.png"]
            if txt_files and image_files:
                for txt_file, image_file in zip(txt_files, image_files):
                    file_path = os.path.join(subdir, txt_file)
                    image_path = os.path.join(subdir, image_file)
                    scale_file_path = os.path.join(subdir, "scale.txt")
                    if os.path.exists(scale_file_path):
                        with open(scale_file_path, 'r') as scale_file:
                            scale_line = scale_file.readline().strip()
                            scale = eval(scale_line)  # 读取scale值

                    self.process_txt_file(file_path)

                    output_file_path = os.path.join(subdir, "distances_angles.csv")
                    self.calculate_geometry_information(file_path, output_file_path , scale)
                    final_output_path_110 = os.path.join(subdir, "filtered_points_110.csv")
                    final_output_path_111 = os.path.join(subdir, "filtered_points_111.csv")
                    final_output_path_100 = os.path.join(subdir, "filtered_points_100.csv")

                    self.filter_and_save_points(output_file_path, os.path.join(subdir, "filtered_points.csv"), scale)
                    
                    save_path = os.path.join(subdir, "results.png") 
                    self.plot_and_save_results(final_output_path_110, final_output_path_111, final_output_path_100, file_path, image_path, save_path, scale)

    def create_scale_txt_in_subfolders(self):
        for subdir, _, _ in os.walk(self.main_directory):
            scale_txt_path = os.path.join(subdir, 'scale.txt')
            with open(scale_txt_path, 'w') as f:
                f.write(str(self.scale_txt_maker))
    # 只读取txt前两列，并将这两列的信息覆盖原来的txt文件
    def process_txt_file(self,file_path):
        df = pd.read_csv(file_path, sep=",", header=None, usecols=[0, 1])
        df.to_csv(file_path, index=False, header=False, sep=",")
    # 计算几何信息
    def calculate_geometry_information(self,file_path, output_file_path,scale):
        points = self.read_points(file_path)
        tree = cKDTree(points)
        output_data = []
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=7)
            closest_points = points[indices[1:]]  # 排除自己
            closest_ids = indices[1:] + 1  # 加1使其从1开始编号
            sorted_points = self.sort_points_by_angle(point, closest_points)
            distance_mid = distances[1:]
            distance_info = distance_mid * scale
            angle_info = []
            for j in range(len(sorted_points)):
                next_index = (j + 1) % len(sorted_points)
                angle = self.calculate_angle(point, sorted_points[j], sorted_points[next_index])
                angle_info.append(angle)
            output_data.append([i+1, point[0], point[1], 
                                *distance_info, 
                                *angle_info, 
                                *closest_ids])#closest_ids是按距离顺序排列的
        columns = ['Point ID', 'Y', 'X'] + \
                [f'Distance_{i+1}' for i in range(6)] + \
                [f'Angle_{i+1}' for i in range(6)] + \
                [f'Closest_Point_ID_{i+1}' for i in range(6)]
        df = pd.DataFrame(output_data, columns=columns)
        df.to_csv(output_file_path, index=False)
    # 读取点坐标
    def read_points(self,file_path):
        return np.loadtxt(file_path, delimiter=',', dtype=float)
    # 按与x轴的极角排序点
    def sort_points_by_angle(self,center_point, points):
        angles = np.arctan2(points[:, 1] - center_point[1], points[:, 0] - center_point[0])
        return points[np.argsort(angles)]
    # 计算两向量之间的夹角
    def calculate_angle(self,point_1, point_2, point_3):
        v1 = point_2 - point_1
        v2 = point_3 - point_1
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    # 筛选符合条件的点并保存
    def filter_and_save_points(self,file_path, output_path, scale):
        data = pd.read_csv(file_path)
        points = []
        for i, row in data.iterrows():
            point = {
                'Point ID': row['Point ID'],
                'coordinates': (row['Y'], row['X']),
                'distances': [row[f'Distance_{j}'] for j in range(1, 7)],
                'angles': [row[f'Angle_{j}'] for j in range(1, 7)],
                'closest_ids': [row[f'Closest_Point_ID_{j}'] for j in range(1, 7)]
            }
            points.append(point)
        # 110类点筛选
        filtered_points_110 = []
        angle_range_1 = (self.params_110['angle_4'] - self.loss, self.params_110['angle_4'] + self.loss)
        angle_range_2 = (self.params_110['angle_2'] - self.loss, self.params_110['angle_2'] + self.loss)
        for point in points:
            angles = point['angles']
            count_range_1 = sum(angle_range_1[0] <= angle <= angle_range_1[1] for angle in angles)
            count_range_2 = sum(angle_range_2[0] <= angle <= angle_range_2[1] for angle in angles)
            if count_range_1 == 4 and count_range_2 == 2:
                filtered_points_110.append(point)
        final_points_110 = filtered_points_110
        # final_points_110 = []
        # distance_range_1 = (self.params_110['distance_4'] * (1 - self.rate), self.params_110['distance_4'] * (1 + self.rate))
        # distance_range_2 = (self.params_110['distance_2'] * (1 - self.rate), self.params_110['distance_2'] * (1 + self.rate))
        # for point in filtered_points_110:
        #     distances_nm = point['distances']
        #     count_range_1 = sum(distance_range_1[0] <= d <= distance_range_1[1] for d in distances_nm)
        #     count_range_2 = sum(distance_range_2[0] <= d <= distance_range_2[1] for d in distances_nm)
        #     if count_range_1 == 4 and count_range_2 == 2:
        #         final_points_110.append(point)

        # 111类点筛选
        filtered_points_111 = []
        angle_range_111 = (self.params_111['angle'] - self.loss, self.params_111['angle'] + self.loss)
        distance_range_111 = (self.params_111['distance'] * (1 - self.rate), self.params_111['distance'] * (1 + self.rate))
        for point in points:
            angles = point['angles']
            count_111 = sum(angle_range_111[0] <= angle <= angle_range_111[1] for angle in angles)
            if count_111 == 6:
                filtered_points_111.append(point)
        final_points_111 = filtered_points_111
        # final_points_111 = []
        # for point in filtered_points_111:
        #     distances_nm = point['distances']
        #     count_111 = sum(self.distance_range_111[0] <= d <= self.distance_range_111[1] for d in distances_nm)
        #     if count_111 == 6:
        #         final_points_111.append(point)

        # 100类点筛选
        filtered_points_100 = []
        angle_range_100 = (self.params_100['angle'] - self.loss, self.params_100['angle'] + self.loss)
        distance_range_100 = (self.params_100['distance'] * (1 - self.rate), self.params_100['distance'] * (1 + self.rate))
        for point in points:
            angles = point['angles']
            count_100 = sum(angle_range_100[0] <= angle <= angle_range_100[1] for angle in angles)
            if count_100 == 4:
                filtered_points_100.append(point)
        final_points_100 = filtered_points_100
        # final_points_100 = []
        # for point in filtered_points_100:
        #     distances_nm = point['distances']
        #     count_100 = sum(self.distance_range_100[0] <= d <= self.distance_range_100[1] for d in distances_nm)
        #     if count_100 == 4:
        #         final_points_100.append(point)
        def save_points(final_points, output_path, label):
            if final_points:
                df = pd.DataFrame([{
                    'Point ID': point['Point ID'],
                    'Y': point['coordinates'][0],
                    'X': point['coordinates'][1],
                    **{f'Distance_{j+1}': point['distances'][j] for j in range(6)},
                    **{f'Angle_{j+1}': point['angles'][j] for j in range(6)},
                    **{f'Closest_Point_ID_{j+1}': point['closest_ids'][j] for j in range(6)}
                } for point in final_points])
                df.to_csv(output_path, index=False)
                print(f"符合{label}类条件的点已保存至{output_path}")
                return True
            else:
                # print(f"没有符合{label}类条件的点。文件路径: {file_path}")
                return False
        save_points(final_points_110, output_path.replace(".csv", "_110.csv"), "110")
        save_points(final_points_111, output_path.replace(".csv", "_111.csv"), "111")
        save_points(final_points_100, output_path.replace(".csv", "_100.csv"), "100")

    def plot_and_save_results(self,filtered_file_path_110, filtered_file_path_111, filtered_file_path_100, file_path_02, original_image_path, save_path, scale):
        def load_data(file_path):
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                point_ids = data['Point ID'].values
                point_id_to_coords = {row['Point ID']: (row['Y'], row['X']) for _, row in data.iterrows()}
                return data, point_ids, point_id_to_coords
            return None, [], {}
        data_110, point_ids_110, point_id_to_coords_110 = load_data(filtered_file_path_110)
        data_111, point_ids_111, point_id_to_coords_111 = load_data(filtered_file_path_111)
        data_100, point_ids_100, point_id_to_coords_100 = load_data(filtered_file_path_100)
        # 读取原始的所有识别出的点
        points_02 = np.loadtxt(file_path_02, delimiter=',', dtype=float)
        original_image = plt.imread(original_image_path)
        image_height, image_width = original_image.shape[:2]
        fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
        ax.imshow(original_image, cmap='gray')
        ax.axis('off')
        
        def plot_points_and_lines(data, point_ids, point_id_to_coords,extend_length,line_color, point_color):
            lines = []
            for _, row in data.iterrows():
                start_coords = (row['Y'], row['X'])
                for j in range(1, 7):
                    closest_point_id = row[f'Closest_Point_ID_{j}']
                    if closest_point_id in point_ids:
                        mid_coords = point_id_to_coords[closest_point_id]
                        # 查找mid_coords的最近点（排除自己和start_coords）
                        mid_row = data.loc[data['Point ID'] == closest_point_id]
                        for k in range(1, 7):
                            end_point_id = mid_row[f'Closest_Point_ID_{k}'].values[0]
                            if end_point_id in point_ids and end_point_id != row['Point ID']:
                                end_coords = point_id_to_coords[end_point_id]
                                # 绘制start_coords到mid_coords
                                lines.append([start_coords, mid_coords])
                                # 绘制mid_coords到end_coords
                                lines.append([mid_coords, end_coords])
                                # 检查end_coords是否在closest_point_id最近点中，如果是，绘制start_coords到end_coords
                                if row['Point ID'] in data.loc[data['Point ID'] == end_point_id, [f'Closest_Point_ID_{m}' for m in range(1, 7)]].values:
                                    lines.append([start_coords, end_coords])
            # Plot all lines at once
            modified_lines = [[(x, y), (j, k)] for [(y, x), (k, j)] in lines]
            line_collection = LineCollection(modified_lines, colors=line_color, linewidths=2)
            ax.add_collection(line_collection)
            # Plot starting points
            start_coords = data[['Y', 'X']].values
            ax.scatter(start_coords[:, 1], start_coords[:, 0], color=point_color, s=10)

        if len(point_ids_110) > 0:
            plot_points_and_lines(data_110, point_ids_110, point_id_to_coords_110, 0.5, 'green','red')
        if len(point_ids_111) > 0:
            plot_points_and_lines(data_111, point_ids_111, point_id_to_coords_111, 2*self.distance_111,'blue', 'blue')
        if len(point_ids_100) > 0:
            plot_points_and_lines(data_100, point_ids_100, point_id_to_coords_100, 2*self.distance_100,'yellow', 'yellow')
            
        fig.set_size_inches(image_width / 100.0, image_height / 100.0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        # print(f"结果图像已保存至{save_path}")

if __name__ == "__main__":
    main_directory = r"D:\zxk\AtomSegNet-master\tiaojie"
    scale_txt_maker = 5 / 657
    # 创建scale文件未启用
    draw_three_axis_zone(main_directory,scale_txt_maker)