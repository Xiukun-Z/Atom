import os
import torch
from PIL import Image
from noui_Seg import Code_Main
from anlazy import draw_three_axis_zone  

def batch_process_images(root_folder, output_folder, Threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.tif')):
                ori_image_path = os.path.join(subdir, file)
                try:
                    with Image.open(ori_image_path) as img:
                        width, height = img.size
                    if width * height > 3096 * 3096:
                        model_name = "denoise&bgremoval&superres"
                    else:
                        model_name = "Gen1-noNoiseNoBackgroundSuperresolution"
                    Code_Main(ori_image_path, output_folder, model_name, Threshold)
                except IOError:
                    print(f"Cannot open image: {ori_image_path}")
                finally:
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 第一步：处理图片并保存原子点识别结果
    root_folder = r"D:\zxk\results_1_good&bad"  # 输入的大批图片所在路径
    output_folder = r"D:\zxk\results_1_good&bad"  # 保存原子点识别结果的路径
    Threshold = 175  # 设定在100-255，增大不利于STEM点的识别，过小则会识别到比较多的杂点
    batch_process_images(root_folder, output_folder, Threshold)

    # 第二步：处理原子点识别结果
    main_directory = output_folder
    scale_txt_maker = 5 / 657
    draw_three_axis_zone(main_directory, scale_txt_maker)
