import os
from os.path import exists
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import opening, disk, erosion
from skimage.segmentation import watershed

from utils.utils import GetIndexRangeOfBlk, load_model, map01

class Code_Main:
    def __init__(self, ori_image_path, output_folder, model_name,Threshold):
        self.__curdir = os.getcwd()
        self.ori_image = None
        self.ori_content = None
        self.output_image = None
        self.ori_markers = None
        self.out_markers = None
        self.model_output_content = None
        self.result = None
        self.denoised_image = None
        self.props = None
        self.imarray_original = None
        self.Threshold = Threshold
        self.model_name = model_name
        self.__model_dir = "model_weights"
        self.__models = {
            'denoise&bgremoval&superres': os.path.join(self.__model_dir, 'denoise&bgremoval&superres.pth'),
            'Gen1-noNoiseNoBackgroundSuperresolution': os.path.join(self.__model_dir, 'Gen1-noNoiseNoBackgroundSuperresolution.pth')}

        from torch.cuda import is_available
        self.use_cuda = is_available()
        self.cuda = self.use_cuda

        self.imagePath_content = ori_image_path
        self.output_folder = output_folder
        self.BrowseFolder()
        self.LoadModel()
        self.CircleDetect()
        self.Save()

    def BrowseFolder(self):
        path = self.imagePath_content
        if path:
            file_name = os.path.basename(path)
            _, suffix = os.path.splitext(file_name)
            if suffix == '.ser':
                from file_readers.ser_lib.serReader import serReader
                ser_data = serReader(path)
                ser_array = np.array(ser_data['imageData'], dtype='float64')
                self.imarray_original = ser_array
                ser_array = (map01(ser_array) * 255).astype('uint8')
                self.ori_image = Image.fromarray(ser_array, 'L')
            elif suffix == '.dm3':
                from file_readers import dm3_lib as dm3
                data = dm3.DM3(path).imagedata
                self.imarray_original = np.array(data)
                data = np.array(data, dtype='float64')
                data = (map01(data) * 255).astype('uint8')
                self.ori_image = Image.fromarray(data, mode='L')
            elif suffix == '.tif':
                im = Image.open(path).convert('L')
                self.imarray_original = np.array(im, dtype='float64')
                self.ori_image = Image.fromarray((map01(self.imarray_original) * 255).astype('uint8'), mode='L')
            else:
                self.ori_image = Image.open(path).convert('L')
                self.imarray_original = np.array(self.ori_image)

            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image

    def __load_model(self):
        if not self.ori_image:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda
        model_path = os.path.join(self.__curdir, self.__models[self.model_name])
        self.ori_content = self.ori_image
        self.width, self.height = self.ori_content.size
        blk_col, blk_row = 1, 1
        if self.height > 512:
            blk_row = 2 if self.height <= 1024 else 4
        if self.width > 512:
            blk_col = 2 if self.width <= 1024 else 4

        self.result = np.zeros((self.height, self.width)) - 100
        for r in range(0, blk_row):
            for c in range(0, blk_col):
                inner_blk, outer_blk = GetIndexRangeOfBlk(self.height, self.width, blk_row, blk_col, r, c, over_lap=int(self.width * 0.01 * blk_row))
                temp_image = self.ori_content.crop((outer_blk[0], outer_blk[1], outer_blk[2], outer_blk[3]))
                temp_result = load_model(model_path, temp_image, self.cuda, 3)#设置迭代次数
                self.result[outer_blk[1]: outer_blk[3], outer_blk[0]: outer_blk[2]] = np.maximum(temp_result
                                                                                                 , self.result[outer_blk[1]:outer_blk[3]
                                                                                                 , outer_blk[0]:outer_blk[2]])
        self.result[self.result < 0] = 0
        self.model_output_content = map01(self.result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype('uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode='L')
    def LoadModel(self):
        self.__load_model()
        self.Denoise()

    def Denoise(self):
        radius = 0  
        kernel = disk(radius)
        self.denoised_image = opening(self.model_output_content, kernel)
        # if self.denoise_method.currentText == 'Opening':
        #     self.denoised_image = opening(self.model_output_content, kernel)
        # else:
        #     self.denoised_image = erosion(self.model_output_content, kernel)
        
    def CircleDetect(self):
        if not self.imagePath_content:
            raise Exception("No image is selected.")
        elevation_map = sobel(self.denoised_image)
        from scipy import ndimage as ndi
        markers = np.zeros_like(self.denoised_image)
        max_thre = self.Threshold
        min_thre = 30
        markers[self.denoised_image < min_thre] = 1
        markers[self.denoised_image > max_thre] = 2
        seg_1 = watershed(elevation_map, markers)
        filled_regions = ndi.binary_fill_holes(seg_1 - 1)
        label_objects, nb_labels = ndi.label(filled_regions)
        self.props = regionprops(label_objects)
        self.out_markers = Image.fromarray(np.dstack((self.denoised_image, self.denoised_image, self.denoised_image)), mode='RGB')
        ori_array = np.array(self.ori_content)
        self.ori_markers = Image.fromarray(np.dstack((ori_array, ori_array, ori_array)), mode='RGB')
        del elevation_map
        del markers, seg_1, filled_regions, label_objects, nb_labels
        draw_out = ImageDraw.Draw(self.out_markers)
        draw_ori = ImageDraw.Draw(self.ori_markers)
        for p in self.props:
            c_y, c_x = p.centroid
            draw_out.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height])
                              , min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])], fill='red', outline='red')
            draw_ori.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height])
                              , min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])], fill='red', outline='red')
    
    def GetSavePath(self):
        file_name = os.path.basename(self.imagePath_content)
        _, suffix = os.path.splitext(file_name)
        if suffix in ['.ser', '.dm3', '.tif']:
            name_no_suffix = file_name.replace(suffix, '')
            suffix = '.png'
        else:
            name_no_suffix = file_name.replace(suffix, '')
        save_path = os.path.join(self.output_folder, name_no_suffix)
        if not exists(save_path):
            os.mkdir(save_path)
        temp_path = os.path.join(save_path, name_no_suffix)
        return temp_path, suffix

    def Save(self):
        if not self.imagePath_content:
            raise Exception("No image is selected.")
        _path, suffix = self.GetSavePath()
        if _path is None:
            return
        new_save_name = _path + suffix
        self.ori_content.save(new_save_name)
        new_save_name = _path + '_origin_' + self.model_name + suffix
        self.ori_markers.save(new_save_name)
        new_save_name = _path + '_pos' + '.txt'
        with open(new_save_name, 'w') as file:
            for p in self.props:
                c_y, c_x = p.centroid
                locations = [str(c_y), str(c_x)]
                file.write(",".join(locations))
                file.write("\n")

if __name__ == "__main__":
    ori_image_path = r"D:\ZXK\AtomSegNet-master\origin_image\alllllllllllll\1111_STEM\13.19.38 Scanning Acquire_0131.tif"
    output_folder = r"D:\ZXK\AtomSegNet-master\test_1"
    model_name = "Gen1-noNoiseNoBackgroundSuperresolution"
    # model_name = "denoise&bgremoval&superres"
    Threshold = 150
    Code_Main(ori_image_path, output_folder, model_name, Threshold)