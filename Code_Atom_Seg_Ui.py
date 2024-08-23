#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
from os.path import exists

import numpy as np
import scipy.io as scio
from PIL import Image, ImageDraw
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.morphology import opening, disk, erosion
from skimage.morphology import opening, disk
from skimage.segmentation import watershed

from UI_files.Atom_Seg_Ui import Ui_MainWindow
from utils.utils import GetIndexRangeOfBlk, load_model, PIL2Pixmap, map01


class Code_MainWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        super(Code_MainWindow, self).__init__()

        self.setupUi(self)
        self.open.clicked.connect(self.BrowseFolder)
        self.load.clicked.connect(self.LoadModel)
        self.se_num.valueChanged.connect(self.Denoise)

        self.circle_detect.clicked.connect(self.CircleDetect)

        self.revert.clicked.connect(self.RevertAll)

        self.save.clicked.connect(self.Save)

        self.__curdir = os.getcwd()  # current directory

        self.ori_image = None
        self.ori_content = None  # original image, PIL format
        self.output_image = None  # output image of model, PIL format
        self.ori_markers = None  # for saving usage, it's a rgb image of original, and with detection result on it
        self.out_markers = None  # for saving usage, it's a rgb image of result after denoising, and with detection result on it
        self.model_output_content = None  # 2d array of model output

        self.result = None
        self.denoised_image = None
        self.props = None

        self.imarray_original = None

        self.__model_dir = "model_weights"
        self.__models = {
            'circularMask': os.path.join(self.__model_dir, 'circularMask.pth'),
            'circularMask_mse_beta': os.path.join(self.__model_dir, 'circularMask_mse_beta.pth'),
            'circularMask_chi10_beta': os.path.join(self.__model_dir, 'circularMask_chi10_beta.pth'),
            'circularMask_chi100_beta': os.path.join(self.__model_dir, 'circularMask_chi100_beta.pth'),
            'guassianMask': os.path.join(self.__model_dir, 'guassianMask.pth'),
            'gaussianMask+': os.path.join(self.__model_dir, 'gaussianMask+.pth'),
            'denoise': os.path.join(self.__model_dir, 'denoise.pth'),
            'denoise&bgremoval': os.path.join(self.__model_dir, 'denoise&bgremoval.pth'),
            'denoise&bgremoval&superres': os.path.join(self.__model_dir, 'denoise&bgremoval&superres.pth'),
            'denoise&airysuperrez_beta': os.path.join(self.__model_dir, 'denoise&airysuperrez_beta.pth'),
            'Gen1-noNoiseNoBackgroundSuperresolution': os.path.join(self.__model_dir, 'Gen1-noNoiseNoBackgroundSuperresolution.pth'),
            'Gen1-circularMask': os.path.join(self.__model_dir, 'Gen1-circularMask.pth'),
            'Gen1-gaussianMask': os.path.join(self.__model_dir, 'Gen1-gaussianMask.pth'),
            'Gen1-noBackgroundNonoise': os.path.join(self.__model_dir, 'Gen1-noBackgroundNonoise.pth'),
            'Gen1-noNoise': os.path.join(self.__model_dir, 'Gen1-noNoise.pth'),
        }
        from torch.cuda import is_available
        self.use_cuda.setChecked(is_available())
        self.use_cuda.setDisabled(not is_available())

        self.imagePath_content = None

    def BrowseFolder(self):
        path, _ = QFileDialog.getOpenFileName(self,
                                              "open",
                                              "/media/Elements/hewei/",
                                              "All Files (*);; Image Files (*.png *.tif *.jpg *.ser *.dm3)")
        self.imagePath_content = self.imagePath_content if not path else path

        if self.imagePath_content:
            self.imagePath.setText(self.imagePath_content)
            file_name = os.path.basename(self.imagePath_content)
            _, suffix = os.path.splitext(file_name)
            # file_name = self.imagePath_content.split('/')[-1]
            # suffix = '.' + file_name.split('.')[-1]
            if suffix == '.ser':
                from file_readers.ser_lib.serReader import serReader
                ser_data = serReader(self.imagePath_content)
                ser_array = np.array(ser_data['imageData'], dtype='float64')
                self.imarray_original = ser_array
                ser_array = (map01(ser_array) * 255).astype('uint8')
                self.ori_image = Image.fromarray(ser_array, 'L')
            elif suffix == '.dm3':
                from file_readers import dm3_lib as dm3
                data = dm3.DM3(self.imagePath_content).imagedata
                self.imarray_original = np.array(data)
                data = np.array(data, dtype='float64')
                data = (map01(data) * 255).astype('uint8')
                self.ori_image = Image.fromarray(data, mode='L')
            elif suffix == '.tif':
                im = Image.open(self.imagePath_content).convert('L')
                self.imarray_original = np.array(im, dtype='float64')
                self.ori_image = Image.fromarray((map01(self.imarray_original) * 255).astype('uint8'), mode='L')
            else:
                self.ori_image = Image.open(self.imagePath_content).convert('L')
                self.imarray_original = np.array(self.ori_image)

            self.width, self.height = self.ori_image.size
            pix_image = PIL2Pixmap(self.ori_image)
            pix_image.scaled(self.ori.size(), QtCore.Qt.KeepAspectRatio)
            self.ori.setPixmap(pix_image)
            self.ori.show()
            self.ori_content = self.ori_image

    def __load_model(self):
        if not self.ori_image:
            raise Exception("No image is selected.")
        self.cuda = self.use_cuda.isChecked()
        model_path = os.path.join(self.__curdir, self.__models[self.model_name])

        if self.change_size.currentText() == 'Down sample by 2':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width // 2, self.height // 2), Image.BILINEAR)
        elif self.change_size.currentText() == 'Up sample by 2':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width * 2, self.height * 2), Image.BICUBIC)
        elif self.change_size.currentText() == 'Down sample by 3':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width // 3, self.height // 3), Image.BILINEAR)
        elif self.change_size.currentText() == 'Up sample by 3':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width * 3, self.height * 3), Image.BICUBIC)
        elif self.change_size.currentText() == 'Down sample by 4':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width // 4, self.height // 4),
                                                     Image.BILINEAR)
        elif self.change_size.currentText() == 'Up sample by 4':
            self.width, self.height = self.ori_image.size
            self.ori_content = self.ori_image.resize((self.width * 4, self.height * 4),
                                                     Image.BICUBIC)
        else:
            self.ori_content = self.ori_image

        pix_image = PIL2Pixmap(self.ori_content)
        pix_image.scaled(self.ori.size(), QtCore.Qt.KeepAspectRatio)
        self.ori.setPixmap(pix_image)
        self.ori.show()

        self.width, self.height = self.ori_content.size

        if self.split.isChecked():

            if self.height > 512 and self.height <= 1024:
                blk_row = 2
            else:
                if self.height > 1024:
                    blk_row = 4
                else:
                    blk_row = 1

            if self.width > 512 and self.width <= 1024:
                blk_col = 2
            else:
                if self.width > 1024:
                    blk_col = 4
                else:
                    blk_col = 1
        else:
            blk_col = 1
            blk_row = 1

        self.result = np.zeros((self.height, self.width)) - 100

        for r in range(0, blk_row):
            for c in range(0, blk_col):
                inner_blk, outer_blk = GetIndexRangeOfBlk(self.height, self.width, blk_row, blk_col, r, c,
                                                          over_lap=int(self.width * 0.01))
                temp_image = self.ori_content.crop((outer_blk[0], outer_blk[1], outer_blk[2], outer_blk[3]))
                temp_result = load_model(model_path, temp_image, self.cuda, self.set_iter.value())
                #                temp_result = map01(temp_result)
                self.result[outer_blk[1]: outer_blk[3], outer_blk[0]: outer_blk[2]] = np.maximum(temp_result,
                                                                                                 self.result[
                                                                                                 outer_blk[1]:outer_blk[
                                                                                                     3], outer_blk[0]:
                                                                                                         outer_blk[2]])

        self.result[self.result < 0] = 0
        self.model_output_content = map01(self.result)
        self.model_output_content = (self.model_output_content * 255 / np.max(self.model_output_content)).astype(
            'uint8')
        self.output_image = Image.fromarray((self.model_output_content), mode='L')
        pix_image = PIL2Pixmap(self.output_image)
        pix_image.scaled(self.model_output.size(), QtCore.Qt.KeepAspectRatio)
        self.model_output.setPixmap(pix_image)
        self.model_output.show()
        del temp_image
        del temp_result

    def LoadModel(self):
        self.model_name = self.modelPath.currentText()
        if not self.ori_image:
            QMessageBox.warning(self, "必须选择一张图片", self.tr("必须选择一张图片!"))
            return
        self.__load_model()
        self.Denoise()

    def Denoise(self):
        radius = self.se_num.value()
        """changes should be done on the kernel generation"""
        kernel = disk(radius)

        if self.denoise_method.currentText == 'Opening':
            self.denoised_image = opening(self.model_output_content, kernel)
        else:
            self.denoised_image = erosion(self.model_output_content, kernel)

        temp_image = Image.fromarray(self.denoised_image, mode='L')

        pix_image = PIL2Pixmap(temp_image)
        self.preprocess.setPixmap(pix_image)
        self.preprocess.show()
        del temp_image

    def CircleDetect(self):
        if not self.imagePath_content:
            QMessageBox.warning(self, "必须选择一张图片", self.tr("必须选择一张图片!"))
            return
        elevation_map = sobel(self.denoised_image)

        from scipy import ndimage as ndi
        markers = np.zeros_like(self.denoised_image)
        if self.set_thre.isChecked() and self.thre.text():
            max_thre = int(self.thre.text()) * 2.55
        else:
            max_thre = 100

        min_thre = 30
        markers[self.denoised_image < min_thre] = 1
        markers[self.denoised_image > max_thre] = 2

        seg_1 = watershed(elevation_map, markers)

        filled_regions = ndi.binary_fill_holes(seg_1 - 1)

        label_objects, nb_labels = ndi.label(filled_regions)

        self.props = regionprops(label_objects)

        self.out_markers = Image.fromarray(np.dstack((self.denoised_image, self.denoised_image, self.denoised_image)),
                                           mode='RGB')

        ori_array = np.array(self.ori_content)
        self.ori_markers = Image.fromarray(np.dstack((ori_array, ori_array, ori_array)), mode='RGB')

        del elevation_map
        del markers, seg_1, filled_regions, label_objects, nb_labels

        draw_out = ImageDraw.Draw(self.out_markers)
        draw_ori = ImageDraw.Draw(self.ori_markers)

        for p in self.props:
            c_y, c_x = p.centroid
            draw_out.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height]),
                              min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])],
                             fill='red', outline='red')
            draw_ori.ellipse([min([max([c_x - 2, 0]), self.width]), min([max([c_y - 2, 0]), self.height]),
                              min([max([c_x + 2, 0]), self.width]), min([max([c_y + 2, 0]), self.height])],
                             fill='red', outline='red')

        pix_image = PIL2Pixmap(self.out_markers)
        self.preprocess.setPixmap(pix_image)
        self.preprocess.show()
        pix_image = PIL2Pixmap(self.ori_markers)
        self.detect_result.setPixmap(pix_image)
        self.detect_result.show()

    #        del props

    def RevertAll(self):
        self.model_output.clear()
        self.se_num.setValue(0)
        self.preprocess.clear()
        self.detect_result.clear()
        del self.result

        self.result = None

    def GetSavePath(self):
        file_name = os.path.basename(self.imagePath_content)
        _, suffix = os.path.splitext(file_name)
        if suffix in ['.ser', '.dm3', '.tif']:
            name_no_suffix = file_name.replace(suffix, '')
            suffix = '.png'
        else:
            name_no_suffix = file_name.replace(suffix, '')

        if not self.change_size.currentText() == 'Do Nothing':
            name_no_suffix = name_no_suffix + '_' + self.change_size.currentText()
        has_content = True

        if self.auto_save.isChecked():
            save_path = os.path.join(self.__curdir, name_no_suffix)
        else:
            path = QFileDialog.getExistingDirectory(self, "save", self.__curdir,
                                                    QFileDialog.ShowDirsOnly
                                                    | QFileDialog.DontResolveSymlinks)
            if not path:
                has_content = False
            save_path = os.path.join(path, name_no_suffix)

        if has_content:
            if not exists(save_path):
                os.mkdir(save_path)
            temp_path = os.path.join(save_path, name_no_suffix)
        else:
            temp_path = None

        return temp_path, suffix

    def Save(self):
        if not self.imagePath_content:
            QMessageBox.warning(self, "必须选择一张图片", self.tr("必须选择一张图片!"))
            return
        opt = self.save_option.currentText()
        _path, suffix = self.GetSavePath()
        if _path is None:
            return
        new_save_name = _path + '_output_' + self.model_name + '.mat'
        scio.savemat(new_save_name, {'result': self.result})
        new_save_name = _path + '_ori_' + self.model_name + '.mat'
        scio.savemat(new_save_name, {'origin': self.imarray_original})

        if not _path:
            return

        if opt == 'Model output':
            new_save_name = _path + '_output_' + self.model_name + suffix
            self.output_image.save(new_save_name)

        if opt == 'Original image with markers':
            new_save_name = _path + '_origin_' + self.model_name + suffix
            self.ori_markers.save(new_save_name)

        if opt == 'Four-panel image':
            new_save_name = _path + '_four_panel_' + self.model_name + suffix
            im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
            im_save.paste(self.ori_content, (0, 0))
            im_save.paste(self.output_image, (self.width + 2, 0))
            im_save.paste(self.ori_markers, (0, self.height + 2))
            im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
            im_save.save(new_save_name)
            del im_save

        if opt == 'Atom positions':
            new_save_name = _path + '_pos_' + self.model_name + '.txt'
            file = open(new_save_name, 'w')
            for p in self.props:
                c_y, c_x = p.centroid
                min_row, min_col, max_row, max_col = p.bbox
                c_y_int = int(min(max(round(c_y), 0), self.height))
                c_x_int = int(min(max(round(c_x), 0), self.width))
                locations = [str(i) for i in
                             (c_y, c_x, min_row, min_col, max_row, max_col, self.result[c_y_int, c_x_int])]
                file.write(",".join(locations))
                file.write("\n")
            file.close()

        if opt == 'Save ALL':
            new_save_name = _path + suffix
            self.ori_content.save(new_save_name)
            new_save_name = _path + '_output_' + self.model_name + suffix
            self.output_image.save(new_save_name)
            new_save_name = _path + '_origin_' + self.model_name + suffix
            self.ori_markers.save(new_save_name)
            new_save_name = _path + '_four_panel_' + self.model_name + suffix
            im_save = Image.new('RGB', ((self.width + 1) * 2, (self.height + 1) * 2))
            im_save.paste(self.ori_content, (0, 0))
            im_save.paste(self.output_image, (self.width + 2, 0))
            im_save.paste(self.ori_markers, (0, self.height + 2))
            im_save.paste(self.out_markers, (self.width + 2, self.height + 2))
            im_save.save(new_save_name)
            del im_save
            
            
            new_save_name = _path + '_pos_' + self.model_name + '.txt'
            file = open(new_save_name, 'w')
            for p in self.props:
                c_y, c_x = p.centroid
                min_row, min_col, max_row, max_col = p.bbox
                c_y_int = int(min(max(round(c_y), 0), self.height))
                c_x_int = int(min(max(round(c_x), 0), self.width))
                locations = [str(i) for i in
                             (c_y, c_x, min_row, min_col, max_row, max_col, self.result[c_y_int, c_x_int])]
                file.write(",".join(locations))
                file.write("\n")
            file.close()

    def drawPoint(self, event):
        self.pos = event.pos()
        self.update()

    def release(self):
        self.model_output.clear()
        self.se_num.setValue(0)
        self.preprocess.clear()
        self.detect_result.clear()
        self.ori.clear()
        del self.props
        del self.output_image
        del self.ori_markers
        del self.out_markers
        return

    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self,
                                                "Confirm Exit...",
                                                "Are you sure you want to exit?",
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            self.release()
            event.accept()


qtCreatorFile = os.path.join("UI_files", "AtomSeg_V1.ui")

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Code_MainWindow()
    window.show()
    sys.exit(app.exec_())
