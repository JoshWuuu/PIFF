# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import re
import numpy as np
import enum
import pandas as pd 
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .jpeg import jpeg_decode, jpeg_encode
from .blur import Deblurring
from .superresolution import build_sr_bicubic, build_sr_pool
from .inpaint import get_center_mask, load_freeform_masks

from ipdb import set_trace as debug


class AllCorrupt(enum.IntEnum):
    JPEG_5 = 0
    JPEG_10 = 1
    BLUR_UNI = 2
    BLUR_GAUSS = 3
    SR4X_POOL = 4
    SR4X_BICUBIC = 5
    INPAINT_CENTER = 6
    INPAINT_FREE1020 = 7
    INPAINT_FREE2030 = 8

class MixtureCorruptMethod:
    def __init__(self, opt, device="cpu"):

        # ===== blur ====
        self.blur_uni = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, opt.image_size, device)

        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, device)

        # ===== sr4x ====
        factor = 4
        self.sr4x_pool = build_sr_pool(factor, device, opt.image_size)
        self.sr4x_bicubic = build_sr_bicubic(factor, device, opt.image_size)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')

        # ===== inpaint ====
        self.center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1, 1, 256, 256]
        self.free1020_masks = torch.from_numpy((load_freeform_masks("freeform1020"))) # [10000, 1, 256, 256]
        self.free2030_masks = torch.from_numpy((load_freeform_masks("freeform2030"))) # [10000, 1, 256, 256]

    def jpeg(self, img, qf):
        return jpeg_decode(jpeg_encode(img, qf), qf)

    def blur(self, img, kernel):
        img = (img + 1) / 2
        if kernel == "uni":
            _img = self.blur_uni.H(img).reshape(*img.shape)
        elif kernel == "gauss":
            _img = self.blur_gauss.H(img).reshape(*img.shape)
        # [0,1] -> [-1,1]
        return _img * 2 - 1

    def sr4x(self, img, filter):
        b, c, w, h = img.shape
        if filter == "pool":
            _img = self.sr4x_pool.H(img).reshape(b, c, w // 4, h // 4)
        elif filter == "bicubic":
            _img = self.sr4x_bicubic.H(img).reshape(b, c, w // 4, h // 4)

        # scale to original image size for I2SB
        return self.upsample(_img)

    def inpaint(self, img, mask_type, mask_index=None):
        if mask_type == "center":
            mask = self.center_mask
        elif mask_type == "free1020":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free1020_masks))
            mask = self.free1020_masks[[mask_index]]
        elif mask_type == "free2030":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free2030_masks))
            mask = self.free2030_masks[[mask_index]]
        return img * (1. - mask) + mask * torch.randn_like(img)

    def mixture(self, img, corrupt_idx, mask_index=None):
        if corrupt_idx == AllCorrupt.JPEG_5:
            corrupt_img = self.jpeg(img, 5)
        elif corrupt_idx == AllCorrupt.JPEG_10:
            corrupt_img = self.jpeg(img, 10)
        elif corrupt_idx == AllCorrupt.BLUR_UNI:
            corrupt_img = self.blur(img, "uni")
        elif corrupt_idx == AllCorrupt.BLUR_GAUSS:
            corrupt_img = self.blur(img, "gauss")
        elif corrupt_idx == AllCorrupt.SR4X_POOL:
            corrupt_img = self.sr4x(img, "pool")
        elif corrupt_idx == AllCorrupt.SR4X_BICUBIC:
            corrupt_img = self.sr4x(img, "bicubic")
        elif corrupt_idx == AllCorrupt.INPAINT_CENTER:
            corrupt_img = self.inpaint(img, "center")
        elif corrupt_idx == AllCorrupt.INPAINT_FREE1020:
            corrupt_img = self.inpaint(img, "free1020", mask_index=mask_index)
        elif corrupt_idx == AllCorrupt.INPAINT_FREE2030:
            corrupt_img = self.inpaint(img, "free2030", mask_index=mask_index)
        return corrupt_img


class MixtureCorruptDatasetTrain(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetTrain, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        rand_idx = np.random.choice(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), rand_idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y

from pathlib import Path
class MixtureCorruptDatasetVal(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetVal, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        idx = index % len(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), idx, mask_index=idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y

class floodDataset(Dataset):
    def __init__(self, opt, val=False, test=False, train_dem_num=None, testing_rainfall=None):
        super(floodDataset, self).__init__()
        self.opt = opt
        
        self.test = test
        if not test:
            self.dem_folder = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png'
            self.flood_path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\d'
            self.vx = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\vx'
            self.vy = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\vy'
            self.dem_stat = "C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\elevation_stats.csv"
            self.dem_stat = pd.read_csv(self.dem_stat)
            if train_dem_num is not None:
                dem_folder = [train_dem_num]
            else:
                dem_folder = [int(f)for f in os.listdir(self.flood_path) if os.path.isdir(os.path.join(self.flood_path, f))]
            rainfall_path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\scenario_rainfall.csv'
            self.maxmin_duv = "C:\\Users\\User\\Desktop\\dev\\50PNG\\maxmin_duv.csv"
            self.ca4d_d = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\d'
            self.ca4d_vx = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\vx'
            self.ca4d_vy = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\vy'
        if test:
            dem_folder = "C:\\Users\\User\\Desktop\\dev\\yilan\\dem"
            self.flood_path = "C:\\Users\\User\\Desktop\\dev\\yilan\\d"
            self.vx = "C:\\Users\\User\\Desktop\\dev\\yilan\\vx"
            self.vy = "C:\\Users\\User\\Desktop\\dev\\yilan\\vy"
            self.dem_stat = "C:\\Users\\User\\Desktop\\dev\\yilan\\maxmin_dem.csv"
            self.dem_stat = pd.read_csv(self.dem_stat)
            self.maxmin_duv = "C:\\Users\\User\\Desktop\\dev\\yilan\\maxmin_duv.csv"
            rainfall_path = 'C:\\Users\\User\\Desktop\\dev\\yilan\\scenario_rainfall.csv'
            dem_folder = [1,2,3]
            self.ca4d_d = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\yilan_ca4d\\d'
            self.ca4d_vx = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\yilan_ca4d\\vx'
            self.ca4d_vy = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\yilan_ca4d\\vy'

        rainfall = pd.read_csv(rainfall_path)
        # remove first row, no 0 row 
        rainfall = rainfall.iloc[:, :]
        self.duv_stat = pd.read_csv(self.maxmin_duv)
        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []

        val = False
        # Iterate through each column
        for dem_num in dem_folder:
            if dem_num in [11, 43, 47, 57, 65,
                           18, 35, 37, 42, 45]:
                continue
            for col in rainfall.columns:
                if col == 'time':
                    continue
                col_num = int(col.split("_")[1])
                if (val and col_num not in [2]) or (not val and col_num in []):
                    continue
                if (col_num in testing_rainfall and not test) or (test and col_num not in testing_rainfall):
                    continue
                cell_values = []
                # Iterate through each row in the current column
                for row in range(len(rainfall)):
                    cell_value = rainfall.iloc[row][col]
                    cell_values.append(np.ceil(cell_value))
                    # make it a len 24 list if not append 0 in front
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    sum_rainfall = sum(temp[24:])
                    # if all the value is 0, then skip
                    if not test and sum_rainfall <= 5:
                        if np.random.rand() < 0.8:
                            continue
                    rainfall_cum_value.append(temp[:])
                    # col_num is the rainfall index, and row is the time index
                    cell_positions.append((dem_num, col_num, row))   

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        # print training data len
        print(f"Training data length: {len(self.cell_positions)}")
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __find_flood_image(self, cell_position, flood_path):
        dem, col, row = cell_position
        path_code = {self.flood_path: 'd', self.vx: 'vx', self.vy: 'vy'}
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                folder_name = f"RF{col:02d}"
            else:
                folder_name = f"RF{col}"
        image_name = f"{dem_folder_name}_{folder_name}_{path_code[flood_path]}_{row:03d}_00.png"
        image_path = os.path.join(flood_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_ca4d_image(self, cell_position, flood_path):
        dem, col, row = cell_position
        path_code = {self.ca4d_d: 'd', self.ca4d_vx: 'vx', self.ca4d_vy: 'vy'}
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"rf{col:02d}"
        else:
            folder_name = f"rf{col}"
        image_name = f"ca4d_{dem_folder_name}_{folder_name}_{path_code[flood_path]}_{row:03d}.png"
        image_path = os.path.join(flood_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_dem_image(self, cell_position):
        dem_num = cell_position[0]
        dem_path = os.path.join(self.dem_folder, f'{dem_num}.png')
        return dem_path
    
    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        # get rainfall
        rainfall = self.rainfall[index]
        rainfall = np.array(rainfall, dtype=np.int64)
        cur_rainfall = rainfall[-1]

        # get the dem, rainfall index and time index 
        cell_position = self.cell_positions[index]

        # get dem
        dem_path = self.__find_dem_image(cell_position)
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
        # normalize different dem to 0-255 based on min max elevation
        min_elev = int(dem_cur_state['Min Elevation'].iloc[0])
        max_elev = int(dem_cur_state['Max Elevation'].iloc[0])
        real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
        dem_image = (real_height - (-3)) / (125 + 3) * 255
        # clamp dem_image to 0-255
        dem_image = np.clip(dem_image, 0, 255)
        dem_image = np.array(dem_image, dtype=np.uint8)

        vx_vy_cur_state = self.duv_stat[self.duv_stat['terrain'] == cell_position[0]]
        min_vx = vx_vy_cur_state['vx_min'].iloc[0]
        max_vx = vx_vy_cur_state['vx_max'].iloc[0]
        min_vy = vx_vy_cur_state['vy_min'].iloc[0]
        max_vy = vx_vy_cur_state['vy_max'].iloc[0]
        min_depth = vx_vy_cur_state['depth_min'].iloc[0]
        max_depth = vx_vy_cur_state['depth_max'].iloc[0]
        
        image_path = self.__find_flood_image(cell_position, self.flood_path)
        vx_path = self.__find_flood_image(cell_position, self.vx)
        vy_path = self.__find_flood_image(cell_position, self.vy)
        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)

        flood_image_real = (1-flood_image.astype(np.float32) / 255.0) * (max_depth - min_depth) + min_depth
        flood_image = 255 - (flood_image_real - 0) / (6 - 0) * 255

        vx_image = np.array(vx_image, dtype=np.float16) 
        vy_image = np.array(vy_image, dtype=np.float16)
        vx_image = (vx_image/255) * (max_vx - min_vx) + min_vx
        vx_image = (vx_image - (-4)) / (4 - (-4)) * 255
        vy_image = (vy_image/255) * (max_vy - min_vy) + min_vy
        vy_image = (vy_image - (-4)) / (4 - (-4)) * 255

        vx_image_real = vx_image/255 * (max_vx - min_vx) + min_vx
        vy_image_real = vy_image/255 * (max_vy - min_vy) + min_vy
        prev_cell = (cell_position[0], cell_position[1], cell_position[2]-1) if cell_position[2] > 0 else (cell_position[0], cell_position[1], cell_position[2])
        prev_vx_path = self.__find_flood_image(prev_cell, self.vx)
        prev_vx_image = cv2.imread(prev_vx_path, cv2.IMREAD_UNCHANGED)
        prev_vy_path = self.__find_flood_image(prev_cell, self.vy)
        prev_vy_image = cv2.imread(prev_vy_path, cv2.IMREAD_UNCHANGED)
        prev_h_path = self.__find_flood_image(prev_cell, self.flood_path)
        prev_h_image = cv2.imread(prev_h_path, cv2.IMREAD_UNCHANGED)
        prev_vx_image = np.array(prev_vx_image, dtype=np.float16)
        prev_vy_image = np.array(prev_vy_image, dtype=np.float16)
        prev_h_image = np.array(prev_h_image, dtype=np.float16)
        prev_vx_image_real = prev_vx_image/255 * (max_vx - min_vx) + min_vx
        prev_vy_image_real = prev_vy_image/255 * (max_vy - min_vy) + min_vy
        prev_h_image_real = (1-prev_h_image.astype(np.float32) / 255.0) * (max_depth - min_depth) + min_depth
        
        ca4d_d_path = self.__find_ca4d_image(cell_position, self.ca4d_d)
        ca4d_d_image = cv2.imread(ca4d_d_path, cv2.IMREAD_UNCHANGED)
        ca4d_vx_path = self.__find_ca4d_image(cell_position, self.ca4d_vx)
        ca4d_vx_image = cv2.imread(ca4d_vx_path, cv2.IMREAD_UNCHANGED)
        ca4d_vy_path = self.__find_ca4d_image(cell_position, self.ca4d_vy)
        ca4d_vy_image = cv2.imread(ca4d_vy_path, cv2.IMREAD_UNCHANGED)

        ca4d_d_image = (1-ca4d_d_image.astype(np.float32) / 255.0) * (max_depth - min_depth) + min_depth
        ca4d_d_image = 255 - (ca4d_d_image - 0) / (6 - 0) * 255
        ca4d_vx_image = np.array(ca4d_vx_image, dtype=np.float16)
        ca4d_vy_image = np.array(ca4d_vy_image, dtype=np.float16)
        ca4d_vx_image = (ca4d_vx_image/255) * (max_vx - min_vx) + min_vx
        ca4d_vx_image = (ca4d_vx_image - (-4)) / (4 - (-4)) * 255
        ca4d_vy_image = (ca4d_vy_image/255) * (max_vy - min_vy) + min_vy
        ca4d_vy_image = (ca4d_vy_image - (-4)) / (4 - (-4)) * 255

        dem_image = np.array(dem_image, dtype=np.uint8)
        flood_image = np.array(flood_image, dtype=np.uint8)
        vx_image = np.array(vx_image, dtype=np.uint8)
        vy_image = np.array(vy_image, dtype=np.uint8)
        ca4d_d_image = np.array(ca4d_d_image, dtype=np.uint8)
        ca4d_vx_image = np.array(ca4d_vx_image, dtype=np.uint8)
        ca4d_vy_image = np.array(ca4d_vy_image, dtype=np.uint8)
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        vx_image = self.transform(vx_image)
        vy_image = self.transform(vy_image)
        ca4d_d_image = self.transform(ca4d_d_image)
        ca4d_vx_image = self.transform(ca4d_vx_image)
        ca4d_vy_image = self.transform(ca4d_vy_image)

        dem_image = (dem_image - 0.18) / 0.22
        flood_image = (flood_image - 0.986) / 0.043
        vx_image = (vx_image - 0.498) / 0.0049
        vy_image = (vy_image - 0.499) / 0.0043
        ca4d_d_image = (ca4d_d_image - 0.985) / 0.05
        ca4d_vx_image = (ca4d_vx_image - 0.498) / 0.017
        ca4d_vy_image = (ca4d_vy_image - 0.499) / 0.017

        physics_features = [cur_rainfall, flood_image_real, vx_image_real, vy_image_real, 
                            prev_h_image_real, prev_vx_image_real, prev_vy_image_real]

        return dem_image, flood_image, vx_image, vy_image, ca4d_d_image, ca4d_vx_image, ca4d_vy_image, physics_features

class singleDEMFloodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(singleDEMFloodDataset, self).__init__()
        self.opt = opt
        dem_path = 'C:\\Users\\User\\Desktop\\dev\\new_train\\tainan_new.png'
        self.spm_folder = 'C:\\Users\\User\\Desktop\\dev\\SPM_output'
        self.dem = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        # turn 255 edge to dem.mean()
        # self.dem[:,0] = int(np.mean(self.dem))
        # self.dem[-2:,:] = int(np.mean(self.dem))
        self.test = test

        self.flood_path = 'C:\\Users\\User\\Desktop\\dev\\new_train\\tainan_png'
        self.vx = 'C:\\Users\\User\\Desktop\\dev\\new_train\\Vx'
        self.vy = 'C:\\Users\\User\\Desktop\\dev\\new_train\\Vy'

        rainfall_path = 'C:\\Users\\User\\Desktop\\dev\\new_train\\train.csv'
        if test:
            rainfall_path = 'C:\\Users\\User\\Desktop\\dev\\new_test\\test.csv'
            self.flood_path = 'C:\\Users\\User\\Desktop\\dev\\new_test\\TEST_png'

        rainfall = pd.read_csv(rainfall_path)
        # remove first row, no 0 row 
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cur_rainfall = []
        cell_positions = []
        spm = []
        val = False
        # Iterate through each column
        for col in rainfall.columns:
            if col == 'time':
                continue
            col_num = int(col.split("_")[1])
            if (val and col_num not in [2]) or (not val and col_num in []):
                continue
            cell_values = []
            # Iterate through each row in the current column
            for row in range(len(rainfall)):
                cell_value = rainfall.iloc[row][col]
                cell_values.append(np.floor(cell_value))
                cur_rainfall.append(cell_value)
                # make it a len 24 list if not append 0 in front
                temp = [0] * (24 - len(cell_values))
                temp.extend(cell_values)
                if len(temp) == 25:
                    temp = temp[1:]
                # Decayed weighted sum: first elem * 1, second * 0.95, third * 0.95^2, ...
                # decay = 0.95
                # weights = [decay ** (len(temp)-i) for i in range(len(temp))]
                # sum_rainfall = sum([v * w for v, w in zip(temp, weights)])
                sum_rainfall = sum(temp)
                spm.append(int(np.ceil(sum_rainfall / 5) * 5))
                rainfall_cum_value.append(temp)
                cell_positions.append((col_num, row))

        self.rainfall = rainfall_cum_value
        self.cur_rainfall = cur_rainfall
        self.cell_positions = cell_positions
        self.spm = spm
        print(f"Training data length: {len(self.cell_positions)}")
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Lambda(lambda t: (t * 2) - 1)
        ])

    def __find_image(self, cell_position, path):
        col, row = cell_position
        path_code = {self.flood_path: 'd', self.vx: 'vx', self.vy: 'vy'}
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                folder_name = f"RF{col:02d}"
            else:
                folder_name = f"RF{col}"
        cur_path_code = path_code[path]
        image_name = f"{folder_name}_{cur_path_code}_{row:03d}_00.png"
        image_path = os.path.join(path, folder_name, image_name)
        return image_path

    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        dem_image = self.dem
        rainfall = self.rainfall[index]
        cur_rainfall = rainfall[-1]
        spm = self.spm[index]
        rainfall = np.array(rainfall, dtype=np.int64)
        # rainfall = rainfall.reshape(1, 24)
        spm_path = os.path.join(self.spm_folder, f'SPM_1_{spm}.png')
        spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
        cell_position = self.cell_positions[index]

        image_path = self.__find_image(cell_position, self.flood_path)
        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        vx_path = self.__find_image(cell_position, self.vx)
        vx_image = cv2.imread(vx_path, cv2.IMREAD_UNCHANGED)

        vy_path = self.__find_image(cell_position, self.vy)
        vy_image = cv2.imread(vy_path, cv2.IMREAD_UNCHANGED)

        prev_cell = (cell_position[0], cell_position[1]-1) if cell_position[1] > 0 else (cell_position[0], cell_position[1])
        prev_vx_path = self.__find_image(prev_cell, self.vx)
        prev_vx_image = cv2.imread(prev_vx_path, cv2.IMREAD_UNCHANGED)
        prev_vy_path = self.__find_image(prev_cell, self.vy)
        prev_vy_image = cv2.imread(prev_vy_path, cv2.IMREAD_UNCHANGED)
        prev_h_path = self.__find_image(prev_cell, self.flood_path)
        prev_h_image = cv2.imread(prev_h_path, cv2.IMREAD_UNCHANGED)

        binary_mask = (flood_image <= 250).astype('uint8')
        binary_mask = np.expand_dims(binary_mask, axis=0)
        flood_image = np.array(flood_image, dtype=np.uint8)
        
        spm_image = self.transform(spm_image)
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)

        spm_image = (spm_image * 2) - 1
        # print(dem_image.mean(), dem_image.std())
        dem_image = (dem_image - dem_image.mean()) / dem_image.std()
        flood_image = (flood_image - 0.98) / 0.035

        # vx_image = self.transform(vx_image)
        # vy_image = self.transform(vy_image)
        # prev_vx_image = self.transform(prev_vx_image)
        # prev_vy_image = self.transform(prev_vy_image)

        # vx_image = (vx_image - 0.5) / 0.0043
        # prev_vx_image = (prev_vx_image - 0.5) / 0.0043
        # vy_image = (vy_image - 0.5) / 0.0047
        # prev_vy_image = (prev_vy_image - 0.5) / 0.0047

        # scale vx and vy from 0-256 to -4-4
        # convert vx_image, vy_image, prev_vx_image, prev_vy_image to numpy with float16
        vx_image = torch.tensor(vx_image, dtype=torch.float32)
        vy_image = torch.tensor(vy_image, dtype=torch.float32)
        prev_vx_image = torch.tensor(prev_vx_image, dtype=torch.float32)
        prev_vy_image = torch.tensor(prev_vy_image, dtype=torch.float32)
        vx_image = (vx_image - 127) / 32
        vy_image = (vy_image - 127) / 32
        prev_vx_image = (prev_vx_image - 127) / 32
        prev_vy_image = (prev_vy_image - 127) / 32

        prev_h_image = torch.tensor(prev_h_image, dtype=torch.float32)
        prev_h_image = (1-prev_h_image / 255) * 4

        # dem_image = (dem_image *2) - 1
        # flood_image = (flood_image *2) - 1
        # dem_image = torch.randn_like(dem_image)

        return flood_image, dem_image, binary_mask, rainfall, image_path, spm_image, \
               vx_image, vy_image, prev_vx_image, prev_vy_image, prev_h_image, cur_rainfall