import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np
import torchvision.utils as vutils

from logger import Logger
from i2sb.embedding import RainfallEmbedder
from i2sb.network import Image256Net
import torchvision.utils as tu
# =================================================================================
# 2. DATASET (Modified for GANs)
# =================================================================================
class singleDEMFloodDataset(Dataset):
    """Modified dataset to return a (condition, target) pair."""
    def __init__(self, val=False, test=False):
        super(singleDEMFloodDataset, self).__init__()
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
        spm = self.spm[index]
        rainfall = np.array(rainfall, dtype=np.int64)
        print(rainfall)
        print(spm)
        # rainfall = rainfall.reshape(1, 24)
        spm_path = os.path.join(self.spm_folder, f'SPM_1_{spm}.png')
        spm_image = cv2.imread(spm_path, cv2.IMREAD_GRAYSCALE)
        cell_position = self.cell_positions[index]

        image_path = self.__find_image(cell_position, self.flood_path)
        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        spm_image = self.transform(spm_image)
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)

        spm_image = (spm_image * 2) - 1
        # print(dem_image.mean(), dem_image.std())
        dem_image = (dem_image - dem_image.mean()) / dem_image.std()
        flood_image = (flood_image - 0.98) / 0.035
        
        return flood_image, dem_image, rainfall, spm_image, image_path


def inference():
    print("\n--- Starting Inference ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("spm_outputs/test_images", exist_ok=True)
    condition_channels = 6
    target_channels = 1

    # --- Get a sample for conditioning ---
    test_dataset = singleDEMFloodDataset(test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # --- Inference Loop ---
    file_index = 0
    for i, (flood, dem, rainfall, spm, image_path) in enumerate(tqdm(test_dataloader, desc="Generating Images")):
        flood = flood.to(device)
        dem = dem.to(device)
        rainfall = rainfall.to(device)
        spm = spm.to(device)
        # print(f"Processing batch {i+1}/{len(test_dataloader)} with image paths: {image_path}")

            # Loop through each image in the current batch
        for j in range(flood.size(0)):
            # Get the j-th image from the batch
            gen_img = spm[j]
            gen_img = (gen_img + 1) / 2
            vutils.save_image(gen_img, f"spm_outputs/test_images/{file_index:04d}.png")
            file_index += 1

    print(f"\n--- Inference Finished. Saved {file_index} images. ---")


if __name__ == '__main__':
    # train()
    inference()