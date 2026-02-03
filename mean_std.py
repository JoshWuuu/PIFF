# read all the images in the floder and calculate mean std
import os

import cv2
import numpy as np
import pandas as pd

# path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png'

# files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.endswith('.csv') and not f.endswith('025_00.png')]
# mean = 0
# std = 0
# dem_stat = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\elevation_stats.csv'
# dem_stat = pd.read_csv(dem_stat)
# pixel_list = []
# for file_name in files:
#     image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_UNCHANGED)[:,:,0]
#     filename = file_name.split('.')[0]
#     dem_cur_state = dem_stat[dem_stat['Filename'] == int(filename)]
#     min_elev = int(dem_cur_state['Min Elevation'])
#     max_elev = int(dem_cur_state['Max Elevation'])
#         # do a normalization with max = 410 min = -3, with current max = max_elev, min = min_elev
#     real_height = image / 255 * (max_elev - min_elev) + min_elev
#     dem_image = (real_height - (-3)) / (125 + 3) * 255
#     # perform totensor()
#     dem_image = dem_image.astype(np.float32)
#     dem_image = np.clip(dem_image, 0, 255)
#     dem_image = dem_image / 255.0  # normalize to [0, 1]
#     # mean += nstd(real_height)
#     pixel_list.extend(dem_image.flatten())

# pixel_list = np.array(pixel_list)
# mean = np.mean(pixel_list)
# std = np.std(pixel_list)
# print(f'Mean: {mean}, Std: {std}')

# pixel_list = []
path = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\d'
vx_path = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\vx'
vy_path = 'C:\\Users\\User\\Desktop\\dev\\ca4d\\Multi_ca4d\\vy'

# get all the image in the path, the subfolder is 1 and one more subfolder RF01 then the png files, get all the png file name
files = [(f, os.path.join(path, f, rainfall, png_file)) for f in os.listdir(path) for rainfall in os.listdir(os.path.join(path, f)) for png_file in os.listdir(os.path.join(path, f, rainfall)) if png_file.endswith('.png')]
vx_path = [(f,  os.path.join(vx_path, f, rainfall, png_file)) for f in os.listdir(vx_path) for rainfall in os.listdir(os.path.join(vx_path, f)) for png_file in os.listdir(os.path.join(vx_path, f, rainfall)) if png_file.endswith('.png')]
vy_path = [(f,  os.path.join(vy_path, f, rainfall, png_file)) for f in os.listdir(vy_path) for rainfall in os.listdir(os.path.join(vy_path, f)) for png_file in os.listdir(os.path.join(vy_path, f, rainfall)) if png_file.endswith('.png')]

mean_sum = 0
std_sum = 0
vx_mean_sum = 0
vx_std_sum = 0
vy_mean_sum = 0
vy_std_sum = 0 
pixel_count = 0
duv_file = "C:\\Users\\User\\Desktop\\dev\\50PNG\\maxmin_duv.csv"
duv_stat = pd.read_csv(duv_file)


for (dem, file), (_, vx_file), (_, vy_file) in zip(files, vx_path, vy_path):
    if int(dem) in [11, 43, 47, 57, 65,
                    18, 35, 37, 42, 45]:
        continue 
    cur_state = duv_stat[duv_stat['terrain'] == int(dem)]
    min_h = cur_state['depth_min'].iloc[0]
    max_h = cur_state['depth_max'].iloc[0]
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    flood_image = (1-image.astype(np.float32) / 255.0) * (max_h - min_h) + min_h
    flood_image = 255 - (flood_image - 0) / (6 - 0) * 255
    if image is None:
        continue
    image = image.astype(np.float64) / 255.0
    
    mean_sum += np.sum(image)
    std_sum += np.sum(np.square(image))
    pixel_count += image.size

    min_vx = cur_state['vx_min'].iloc[0]
    max_vx = cur_state['vx_max'].iloc[0]
    min_vy = cur_state['vy_min'].iloc[0]
    max_vy = cur_state['vy_max'].iloc[0]

    vx_image = cv2.imread(vx_file, cv2.IMREAD_UNCHANGED)
    vy_image = cv2.imread(vy_file, cv2.IMREAD_UNCHANGED)

    vx_image = (vx_image/255) * (max_vx - min_vx) + min_vx
    vx_image = (vx_image - (-4)) / (4 - (-4)) * 255
    vy_image = (vy_image/255) * (max_vy - min_vy) + min_vy
    vy_image = (vy_image - (-4)) / (4 - (-4)) * 255

    vx_image = vx_image.astype(np.float64) / 255.0
    vy_image = vy_image.astype(np.float64) / 255.0

    vx_mean_sum += np.sum(vx_image)
    vx_std_sum += np.sum(np.square(vx_image))
    vy_mean_sum += np.sum(vy_image)
    vy_std_sum += np.sum(np.square(vy_image))

mean = mean_sum / pixel_count
std = np.sqrt(std_sum / pixel_count - mean**2)

print(f'Mean: {mean}, Std: {std}')

mean_vx = vx_mean_sum / pixel_count
std_vx = np.sqrt(vx_std_sum / pixel_count - mean_vx**2)
mean_vy = vy_mean_sum / pixel_count
std_vy = np.sqrt(vy_std_sum / pixel_count - mean_vy**2)
print(f'Vx Mean: {mean_vx}, Vx Std: {std_vx}')
print(f'Vy Mean: {mean_vy}, Vy Std: {std_vy}')



# pixel_list = []
# path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\d'
# duv_file = "C:\\Users\\User\\Desktop\\dev\\50PNG\\maxmin_duv.csv"
# duv_stat = pd.read_csv(duv_file)

# # get all the image in the path, the subfolder is 1 and one more subfolder RF01 then the png files, get all the png file name
# files = [(f, os.path.join(path, f, rainfall, png_file)) for f in os.listdir(path) for rainfall in os.listdir(os.path.join(path, f)) for png_file in os.listdir(os.path.join(path, f, rainfall)) if png_file.endswith('.png')]

# mean_sum = 0
# std_sum = 0
# pixel_count = 0
# vals_list = []
# for (dem,file) in files: 
#     if int(dem) in [11, 43, 47, 57, 65]:
#         continue
#     cur_state = duv_stat[duv_stat['terrain'] == int(dem)]
#     min_h = cur_state['depth_min'].iloc[0]
#     max_h = cur_state['depth_max'].iloc[0]
#     raw = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#     if raw is None:
#         continue
#     # Mask out nodata pixels (raw==0)this image contributes nothing

#     # Map to heights only once, then index with mask
#     mapped = (1.0 - raw.astype(np.float64) / 255.0) * (max_h - min_h) + min_h
#     mask = mapped != 0
#     vals = mapped[mask]
#     vals_list.extend(vals)
#     mean_sum += vals.sum()
#     std_sum  += np.square(vals).sum()
#     pixel_count += vals.size

# mean = mean_sum / pixel_count
# std = np.sqrt(std_sum / pixel_count - mean**2)

# print(f'Mean: {mean}, Std: {std}')

# # plot the histogram of vals_list
# import matplotlib.pyplot as plt
# plt.hist(vals_list, bins=100, color='blue', alpha=0.7)
# plt.title('Histogram of Flood Depth Values')
# plt.xlabel('Flood Depth (cm)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig('depth_histogram.png')
# path = 'C:\\Users\\User\\Desktop\\dev\\new_train\\Vy'

# file_path = [os.path.join(path, rainfall, png_file) for rainfall in os.listdir(path) for png_file in os.listdir(os.path.join(path, rainfall)) if png_file.endswith('.png')]
# mean_sum = 0
# std_sum = 0
# pixel_count = 0

# for file in file_path:
#     image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#     if image is None:
#         continue
#     image = image.astype(np.float64) / 255.0
    
#     mean_sum += np.sum(image)
#     std_sum += np.sum(np.square(image))
#     pixel_count += image.size

# mean = mean_sum / pixel_count
# std = np.sqrt(std_sum / pixel_count - mean**2)
# print(f'Mean: {mean}, Std: {std}')

# count the percentage of the pixel value that is larger than 100
# percentage = len(np.where(pixel_list > 55)[0]) / len(pixel_list)
# print(percentage)
# print(mean, std)

# depth_folder = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\DEPTH_png'
# dem_depth_folder = [f for f in os.listdir(depth_folder) if os.path.isdir(os.path.join(depth_folder, f))]

# for folder in 