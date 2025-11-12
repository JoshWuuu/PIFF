import cv2
from matplotlib import pyplot as plt
import numpy as np


def contrast_stretching(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - 200) / (255 - 200) * 255
    img = img.astype(np.uint8)
    return img

# image_path = 'C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\results\\piff\\test3_nfe10_euler-dcvar\\recon_RF133_d_016_00.png'
# image_path = 'C:\\Users\\User\\Desktop\\dev\\new_test\\TEST_png\\RF133\\RF133_d_016_00.png'
# 254 258 262 266 270 274
image_path = "C:\\Users\\User\\Desktop\\dev\\gan_outputs\\test_images\\0274.png"

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# image_test_path = 'C:\\Users\\User\\Desktop\\dev\\test_dem\\61\\RF08\\61_RF08_024.png'
# image_test = cv2.imread(image_test_path, cv2.IMREAD_UNCHANGED)

# diff_image = cv2.absdiff(image, image_test)

# dem_image_path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\61.png'
# dem = cv2.imread(dem_image_path, cv2.IMREAD_UNCHANGED)[:,:,0]

# zeros = np.where(image == 0)
# print("Zero-value pixels:", zeros)

image = contrast_stretching(image)

cv2.imwrite('plots\\gan_contrast_133_24_gen_nfe10.png', image)

# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(image, cmap='gray')
# ax.axis('off')
# cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Flood Depth (cm)')
# plt.savefig('plots\\colorbar.png', bbox_inches='tight', dpi=300)

# upper_left = (100, 20)
# lower_right = (200, 120)
# upper_left = (30,20)
# lower_right = (130, 120)
# upper_left = (40, 5)
# lower_right = (90, 50)

# crop = image[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

# # # # save the crop 
# cv2.imwrite('plots\\gt_gen_16_crop.png', crop)


test_rainfall = 'C:\\Users\\User\\Desktop\\dev\\new_test\\test.csv'

# # plot the column of inflow_132 as time series plot
import matplotlib.pyplot as plt
import pandas as pd

time = np.arange(1, 25)
rainfall = pd.read_csv(test_rainfall)
uniform_rainfall = rainfall['inflow_02'][1:]
nonuniform_rainfall = rainfall['inflow_04'][1:]
cur_rainfall = rainfall['inflow_133'][1:]
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

axes[0].plot(time, uniform_rainfall, '-o', color='tab:green')
# axes[0].set_ylabel('Rainfall (mm/hr)', fontsize=14)
axes[0].set_title('Uniform Rainfall', fontsize=18)
axes[0].set_ylim(0, 60)
axes[0].grid(True)

axes[1].plot(time, nonuniform_rainfall, '-o', color='tab:red')
axes[1].set_ylabel('Rainfall (mm/hr)', fontsize=18)
axes[1].set_title('Non-Uniform Rainfall', fontsize=18)
axes[1].set_ylim(0, 60)
axes[1].grid(True)

axes[2].plot(time, cur_rainfall, '-o', color='tab:blue')
# axes[2].set_ylabel('Rainfall (mm/hr)', fontsize=14)
axes[2].set_title('Real-Event Rainfall', fontsize=18)
axes[2].set_xlabel('Rainfall Timestep (H)', fontsize=18)
axes[2].set_ylim(0, 60)
axes[2].grid(True)

for ax in axes:
    ax.set_xticks(time)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

plt.tight_layout()
plt.savefig('plots\\rainfall_comparison.png', dpi=300)
plt.close(fig)

import matplotlib.colors as mcolors
import matplotlib.cm as cm

colorbar_min_depth = 0
colorbar_max_depth = 90

# Note: These pixel values (0-255) are for the internal data of your image,
# which `imshow` will map to colors. The colorbar directly maps *physical values*.
min_pixel_value = 0
max_pixel_value = 255

# To make darker color (black) correspond to 90cm (max depth)
# and lighter color (white) correspond to 0cm (min depth)
# We need the 'gray_r' (reversed gray) colormap.
# 'gray_r' maps low normalized data values (e.g., vmin) to white, and high normalized data values (e.g., vmax) to black.
cmap_to_use = 'gray_r'

colorbar_label = 'Flood Depth (cm)'
output_filename = 'plots\\colorbar_dark_is_90cm_1_horizontal.png' # Changed filename for clarity

# 1. Create a "mappable" object that defines the colormap and normalization.
# The `vmin` and `vmax` define the physical range that the colorbar represents.
norm = mcolors.Normalize(vmin=colorbar_min_depth, vmax=colorbar_max_depth)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap_to_use)
mappable.set_array([]) # Give it an empty array

# 2. Create a new figure and an axes object specifically for the colorbar
fig, ax = plt.subplots(figsize=(10, 0.2)) # Adjust size as needed

label_fontsize = 20 # Adjust as desired (e.g., 12, 14, 16)
tick_labels_fontsize = 17 
# 3. Draw the colorbar into the new axes
cbar = fig.colorbar(mappable, cax=ax, orientation='horizontal', label=colorbar_label)
cbar.set_label(colorbar_label, fontsize=label_fontsize) # Set font size for the main label
cbar.ax.tick_params(labelsize=tick_labels_fontsize) #
# 4. Save the figure (which now only contains the colorbar)

plt.savefig(output_filename, bbox_inches='tight', dpi=300)