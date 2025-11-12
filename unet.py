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

# =================================================================================
# 1. MODELS (Generator and Discriminator)
# =================================================================================

class UNetDownBlock(nn.Module):
    """A downsampling block for the U-Net Generator."""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDownBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUpBlock(nn.Module):
    """An upsampling block for the U-Net Generator."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UnetGenerator(nn.Module):
    """The U-Net Generator Architecture (Pix2Pix)."""
    def __init__(self, in_channels=6, out_channels=1):
        super(UnetGenerator, self).__init__()
        # Encoder (Downsampling path)
        self.down1 = UNetDownBlock(in_channels, 64, normalize=False)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256)
        self.down4 = UNetDownBlock(256, 512, dropout=0.5)
        self.down5 = UNetDownBlock(512, 512, dropout=0.5)
        self.down6 = UNetDownBlock(512, 512, dropout=0.5)
        
        # Bottleneck
        self.center = UNetDownBlock(512, 512, dropout=0.5)

        # Decoder (Upsampling path)
        self.up1 = UNetUpBlock(512, 512, dropout=0.5)
        self.up2 = UNetUpBlock(1024, 512, dropout=0.5)
        self.up3 = UNetUpBlock(1024, 512, dropout=0.5)
        self.up4 = UNetUpBlock(1024, 256)
        self.up5 = UNetUpBlock(512, 128)
        self.up6 = UNetUpBlock(256, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(), # Output is in range [-1, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        c = self.center(d6)
        
        u1 = self.up1(c, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        
        return self.final_up(u6)

class PatchGANDiscriminator(nn.Module):
    """The PatchGAN Discriminator Architecture."""
    def __init__(self, in_channels=7): # 6 (condition) + 1 (image)
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False) # Final 1-channel output
        )

    def forward(self, img):
        # Concatenate condition and image channels
        # combined = torch.cat((condition, img), 1)
        return self.model(img)

def weights_init_normal(m):
    """Initializes model weights."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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
        
        return flood_image, dem_image, rainfall, spm_image

# =================================================================================
# 3. TRAINING SCRIPT
# =================================================================================
def train():
    # --- Hyperparameters ---
    epochs = 50 # Pix2Pix needs a good number of epochs
    batch_size = 4
    lr = 0.0002
    beta1 = 0.5
    lambda_l1 = 100.0 # Weight for L1 loss
    img_size = 256
    condition_channels = 1
    target_channels = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    log = Logger(0, "logs")
    log.info("=======================================================")
    log.info("         unet")
    log.info("=======================================================")
    # --- Models ---
    noise_level = torch.linspace(1, 0, steps=1000).to(device)
    generator = Image256Net(log, noise_levels=noise_level).to(device)
    rainfall_emb = RainfallEmbedder(256, 1).to(device)
    discriminator = PatchGANDiscriminator(in_channels=1).to(device)
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    # --- Loss and Optimizers ---
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    params = [
        {"params": generator.parameters(), "lr": lr},
        {"params": rainfall_emb.parameters(), "lr": 1e-4},
        # {"params": spm.parameters(), "lr": 1e-3, "weight_decay": opt.l2_norm},
    ]
    optimizer_G = Adam(params)
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # --- Data ---
    dataset = singleDEMFloodDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Folders for results ---
    os.makedirs("unet_sp_outputs/images", exist_ok=True)
    os.makedirs("unet_sp_outputs/models", exist_ok=True)

    # --- Training Loop ---
    for epoch in range(epochs):
        for i, (flood, dem, rainfall, spm) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            flood = flood.to(device)
            dem = dem.to(device)
            rainfall = rainfall.to(device)
            spm = spm.to(device)

            # Create labels for adversarial loss
            # The discriminator output is a patch grid, e.g., (B, 1, 30, 30)
            # We create labels of the same size.
            # patch_size = (1, img_size // 2**4, img_size // 2**4)
            # real_labels = torch.ones((flood.size(0), *patch_size), device=device)
            # fake_labels = torch.zeros((flood.size(0), *patch_size), device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # optimizer_D.zero_grad()

            # Loss for real images
            # pred_real = discriminator(flood)
            # loss_real = adversarial_loss(pred_real, real_labels)

            # Generate a fake image
            # rainfall_embedding = rainfall_emb(rainfall)
            # step = torch.ones((flood.size(0), 1, 1, 1), device=device)
            # Generate fake target and detach it immediately.
            # This prevents gradients from flowing to the generator during the discriminator update.
            # fake_target_for_D = generator(dem, step, rainfall_embedding).detach()

            # Loss for fake images
            # pred_fake = discriminator(fake_target_for_D)
            # loss_fake = adversarial_loss(pred_fake, fake_labels)

            # Total discriminator loss
            # loss_D = (loss_real + loss_fake) * 0.5
            # loss_D.backward()
            # optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # We perform a new forward pass for the generator's training step.
            # This creates a new computational graph that we can backpropagate through.
            rainfall_embedding = rainfall_emb(rainfall) # Redo embedding if it has learnable params, otherwise optional
            step = torch.ones((flood.size(0), 1, 1, 1), device=device)
            fake_target_for_G = generator(dem, step, rainfall_embedding, spm=spm)
            # pred_fake_for_G = discriminator(fake_target_for_G)

            # Adversarial loss
            # loss_G_adv = adversarial_loss(pred_fake_for_G, real_labels)

            # L1 reconstruction loss
            loss_G_l1 = l1_loss(fake_target_for_G, flood)

            # Total generator loss
            loss_G = loss_G_l1
            loss_G.backward()
            optimizer_G.step()                                       
            
        # --- Logging and Saving ---
        print(f"\n[Epoch {epoch+1}/{epochs}] [G loss: {loss_G.item():.4f}, l1: {loss_G_l1.item():.4f}]")

        # Save a sample of generated images
        sample_imgs = fake_target_for_G
        # Un-normalize from [-1, 1] to [0, 1]
        sample_imgs = (sample_imgs * 0.035) + 0.98
        vutils.save_image(sample_imgs, f"unet_sp_outputs/images/{epoch+1:03d}.png", nrow=2, normalize=False)

    # Save final model
    torch.save(generator.state_dict(), "unet_sp_outputs/models/generator.pth")
    torch.save(rainfall_emb.state_dict(), "unet_sp_outputs/models/rainfall_emb.pth")
    print("--- Training Finished. Model saved. ---")


def inference():
    print("\n--- Starting Inference ---")
    os.makedirs("unet_sp_outputs/test_images", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    condition_channels = 6
    target_channels = 1
    
    # --- Load Model ---
    log = Logger(0, "logs")
    # --- Models ---
    noise_level = torch.linspace(1, 0, steps=1000).to(device)
    generator = Image256Net(log, noise_levels=noise_level).to(device)
    generator.load_state_dict(torch.load("unet_sp_outputs/models/generator.pth"))
    generator.eval()
    rainfall_emb = RainfallEmbedder(256, 1).to(device)
    rainfall_emb.load_state_dict(torch.load("unet_sp_outputs/models/rainfall_emb.pth"))
    rainfall_emb.eval()

    # --- Get a sample for conditioning ---
    test_dataset = singleDEMFloodDataset(test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # --- Inference Loop ---
    file_index = 0
    with torch.no_grad():
        for i, (flood, dem, rainfall, spm) in enumerate(tqdm(test_dataloader, desc="Generating Images")):
            flood = flood.to(device)
            dem = dem.to(device)
            rainfall = rainfall.to(device)
            spm = spm.to(device)
            step = torch.ones((flood.size(0), 1, 1, 1), device=device)
            # Generate the fake/predicted image batch
            rainfall_embedding = rainfall_emb(rainfall)
            generated_images = generator(flood, step, rainfall_embedding)

            # Loop through each image in the current batch
            for j in range(flood.size(0)):
                # Get the j-th image from the batch
                gen_img = generated_images[j]
                sample_imgs = (gen_img * 0.035) + 0.98
                vutils.save_image(sample_imgs, f"unet_sp_outputs/test_images/{file_index:04d}.png")
                file_index += 1

    print(f"\n--- Inference Finished. Saved {file_index} images. ---")


if __name__ == '__main__':
    train()
    inference()