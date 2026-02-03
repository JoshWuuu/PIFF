# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
from tqdm import tqdm

# from PIFF import loss
import distributed_util as dist_util
from evaluation import build_resnet50
from matplotlib import pyplot as plt

from . import util
from .network import Image256Net
from .diffusion import Diffusion, disabled_train, create_model_config
from .nv_loss import navier_stokes_operators, navier_stokes_operators_torch, continuity_residual_torch
# from i2sb.VQGAN.vqgan import VQModel
from i2sb.base.modules.encoders.modules import SpatialRescaler
from torch.utils.data import DataLoader
from corruption.mixture import floodDataset
from .embedding import RainfallEmbedder

from ipdb import set_trace as debug
from fvcore.nn import FlopCountAnalysis

def build_optimizer_sched(opt, rainfall_embber, net, log, spm=None):

    # optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    # params = list(net.parameters()) + list(rainfall_embber.parameters()) + list(spm.parameters())
    params = [
        {"params": net.parameters(), "lr": opt.lr, "weight_decay": opt.l2_norm},
        {"params": rainfall_embber.parameters(), "lr": 1e-4, "weight_decay": opt.l2_norm},
        # {"params": spm.parameters(), "lr": 1e-3, "weight_decay": opt.l2_norm},
    ]
    optimizer = AdamW(params)
    # log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        # sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=opt.num_itr,   # full run, no restarts
            eta_min=1e-8
        )
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_grad_flow(model):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    fig = plt.figure(figsize=(10,5))
    plt.plot(ave_grads, alpha=0.5, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    # plt.xticks(range(0,len(ave_grads),1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.show()

class WeightedSumHead(nn.Module):
    """
    Predicts a scalar SPM rainfall from a 24-dim rainfall vector:
      spm_pred = x @ w   (no bias, no softmax)
    w initialized to all ones.
    """
    def __init__(self, dim: int = 24, jitter: float = 1e-2):
        super().__init__()
        self.lin = nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            self.lin.weight.fill_(1.0)
            self.lin.weight.add_(jitter * torch.randn_like(self.lin.weight))
        
    def forward(self, rainfall_vec: torch.Tensor):  # (B, 24)
        # returns (B, 1) continuous rainfall amount (same unit as your SPM bins)
        # convert rainfall to float 64
        dtype = self.lin.weight.dtype
        device = self.lin.weight.device
        rainfall_vec = rainfall_vec.to(device=device, dtype=dtype)
        return self.lin(rainfall_vec)

class WeightedSumHeadUpdated(nn.Module):
    """
    Delta-with-base version.
    - Learn a shared base weight vector w_base (size = dim)
    - Each DEM id has a learned delta vector; unseen DEMs use delta=0 (fallback to base)
    - Prediction: y = sum_i x_i * (w_base[i] + delta_dem[i])
    """
    def __init__(self, dim: int = 24, dem_num: int = 60, jitter: float = 1e-2):
        super().__init__()
        self.dim = dim
        self.dem_num = dem_num
        self.test_dem_num = [61, 62, 65, 67, 69]

        # Shared base weights
        self.w_base = nn.Parameter(torch.ones(dim))
        with torch.no_grad():
            self.w_base.add_(jitter * torch.randn_like(self.w_base))

        # DEM-specific deltas (start at zero so base is the initial fallback)
        self.w_dem_delta = nn.Embedding(dem_num, dim)
        with torch.no_grad():
            self.w_dem_delta.weight.zero_()

    def forward(self, rainfall_vec: torch.Tensor, dem_id: torch.Tensor) -> torch.Tensor:
        """
        rainfall_vec: (B, dim)
        dem_id:       (B,) LongTensor of DEM indices
        - If dem_id is outside [0, dem_num-1], delta=0 (fallback to base)
        Returns: (B, 1)
        """
        if dem_id.dtype != torch.long:
            dem_id = dem_id.long()

        device = self.w_base.device
        dtype = self.w_base.dtype
        x = rainfall_vec.to(device=device, dtype=dtype)
        dem_id = dem_id.to(device=device)

        # Known mask and safe indices (clamped to valid range for embedding lookup)
        known = (dem_id >= 0) & (dem_id < self.dem_num)
        idx = torch.clamp(dem_id, 0, max(self.dem_num, 0)) - 1

        # Gather delta; zero it where DEM is unknown
        delta = self.w_dem_delta(idx)                    # (B, dim)
        delta = torch.where(known.unsqueeze(-1), delta, torch.zeros_like(delta))

        # Combine base + delta and do weighted sum
        w = self.w_base.unsqueeze(0) + delta             # (B, dim)
        y = (x * w).sum(dim=-1, keepdim=True)            # (B, 1)
        return y
    
class DeviationsAroundOnes(nn.Module):
    """
    w = 1 + P u, where P projects to zero-sum subspace so sum(w) stays near 24.
    (Keeps a 'sum' baseline but learns variations.)
    """
    def __init__(self, dim=24, nonneg=False):
        super().__init__()
        self.dim = dim
        # u are free params; init small to break symmetry
        self.u = nn.Parameter(1e-2 * torch.randn(dim))
        # projection matrix P = I - (1/d) 11^T ensures sum(Pu)=0
        P = torch.eye(dim) - torch.full((dim, dim), 1.0/dim)
        self.register_buffer("P", P)
        self.nonneg = nonneg

    def weight(self):
        w = 1.0 + self.P @ self.u
        if self.nonneg:
            w = F.softplus(w)   # keep weights ≥0, optional
        return w

    def forward(self, x):  # x: (B, dim)
        return x @ self.weight().unsqueeze(-1)
    
class ScaledSoftmaxHead(nn.Module):
    """
    w = c * softmax(a).  Init a=0 => equal weights; set c_init=dim => sum(x) at init.
    """
    def __init__(self, dim=24, c_init=None):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.zeros(dim))          # logits
        self.c = nn.Parameter(torch.tensor(float(dim if c_init is None else c_init)))

    def weight(self):
        w = torch.softmax(self.a, dim=0) * self.c
        return w

    def forward(self, x):  # (B, dim)
        return x @ self.weight().unsqueeze(-1)

def soft_bin_blend(spm_bank: torch.Tensor,  # (K, 1, H, W)
                   spm_bins: torch.Tensor,  # (K,)
                   spm_pred: torch.Tensor,  # (B, 1)
                   tau: float = 1):
    """
    Softly blend SPM images based on predicted rainfall amount.
    Uses distance-based softmax over bins for differentiability.

    Returns:
      spm_blended: (B, 1, H, W)
      weights:     (B, K) blending weights per sample
    """
    # Ensure shapes are device/dtype compatible
    K = spm_bins.shape[0]
    # distances (B, K)
    d = (spm_pred.unsqueeze(1) - spm_bins.unsqueeze(0)).abs()
    weights = torch.softmax(-d.squeeze(0) / tau, dim=1)                  # (B, K)
    # Blend: einsum over K
    spm_bank = spm_bank.to(spm_pred.device)                   # (K, 1, H, W)
    spm_blended = torch.einsum('bk,bkchw->bchw', weights, spm_bank)
    return spm_blended, weights

def count_flops(model: nn.Module, rainfall_emb:nn.Module, inputs: tuple) -> float:
    """
    Counts the number of FLOPs (specifically, MACs) for a given model
    and a tuple of input tensors.

    Args:
        model (nn.Module): The PyTorch model.
        inputs (tuple): A tuple of input tensors to the model.

    Returns:
        float: The total number of GFLOPs.
    """
    # Note: fvcore counts MACs (Multiply-Accumulate operations), where 1 MAC is 
    # often considered equivalent to 2 FLOPs (1 multiplication + 1 addition).
    # The result is returned as the number of MACs.
    rainfall_inputs = (1, 24)
    rainfall_input = torch.ones(rainfall_inputs).to(next(model.parameters()).device)
    rainfall_input = rainfall_input.to(dtype=torch.long)
    flop_analyzer_rainfall = FlopCountAnalysis(rainfall_emb, (rainfall_input,))
    print(f"Rainfall Embedder FLOPs: {flop_analyzer_rainfall.total() / 1e9:.2f} GFLOPs")
    step = torch.tensor([10], dtype=torch.float32).to(next(model.parameters()).device)
    model_input = torch.randn(inputs).to(next(model.parameters()).device)
    rainfall_emb_shape = (1, 24, 256)
    rainfall_emb = torch.randn(rainfall_emb_shape).to(next(model.parameters()).device)
    flop_analyzer = FlopCountAnalysis(model, (model_input, step, rainfall_emb))
    print(f"Network FLOPs: {flop_analyzer.total() / 1e9:.2f} GFLOPs")
    return flop_analyzer.total()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        self.opt = opt
        self.model_config = create_model_config()
        # if opt.latent_space:
        #     self.vqgan = VQModel(**vars(self.model_config.VQGAN.params)).eval()
        #     self.vqgan.train = disabled_train
        #     for param in self.vqgan.parameters():
        #         param.requires_grad = False
        #     print(f"load vqgan from {self.model_config.VQGAN.params.ckpt_path}")
            
        #     self.cond_stage_model = SpatialRescaler(**vars(self.model_config.CondStageParams))
        #     self.vqgan.to(opt.device)
        #     self.cond_stage_model.to(opt.device)

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1, spm=opt.spm)
        self.rainfall_emb = RainfallEmbedder(256, 1)
        # print out the flops of self.net and self.rainfall_emb
        # print(f"Network FLOPs: {count_flops(self.net, self.rainfall_emb, (1, 3, 256, 256)) / 1e9:.2f} GFLOPs")
        # print(f"Rainfall Embedder FLOPs: {count_flops(self.rainfall_emb, (1, 4, 256)) / 1e9:.2f} GFLOPs")
        # self.spm_model = WeightedSumHead(dim=24)
        self.spm_model = WeightedSumHeadUpdated(dim=24)
        # print("weight shape:", self.spm_model.lin.weight)
        print("SPM model weights:", self.spm_model.w_base)
        print("SPM model bias:", self.spm_model.w_dem_delta.weight)
        params = list(self.net.parameters()) + list(self.rainfall_emb.parameters() )
        params += list(self.spm_model.parameters()) if opt.auto_spm else []
        # params = [
        #     {"params": self.net.parameters(), "weight_decay": opt.l2_norm},
        #     {"params": self.rainfall_emb.parameters(), "weight_decay": opt.l2_norm},
        #     {"params": self.spm_model.parameters(), "weight_decay": 0},
        # ]
        # [[0.8688, 0.8978, 0.9912, 0.8514, 1.0660, 0.8630, 1.0805, 1.0857, 0.8951,
        #  1.0177, 0.9929, 0.9000, 1.1266, 1.0872, 0.9186, 0.9086, 0.9189, 1.0403,
        #  0.9085, 1.1300, 1.1367, 1.1165, 0.9820, 0.9652]
        self.ema = ExponentialMovingAverage(params, decay=opt.ema)
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")
            self.rainfall_emb.load_state_dict(checkpoint['embedding'])
            log.info(f"[Embedding] Loaded embedding ckpt: {opt.load}!")
            # self.spm_model.load_state_dict(checkpoint['spm'])
            # log.info(f"[SPM] Loaded SPM model ckpt: {opt.load}!")
            # print out the weight of the spm model
            # print("SPM model weights:", self.spm_model.lin.weight)
            # print("SPM model bias:", self.spm_model.w_dem_delta.weight)
            if opt.normalize_latent:
                self.net.ori_latent_mean = checkpoint["ori_latent_mean"]
                self.net.ori_latent_std = checkpoint["ori_latent_std"]
                self.net.cond_latent_mean = checkpoint["cond_latent_mean"]
                self.net.cond_latent_std = checkpoint["cond_latent_std"]
                log.info(f"[Latent] Loaded latent mean/std ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.rainfall_emb.to(opt.device)
        # self.spm_model.to(opt.device)

        self.log = log

        if opt.eval:
            self.net.ori_latent_mean = self.net.ori_latent_mean.to(opt.device)
            self.net.ori_latent_std = self.net.ori_latent_std.to(opt.device)
            self.net.cond_latent_mean = self.net.cond_latent_mean.to(opt.device)
            self.net.cond_latent_std = self.net.cond_latent_std.to(opt.device)

    def logger(self, msg, **kwargs):
        print(msg, **kwargs)

    def get_latent_mean_std(self):
        train_dataset = floodDataset(True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        # max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_cond, _) = batch
            x = x.to(self.opt.device)
            x_cond = x_cond.to(self.opt.device)

            x_latent = self.vqgan.encoder(x)
            x_cond_latent = self.vqgan.encoder(x_cond)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        # self.logger(self.net.ori_latent_mean)
        # self.logger(self.net.ori_latent_std)
        # self.logger(self.net.cond_latent_mean)
        # self.logger(self.net.cond_latent_std)
    
    @torch.no_grad()
    def encode(self, x, cond=True):
        normalize = self.opt.normalize_latent 
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.net.cond_latent_mean) / self.net.cond_latent_std
            else:
                x_latent = (x_latent - self.net.ori_latent_mean) / self.net.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True):
        normalize = self.opt.normalize_latent
        if normalize:
            if cond:
                x_latent = x_latent * self.net.cond_latent_std + self.net.cond_latent_mean
            else:
                x_latent = x_latent * self.net.ori_latent_std + self.net.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out
    
    def compute_label(self, step, x0, xt, x1):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        # label = x1 - x0
        return label.detach()

    def compute_pred_x0(self, step, xt, x1, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        # pred_x0 = x1 - net_out
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, corrupt_method):
        dem_image, rainfall, flood_image, vx_image, vy_image, ca4d_d_image, ca4d_vx_image, ca4d_vy_image, physics_features = next(loader)
        x0 = torch.cat([flood_image, vx_image, vy_image], dim=1).detach().to(opt.device)
        x1 = dem_image.detach().to(opt.device)
        # make x1 have the same channel as x0
        x1 = x1.repeat(1, 3, 1, 1)
        cond = torch.cat([ca4d_d_image, ca4d_vx_image, ca4d_vy_image], dim=1).detach().to(opt.device)
        x1 = torch.randn_like(x1) if opt.fm else x1
        rainfall = rainfall.detach().to(opt.device)

        # assert x0.shape == x1.shape

        return x0, x1, cond, rainfall, physics_features

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        gradient_list = []
        embedder_gradient_list = []
        plot_loss = []
        plot_lpdes_loss = []

        def plot_losses(losses, lpdes_loss):
            plt.figure(figsize=(12, 8))
            plt.plot(np.log(losses))
            plt.xlabel("Iterations")
            plt.ylabel("Log Loss")
            plt.title("Log Loss per Iteration")
            plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\loss_plot.png")

            plt.figure(figsize=(12, 8))
            plt.plot(np.log(lpdes_loss))
            plt.xlabel("Iterations")
            plt.ylabel("Log LPDEs Loss")
            plt.title("Log LPDEs Loss per Iteration")
            plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\lpdes_loss_plot.png")
    
        def plot_model_gradients(model, embedder):
            def format_scientific(val):
                # Format the number using scientific notation with no decimal places.
                s = f"{val:.0e}"  # e.g., 0.000001 --> "1e-06"
                # Remove any extra zero in the exponent, e.g., change "e-06" to "e-6"
                s = s.replace("e-0", "e-")
                s = s.replace("e+0", "e+")
                return s
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Compute the mean absolute value of the gradient
                    grad_norm = param.grad.abs().mean().item()
                    gradients[name] = grad_norm
            gradient_list.append(gradients)
            # print(model)
            print("Top 25 parameters with the largest gradient magnitudes:")
            sorted_gradients = sorted(gradients.items(), key=lambda x: x[1], reverse=True)[:25]
            for param_name, grad_val in sorted_gradients:
                print(f"{param_name}: {format_scientific(grad_val)}")

            print("\nBottom 25 parameters with the smallest gradient magnitudes:")
            sorted_gradients_low = sorted(gradients.items(), key=lambda x: x[1])[:25]
            for param_name, grad_val in sorted_gradients_low:
                print(f"{param_name}: {format_scientific(grad_val)}")

            plt.figure(figsize=(12, 8))
            for epoch, gradients in enumerate(gradient_list):
                keys = list(gradients.keys())
                values = [gradients[key] for key in keys]

                plt.plot(range(len(keys)), np.log(values), label=f"Iter {(epoch+1)*200}")
            
            # plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.xlabel("Parameters")
            plt.ylabel("Log Mean Gradient Magnitude")
            plt.title("Log Gradient Magnitude per Parameter Across 200 iterations")
            plt.legend()
            plt.tight_layout()
            plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\gradient_plot.png")

            embedder_gradients = {}
            for name, param in embedder.named_parameters():
                if param.grad is not None:
                    # Compute the mean absolute value of the gradient
                    grad_norm = param.grad.abs().mean().item()
                    embedder_gradients[name] = grad_norm
            
            embedder_gradient_list.append(embedder_gradients)
            print("Top 25 parameters with the largest gradient magnitudes:")
            sorted_gradients = sorted(embedder_gradients.items(), key=lambda x: x[1], reverse=True)[:25]
            for param_name, grad_val in sorted_gradients:
                print(f"{param_name}: {format_scientific(grad_val)}")
            print("\nBottom 25 parameters with the smallest gradient magnitudes:")
            sorted_gradients_low = sorted(embedder_gradients.items(), key=lambda x: x[1])[:25]
            for param_name, grad_val in sorted_gradients_low:
                print(f"{param_name}: {format_scientific(grad_val)}")

            fig = plt.figure(figsize=(12, 8))
            for epoch, gradients in enumerate(embedder_gradient_list):
                keys = list(gradients.keys())
                values = [gradients[key] for key in keys]

                plt.plot(range(len(keys)), np.log(values), label=f"Iter {(epoch+1)*200}")
            
            # plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.xlabel("Parameters")
            plt.ylabel("Log Mean Gradient Magnitude")
            plt.title("Log Gradient Magnitude per Parameter Across 200 iterations")
            plt.legend()
            plt.tight_layout()
            plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\embedder_gradient_plot.png")
            
        self.writer = util.build_log_writer(opt)
        log = self.log
        spm_weights = []
        dem_weights = []
        net = self.net
        ema = self.ema
        # print(net)
        rainfall_embber = self.rainfall_emb
        # print(rainfall_embber)
        optimizer, sched = build_optimizer_sched(opt, rainfall_embber, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()
            losses = []
            lpdes = []
            lpdes_norm = []
            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                x0, x1, cond, rainfall, physics_features = self.sample_batch(opt, train_loader, corrupt_method)

                # ===== compute loss =====
                if opt.timestep_importance == 'continuous':
                    t1, t0 = 1, 0
                    step = torch.rand((x0.shape[0],)) * (t1 - t0)
                    # make step shape = (x0.shape[0], 1, 1, 1)
                    step = step.view(-1, 1, 1, 1).to(x0.device)
                    if opt.ot_ode:
                        xt = (1-step) * x0 + step * x1
                        label = x1 - x0

                    if not opt.ot_ode:
                        var = (step**2 * (1-step)**2) / (step**2 + (1-step)**2)
                        sqrt_var = var.to('cpu')
                        sqrt_var = torch.tensor(sqrt_var, device=x0.device)
                        rand = torch.randn_like(x0) * 0.5
                        xt = (1-step) * x0 + step * x1 + sqrt_var * rand
                        label = x1 - x0 
                else:
                    step = torch.randint(0, opt.interval, (x0.shape[0],))
                    step = step.to(x0.device)
                    xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                    label = self.compute_label(step, x0, xt, x1)

                rainfall_emb = rainfall_embber(rainfall)
                pred = net(xt, step, rainfall_emb, cond=x0, ca4d=cond)
                assert xt.shape == label.shape == pred.shape

                if opt.nv_loss:
                    cur_rainfall, flood_image_real, vx_image_real, vy_image_real, prev_h_image_real, prev_vx_image_real, prev_vy_image_real,_ = physics_features
                    pred_x0 = x1 - pred
                    pred_h0 = pred_x0[:, 0, :, :] * 0.043 + 0.986
                    pred_h0 = torch.clamp(pred_h0, 0, 1)
                    pred_h0_flood_depth = (1 - pred_h0) * (6 - 0) + 0  # (B,1,H,W)

                    pred_vx = pred_x0[:, 1, :, :] * 0.0049 + 0.498
                    pred_vx = torch.clamp(pred_vx, 0, 1)
                    pred_vx_real = pred_vx * (4 - (-4)) + (-4)  # (B,1,H,W)

                    pred_vy = pred_x0[:, 2, :, :] * 0.0043 + 0.499
                    pred_vy = torch.clamp(pred_vy, 0, 1)
                    pred_vy_real = pred_vy * (4 - (-4)) + (-4)  # (B,1,H,W)     

                    dem_height = x1 * 0.22 + 0.18
                    dem_height = torch.clamp(dem_height, 0, 1)
                    dem_height_norm = dem_height * (125 - (-3)) + (-3)
                    dem_height_t = dem_height_norm.squeeze(1)  # (B,H,W)

                    # total_height = dem_height_t + pred_h0_t  # (B,H,W)

                    # # compute the velocity field from pred_h0_np
                    # lu, lv, lg = navier_stokes_operators_torch(prev_u=prev_vx, prev_v=prev_vy,
                    #                                      cur_u=vx, cur_v=vy, h=total_height)
                    lu = continuity_residual_torch(prev_h=prev_h_image_real, cur_h=pred_h0_flood_depth,
                                                   cur_ux=pred_vx_real, cur_uy=pred_vy_real, elevation=dem_height_t, rainfall=cur_rainfall)
                loss = F.mse_loss(pred, label)
                if opt.nv_loss:
                    # NV loss weighting decays by factor 0.9 every 1000 iterations
                    base_nv_loss_weight = 0.1
                    # decay_factor = 0.9
                    # decay_every = 1000
                    # nv_loss_weighting = base_nv_loss_weight * (decay_factor ** (it // decay_every))
                    # lu_norm = torch.linalg.vector_norm(lu, dim=(1,2))
                    # lv_norm = torch.linalg.vector_norm(lv, dim=(1,2))
                    # lpde_mean = (lu_norm + lv_norm).mean()
                    # lpde_mean_norm = torch.abs(lpde_mean - 0.152)
                    lpdes_mean = torch.mean(torch.sum(lu**2, dim=(1,2)))
                    # lpdes_mean_norm = torch.abs(lpdes_mean - 0.0352)
                    loss += base_nv_loss_weight * lpdes_mean
                    lpdes.append(lpdes_mean.item())
                    lpdes_norm.append(lpdes_mean.item())
                loss.backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(rainfall_embber.parameters(), max_norm=1)
            # print out rainfall embedder gradients
            # print("Rainfall Embedder Gradients:")
            # for name, param in rainfall_embber.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.abs().mean().item():.6f}")
            #         # print out the gradients for each layer
            #         for i, layer in enumerate(param.grad):
            #             print(f"Layer {i}: {layer.abs().mean().item():.6f}")
            # print("net Gradients:")
            # for name, param in net.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.abs().mean().item():.6f}")
            # plot_grad_flow(net)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{} | lpde:{} | lpde_norm:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(np.mean(losses)),
                "{:+.4f}".format(np.mean(lpdes)),
                "{:+.4f}".format(np.mean(lpdes_norm)),
            ))
            plot_loss.append(np.mean(losses))
            plot_lpdes_loss.append(np.mean(lpdes_norm))

            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 100 == 0 and opt.auto_spm:
                print("SPM model weights:", self.spm_model.w_base)
                spm_weights.append(self.spm_model.w_base.detach().cpu().numpy().flatten())
                # help me plot the 24 weight in a plot, each weight is a line
                plt.figure(figsize=(12, 6))
                for i in range(24):
                    plt.plot(range(1, len(spm_weights)+1), [w[i] for w in spm_weights], marker='o', label=f'Weight {i+1}')
                plt.xlabel('Iteration')
                plt.ylabel('Weight Value')
                plt.title('SPM Model Weights Over Time')
                # put legend outside the plot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\spm_w_base_weights_over_time.png")

                # print("DEM model weights:", self.spm_model.w_dem_delta.weight)
                # dem_weights.append(self.spm_model.w_dem_delta.weight.detach().cpu().numpy())
                # # help me plot the 60 weight in a plot, each weight is a lin
                # plt.figure(figsize=(12, 6))
                # for i in range(self.spm_model.dem_num):
                #     plt.plot(range(1, len(dem_weights)+1), [w[i] for w in dem_weights], marker='o', label=f'DEM Weight {i+1}')
                # plt.xlabel('Iteration')
                # plt.ylabel('Weight Value')
                # plt.title('SPM Model DEM Weights Over Time')
                # # put legend outside the plot
                # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                # plt.savefig("C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\res\\spm_dem_weights_over_time.png")

            if it % 200 == 0 and it != 0:
                plot_losses(plot_loss, plot_lpdes_loss)
            #     plot_model_gradients(net, rainfall_embber)

            if it % 1000 == 0 and it != 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        'embedding': self.rainfall_emb.state_dict(),
                        # "spm": self.spm_model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / "latest.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            # if it == 500 or it % 3000 == 0: # 0, 0.5k, 3k, 6k 9k
            #     net.eval()
            #     self.evaluation(opt, it, val_loader, corrupt_method)
            #     net.train()
        self.writer.close()

    
    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, y, mask=None, cond=None, spm=None, spm_bank=None, 
                      spm_bins=None, dem_num=None, clip_denoise=False, 
                      nfe=None, log_count=10, verbose=True, eval=False, ode_method=None, 
                      vx_image=None, prev_vx_image=None, vy_image=None, prev_vy_image=None):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        if opt.latent_space and eval:
            x1 = self.encode(x1, cond=False)
            cond = self.cond_stage_model(cond)
            
        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if spm is not None: spm = spm.to(opt.device)
        if spm_bank is not None: spm_bank = spm_bank.to(opt.device)
        if spm_bins is not None: spm_bins = spm_bins.to(opt.device)

        mask = None
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(x1, xt, rainfall_emb, step, ode=None):
                if not ode:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                    out = self.net(xt, step, rainfall_emb, cond=cond, spm=spm)
                    return self.compute_pred_x0(step, x1, xt, out, clip_denoise=clip_denoise)
                else:
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.float32)
                    out = self.net(xt, step, rainfall_emb, cond=x1, ca4d=cond)
                    return out

            rainfall_emb = self.rainfall_emb(y)
            # if opt.auto_spm:
            #     spm_pred = self.spm_model(y, dem_num).squeeze(1)
            #     spm, weights = soft_bin_blend(spm_bank, spm_bins, spm_pred)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, rainfall_emb, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose, ode_method=ode_method,
                vx_image=vx_image, prev_vx_image=prev_vx_image, vy_image=vy_image, prev_vy_image=prev_vy_image
            )

        b, *xdim = x1.shape
        # assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        if opt.latent_space and eval:
            xs = xs[:, 0, ...].to(opt.device)
            # xs = xs.squeeze(1)
            xs = self.decode(xs, cond=False)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)
        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, y, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        y           = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        # assert y.shape == (batch,)
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

        # def log_accuracy(tag, img):
        #     pred = self.resnet(img.to(opt.device)) # input range [-1,1]
        #     accu = self.accuracy(pred, img_clean.to(opt.device))
        #     self.writer.add_scalar(it, tag, accu)

        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)

        # log.info("Logging accuracies ...")
        # log_accuracy("accuracy/clean",   img_clean)
        # log_accuracy("accuracy/corrupt", img_corrupt)
        # log_accuracy("accuracy/recon",   img_recon)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
