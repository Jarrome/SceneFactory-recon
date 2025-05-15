import os, sys
import glob
import tqdm
import math
import imageio
import random
import warnings
#import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
#import mcubes
from rich.console import Console
#from torch_ema import ExponentialMovingAverage

from packaging import version as pver
#import lpips
#from torchmetrics.functional import structural_similarity_index_measure
#from torchmetrics.functional import image_gradients

from icecream import ic
from rich import print
import pdb

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


w_patch = 32
i_ = np.linspace(0, 32, 32)
j_ = np.linspace(0, 32, 32)
iv, jv = np.meshgrid(i_, j_, indexing='ij')
iv = iv.reshape(1,-1)
jv = jv.reshape(1,-1)

def get_patch(H,W,N):
    n_patch = N // (w_patch**2)
    lt = [np.random.randint(0, H-w_patch, (n_patch,1)), np.random.randint(0, W-w_patch, (n_patch,1))] 
    inds = (iv+lt[0])*W+(jv+lt[1])
    inds = inds.reshape(-1)
    return torch.from_numpy(inds).long().cuda()
    

    

@torch.cuda.amp.autocast(enabled=False)
def get_rays_w_mask_by_patch(poses, masks, intrinsics, H, W, N=-1, error_map=None, patch_size=1, images=None):
    ''' get rays with mask, this only work for dataloader for trainging
        with patch because this way make it easier to get smooth appearance
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
        images: [B,H,W,3]
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)
        inds = get_patch(H,W,N)#torch.randint(0, H*W, size=[N], device=device) # may duplicate
        try:
            inds = inds[masks[0,inds]]
        except Exception as e:
            print(e)
            print(inds, masks)
            exit(-1)
        N = inds.shape[0]
        inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds
    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
        results['inds'] = inds


    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    xyzs_norm = torch.norm(directions, dim=-1, keepdim=True)
    directions = directions / xyzs_norm #torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

        

@torch.cuda.amp.autocast(enabled=False)
def get_rays_w_mask(poses, masks, intrinsics, H, W, N=-1, error_map=None, patch_size=1, images=None):
    ''' get rays with mask, this only work for dataloader for trainging
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
        images: [B,H,W,3]
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)
        inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
        try:
            inds = inds[masks[0,inds]]
        except Exception as e:
            print(e)
            print(inds, masks)
            exit(-1)
        N = inds.shape[0]
        inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds
    else:
        #grid_h, grid_w = torch.meshgrid(torch.range(0,H-1,2), torch.range(0,W-1,2), indexing='ij')
        #inds = (grid_h * W + grid_w).reshape(-1).long().cuda()
        inds = torch.arange(H*W, device=device)
        inds = inds[masks[0,inds]]
        N = inds.shape[0]
        inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds


    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, images=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
        images: [B,H,W,3]
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)
        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            if True:#images is None:
                inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
                inds = inds.expand([B, N])
            else:
                # 1. compute gradient maginitude on B,C,H,W image
                dy, dx = image_gradients(images.permute(0,3,1,2))
                mag = (dy**2+dx**2).mean(1) # B,H,W

                # 2. follow point-slam to add neural points
                ps_X = 200000
                ps_Y = 10000
                inds_X = torch.randint(0, H*W, size=[ps_X], device=device)
                inds_X = inds_X.expand([B, ps_X])

                inds_Y = torch.sort(mag.view(-1,H*W),axis=1,descending=True)[1][:,:10*ps_Y]
                inds_Y = inds_Y[:,torch.randint(10*ps_Y,(ps_Y,))]
                inds = torch.cat([inds_X,inds_Y],axis=-1)
        else:
            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)
        results['inds'] = inds
    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
        results['inds'] = inds


    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step

                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.trained_mask = None

        self.c = 3

        if True: #model is not None:
            model.to(self.device)
            if self.world_size > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            self.model = model

            if isinstance(criterion, nn.Module):
                criterion.to(self.device)
            self.criterion = criterion
            self.sem_criterion = nn.CrossEntropyLoss()
            self.inst_criterion = nn.CrossEntropyLoss()



            # optionally use LPIPS loss for patch-based training
            if self.opt.patch_size > 1:
                import lpips
                self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)

            if optimizer is None:

                self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4) # naive adam
                #self.optimizer = optim.Adam(self.model.get_params_p(start=True))


            else:
                self.in_optimizer = optimizer
                self.optimizer = optimizer(self.model, 1e-1, True)
            '''
            if lr_scheduler is None:
                self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
            else:
                self.lr_scheduler = lr_scheduler(self.optimizer)
                self.lr_scheduler_func = lr_scheduler
            '''

            if False:#ema_decay is not None:
                self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            else:
                self.ema = None

            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

            # variable init
            self.epoch = 0
            self.global_step = 0
            self.local_step = 0
            self.stats = {
                "loss": [],
                "valid_loss": [],
                "results": [], # metrics[0], or valid_loss
                "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
                "best_result": None,
                }

            # auto fix
            if len(metrics) == 0 or self.use_loss_as_metric:
                self.best_mode = 'min'

            # workspace prepare
            self.log_ptr = None
            if self.workspace is not None:
                os.makedirs(self.workspace, exist_ok=True)        
                self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
                self.log_ptr = open(self.log_path, "a+")

                #self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
                #self.best_path = f"{self.ckpt_path}/{self.name}.pth"
                #os.makedirs(self.ckpt_path, exist_ok=True)
                
            self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
            self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
            '''
            if self.workspace is not None:
                if self.use_checkpoint == "scratch":
                    self.log("[INFO] Training from scratch ...")
                elif self.use_checkpoint == "latest":
                    self.log("[INFO] Loading latest checkpoint ...")
                    self.load_checkpoint()
                elif self.use_checkpoint == "latest_model":
                    self.log("[INFO] Loading latest checkpoint (model only)...")
                    self.load_checkpoint(model_only=True)
                elif self.use_checkpoint == "best":
                    if os.path.exists(self.best_path):
                        self.log("[INFO] Loading best checkpoint ...")
                        self.load_checkpoint(self.best_path)
                    else:
                        self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                        self.load_checkpoint()
                else: # path to ckpt
                    self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                    self.load_checkpoint(self.use_checkpoint)
            '''
            
            # clip loss prepare
            if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
                from nerf.clip_utils import CLIPLoss
                self.clip_loss = CLIPLoss(self.device)
                self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_NSLF_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        images = data['images'] # [B, N, 3/4]
        gt_depth = data['depths'].float()

        B, N, C = images.shape


        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        xyzs = rays_o + rays_d * gt_depth.view(-1, 1)


        pred_rgb = self.model.nslf_color(xyzs, rays_d)

        # MSE loss
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)


        loss = loss.mean()

        return pred_rgb, gt_rgb, loss


    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        images = data['images'] # [B, N, 3/4]
        gt_depth = data['depths'].float()

        rast_info = [data['pose'][:3,:3], data['pose'][:3,3], data['K'], data['H'], data['W'], data['inds']]


        #B, N, C = images.shape
        if data['inverse_assign']:
            gt_rgb = images
        else:
            gt_rgb = None

        bg_color = None
        outputs = self.model.render_w_depth(rays_o, rays_d, gt_depth, frame_id=data['index'][0], rast_info=rast_info, color=gt_rgb,staged=False, bg_color=bg_color, perturb=True, force_all_rays=False if self.opt.patch_size == 1 else True, **vars(self.opt))
    
        pred_rgb = outputs['image']
        #pred_depth = outputs['depth_im'] # this is a image
        #pred_color = outputs['color_im'] # this is a image

        #gt_depth = data['depth_im'].view(*(pred_depth.shape))
        #gt_color = data['image_im']
        #valid_depth = (pred_depth > 0 ) #* (gt_depth > 0) 



        # MSE loss
        if pred_rgb is not None: # no valid tensor to train 
            #if not data['inverse_assign']:
            gt_rgb = images
            loss = self.criterion(pred_rgb[...,:3], gt_rgb[...,:3]).mean(-1).mean()
            #else:
            #    loss = torch.zeros(1).cuda()
            '''
            loss = self.criterion(pred_rgb[...,:3], gt_rgb[...,:3]).mean(-1).mean() + \
                   self.criterion(pred_color, gt_color).mean(-1).mean() + \
                   self.criterion(pred_depth*valid_depth, gt_depth*valid_depth).mean()
            '''

        else:
            loss = None

        if False:#data['add_neural_point']: 
            torch.save({'gt':gt_depth, 'pred':pred_depth},'tmp_depth.pth')    
            mse = self.criterion(pred_depth*valid_depth, gt_depth*valid_depth)
            print(mse.max(), mse.median())
            miss_depth_mask = mse > .1**2
            loss += mse[miss_depth_mask].mean()
        #if self.global_step < 200:

        # here gt_depth is actually t
        '''
        d_valid= ~torch.isnan(pred_depth)
        d_valid *= ~torch.isinf(pred_depth)
        loss_d = self.criterion(pred_depth[d_valid],gt_depth[:,:,0][d_valid])
        #else:
        #loss += self.criterion(pred_depth,gt_depth[:,:,0]) * .1 
        #loss+=            1 - torch.exp(-(pred_depth-gt_depth[:,:,0])**2 / (2))# [B, N, 3] --> [B, N]
        loss = loss.mean() + 1e-4*loss_d.mean()# + loss_direct/4000
        '''

        # extra loss
        '''
        pred_weights_sum = outputs['weights_sum'] + 1e-8
        loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        loss = loss + loss_ws.mean()
        '''
        #print('loss',loss)

        return pred_rgb, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, return_belongings=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        rast_info = (data['R'],data['t'],data['K'], H, W)

        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        with torch.no_grad():
            outputs = self.model.render(rays_o, rays_d, rast_info, trained_mask = self.trained_mask if hasattr(self,'trained_mask') else None, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))
        if outputs['image'] is None:
            pred_rgb = torch.zeros(1,H,W,self.c).to(rays_o)+.5
        else:
            pred_rgb = outputs['image'].reshape(-1, H, W, self.c)
        if outputs['depth'] is None:
            pred_depth = None
        else:
            pred_depth = outputs['depth'].reshape(-1, H, W)
        if return_belongings:
            belonging_ids = outputs['belonging_ids']
        else:
            belonging_ids = None

        return pred_rgb, pred_depth, belonging_ids

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16, use_nslf=False, jump=False):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        #loader = iter(train_loader)


        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            '''
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)
            '''
            with train_loader.add_lock:
                data = train_loader.collate_fair()

                if data['rays_o'].shape[1] == 0: # no ray input
                    continue


                # update grid every 16 steps
                '''
                if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        self.model.update_extra_state()
                '''
                self.global_step += 1

                # add neural points
                if data['add_neural_point']:
                  with self.model.data_lock:
                    print('[bold purple]Add neural points...')
                    rays_o = data['rays_o'] # [B, N, 3]
                    rays_d = data['rays_d'] # [B, N, 3]

                    color = data['images'] # [B, N, 3/4]
                    depth = data['depths'].float()

                    frame_id = data['index'][0]
                    if torch.isnan(rays_o).sum()>0:
                            print('before contiguous', torch.isnan(rays_o).sum(), torch.isnan(rays_d).sum())


                    rays_o = rays_o.contiguous().view(-1, 3)
                    rays_d = rays_d.contiguous().view(-1, 3)
                    if depth is not None:
                        t = depth.contiguous().view(-1,1)
                        ps = rays_o + t * rays_d
                        #print(torch.isnan(ps).sum())
                        if torch.isnan(ps).sum()>0:
                            print('has Nan ps')
                            print(torch.isnan(depth).sum(), torch.isnan(t).sum(), torch.isnan(rays_o).sum(), torch.isnan(rays_d).sum())
                            print(depth)
                        self.model.add_neural_points(ps, frame_id, color[...,:3].view(-1,3), data['pose'].view(4,4))

                    # update optimizer
                    if self.in_optimizer is None:
                        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4) # naive adam
                    else:
                        if jump:
                            lr = 1e-1
                        else:
                            lr = 1e-2
                        if False:# self.global_step > 500:
                            self.optimizer = self.in_optimizer(self.model,lr,with_mlp=False)
                        else:
                            self.optimizer = self.in_optimizer(self.model,lr,with_mlp=True)
            with self.model.data_lock:

                self.optimizer.zero_grad()
            #with torch.cuda.amp.autocast(enabled=self.fp16):
                #if use_nslf:
                #    preds, truths, loss = self.train_NSLF_step(data)
                #else:
                #torch.cuda.synchronize()
                #tst=time.time() 

                if self.global_step > 500:
                    data['inverse_assign'] = True
                else:
                    data['inverse_assign'] = False
                preds, truths, loss = self.train_step(data)
                #torch.cuda.synchronize()
                #print('step take', time.time()-tst)
                if loss is None:
                    print('loss is None because no valid tensor to train!!!')
                    print('rays_o',data['rays_o'])
                    print('rays_d',data['rays_d'])
                    print('depths',data['depths'])
                    print('pose',data['pose'])
                    #self.optimizer.zero_grad()
                    break

                if True:#self.global_step <= 500:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                '''
                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()
                '''

                total_loss += loss.detach()
        #if self.ema is not None:
        #    self.ema.update()

        average_loss = total_loss.item() / step
        '''
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        '''
        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, return_belongings=False, view_context=0):
        '''
            return_belonging: if set True, will return the keyframe ID whose nps are used in randering
        '''
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device).float()

        rays = get_rays(pose, intrinsics, rH, rW, -1)



        fx, fy, cx, cy = intrinsics
        K = torch.eye(4)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy

        pose = pose[0]

        #inv_pose = torch.inverse(pose)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'K': K,
            'R': pose[:3,:3],
            't': pose[:3,3]
        }
        
        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth, belonging_ids = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp, return_belongings=return_belongings)
        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if view_context == 0: # rgb
            preds = preds[:,:,:,:3]
            pred = preds[0].detach().cpu().numpy() * 255




        if preds_depth is not None:
            pred_depth = preds_depth[0].detach().cpu().numpy()
        else:
            pred_depth = None


        outputs = {
            'image': pred,
            'depth': pred_depth,
            'belonging_ids': belonging_ids,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            # update grid every 16 steps
            '''
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            '''
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            if self.model.encoder.only_train_mask is None:
                self.model.encoder.only_train_mask = self.model.encoder.embeddings == 0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            '''
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            '''

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()
        '''
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        '''
        self.log(f"==> Finished Epoch {self.epoch}.")

