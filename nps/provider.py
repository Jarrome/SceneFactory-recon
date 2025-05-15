import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import open3d as o3d
from threading import Lock

import trimesh
import torch
from torch.utils.data import DataLoader

from .utils_SLARE_rast import get_rays_w_mask, get_rays_w_mask_by_patch

from .ext3d import unproject_depth, remove_radius_outlier, estimate_normals



from icecream import ic

import pdb

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, semantic=False):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        assert downscale == 1, "now only support 1"
        #self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.mode = 'blender' # provided split
        self.intrinsics = None
        self.K = None
        self.poses = None
        self.images = None
        self.depths = None # for ray depth, not z depth
        self.valid_masks = None

        self.trained_times = []

        
        self.data_len = 0
        self.add_lock = Lock()
        

    def add_frame(self, frame, remove_outlier=False):
        rgb, depth, pose, intrin, depth_scale = frame
        if self.intrinsics is None:
            self.intrinsics = np.array([intrin[0,0], intrin[1,1],intrin[0,2],intrin[1,2]])
            self.K = intrin

        self.H, self.W = depth.shape
        
        image = rgb / 255
        depth = depth / depth_scale
        
        # get valid index (remove outlier)
        if remove_outlier:
            torch_depth = torch.from_numpy(depth).to(self.device).float()
            dist_mask = torch_depth.view(-1) > .5
            pc_data = unproject_depth(torch_depth, *(self.intrinsics))
            pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
            pc_data = pc_data.reshape(-1,4)
            nan_mask = ~torch.isnan(pc_data[..., 0])

            with torch.cuda.device(self.device):
                valid_mask = remove_radius_outlier(pc_data, 16*4, 0.05) * dist_mask * nan_mask
                pc_data = pc_data[valid_mask]
                normal_data = estimate_normals(pc_data, 16*4, 0.1, [0.0, 0.0, 0.0])
                normal_valid_mask = ~torch.isnan(normal_data[..., 0])
                valid_mask[valid_mask.clone()] = normal_valid_mask
                valid_mask = valid_mask.unsqueeze(0)
        else:
            valid_mask = torch.from_numpy((depth > .1)).view(1,-1).cuda(self.device) #torch.ones(self.H*self.W,dtype=bool).cuda().unsqueeze(0)


        # get ray depth
        u = np.arange(depth.shape[1])
        v = np.arange(depth.shape[0])
        u,v = np.meshgrid(u, v) # H,W and H,W

        xyz = np.ones((depth.shape[0],depth.shape[1],3))

        fx, fy, cx, cy = self.intrinsics
        xyz[:,:,0] = (u-cx)/fx
        xyz[:,:,1] = (v-cy)/fy

        xyz_occ = xyz.copy()
        xyz_occ *= depth[...,np.newaxis]#[:,:,2] = depth

        xyz = xyz / np.sqrt((xyz**2).sum(-1,keepdims=True))
        depth = depth / xyz[:,:,2] # the length on ray direction

        pose = torch.from_numpy(pose).unsqueeze(0).float() # [1, 4, 4]
        image = torch.from_numpy(image).unsqueeze(0).float() # [1, H, W, C]
        depth = torch.from_numpy(depth).unsqueeze(0).float() # [1, H, W, 1]
 
        

        if self.poses is None:
            self.poses = pose
            self.images = image
            self.depths = depth
            self.valid_masks = valid_mask
        else:
            self.poses = torch.cat([self.poses, pose],axis=0)
            self.images = torch.cat([self.images, image],axis=0)
            self.depths = torch.cat([self.depths, depth],axis=0)
            self.valid_masks = torch.cat([self.valid_masks, valid_mask],axis=0)

        self.error_map = None
        self.data_len += 1

    def transform_to(self, frame_ids, Ts):
        dTs = []
        for frame_id, T in zip(frame_ids, Ts):
            dT = torch.from_numpy(T).float() @ torch.inverse(self.poses[frame_id,:,:])
            self.poses[frame_id,:,:] = torch.from_numpy(T).float() #@ self.poses[frame_id,:,:]
            dTs.append(dT)
        return dTs



    def collate_fair(self, index=None):
        #B = len(index) # a list of length 1
        B = 1   
        add_neural_point = False
        # choose the least trained  
        while len(self)>len(self.trained_times):
            self.trained_times.append(0)
        index_ = np.argmin(self.trained_times)

        rand_train = False
        if np.min(self.trained_times) < 5:#[index_] < 5:
            if np.min(self.trained_times) < 1:
                add_neural_point = True
            index = index_
        #elif self.trained_times[index_] <5:#2:
        #    index = index_
        else: # greater than 10 will all the same
            index = np.random.randint(len(self)) #index[0]
            rand_train = True



        self.trained_times[index] += 1
        index = [index]




        poses = self.poses[index].to(self.device).float() # [B, 4, 4]
        valid_masks = self.valid_masks[index]
        error_map = None if self.error_map is None else self.error_map[index]

        if add_neural_point: # add alot points
            rays = get_rays_w_mask(poses, valid_masks, self.intrinsics, self.H, self.W, -1, error_map, self.opt.patch_size, images = self.images[index].to(self.device))
        else:
            if rand_train:
                rays = get_rays_w_mask_by_patch(poses, valid_masks, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size, images = self.images[index].to(self.device))
            else:
                rays = get_rays_w_mask(poses, valid_masks, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size, images = self.images[index].to(self.device))

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'].clone(),
            'rays_d': rays['rays_d'].clone(),
            'pose': poses[0].clone(),
            'K': self.K,
            'inds': rays['inds'].clone(),
            'add_neural_point':add_neural_point
        }
        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images.clone()
        if self.depths is not None:
            depths = self.depths[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                depths = torch.gather(depths.view(B, -1, 1), 1, torch.stack([rays['inds']], -1)) # [B, N, 3/4]

            results['depths'] = depths.clone()
            results['depth_im'] = self.depths[index].to(self.device)
            results['image_im'] = self.images[index].to(self.device)




        results['index'] = index
            
        return results


    def collate(self, index):
        B = len(index) # a list of length 1
        poses = self.poses[index].to(self.device).float() # [B, 4, 4]
        valid_masks = self.valid_masks[index]
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays_w_mask(poses, valid_masks, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size, images = self.images[index].to(self.device))
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'pose': poses[0],
            'K': self.K,
            'inds': rays['inds']
        }
        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        if self.depths is not None:
            depths = self.depths[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                depths = torch.gather(depths.view(B, -1, 1), 1, torch.stack([rays['inds']], -1)) # [B, N, 3/4]

            results['depths'] = depths



        results['index'] = index
            
        return results
    def __len__(self):
        #return len(self.poses)
        return self.data_len

    def dataloader(self):
        if self.poses is None:
            size = 1
        else:
            size = self.poses.shape[0]
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate_fair, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
