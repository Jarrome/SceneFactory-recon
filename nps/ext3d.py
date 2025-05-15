import os,sys
import torch
import numpy as np
import open3d as o3d

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_path))
from ext import remove_radius_outlier as ext_remove_radius_outlier


def unproject_depth(torch_depth, fx, fy, cx, cy):
    H,W = torch_depth.shape
    _x = torch.arange(W)
    _y = torch.arange(H)
    grid_y, grid_x = torch.meshgrid(_y, _x, indexing='ij') # H,W  H,W

    X = torch.ones(H,W,3).to(torch_depth) * torch_depth[:,:,None]
    grid_y, grid_x = grid_y.to(torch_depth), grid_x.to(torch_depth)
    X[:,:,0] *= (grid_x -cx) / fx  
    X[:,:,1] *= (grid_y -cy) / fy 
    return X


def remove_radius_outlier(pc, nb, radius):
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3].cpu().numpy().astype(np.float64))
    cl, ind = pcd.remove_radius_outlier(nb_points=nb, radius=radius)
    mask = torch.ones(pc.shape[0], dtype=bool).to(pc.device)
    mask[ind] = False
    '''
    return ext_remove_radius_outlier(pc, nb, radius)
    '''
    D, I, _, _ = frnn.frnn_grid_points(
              pc[:,:3].unsqueeze(0), pc[:,:3].unsqueeze(0), K=1, r=radius, grid=None, return_nn=True, return_sorted=False)
    I = I.view(-1)
    mask = I != -1
    return mask
    '''

def estimate_normals(pc, nb, radius, center):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3].cpu().numpy().astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nb))
    return torch.as_tensor(np.asarray(pcd.normals)).float().to(pc)


