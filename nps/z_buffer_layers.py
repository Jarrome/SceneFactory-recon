'''
modified from https://github.com/facebookresearch/synsin
'''
import os

import torch
from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

import pdb
torch.manual_seed(42)

EPS=1e-2
def project_pts(cam_X, K, H,W):
    '''
        cam_X: N,3
    '''
    planar_X = cam_X / cam_X[:,2:3]
    #planar_X[:,:2]*=-1
    #homo_X = torch.cat((cam_X[:,0:2], torch.ones_like(cam_X[:,0:1]).to(cam_X), xy_proj[:,2:3]), axis=0) # N,4
    # And intrinsics
    xy_proj = K@planar_X.T #3,N

    # And finally we project to get the final result
    #mask = (cam_X.T[2:3, :].abs() < EPS).detach()

    # Remove invalid zs that cause nans
    #zs = xy_proj[2:3, :]
    zs = cam_X.T[2:3,:]
    #zs[mask] = EPS

    #sampler = torch.cat((xy_proj[0:2, :] / zs, xy_proj[2:3, :]), 0).unsqueeze(0)
    sampler = torch.cat((xy_proj[0:2,:], zs),0).unsqueeze(0)

    sampler[:, 0,:] /= W
    sampler[:, 1,:] /= H

    
    # Flip the ys
    '''
    sampler = sampler * torch.Tensor([-1, -1, 1]).unsqueeze(0).unsqueeze(
        2
    ).to(sampler.device)
    '''

    # space [0,1] to [-1,1]
    sampler[:,0:2,:] = sampler[:,0:2,:] * 2 - 1
    
    return sampler


class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        radius=1.5,
        size=(480, 640),
        points_per_pixel=8,
        rad_pow = 2,
        tau = 1.0,
        accumulation = 'wsum',# "wsumnorm", "alphacomposite"
    ):
        super().__init__()

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.rad_pow = rad_pow
        self.tau = tau
        self.accumulation = accumulation




    def forward(self, pts3D, src):
        '''
            pts3D: pointcloud


            pts3D: N,3
            src: [color] N,3
        '''

        # project to pytorch3d's space


        pts3D[:,1] = - pts3D[:,1]
        pts3D[:,0] = - pts3D[:,0]

        radius = float(self.radius) / float(self.size[0]) * 2.0
        pts3D = Pointclouds(points=[pts3D], features=[src])
        points_idx, zbuf, dist = rasterize_points(
            pts3D, self.size, radius, self.points_per_pixel
        )


        dist = dist / pow(radius, self.rad_pow)


        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.tau)
            .permute(0, 3, 1, 2)
        )

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

        return transformed_src_alphas
