import os, sys
import math
import numpy as np
import torch
import torch.nn as nn

import copy

import open3d as o3d
from time import time, sleep 
from timeit import  default_timer as time

from threading import Lock

import argparse

from .renderer_SLARE2_rast_multi import NeRFRenderer
from .ray_cast import RayCaster
from .ext3d import unproject_depth
from .point_raster import point_raster


import matplotlib.pyplot as plt
from pykdtree.kdtree import KDTree
from scipy.spatial import Delaunay


import pdb
from icecream import ic
from rich import print


class SLARE(NeRFRenderer):
    def __init__(self, device):
        super().__init__(bound=2, cuda_ray=False)

        self.is_train = True
        # sh
        self.degree = 2
        self.n_sphere = 3

        self.feat_dim = 2 #16 #self.n_sphere*(self.degree+1)**2 # 16
        ''' worked version
        self.level = 3#8#6
        self.reso_factor = 4#2 #4
        self.r = .005
        '''
        ''' v1 good version
        self.level = 4
        self.reso_factor = 4
        self.r = .005
        '''
        self.level = 3
        self.reso_factor = 4
        self.r = .005



        self.nps_ps = [torch.empty((0, 3)) for _ in range(self.level)]
        self.nps_o = nn.Parameter(torch.empty((0, 1))) # opacity for 0 level nps
        self.nps_s = nn.Parameter(torch.empty((0, 1))) # scale for 0 level nps
        self.nps_rot = nn.Parameter(torch.empty((0, 4))) # scale for 0 level nps


        # triangles for level=1 nps_ps for meshing
        self.triangles = torch.empty((0,3),dtype=torch.int32)
        self.nps_ds = [torch.empty((0, 3)) for _ in range(self.level)]
        self.nps_cs = [torch.empty((0, 3)) for _ in range(self.level)]
#[nn.Parameter(torch.empty((0, 3))) for _ in range(self.level)]
        self.nps_Fs = [nn.Parameter(torch.empty(0, self.feat_dim)) for _ in range(self.level)]
        self.nps_num = [0 for _ in range(self.level)]

        self.kdtrees = [None for _ in range(self.level)]

        self.mesh = None

        self.c = 3


        self.frame2nps = [dict() for _ in range(self.level)]

        self.bg_color = .5

        self.device = device

        self.k = 4
        net_width=32
        self.c_mlp = nn.Sequential(
            nn.Linear(self.feat_dim*self.level, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3),
            )       
        '''
        self.c_mlp = nn.Sequential(
            nn.Linear(self.feat_dim*self.level, 3),
            ) 
        '''

        '''
        self.in_dim = self.feat_dim//2*self.level
        self.late_mlp = nn.Sequential(
            nn.Linear(self.in_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, (self.degree+1)**2*self.n_sphere),
            )

        self.weight_mlp = nn.Sequential(
            nn.Linear(self.in_dim, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3*3*2+3*2 + self.n_sphere*3+3),
            )



        self.sem_mlp = nn.Sequential(
            nn.Linear(self.feat_dim*1, 134+100),
            )

        '''

        # grid to speed up frnn
        self.frnn_grid = {}
        self.frnn_grid['find_valid'] = None
        for level in range(self.level):
            self.frnn_grid[level] = None

        def init_param(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        #self.g_mlp.apply(init_param)
        self.c_mlp.apply(init_param)
        '''
        self.late_mlp.apply(init_param)
        self.weight_mlp.apply(init_param)
        self.sem_mlp.apply(init_param)

        '''
        #self.F_mlp.apply(init_param)

        # touch grad mask
        self.vbound = 100 #bound 
        self.voxel_size = 0.5
        self.n_xyz = [int(2*self.vbound/self.voxel_size)]*3
        self.touched = torch.zeros(np.product(self.n_xyz), device=device, dtype=bool)

        self.data_lock = Lock()
        self.cuda_ray = False

        self.pad_one_image_cuda = {} #None #torch.ones((1,H,W,1)).cuda()

    def get_model_copy(self):
        model = SLARE(self.device)
        model.g_mlp = copy.deepcopy(self.g_mlp)
        model.c_mlp = copy.deepcopy(self.c_mlp)
        model.nps_p = copy.deepcopy(self.nps_p)
        model.nps_c = copy.deepcopy(self.nps_c)
        model.nps_F = copy.deepcopy(self.nps_F)
        model.tree = KDTree(self.nps_p)
        model.touched = copy.deepcopy(self.touched)
        return model

    def get_mask_add(self, ps):
        if type(ps) == torch.Tensor:
            ps = ps.cpu().numpy()
        if self.nps_p.shape[0] > 0:
            D, I = self.tree.query(ps,k=1)
        else:
            D = torch.ones((ps.shape[0]))
        mask_add = D[:] > self.r
        return mask_add
 
    def add_neural_points(self, ps, frame_id, color=None, Twc=torch.eye(4), check_valid=True):#rays_o, rays_d, t, color=None):
     #with self.data_lock:
      # voxelize ps with self.r
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(ps.cpu().numpy().astype(np.float64))
      pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy().astype(np.float64))
      

      for level in range(self.level):
        try:
            pcd = pcd.voxel_down_sample(voxel_size=self.r*self.reso_factor**level)#, min_bound=np.array([-self.vbound]*3), max_bound=np.array([self.vbound]*3))
            ps_ = torch.from_numpy(np.asarray(pcd.points)).to(ps)
            color_ = torch.from_numpy(np.asarray(pcd.colors)).to(ps)
        except Exception as e:
            print(e)
            ps_ = torch.from_numpy(np.asarray(pcd.points)).to(ps)
            color_ = torch.from_numpy(np.asarray(pcd.colors)).to(ps)

            


        if self.nps_ps[level].shape[0] > 0:
            #D, I = self.index.search(ps, k=1)
            #D, I = self.tree.query(ps.cpu().numpy(),k=1)
            '''
            pt = pytorch3d.ops.knn_points(ps_.unsqueeze(0), self.nps_ps[level].to(ps).unsqueeze(0), K=1,return_nn=True)
            D,I = torch.sqrt(pt.dists.cpu()), pt.idx.cpu()
            D = D.view(-1).numpy()
            I = I.view(-1).numpy()
            '''
            '''
            D, I, _, _ = frnn.frnn_grid_points(
              ps_.unsqueeze(0), self.nps_ps[level].to(ps).unsqueeze(0), K=1, r=self.r*20, grid=None, return_nn=True, return_sorted=False)
            '''
            D, I = self.kdtrees[level].query(ps_.cpu().numpy().astype(np.float32),k=1)
            #D = torch.as_tensor(D).to(ps_.device)
            #I = torch.as_tensor(I.astype(int)).to(ps_.device)
            I = I.reshape(-1).astype(int)
            D = D.reshape(-1).astype(np.float64)
            I[D>(self.r*20)] = -1
            D = D*D




            D[I==-1] = 1e6
            D = np.sqrt(D)
        else:
            # first time run
            D = torch.ones((ps_.shape[0]))
        
        if check_valid: 
            mask_add = D[:] > self.r *self.reso_factor**level
        else:
            mask_add = D[:] > 0
        points_new = ps_.cpu()[mask_add,:]

        # remove the nan points
        #points_new = points_new[~torch.isnan(points_new.sum(-1)),:]
        points_new = points_new[(~torch.isnan(points_new.sum(-1))) * (~torch.isinf(points_new.sum(-1))),:]

        N_new = points_new.shape[0]
        # NO NEW POINTS
        if N_new == 0:
            continue
        # ADD NEW POINTS
        else:
            for key in self.frnn_grid.keys():
                self.frnn_grid[key] = None
        if self.nps_ps[level].shape[0] > 0:
            feats_F_new = torch.zeros(N_new, self.feat_dim).to(self.device)+.5
            I = torch.from_numpy(I.astype(int)).to(self.device).long()
            mask_near = torch.from_numpy(D[mask_add]<self.r*4*self.reso_factor**level).to(self.device)
            feats_F_new[mask_near] = self.nps_Fs[level].data[I[mask_add][mask_near],:].clone()

        else:
            feats_F_new = torch.zeros(N_new, self.feat_dim)+.5#.uniform_(-1e-4,1e-4)#+.5

        '''
        # build mesh with level == 1
        if level == 1:
            new_triangles = self.frame_mesh(points_new, Twc) + self.nps_ps[level].shape[0]
            self.triangles = torch.cat([self.triangles, new_triangles],axis=0)
        '''
        
        self.nps_ps[level] = torch.cat([self.nps_ps[level].to(self.device), points_new.to(self.device)],axis=0)
        self.kdtrees[level] = KDTree(self.nps_ps[level].cpu().numpy().copy().astype(np.float32))
        

        ''' 
        if level == 0:
            self.nps_o = nn.Parameter(torch.cat([self.nps_o.data.to(self.device), torch.zeros((N_new,1)).to(self.nps_o).to(self.device)],axis=0))
            self.nps_s = nn.Parameter(torch.cat([self.nps_s.data.to(self.device), torch.zeros((N_new,1)).to(self.nps_o).to(self.device)-5],axis=0))
            #self.nps_rot = nn.Parameter(torch.cat([self.nps_rot.data.to(self.device), torch.as_tensor(np.tile([1, 0, 0, 0], (N_new, 1))).to(self.nps_o).to(self.device)],axis=0))
        '''



        
   
        '''
        # meshing
        if level == 1:
            if self.mesh is None:
                self.mesh = o3d.geometry.TriangleMesh()
            self.mesh.vertices = o3d.utility.Vector3dVector(self.nps_ps[level].cpu().numpy().astype(np.float64))
            self.mesh.triangles = o3d.utility.Vector3iVector(self.triangles.numpy().astype(np.int32))
        '''

        ''' 
        # dir
        ds_new = torch.zeros_like(points_new)
        ds_new[:,0] = 1
        self.nps_ds[level] = torch.cat([self.nps_ds[level], ds_new],axis=0)
        '''


        if color is not None:
            color_new = color_.cpu().numpy()[mask_add,:]
            self.nps_cs[level] = torch.cat([self.nps_cs[level], torch.as_tensor(color_new).float()],axis=0)#nn.Parameter(torch.cat([self.nps_cs[level].data.to(self.device), torch.as_tensor(color_new).float().cuda()],axis=0))

        # assign each neural point a frame_id
        '''
        N_nps = self.nps_ps[level].shape[0]
        if frame_id not in self.frame2nps[level]:
            self.frame2nps[level][frame_id] = np.arange(N_nps-N_new,N_nps)
        else:
            self.frame2nps[level][frame_id] = np.concatenate([self.frame2nps[level][frame_id], np.arange(N_nps-N_new,N_nps)])
        '''
        
        self.nps_Fs[level] = nn.Parameter(torch.cat([self.nps_Fs[level].data.to(self.device), feats_F_new.to(self.device)],axis=0))

        # now update num
        self.nps_num[level] = self.nps_ps[level].shape[0]
        #del color_, feats_F_new, ps_
      torch.cuda.empty_cache()
      print('added')

    def frame_mesh(self, ps, Twc):
        '''
            generate mesh for frame
        '''
        # 2D triangulation
        Tcw = torch.inverse(Twc).to(ps) 
        ps_ = ps@Tcw[:3,:3].T+Tcw[:3,(3,)].T
        ps_proj = ps_[:,:2] / ps_[:,(2,)]
        triangle = Delaunay(ps_proj.cpu().numpy())
        simplices = triangle.simplices # N_triangle,3
        # filtering z
        zs = ps[simplices.reshape(-1),2].reshape(-1,3)
        zs_valid = (torch.max(zs,dim=-1)[0] - torch.min(zs,dim=-1)[0] ) < .1
        # filtering 2D 
        xys = ps_proj[simplices.reshape(-1), :].reshape(-1,3,2)
        ls = ((xys - xys[:,(1,2,0),:])**2).sum(-1) # N,3
        ls_valid = ls.max(dim=-1)[0] < .01
        #print(zs_valid.shape, zs_valid.sum(), ls_valid.sum())
        return torch.as_tensor(simplices[zs_valid*ls_valid,:],dtype=torch.int32)
        

        
        


    def transform(self, frame_ids, dTs):
        ic('Transforming renderer frame with ids', frame_ids)
        for level in range(self.level):
            for frame_id, dT in zip(frame_ids, dTs):
                #dT = torch.from_numpy(dT).float()
                if frame_id in self.frame2nps[level].keys():
                    self.nps_ps[level][self.frame2nps[level][frame_id]] = \
                            (dT[:3,:3]@self.nps_ps[level][self.frame2nps[level][frame_id]].T + dT[:3,(3,)]).T
        ic('Transformed!')
    def _linearize_id(self, xyz):
        return xyz[:, 2] + self.n_xyz[-1] * xyz[:, 1] + (self.n_xyz[-1] * self.n_xyz[-2]) * xyz[:, 0]


    def find_valid(self, ps):
        '''
            return the mask of points that is in touched grid
        '''
        level = 0
        grid = None # self.frnn_grid[level]#['find_valid'] 
        if False:#not self.is_train:
            D, I, _, grid = frnn.frnn_grid_points(
                      ps.unsqueeze(0), self.nps_ps[level][:self.nps_num[level],:].to(ps).unsqueeze(0), K=1, r=self.r*20*self.reso_factor**level, return_nn=True, return_sorted=False)
            #self.frnn_grid['find_valid'] = grid

            D,I = torch.sqrt(D.cpu()), I.cpu()
        else:
            D, I = self.kdtrees[level].query(ps.cpu().numpy().astype(np.float32),1)
            D = torch.as_tensor(D).view(-1)
            I = torch.as_tensor(I.astype(int)).view(-1)
            I[D>self.r*20*self.reso_factor**level] = -1


        D[I==-1] == 1e4

        mask_valid = (D.view(-1)<self.r*10)# * (~torch.isnan(ps.sum(-1)).cpu())
        return mask_valid
    def ray_tracing(self, R,t,K,H,W):
        #print(K)
        ray_caster = RayCaster(self.mesh, H, W, K[:3,:3])
        pose = torch.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = t
        # 2. predict
        pc = ray_caster.ray_cast(pose) # N,3
        '''
        depth_on_ray = ans['t_hit'].numpy().reshape((H,W))
        facing_direction = pose[:3,:3].dot(np.array([[0.,0.,1.]]).T).T # 1,3
        facing_direction = facing_direction / np.linalg.norm(facing_direction)
        # depth_im is on z axis
        #depth_im = (ray_direction * facing_direction).sum(-1).reshape((H,W)) * depth_on_ray
        pc = depth_on_ray * ray_direction 
        '''
        return pc



        


    def ray_rasterize(self, R, t, K, H, W, inds=None):
        '''
            follow 
            https://github.com/wangg12/pytorch3d_render_linemod/blob/master/test.py
            in 
            https://github.com/facebookresearch/pytorch3d/issues/934#issuecomment-976695419
        '''
        points_per_pixel = 8

        pc = self.nps_ps[0].to(self.device).float()
        #color = torch.from_numpy(self.nps_cs[0]).to(self.device).float()

        '''
        point_cloud = Pointclouds(points=[pc])#,features=[color])

        f_ndc = -torch.tensor([K[0, 0], K[1, 1]]).to(self.device).float()
        p_ndc = torch.tensor([K[0, 2], K[1, 2]]).to(self.device).float()

        T = t

        RT = torch.zeros((4, 4))
        RT[3, 3] = 1
        RT[:3, :3] = R
        RT[:3, 3] = T
        #Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
        #RT = torch.matmul(Rz, RT)
        RT = torch.inverse(RT)
        R = RT[:3, :3].t().reshape(1, 3, 3)
        T = RT[:3, 3].reshape(1, 3)

        cam = PerspectiveCameras(R=R,T=T,focal_length=f_ndc.unsqueeze(0), principal_point = p_ndc.unsqueeze(0), device=self.device,in_ndc=False, image_size=((H,W),))
        raster_settings = PointsRasterizationSettings(\
                (H,W),
                radius=self.r, 
                points_per_pixel = points_per_pixel,
                )
        rasterizer = PointsRasterizer(cameras = cam,
                raster_settings = raster_settings)
        r = rasterizer.raster_settings.radius
        '''
        r = self.r

        torch.cuda.synchronize()
        st = time()
        if inds is None: #full image
            Twc = torch.eye(4)
            Twc[:3,:3] = R
            Twc[:3,3] = t
            zbuf, idx, dists2 = point_raster(pc, H,W, K=torch.as_tensor(K[:3,:3]).to(self.device).float(),
                                                Twc=Twc.cuda())

            ''' 
            points_proj = rasterizer.transform(point_cloud)

            idx, zbuf, dists2 = rasterize_points(
                points_proj,
                image_size=raster_settings.image_size,
                radius=raster_settings.radius,
                points_per_pixel=raster_settings.points_per_pixel,
                bin_size=raster_settings.bin_size,
                max_points_per_bin=raster_settings.max_points_per_bin)
            '''
            #fragments = rasterizer(point_cloud)
            '''
            torch.cuda.synchronize()
            sys.stdout.write('rast %f'%(time()-st))
            sys.stdout.flush()
            '''


            idx = idx.long()#fragments.idx.long()#1,H,W,_ 
            dist2 = dists2 #fragments.dists
            #zbuf = fragments.zbuf

            belonging_ids = None #np.unique(self.nps2frame[idx.view(-1).cpu().numpy()])

        else: # on certain index
            points_proj = rasterizer.transform(point_cloud)
            idx, zbuf, dist2 = rasterize_points_w_inds_FRNN(
                                        points_proj,
                                        inds,
                                        image_size=raster_settings.image_size,
                                        radius=raster_settings.radius,
                                        points_per_pixel=raster_settings.points_per_pixel,
                                    )
            belonging_ids = None
        '''
        infer = False
        if inds is None: #full image
            infer = True
            inds = torch.arange(H*W).to(self.device).view(1,-1)
        # FRNN rasterize
        points_proj = rasterizer.transform(point_cloud)
        idx, zbuf, dist2 = rasterize_points_w_inds_FRNN(
                                    points_proj,
                                    inds,
                                    image_size=raster_settings.image_size,
                                    radius=raster_settings.radius,
                                    points_per_pixel=raster_settings.points_per_pixel,
                                )
        idx = idx.long()
        dist2 = dist2.detach()
        belonging_ids = None #np.unique(self.nps2frame[idx.view(-1).cpu().numpy()])
        ''' 
            
        valid_mask8 = idx != -1
        valid_mask = (valid_mask8.sum(-1) > 0) #(idx == -1).sum(-1) != 8 
        
        invalid_mask = ~valid_mask
        has_hole = invalid_mask.sum()>0
        
        # interpolate
        dist2[~valid_mask8] = 1e8

        # Q:solve the inflation problem
        # A: 1. use small radius
        #    2. use alpha composition to cover most surface
        # with 1 and 2 got 25.4 psnr on room0
        #    3. fill in small black hole by using rasterize_frnn

        use_first_surf = True # first layer with first point
        use_first_cluster = False # first cluster


        if use_first_surf:
            ### Find the first layers of points
            # assumes the first point is with 1
            # example with 111011

            zbuf[~valid_mask8] = 1e8

            zbuf_diff_occ = (zbuf[...,1:] - zbuf[...,:-1] ) < 1e-2

            if not H*W in self.pad_one_image_cuda:# or self.pad_one_image_cuda.shape[1] != H:
                self.pad_one_image_cuda[H*W] = torch.ones((1,H,W,1)).cuda()
            occ = torch.cat([self.pad_one_image_cuda[H*W], zbuf_diff_occ], dim = -1)

            #occ = torch.cat([torch.ones_like(zbuf[...,(0,)]), 
            #                    (zbuf[...,1:] - zbuf[...,:-1] ) < 1e-2,
            #        ], dim = -1)



            occ[~valid_mask8] = 0
            # 111000
            occ_first_layer = torch.cumprod(occ,dim=-1) 

            dist2[~occ_first_layer.bool()] = 1e8
    


            ###
        elif use_first_cluster:
            ### Find the first cluster of points
            # example with 011011
            zbuf[~valid_mask8] = 1e8
            occ = torch.cat([ (zbuf[...,1:] - zbuf[...,:-1] ) < 1e-2,
                                torch.zeros_like(zbuf[...,(0,)]), 
                    ], dim = -1)
            # 01110110
            occ[~valid_mask8] = 0
            # 01233455
            occ_cum = torch.cumsum(occ,dim=-1)
            # 01111111 
            mask_start = occ_cum>0
            # 01234567
            mask_start_cum = torch.cumsum(mask_start,dim=-1)
            # 01110000
            occ_first_cluster = mask_start_cum == occ_cum
            # from time interval to points 
            # 01221000
            occ_first_cluster[...,1:] += occ_first_cluster[...,:-1].clone()
            # 01111000
            occ_first_cluster = occ_first_cluster > 0

            # if only have first point, make first point the one
            occ_first_cluster_sum = occ_first_cluster.sum(-1)
            _one = torch.zeros(1,occ_first_cluster.shape[-1]).to(occ_first_cluster)
            _one[0,0] = True 
            occ_first_cluster[occ_first_cluster_sum==0] *= _one
            


            dist2[~occ_first_cluster.bool()] = 1e8
            ###



        '''
        # dist2 is in NDC space
        #dist_3d = torch.sqrt(dist2) * zbuf
        #dist2[dist_3d>self.r*2] = 1e8
        #dist2[dist2>self.r*2] = 1e8 # solve the inflation problem 
        w = torch.exp(-dist2/(r**2)/100)#1/(dist2/r**2+1e-8)
        #w = 1/(dist2/self.r**2+1e-8)
        #w = 1-dist2/r**2
        alphas = w 
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        w = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] 

        w /= w.sum(-1,keepdims=True)
        '''
        w = torch.exp(-dist2/(r**2)/100)#1/(dist2/r**2+1e-8)
        w /= w.sum(-1,keepdims=True)


            
        







        pc_ = (pc[idx.view(-1),:].reshape(*(idx.shape),3) * w.unsqueeze(-1)).sum(-2) # 1,H,W,3


        # 3. fill in not observed place

        if False: #(~valid_mask).sum()>0:
            '''
            Twc = torch.eye(4)
            Twc[:3,:3] = R
            Twc[:3,3] = t
            print((~valid_mask).sum())
            
            zbuf, idx, dists2 = point_raster(pc, H,W, K=torch.as_tensor(K[:3,:3]).to(self.device).float(),
                                                Twc=Twc.cuda(), mask_fill=(~valid_mask), r=16)
            zbuf = zbuf[~valid_mask,:][None]
            idx = idx[~valid_mask,:][None]
            dist2 = dists2[~valid_mask,:][None]

            '''

            #    self.HWrange = torch.arange(H*W).long().to(self.device).reshape(-1)
            # multiply to replace torch where for speedup
            points_proj = rasterizer.transform(point_cloud)
            inds = torch.nonzero(invalid_mask.view(-1), as_tuple=True)[0][None]
            idx, zbuf, dist2 = rasterize_points_w_inds_FRNN(
                                        points_proj,
                                        inds,
                                        image_size=raster_settings.image_size,
                                        radius=raster_settings.radius*16,
                                        points_per_pixel=raster_settings.points_per_pixel,
                                    )
            '''
            torch.cuda.synchronize()
            sys.stdout.write('frnn rast %f'%(time()-st))
            sys.stdout.flush()
            '''
            if idx is not None:
                _, _M, _k = idx.shape 
                idx_ = idx.long()
                dist2_ = dist2.detach()

                # -> now rasterize in 3D space
                idxed_pc = pc[idx.view(-1),:].view(_M,_k,3) - t
                # Tcw = R.T,t
                x = torch.arange(0,W)
                y = torch.arange(0,H)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # H,W
                grid = torch.stack([grid_y, grid_x],dim=-1).reshape(-1,2).to(self.device)
                grid = grid[inds,:] #N,2
                invK = torch.inverse(K).cuda()
                px_xy = grid[0, :,[1,0]].float()
                p_xy = (invK[:2,:2]@px_xy.T+invK[:2,(2,)] ).T
                p_xyz = torch.ones((px_xy.shape[0],3))
                p_xyz[:,:2] = p_xy
                ray_d = p_xyz / torch.norm(p_xyz,dim=-1,keepdim=True) # M,3
                ray_d = (ray_d@R[0].T).cuda()
                dist_3d = (idxed_pc**2).sum(-1) - ((idxed_pc * ray_d[:,None,:])).sum(-1)**2
                valid_3d = dist_3d < 0.05**2

                idx_[~valid_3d.view(idx.shape)] = -1
                # <-


                valid_mask8_ = idx_ != -1
                valid_mask_ = valid_mask8_.sum(-1) > 0 #(idx == -1).sum(-1) != 8 
                dist2_[~valid_mask8_] = 1e8

                if use_first_surf:
                    ### only use the first layers of points
                    # 1. ordered in z-ascend order
                    zbuf[~valid_mask8_] = 1e8
                    zbuf, sorted_ids = torch.sort(zbuf, dim=-1, descending=False)

                    valid_mask8_ = torch.gather(valid_mask8_,dim=-1, index=sorted_ids)
                    idx_ = torch.gather(idx_,dim=-1, index=sorted_ids)
                    dist2_ = torch.gather(dist2_,dim=-1, index=sorted_ids)
                    # 2. get the first layer of points
                    # assumes the first point is with 1
                    # use cuda-ed ones because tensor to cuda is too slow
                    occ = torch.cat([torch.ones_like(zbuf[...,(0,)]), 
                                        (zbuf[...,1:] - zbuf[...,:-1] ) < 1e-2,
                            ], dim = -1)
                    occ[~valid_mask8_] = 0
                    occ_first_layer = torch.cumprod(occ,dim=-1) 
                    dist2_[~occ_first_layer.bool()] = 1e8
                    ###
                elif use_first_cluster:
                    ### only use the first cluster of points
                    # 1. ordered in z-ascend order
                    zbuf[~valid_mask8_] = 1e8
                    zbuf, sorted_ids = torch.sort(zbuf, dim=-1, descending=False)

                    valid_mask8_ = torch.gather(valid_mask8_,dim=-1, index=sorted_ids)
                    idx_ = torch.gather(idx_,dim=-1, index=sorted_ids)
                    dist2_ = torch.gather(dist2_,dim=-1, index=sorted_ids)
                    # 2. get the first cluster of points
                    occ = torch.cat([ (zbuf[...,1:] - zbuf[...,:-1] ) < 1e-2,
                                        torch.zeros_like(zbuf[...,(0,)]), 
                            ], dim = -1)
                    # 01110110
                    occ[~valid_mask8_] = 0
                    # 01233455
                    occ_cum = torch.cumsum(occ,dim=-1)
                    # 01111111 
                    mask_start = occ_cum>0
                    # 01234567
                    mask_start_cum = torch.cumsum(mask_start,dim=-1)
                    # 01110000
                    occ_first_cluster = mask_start_cum == occ_cum
                    # from time interval to points 
                    # 01221000
                    occ_first_cluster[...,1:] += occ_first_cluster[...,:-1].clone()
                    # 01111000
                    occ_first_cluster = occ_first_cluster > 0


                    # if only have first point, make first point the one
                    occ_first_cluster_sum = occ_first_cluster.sum(-1)
                    _one = torch.zeros(1,occ_first_cluster.shape[-1]).to(occ_first_cluster)
                    _one[0,0] = True 
                    occ_first_cluster[occ_first_cluster_sum==0] *= _one
                    


                    dist2_[~occ_first_cluster.bool()] = 1e8
                   

                '''
                alphas = torch.exp(-dist2_/(r**2)/100)
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
                w_ = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] 
                '''
                w_ = torch.exp(-dist2_/(r**2)/100)#1/(dist2/r**2+1e-8)

                w_ /= w_.sum(-1,keepdims=True)
                pc_[~valid_mask,:] = (pc[idx_.view(-1),:].reshape(*(idx_.shape),3) * w_.unsqueeze(-1)).sum(-2) #
                valid_mask[~valid_mask] = valid_mask_

        return valid_mask, pc_[valid_mask,:], belonging_ids #pc[idx]#fragments.zbuf[valid_mask,0]


        #z_ = (zbuf * w).sum(-1)
        #return valid_mask, z_[valid_mask], belonging_ids
        
    def encode(self, x, d):
        N = x.shape[0]
        #D, I = self.tree.query(x.detach().cpu().numpy(), k=self.k)
        '''
        pt = pytorch3d.ops.knn_points(x.unsqueeze(0), torch.from_numpy(self.nps_p).to(x).unsqueeze(0),K=self.k,return_nn=False)
        D2,I = pt.dists, pt.idx
        '''
        #x = x.detach()
        #ic(torch.isnan(x).sum(), torch.isinf(x).sum())
        f_c_lst, f_d_lst = [], []
        for level in range(self.level):
            #grid = None #self.frnn_grid[level]
            #print(level,self.nps_num[level],self.nps_ps[level].shape, torch.isnan(self.nps_ps[level][:self.nps_num[level],:].unsqueeze(0)).sum())
            #print(self.nps_ps[level][0,0], self.nps_ps[level].shape, self.nps_num[level])
            if False:#not self.is_train:
                D, I, _, grid = frnn.frnn_grid_points(
                              x.unsqueeze(0), self.nps_ps[level][:self.nps_num[level],:].unsqueeze(0), K=self.k, r=self.r*20*self.reso_factor**level, return_nn=True, return_sorted=False)
            else:
                #st =time()
                D, I = self.kdtrees[level].query(x.cpu().numpy().astype(np.float32),self.k)
                #print('encode query', 'level', level, time()-st)
                I = torch.as_tensor(I.astype(int)).to(x.device)
                D = torch.as_tensor(D*D).to(x.device)
                I[D>(self.r*20*self.reso_factor**level)**2] = -1
            #self.frnn_grid[level] = grid
            D2,I = D.view(-1,self.k), I.view(-1,self.k)
            D2[I==-1] = 1e4
            w = 1/(D2/(self.r*self.reso_factor**2)+1e-8)
            w_sum = w.sum(-1)
            w = w / w_sum.unsqueeze(-1)
            w = w.detach()

            f_c = self.nps_Fs[level][I.view(-1), :].reshape(N,self.k,self.feat_dim)
            f_c = (f_c * w.unsqueeze(-1)).sum(1)

            #f_d = self.nps_ds[level][I.view(-1), :].reshape(N,self.k,3)
            # TODO: transform
            f_d = d

            #f_c_lst.append(f_c[:,:self.feat_dim//2])
            #f_w_lst.append(f_c[:,self.feat_dim//2:])
            f_c_lst.append(f_c.unsqueeze(0))
            f_d_lst.append(f_d.unsqueeze(0))


        f_c = torch.cat(f_c_lst, axis=-1)
        f_d = torch.stack(f_d_lst).mean(0)

        return f_c, f_d


    def forward(self, x, d=None, make_density=True, make_color=True, is_train=True, jacobian=False):
        '''
          x: Nx3
        '''
        mask_valid = None
        st = time()
        #if not is_train:

        if self.nps_ps[0].shape[0]>0:
            mask_valid = self.find_valid(x.view(-1,3))

            x = x[mask_valid,:]
            d = d[mask_valid,:]
        #d = d / d.norm(p=2, dim=-1).unsqueeze(-1)
        #else:

        N = x.shape[0]
        if self.nps_ps[0].shape[0] == 0 or mask_valid.sum() == 0:
            return torch.zeros((N)).to(x), None, None

        f_c, f_d = self.encode(x,d)
        color = torch.sigmoid(self.c_mlp(f_c))
        if True:#mask_valid is not None:
            sigma_, color_ = None, None
            color_grad_ = None
            if make_density:
                sigma_ = torch.zeros((mask_valid.shape[0],1)).to(x)
                sigma_[mask_valid,:] = sigma
            if make_color:
                color_ = torch.zeros((mask_valid.shape[0],self.c)).to(x) + self.bg_color
                color_[mask_valid,:] = color

            return sigma_, color_, color_grad_

        #return sigma, color

    def density(self,x,is_train=True):
        return {'sigma':self(x,make_color=False,is_train=is_train)[0]}

    def color(self,x,d=None,is_train=True, jacobian=False):
        return self(x,d=d,make_density=False,is_train=is_train, jacobian=jacobian)

    # optimizer utils 
    def get_params(self, lr, with_mlp=True): 
        params = [ 
            *[{'params': self.nps_Fs[level], 'lr': lr} for level in range(self.level)]]
        if with_mlp:
            params.extend([
            #{'params':self.nps_o, 'lr':lr},
            #{'params':self.nps_s, 'lr':1e-2},
            #{'params':self.nps_cs[0], 'lr':lr},
            #{'params':self.nps_rot, 'lr':1e-4},
            {'params':self.c_mlp.parameters(), 'lr':1e-4},
            ])

        return params


    def save(self, path):
        state_dicts = {}
        state_dicts['ps'] = self.nps_ps
        state_dicts['cs'] = self.nps_cs

        state_dicts['o'] = self.nps_o
        state_dicts['s'] = self.nps_s
        state_dicts['rot'] = self.nps_rot
        state_dicts['triangles'] = self.triangles
        state_dicts['Fs'] = self.nps_Fs
        state_dicts['c_mlp'] = self.c_mlp.state_dict()
        
        '''
        torch.save(self.c_mlp.state_dict(), path+'.c_mlp')
        torch.save(self.embedder_rel_pos.state_dict(), path+'.embedder_rel_pos')
        torch.save(self.space_mlp.state_dict(), path+'.space_mlp')
        '''


        #torch.save(self.weight_mlp.state_dict(), path+'.weight_mlp')

        torch.save(state_dicts,path)
        print('nps ckpt saved!')
        '''
        o3d.io.write_triangle_mesh(os.path.dirname(path)+'/mesh.ply', self.mesh)
        print('mesh saved!')
        '''




    def load(self, path):
        state_dicts = torch.load(path)
        self.nps_ps = state_dicts['ps']
        for level in range(len(self.nps_ps)):
            self.kdtrees[level] = KDTree(self.nps_ps[level].cpu().numpy().astype(np.float32))
        

        #self.nps_cs = state_dicts['cs']

        if 'o' in state_dicts.keys():
            self.nps_o = state_dicts['o']
            self.nps_s = state_dicts['s']
            self.nps_rot = state_dicts['rot']
        self.triangles = state_dicts['triangles']
        self.nps_num = [int(self.nps_ps[i].shape[0]) for i in range(self.level)]
        self.nps_Fs = state_dicts['Fs']
        self.c_mlp.load_state_dict(state_dicts['c_mlp'])
        '''
        self.late_mlp.load_state_dict(torch.load(path+'.late_mlp'))
        self.weight_mlp.load_state_dict(torch.load(path+'.weight_mlp'))
        '''

        ic('nps ckpt loaded!')


    def nps2mesh(self):
        #from .ext import estimate_normals
        device = torch.device(self.device)

        input_xyz = self.nps_ps[1].float().to(device)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_xyz.cpu().numpy().astype(np.float64))
        #pcd.remove_statistical_outlier(nb_neighbors=50,
        #                                   std_ratio=2.0)
        pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                                  max_nn=30))
        #input_normal = estimate_normals(input_xyz, 16, .1, [.0, .0, .0])
        input_xyz = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
        input_normal = torch.from_numpy(np.asarray(pcd.normals)).float().to(device)




        main_device = input_xyz.device
        uni_model = get_uni_model(main_device)
        surface_mapping = argparse.Namespace()
        surface_mapping.GPIS_mode= "sample"
        surface_mapping.margin= .1
        surface_mapping.bound_min= [-10., -5., -10.]
        surface_mapping.bound_max= [10., 5., 10.]
        surface_mapping.voxel_size= 0.05
        surface_mapping.prune_min_vox_obs= 1
        surface_mapping.ignore_count_th= 1.0
        surface_mapping.encoder_count_th= 60000.0
        sm = SurfaceMap(uni_model, None,
                        surface_mapping, uni_model.surface_code_length, device=main_device,
                                enable_async=False)
        sm.integrate_keyframe(input_xyz, input_normal) 
        final_mesh = sm.extract_mesh(4, int(4e6), max_std=0.15,
                                                  extract_async=False, interpolate=True)




        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_xyz.cpu().numpy().astype(np.float64))
        pcd.normals = o3d.utility.Vector3dVector(input_normal.cpu().numpy().astype(np.float64))
        radii = [0.005, 0.01, 0.02, 0.04]
        final_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        '''
        '''
        import nksr
        reconstructor = nksr.Reconstructor(device)
        field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=0.05)
        mesh = field.extract_dual_mesh(mise_iter=2)
        v, f= mesh.v, mesh.f
        final_mesh = o3d.geometry.TriangleMesh()
        final_mesh.vertices = o3d.utility.Vector3dVector(v.cpu().numpy().astype(np.float64))
        final_mesh.triangles = o3d.utility.Vector3iVector(f.cpu().numpy().astype(np.int32))
        '''
        '''
        current_path = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_path+'/../external/neural-galerkin/')

        import torch_spsr

        v, f = torch_spsr.reconstruct(
          input_xyz/10,          # torch.Tensor (N, 3)
          input_normal,   # torch.Tensor (N, 3)
          depth=4, 
          voxel_size=0.005,
          screen_alpha=32.0
        )
        v *= 10

        from pykdtree.kdtree import KDTree

        tree = KDTree(input_xyz.cpu().numpy())
        D, I = tree.query(v.cpu().numpy(),k=1)
        def to_o3d_mesh(vertices: torch.Tensor, triangles: torch.Tensor):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
            mesh.compute_vertex_normals()
            return mesh

        final_mesh = to_o3d_mesh(v,f)
        idx = np.where(D.reshape(-1)>self.r*4)
        final_mesh.remove_vertices_by_index(idx[0].tolist())
        '''
        return final_mesh




 




        
        


