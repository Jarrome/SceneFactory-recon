import math
import trimesh
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from time import sleep, time

def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, depth=None, frame_id=None, rast_info=None, color=None, track=False, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]


        is_train = depth is not None

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        '''
        if depth is not None:
            t = depth.contiguous().view(-1,1)
            if not track:
                ps = rays_o + t * rays_d
                if color is not None:
                    assert frame_id is not None, "training need frame_id"
                    self.add_neural_points(ps, frame_id, color[...,:3].view(-1,3))
                else:
                    assert False, 'color shoulnot be None'
                    self.add_neural_points(ps)
        '''


        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        output = {}

        # choose aabb
        if depth is None:
            R,t,K,H,W = rast_info
            if self.nps_ps[0].shape[0] > 0:
                st_ = time()
                mask, pc, belonging_ids = self.ray_rasterize(*rast_info)


                #pc = self.ray_tracing(*rast_info)

                #mask, z, belonging_ids = self.ray_rasterize(*rast_info)
                print('rasterize_take', time()-st_)
                #print('ray tracing take', time()-st_)
                output['belonging_ids'] = belonging_ids
                mask = mask.view(-1)
                #z = z.view(-1,1)
                #rays_d_ori = rays_d @ rast_info[0]
                #t = z*rays_d_ori[mask,:][:,(2,)]

                #z = ((pc - rays_o[mask,:]) * rays_d[mask,:]).sum(-1,keepdims=True)
                '''
                xy1 = rays_d / rays_d[:,(2,)]
                z = z*torch.sqrt((xy1[mask,:]**2).sum(-1,keepdims=True))
                '''

                xyzs = torch.zeros(N,3).to(rays_o)
                '''
                viewer = o3d.visualization.Visualizer()
                viewer.create_window()

                pdb.set_trace()
                #xyzs[mask,:] = rays_o[mask,:] + t * rays_d[mask,:]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(self.nps_p)
                #o3d.visualization.draw_geometries([pcd])
                #o3d.visualization.draw_geometries([pcd, pcd2])
        
                viewer.add_geometry(pcd)

                opt = viewer.get_render_option()
                opt.show_coordinate_frame = True
                viewer.run()
                viewer.destroy_window()
                '''
                #xyzs[mask,:] = rays_o[mask,:] + z * rays_d[mask,:]
                #xyzs = pc.to(rays_o)
                #print(N, mask.shape, mask.sum(), pc.shape)
                xyzs[mask,:] = pc.clone()
                '''
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyzs[mask,:].cpu().numpy())

                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
                o3d.visualization.draw_geometries([pcd])
                o3d.visualization.draw_geometries([pcd, pcd2])
                '''
        
                #

            else:
                xyzs = torch.zeros(N,3).to(rays_o)

        else:
            xyzs = rays_o + rays_d * depth.view(-1,1)
            #R,t,K,H,W,inds = rast_info
            #pred_depth, pred_color = self.rasterize_gs(*rast_info[:-1], depth_only=True) #1,H,W,   H,W,3 

            #output['depth_im'] = pred_depth[0]
            #output['color_im'] = pred_color



            '''
            R,t,K,H,W,inds = rast_info
            mask, pc, _ = self.ray_rasterize(*rast_info)
            mask = mask.view(-1)
            inds = inds.view(-1)

            #pc_ = torch.zeros(H*W,3).to(rays_o)
            pc_ = torch.zeros(inds.shape[0],3).to(rays_o)
            pc_[mask,:] = pc.clone()
            #pc_ = pc_[inds,:]

            #sub_mask = mask[inds]
            #z = ((pc_[sub_mask,:] - rays_o[sub_mask,:]) * rays_d[sub_mask,:]).sum(-1,keepdims=True)
            z = ((pc_ - rays_o) * rays_d).sum(-1,keepdims=True)


            xyzs = torch.zeros(N,3).to(rays_o)
            xyzs = rays_o + z * rays_d
            '''



        _step = 2100000
        if xyzs.shape[0] > _step:
            # only in test
            xyzs = xyzs.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            rgbs_lst = []
            for p_id in range(0,xyzs.shape[0], _step):
                pred_pc, rgbs, color_grad = self.color(xyzs[p_id:min(p_id+_step, xyzs.shape[0]),:], rays_d[p_id:p_id+min(_step,xyzs.shape[0]),:], jacobian=track)#, dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
                if rgbs is None:
                    rgbs = torch.zeros((min(p_id+_step, xyzs.shape[0])-p_id,3)).to(xyzs)
                rgbs_lst.append(rgbs)
            rgbs = torch.cat(rgbs_lst,axis=0)
            color_grad = None


        else:
            pred_pc, rgbs, color_grad = self.color(xyzs.reshape(-1, 3), rays_d.reshape(-1,3), jacobian=track)#, color=color.reshape(-1,3) if color is not None else color)#, dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        if depth is None:
            # FIXME: this is not depth
            #dist = torch.sqrt(((xyzs - rays_o)**2).sum(-1,keepdims=True))
            T = torch.eye(4).cuda(self.device)
            T[:3,:3] = R
            T[:3,3] = t
            invT = torch.inverse(T)
            depth = (xyzs.reshape(-1,3)@invT[:3,:3].T + invT[:3,(3,)].T)[:,2]

            # fix bug, xyzs contains zeros, need maskout
            try:
                depth[~mask] = 0
            except:
                pass

            #depth = torch.sqrt(((pred_pc - rays_o)**2).sum(-1,keepdims=True))

        # calculate color
        # SLF color
        image = rgbs
        # GS color
        #image = pred_color
        if image is not None:
            image = image.view(*prefix, self.c)
        depth = depth.view(*prefix)

        output.update({
            'depth': depth,
            'image': image,
            'color_grad': color_grad,
            })
        if 'belonging_ids' not in output:
            output['belonging_ids'] = None
        return output


    def render_w_depth(self, rays_o, rays_d, depth, frame_id, rast_info=None, color=None, track=False, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        _run = self.run
        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], depth[b:b+1, head:tail], track, **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, depth, frame_id, rast_info, color, track, **kwargs)

        return results

    def render(self, rays_o, rays_d, rast_info, staged=False, max_ray_batch=21000, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        B, N = rays_o.shape[:2]
        device = rays_o.device
        results = self.run(rays_o, rays_d, None, None, rast_info, **kwargs)

        return results
