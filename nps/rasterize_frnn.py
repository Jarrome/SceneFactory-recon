from typing import List, Optional, Tuple, Union
import torch
#from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc
from pykdtree.kdtree import KDTree

import frnn 
import pdb



# modified from pytorch3d https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/points/rasterize_points.py#L185

def non_square_ndc_range(S1, S2):
    """
    In the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range of 2.0.

    Args:
        S1: dimension along with the NDC range is needed
        S2: the other image dimension

    Returns:
        ndc_range: NDC range for dimension S1
    """
    ndc_range = 2.0
    if S1 > S2:
        ndc_range = (S1 / S2) * ndc_range
    return ndc_range


def pix_to_non_square_ndc_batch(ids, S1, S2):
    '''
        follow https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/mesh/rasterize_meshes.html's single pixel version


        ids: N
        S1: float
        S2: float
    '''
    ndc_range = non_square_ndc_range(S1, S2)
    offset = ndc_range / 2.0
    return -offset + (ndc_range * ids + offset) / S1


def rasterize_points_w_inds_FRNN(
    pointclouds,
    inds = None,
    image_size: Union[int, Tuple[int, int]] = 256,
    radius: Union[float, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
):
    """
    Inputs / Outputs: Same as above

    YIJUN: this version use index to only rasterize certain idx of image
    """
    inds = inds.view(-1)
    M = inds.shape[-1]


    N = len(pointclouds)
    H, W = (
        image_size
        if isinstance(image_size, (tuple, list))
        else (image_size, image_size)
    )
    K = points_per_pixel
    device = pointclouds.device

    points_packed = pointclouds.points_packed()
    #cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    #num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Support variable size radius for each point in the batch
    #radius = _format_radius(radius, pointclouds)

    # Initialize output tensors.
    point_idxs = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    pix_dists = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.float32, device=device
    )

    x = torch.arange(0,W)
    y = torch.arange(0,H)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # H,W
    grid = torch.stack([grid_y, grid_x],dim=-1).reshape(-1,2).to(inds.device)
    grid = grid[inds,:] #N,2

    yfs = pix_to_non_square_ndc_batch(H-1-grid[:,0], H, W)
    xfs = pix_to_non_square_ndc_batch(W-1-grid[:,1], W, H)
    # ndc: x \in [-2,2], y \in [-1,1]



    valid_zforward = points_packed[:,2]>0
    #valid_x = torch.abs(points_packed[:,0]) <= 2.5
    #valid_y = torch.abs(points_packed[:,1]) <= 1.5
    frnn_pc = points_packed[valid_zforward,:].clone()
    frnn_pc[:,-1] = 1

    frnn_q = torch.ones((grid.shape[0],3)).to(inds.device)
    frnn_q[:,0] = xfs
    frnn_q[:,1] = yfs

    if frnn_pc.shape[0] == 0:
        return None, None, None
    tree = KDTree(frnn_pc.cpu().numpy())
    D, I = tree.query(frnn_q.cpu().numpy(),k=K)
    D = torch.as_tensor(D).to(frnn_q)[None]
    I = torch.as_tensor(I.astype(int)).to(frnn_q.device)[None]
    I[D>radius] = -1

    '''
    # frnn on
    D, I, _, _ = frnn.frnn_grid_points(
                      frnn_q.unsqueeze(0), frnn_pc.unsqueeze(0), K=K, r=radius, grid=None, return_nn=True, return_sorted=True) #1xNxK
    '''

    # return is with 1xNxK
    point_idxs = torch.arange(points_packed.shape[0]).to(device)
    point_idxs = point_idxs[valid_zforward][I.view(-1)].view(I.shape)
    point_idxs[I == -1] = -1

    zbuf = points_packed[point_idxs.view(-1),2].view(I.shape)
    pix_dists = D

    return point_idxs, zbuf, pix_dists

