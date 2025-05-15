from typing import List, Optional, Tuple, Union
import torch
from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc

# modified from pytorch3d https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/renderer/points/rasterize_points.py#L185


def _format_radius(
    radius: Union[float, List, Tuple, torch.Tensor], pointclouds
) -> torch.Tensor:
    """
    Format the radius as a torch tensor of shape (P_packed,)
    where P_packed is the total number of points in the
    batch (i.e. pointclouds.points_packed().shape[0]).

    This will enable support for a different size radius
    for each point in the batch.

    Args:
        radius: can be a float, List, Tuple or tensor of
            shape (N, P_padded) where P_padded is the
            maximum number of points for each pointcloud
            in the batch.

    Returns:
        radius: torch.Tensor of shape (P_packed)
    """
    N, P_padded = pointclouds._N, pointclouds._P
    points_packed = pointclouds.points_packed()
    P_packed = points_packed.shape[0]
    if isinstance(radius, (list, tuple)):
        radius = torch.tensor(radius).type_as(points_packed)
    if isinstance(radius, torch.Tensor):
        if N == 1 and radius.ndim == 1:
            radius = radius[None, ...]
        if radius.shape != (N, P_padded):
            msg = "radius must be of shape (N, P): got %s"
            raise ValueError(msg % (repr(radius.shape)))
        else:
            padded_to_packed_idx = pointclouds.padded_to_packed_idx()
            radius = radius.view(-1)[padded_to_packed_idx]
    elif isinstance(radius, float):
        radius = torch.full((P_packed,), fill_value=radius).type_as(points_packed)
    else:
        msg = "radius must be a float, list, tuple or tensor; got %s"
        raise ValueError(msg % type(radius))
    return radius

def rasterize_points_w_inds(
    pointclouds,
    inds = None,
    image_size: Union[int, Tuple[int, int]] = 256,
    radius: Union[float, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
):
    """
    Naive pure PyTorch implementation of pointcloud rasterization.

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
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Support variable size radius for each point in the batch
    radius = _format_radius(radius, pointclouds)

    # Initialize output tensors.
    point_idxs = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    pix_dists = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.float32, device=device
    )

    x = torch.range(0,W-1)
    y = torch.range(0,H-1)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # H,W
    grid = torch.stack([grid_y, grid_x],dim=-1).reshape(-1,2).to(inds.device)
    grid = grid[inds,:]
    



    # NDC is from [-1, 1]. Get pixel size using specified image size.
    radius2 = radius * radius

    # Iterate through the batch of point clouds.
    for n in range(N):
        point_start_idx = cloud_to_packed_first_idx[n]
        point_stop_idx = point_start_idx + num_points_per_cloud[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        #for yi in range(H):
        for i in range(M):
            yi = grid[i,0]
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = H - 1 - yi
            yf = pix_to_non_square_ndc(yfix, H, W)

            # Iterate through pixels on this horizontal line, left to right.
            #for xi in range(W):
            if True:
                xi = grid[i,1]
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = W - 1 - xi
                xf = pix_to_non_square_ndc(xfix, W, H)

                top_k_points = []
                # Check whether each point in the batch affects this pixel.
                for p in range(point_start_idx, point_stop_idx):
                    px, py, pz = points_packed[p, :]
                    r = radius2[p]
                    if pz < 0:
                        continue
                    dx = px - xf
                    dy = py - yf
                    dist2 = dx * dx + dy * dy
                    if dist2 < r:
                        top_k_points.append((pz, p, dist2))
                        top_k_points.sort()
                        if len(top_k_points) > K:
                            top_k_points = top_k_points[:K]
                for k, (pz, p, dist2) in enumerate(top_k_points):
                    zbuf[n, i, k] = pz
                    point_idxs[n, i, k] = p
                    pix_dists[n, i, k] = dist2
    return point_idxs, zbuf, pix_dists

def rasterize_points_w_inds_fast(
    pointclouds,
    inds = None,
    image_size: Union[int, Tuple[int, int]] = 256,
    radius: Union[float, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
):
    """
    Naive pure PyTorch implementation of pointcloud rasterization.

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
    cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
    num_points_per_cloud = pointclouds.num_points_per_cloud()

    # Support variable size radius for each point in the batch
    radius = _format_radius(radius, pointclouds)

    # Initialize output tensors.
    point_idxs = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.int32, device=device
    )
    zbuf = torch.full((N, H, W, K), fill_value=-1, dtype=torch.float32, device=device)
    pix_dists = torch.full(
        (N, M, K), fill_value=-1, dtype=torch.float32, device=device
    )

    x = torch.range(0,W-1)
    y = torch.range(0,H-1)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # H,W
    grid = torch.stack([grid_y, grid_x],dim=-1).reshape(-1,2).to(inds.device)
    grid = grid[inds,:]
    



    # NDC is from [-1, 1]. Get pixel size using specified image size.
    radius2 = radius * radius

    # Iterate through the batch of point clouds.
    for n in range(N):
        point_start_idx = cloud_to_packed_first_idx[n]
        point_stop_idx = point_start_idx + num_points_per_cloud[n]

        # Iterate through the horizontal lines of the image from top to bottom.
        #for yi in range(H):
        for i in range(M):
            yi = grid[i,0]
            # Y coordinate of one end of the image. Reverse the ordering
            # of yi so that +Y is pointing up in the image.
            yfix = H - 1 - yi
            yf = pix_to_non_square_ndc(yfix, H, W)

            # Iterate through pixels on this horizontal line, left to right.
            #for xi in range(W):
            if True:
                xi = grid[i,1]
                # X coordinate of one end of the image. Reverse the ordering
                # of xi so that +X is pointing to the left in the image.
                xfix = W - 1 - xi
                xf = pix_to_non_square_ndc(xfix, W, H)

                top_k_points = []
                # Check whether each point in the batch affects this pixel.
                for p in range(point_start_idx, point_stop_idx):
                    px, py, pz = points_packed[p, :]
                    r = radius2[p]
                    if pz < 0:
                        continue
                    dx = px - xf
                    dy = py - yf
                    dist2 = dx * dx + dy * dy
                    if dist2 < r:
                        top_k_points.append((pz, p, dist2))
                        top_k_points.sort()
                        if len(top_k_points) > K:
                            top_k_points = top_k_points[:K]
                for k, (pz, p, dist2) in enumerate(top_k_points):
                    zbuf[n, i, k] = pz
                    point_idxs[n, i, k] = p
                    pix_dists[n, i, k] = dist2
    return point_idxs, zbuf, pix_dists
