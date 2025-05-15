import torch
import fastpr
import pdb
from time import time
import pdb

#idx, zbuf, dist2 = [None] *3
#dict_key, dict_value, dict_key_sorted, dict_value_sorted, range_ = [None]*5
#px = None


placeholders = {}


def point_raster(pc, H,W, K, Twc, mask_fill=None, r=1., near=.1, far=10):
    '''
        pc: N,3

        # mask is to indcate which pixel to raster
    '''
    #global idx, zbuf, dist2, range_, px
    global placeholders
    #global dict_key, dict_value, dict_key_sorted, dict_value_sorted, range_
    #global px

    torch.cuda.synchronize()
    st = time()




    N_p = pc.shape[0]
    T = Twc[:3,:].reshape(-1) 

    # 1. sort the point by z

    Tcw = torch.inverse(Twc)
    pc_c = pc@Tcw[:3,:3].T+Tcw[:3,(3,)].T
    _, sort_ids = torch.sort(pc_c[:,2], descending=False) 

    pc_sorted = pc[sort_ids,:] 

    # rasterize

    #if idx is None:
    if not H*W in placeholders: 
        
        # create tensor to cuda is slow
        idx = torch.zeros(H,W,8, dtype = torch.int32).cuda()-1
        zbuf = torch.zeros(H,W,8).cuda()
        dist2 = torch.zeros(H,W,8).cuda()

        '''
        dict_key = torch.zeros(N_p, 32, dtype = torch.int32).cuda()+(H*W)
        dict_value = torch.zeros(N_p, 32, dtype = torch.int32).cuda()+N_p
        dict_key_sorted = torch.zeros(N_p, 32, dtype = torch.int32).cuda()+(H*W)
        dict_value_sorted = torch.zeros(N_p, 32, dtype = torch.int32).cuda()+N_p
        '''
        range_ = torch.zeros(H*W, 2, dtype = torch.int32).cuda()

        grid_h, grid_w = torch.meshgrid(torch.range(0,H-1), torch.range(0,W-1), indexing='ij')
        px = torch.stack([grid_w, grid_h],axis=-1).cuda() # H,W,2

        placeholders[H*W] = idx, zbuf, dist2, range_, px
    else:
        idx, zbuf, dist2, range_, px = placeholders[H*W]
        # reuse the place holder is fast
        idx.zero_()
        zbuf.zero_()
        dist2.zero_()
        '''
        dict_key.zero_()
        dict_value.zero_()
        dict_key_sorted.zero_()
        dict_value_sorted.zero_()
        '''
        range_.zero_()

        idx += -1
        '''
        dict_key += (H*W)
        dict_value += N_p
        dict_key_sorted += (H*W)
        dict_value_sorted += N_p
        '''

    if mask_fill is None:
        mask_fill = torch.ones(H,W,dtype=bool).cuda() 
    else:
        mask_fill = mask_fill.to(bool).cuda()
    idx, range_ = fastpr.forward(pc_sorted, K.view(-1), T, near,far,2.5,\
            idx,\
            range_)#,\
            #mask_fill,r)
    # unsort
    valid_mask = idx > -1
    reverse_sort_ids = torch.arange(0,N_p).long().cuda()[sort_ids]
    idx_ = idx.long()
    idx_[valid_mask] = reverse_sort_ids[idx_[valid_mask]]

    
    

    # extract full result
    pc_c[...,:2] /= pc_c[...,(2,)]
    pc_c[...,0] = pc_c[...,0] * K[0,0]+K[0,2] 
    pc_c[...,1] = pc_c[...,1] * K[1,1]+K[1,2]

    pcs = pc_c[idx_.view(-1),:].reshape(*(idx.shape),3) # H,W,8,3
    zbuf = pcs[...,2]


    dist2 = ((pcs[...,:2] - px.unsqueeze(2) ) **2).sum(-1)/(K[0,0]*K[1,1]) # H,W,8

    return zbuf[None], idx_[None], dist2[None]

