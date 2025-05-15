import cv2
import torch
import numpy as np
from pyquaternion import Quaternion
from time import sleep
from tqdm import tqdm
import pdb






from nps.rendering import RenderingThread
renderer = RenderingThread('cuda:0', workspace='workspace/demo', vis=True, remove_outlier=False)

def step(idx, rgb, depth, pose, K, depth_im_scale):
    renderer.add_frame
    renderer.id2pose[idx] = pose

def _parse_traj_file(traj_path):
    poses = []
    idxs = []
    traj_data = np.genfromtxt(traj_path)
    for p in traj_data:
        idx = int(p[0]*10)
        t = p[1:4]
        q = p[4:].tolist()
        q = Quaternion([q[-1], *(q[:-1])])
        T = np.eye(4)
        T[:3,:3] = q.rotation_matrix
        T[:3,3] = t
        poses.append(T)
        idxs.append(idx)
    return idxs, poses



if __name__ == '__main__':
    data_dir = 'data/robotics_hall/'
    rgb_path = data_dir+'/color/%06d.png'
    depth_path =  data_dir+'/depth/%06d.png'
    depth_im_scale = 1000 

    intrinsic = np.genfromtxt(data_dir+'/intrinsic.txt')
    fx, fy, cx, cy = intrinsic
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    idxs, poses = _parse_traj_file(data_dir+'/orbslam2_traj.txt')

    # incrementally run
    N = len(idxs)
    for i in tqdm(range(N)):
        idx = idxs[i]
        pose = poses[i]
        rgb = cv2.cvtColor(cv2.imread(rgb_path%idx,-1),cv2.COLOR_BGR2RGB) 
        depth = cv2.imread(depth_path%idx,-1)
        
        step(idx,rgb,depth,pose,K,depth_im_scale)
                
        sleep(0.1)
        

