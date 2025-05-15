import math
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from threading import Thread
import pygame
from .utils_SLARE_rast import *
from icecream import ic

# control
heading_bias = None#
heading = None
posi = None

facing_dire = np.array([[0,0,1.]]).T
delta_posi = .1

delta_dire_left = R.from_euler('y', -90, degrees=True).as_matrix()
delta_dire_right = R.from_euler('y', 90, degrees=True).as_matrix()
delta_dire_forward = np.eye(3)
delta_dire_back = -np.eye(3)
delta_dire_up = R.from_euler('x', 90, degrees=True).as_matrix()
delta_dire_down = R.from_euler('x', -90, degrees=True).as_matrix()




delta_degree = 2.
delta_dire_turnleft = R.from_euler('y', -delta_degree, degrees=True).as_matrix()
delta_dire_turnright = R.from_euler('y', delta_degree, degrees=True).as_matrix()
delta_dire_turnup = R.from_euler('x', delta_degree, degrees=True).as_matrix()
delta_dire_turndown = R.from_euler('x', -delta_degree, degrees=True).as_matrix()
delta_dire_turnrollleft = R.from_euler('z', -delta_degree, degrees=True).as_matrix()
delta_dire_turnrollright = R.from_euler('z', delta_degree, degrees=True).as_matrix()




delta_dire_collect = [delta_dire_up, delta_dire_forward, delta_dire_down, delta_dire_back, delta_dire_left, delta_dire_right, # for translation of view point 
                        delta_dire_turnleft, delta_dire_turnright, delta_dire_turnup, delta_dire_turndown, delta_dire_turnrollleft, delta_dire_turnrollright] # for rotation of view point
#key_collect = [pygame.K_o, pygame.K_UP, pygame.K_l, pygame.K_DOWN, 
#                    pygame.K_LEFT, pygame.K_RIGHT,pygame.K_w,pygame.K_s,pygame.K_a,pygame.K_d]
key_collect = [pygame.K_PAGEUP, pygame.K_UP, pygame.K_PAGEDOWN, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, 
                    pygame.K_a, pygame.K_d,pygame.K_w,pygame.K_s,pygame.K_q,pygame.K_e]




class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, debug=True, vis=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H

        self.intrinsics = [self.W//2,self.H//2,self.W//2,self.H//2]

        # old GUI's params
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        #if train_loader is not None:
        #    self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']
        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        if vis is True:
            self.vr_thread = Thread(target=self.vr)
            self.vr_thread.start()


    def train_step(self, jump=False):

        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()
        if jump:
            outputs = self.trainer.train_gui(self.train_loader, step=5, jump=jump)#, use_nslf = True if self.iters > 1000 else False)

        else:
            outputs = self.trainer.train_gui(self.train_loader, step=1)#self.train_steps)#, use_nslf = True if self.iters > 1000 else False)


        #ender.record()
        #torch.cuda.synchronize()
        '''
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True
        '''

        #dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        #dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        '''
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps
        '''

    def vr(self):
        #pygame.init()
        pygame.display.init()
        #pygame.font.init()
        #pygame.mixer.init()  # disable sound

        enlarge_s = 1

        clock = pygame.time.Clock()
        display = pygame.display.set_mode((self.W*enlarge_s,self.H*enlarge_s))
        change_pose = np.identity(4)
        heading_bias = change_pose[:3,:3]#.dot(np.array([[1.,0,0]]).T)
        heading = np.eye(3)
        posi = change_pose[:3,(3,)]

        view_context = 0 # 0:rgb; 1: semantic; 2: instance

        while True:
            clock.tick(6)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    print(pygame.key.name(event.key))
            keys = pygame.key.get_pressed()

            # use 2.1.2, dont 2.3 their key is wierd...
            if np.sum(keys) == 0:
                continue

            st = time.time()
            # 1. get controls
            for key, delta_dire in zip(key_collect[:6], delta_dire_collect[:6]):
                if keys[key]:
                    posi += delta_dire.dot(heading_bias.dot(heading).dot(facing_dire)) * delta_posi
            for key, delta_dire in zip(key_collect[6:], delta_dire_collect[6:]):
                if keys[key]:
                    heading = heading.dot(delta_dire)
            pose = np.eye(4)
            pose[:3,:3] = heading_bias.dot(heading)#R.align_vectors(heading.T, heading_init.T)[0].as_matrix()
            pose[:3,(3,)] = posi

            # check if change view context [rgb, semantic, instance]
            if keys[pygame.K_n]:
                view_context = (view_context+1)%3

            # ray rasterize
            outputs = self.trainer.test_gui(pose, self.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale, view_context=view_context)

            #enlarge
            image = cv2.resize(outputs['image'], (self.W*enlarge_s, self.H*enlarge_s)).transpose(1,0,2)

            # RGB to BGR
            #image = image[:,:,::-1]

            #image = outputs['image'].transpose(1,0,2)
            #image = cv2.Laplacian(image,cv2.CV_32F)
            surf = pygame.surfarray.make_surface(image)
            display.blit(surf, (0, 0))
            pygame.display.update()


