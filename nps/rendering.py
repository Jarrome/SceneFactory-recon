import torch

import numpy as np

from queue import Queue
from threading import Thread, Lock, Condition
import time

from itertools import chain
from collections import defaultdict

from .ext3d import unproject_depth
from .provider import NeRFDataset
from .gui_vr import NeRFGUI # based on pygame
from .parser import ngp_parser


torch.set_printoptions(precision=8)


from .utils_SLARE_rast import *
from .SLARE_transformable_v2 import SLARE # multiple reso + transformable with dir feature



import open3d as o3d
from icecream import ic
from rich import print



class RenderingThread(object):
    def __init__(self, device, reso=0.005, workspace=None, vis=True, remove_outlier=False):
        opt = ngp_parser()
        self.device = device
        # model 
        self.model = SLARE(device)
        # fake dataset collect frames
        self.fake_dataset = NeRFDataset(opt, device=device, type='train')
        # trainer train the rendering model
        #criterion = torch.nn.MSELoss(reduction='none')
        criterion = torch.nn.L1Loss(reduction='none')
        #optimizer = lambda model, lr, with_mlp: torch.optim.Adam(model.get_params(lr, with_mlp), betas=(0.9, 0.99), eps=1e-15)
        optimizer = lambda model, lr, with_mlp: torch.optim.AdamW(model.get_params(lr, with_mlp),betas=(0.9, 0.99), eps=1e-15)



        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        metrics = [PSNRMeter()]#, LPIPSMeter(device=device)]
        workspace = opt.workspace if workspace is None else workspace
        self.trainer = Trainer('ngp', opt, self.model, device=device, workspace=workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        self.opt = opt
        self.vis = vis
        #self.gui = NeRFGUI(opt, self.trainer, None)

        self.id2pose = {}


        self._queue = Queue()
        self._requests_cv = Condition()
        self.maintenance_thread = Thread(target=self.maintenance)
        self.maintenance_thread.daemon = True # make sure main thread can exit
        self.maintenance_thread.start()

        self.stopped = False

        self.remove_outlier = remove_outlier # remove depth's outlier

    def add_frame(self, rgb, depth, pose, intrin, depth_scale): 
        # 0. filtering the depth that is too close to the render
        H,W = rgb.shape[:2]
        outputs = self.trainer.test_gui(pose,
                                                [intrin[0,0], intrin[1,1],intrin[0,2],intrin[1,2]],
                                                W, H, return_belongings=True)

        depth_rendered = outputs['depth']
        diff = np.absolute(depth - depth_rendered)
        depth[diff < .2] = 0



        # 2. feed
        self._queue.put((0, (rgb, depth, pose, intrin, depth_scale)))

        #cv2.imwrite('debug/%d.png'%(self.fake_dataset.data_len), (depth*5000).astype(np.uint16))

        with self._requests_cv:
            self._requests_cv.notify()

    def transform(self, frame_ids, Ts): 
        #if self.started:
        self._queue.put((1,(frame_ids,Ts)))
        with self._requests_cv:
            self._requests_cv.notify()




        '''
        else:
            self.fake_dataset.add_keyframe(keyframe)
            self.gui.train_loader = self.fake_dataset.dataloader()
            if self.gui.dpg.is_dearpygui_running():
                self.gui.render_once()
            self.maintenance_thread.start()
        '''
    
    def maintenance(self):
        stopped = False
        jump_start = True

        # wait for first datainput signal                                                            
        if self._queue.empty():
            with self._requests_cv:
                self._requests_cv.wait()    
               
        self.gui = NeRFGUI(self.opt, self.trainer, self.fake_dataset, vis=self.vis) # ATTENTION: GUI and dpg must be in same thread !!!


        #while dpg.is_dearpygui_running() and not stopped:
        while not self.stopped:
            #torch.cuda.synchronize()
            #block_st = time.time()
            while not self._queue.empty():
                signal, data = self._queue.get()
                #if type(data[0]) != list:

                if self.stopped:# == -1:
                    # need jump out of the loop because queue might have more items
                    break

                if signal == 0:
                    st = time.time()
                    # add a keyframe and update to new loader in gui 
                    with self.fake_dataset.add_lock:
                        self.fake_dataset.add_frame(data, remove_outlier=self.remove_outlier)
                    #self.gui.train_loader = self.fake_dataset.dataloader()
                    print('[bold purple]Renderer. Add data take:', time.time()-st)
                elif signal == 1:
                    st = time.time()
                    # transform nps given dTs
                    frame_ids, Ts = data
                    # because all data are from dataset, to avoid new pose wrong, we transform in dataset
                    dTs = self.fake_dataset.transform_to(frame_ids, Ts)
                    # the dataset will help provide dTs to transform the model
                    self.model.transform(frame_ids, dTs)
                    print('[bold purple]Renderer. Transform take:',time.time()-st)

                #if not self._queue.empty():
                #    self.gui.train_step(jump=jump_start)
                print('[bold purple]Queue size:',self._queue.qsize())

            #self.gui.render_once()
            # set high learning rate to jump start

            #print('[bold purple]train_step')
            self.gui.train_step(jump=jump_start)
            jump_start=False



    def stop(self):
        self.stopped = True
        '''
        while not self._queue.empty():
            self._queue.pop()
            time.sleep(1e-4)
        self._queue.put((-1,None))
        '''
        self.maintenance_thread.join()
        print('mapping stopped')

