# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import random
from pathlib import Path 
import os
import math
import random
from glob import glob
import os.path as osp
import sys
from os.path import exists
sys.path.append('./core/')

from utils import frame_utils
from utils.augmentor_sf import FlowAugmentor_double as FlowAugmentor
from utils.augmentor_sf import SparseFlowAugmentor

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disp_list = []
        self.disp_ch_list = []
        self.l_image_list = []
        self.r_image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img_l1 = frame_utils.read_gen(self.l_image_list[index][0])
            img_l2 = frame_utils.read_gen(self.l_image_list[index][1])
            img_r1 = frame_utils.read_gen(self.r_image_list[index][0])
            img_r2 = frame_utils.read_gen(self.r_image_list[index][1])

            img_l1 = np.array(img_l1).astype(np.uint8)[..., :3]
            img_l2 = np.array(img_l2).astype(np.uint8)[..., :3]
            img_r1 = np.array(img_r1).astype(np.uint8)[..., :3]
            img_r2 = np.array(img_r2).astype(np.uint8)[..., :3]

            img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1).float()
            img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1).float()
            img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1).float()
            img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1).float()
            return img_l1, img_l2, img_r1, img_r2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.l_image_list)
        valid = None
        valid_d = None
        valid_dc = None
        #disp = None
        #disp_ch = None
        dataset=''
        if self.sparse:
            dataset='Kitti'
            flow,  valid = frame_utils.readFlowKITTI(self.flow_list[index])
            disparity, valid_d = frame_utils.readDispKITTI(self.disp_list[index])
            disparity_change, valid_dc = frame_utils.readDispKITTI(self.disp_ch_list[index])

        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            disparity = frame_utils.read_gen(self.disp_list[index])
            disparity_change = frame_utils.read_gen(self.disp_ch_list[index])

        img_l1 = frame_utils.read_gen(self.l_image_list[index][0])
        img_l2 = frame_utils.read_gen(self.l_image_list[index][1])
        img_r1 = frame_utils.read_gen(self.r_image_list[index][0])
        img_r2 = frame_utils.read_gen(self.r_image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        disparity = np.array(disparity).astype(np.float32)
        disparity_change = np.array(disparity_change).astype(np.float32)
        img_l1 = np.array(img_l1).astype(np.uint8)
        img_l2 = np.array(img_l2).astype(np.uint8)
        img_r1 = np.array(img_r1).astype(np.uint8)
        img_r2 = np.array(img_r2).astype(np.uint8)

        # grayscale images
        if(len(img_l1.shape) == 2 and len(img_r1.shape) == 2):
            img_l1 = np.tile(img_l1[...,None], (1, 1, 3))
            img_l2 = np.tile(img_l2[...,None], (1, 1, 3))
            img_r1 = np.tile(img_r1[...,None], (1, 1, 3))
            img_r2 = np.tile(img_r2[...,None], (1, 1, 3))
        else:
            img_l1 = img_l1[..., :3]
            img_l2 = img_l2[..., :3]
            img_r1 = img_r1[..., :3]
            img_r2 = img_r2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img_l1, img_l2, img_r1, img_r2, flow, valid, disparity, valid_d, disparity_change, valid_dc = self.augmentor(img_l1, img_l2,img_r1, img_r2, flow, disparity, disparity_change, valid, valid_d, valid_dc)
            else:
                #print("Augmentor Block")
                img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change = self.augmentor(img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change)
        else:
            if(len(disparity.shape) < 3):
                disparity = np.reshape(disparity, (disparity.shape[0], disparity.shape[1], -1))
            if(len(disparity_change.shape) < 3):
                disparity_change = np.reshape(disparity_change, (disparity_change.shape[0], disparity_change.shape[1], -1))
            #disparity = disparity * [1.0, 1.0]
            #disparity_change = disparity_change * [1.0, 1.0]

        
        

        #print('disparity vals')
        #print(disparity[:, :, 0])

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1).float()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1).float()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1).float()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1).float()

        #print(f'flow.shape {flow.shape}')
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        #print(f'flow.shape {flow.shape}')
        disparity = torch.from_numpy(disparity).permute(2, 0, 1).float()
        disparity_change = torch.from_numpy(disparity_change).permute(2, 0, 1).float()
        #print(f'disparity.shape {disparity.shape}')

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if valid_d is not None:
            valid_d = torch.from_numpy(valid_d)
        else:
            valid_d = (disparity[0].abs() < 1000)

        if valid_dc is not None:
            valid_dc = torch.from_numpy(valid_dc)
        else:
            valid_dc = (disparity_change[0].abs() < 1000)
        
        #print(f'disp shape {disparity.shape}')
        #print(f'disp shape {disparity_change.shape}')
        #print('\n')
        caught = 0
        missed = 0
        if dataset=='Kitti':
            #print(index)
            #print(f'flow.shape, {flow.shape}, disparity_shape {disparity.shape}, disparity change shape {disparity_change.shape}')
            
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity_change[0,:,:]], dim = 0)
            #scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity[1,:,:], disparity_change[0,:,:], disparity_change[1,:,:]], dim = 0)

            #print(index)
            #print("Done SF")
            #print(f'valid.shape, {valid.shape}, valid_d shape {valid_d.shape}, valid_dc shape {valid_dc.shape}')
            #print('\n')
            valid = torch.stack([valid.float(), valid_d.float(), valid_dc.float()], dim =0)
            #print("Done VAlid")
            #print("\n")
        else:
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity[0,:,:]+disparity_change[0,:,:]], dim=0)
            #scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity[1,:,:], disparity[0,:,:]+disparity_change[0,:,:], disparity[1,:,:]+disparity_change[1,:,:]], dim=0)
            
            valid = torch.stack([valid, valid_d, valid_dc], dim =0)
            

        return img_l1, img_l2, img_r1, img_r2, scene_flow, valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.l_image_list = v * self.l_image_list
        self.r_image_list = v * self.r_image_list
        return self
        
    def __len__(self):
        return len(self.l_image_list)


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw', dstype='frames_cleanpass', val_split = 5, validation=False, train_im_file = 'FT3d_sf.txt'): # original root='datasets/FlyingThings3D' reaplced: root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw'
        super(FlyingThings3D, self).__init__(aug_params)
        image_dirs = sorted(glob(osp.join(root, dstype, f'TRAIN/*/*')))
        
        if(exists(train_im_file)):
            pass
        else:
            f = open(train_im_file, 'wb')
            f.close()
        
        if(os.stat(train_im_file).st_size == 0):
            file_names = set([f[-4:len(f)] for f in image_dirs])
            image_dirs_val = random.sample(image_dirs, int(len(image_dirs)*(val_split/300)))
            file_names_val = set([f[-4:len(f)] for f in image_dirs_val])
        
            file_names = set(file_names)
            file_names_val = set(file_names_val)
            file_names_train = file_names - file_names_val
            file_names_train = np.array(list(file_names_train))
            with open(train_im_file, 'wb') as wf:
                np.save(wf, file_names_train)

        with open(train_im_file, 'rb') as rf:
            train_files = np.load(rf)
        
        for cam in ('left', 'right'):
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'TRAIN/*/*')))
                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/TRAIN/*/*')))
                disp_dirs = sorted(glob(osp.join(root, f'disparity/TRAIN/*/*')))
                disp_ch_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))

                if(validation):
                    image_dirs = [i for i in image_dirs if i[-4:len(i)] not in train_files]
                    flow_dirs = [f for f in flow_dirs if f[-4:len(f)] not in train_files]
                    disp_dirs = [d for d in disp_dirs if d[-4:len(d)] not in train_files]
                    disp_ch_dirs = [dc for dc in disp_ch_dirs if dc[-4:len(dc)] not in train_files]
                else:
                    image_dirs = [i for i in image_dirs if i[-4:len(i)] in train_files]
                    flow_dirs = [f for f in flow_dirs if f[-4:len(f)] in train_files]
                    disp_dirs = [d for d in disp_dirs if d[-4:len(d)] in train_files]
                    disp_ch_dirs = [dc for dc in disp_ch_dirs if dc[-4:len(dc)] in train_files]
                
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                disp_dirs = sorted([osp.join(f, cam) for f in disp_dirs])
                disp_ch_dirs = sorted([osp.join(f, direction, cam) for f in disp_ch_dirs])
                    
                for idir, fdir, ddir, dch_dir in zip(image_dirs, flow_dirs, disp_dirs, disp_ch_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    disparities = sorted(glob(osp.join(ddir, '*.pfm')) )
                    disparity_change = sorted(glob(osp.join(dch_dir, '*.pfm')) )

                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i], images[i+1]] ]
                            else:
                                self.r_image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.disp_list += [ disparities[i] ] 
                            self.disp_ch_list += [ disparity_change[i] ]

                        elif direction == 'into_past':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i+1], images[i]] ]
                            else:
                                self.r_image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.disp_list += [ disparities[i+1] ] 
                            self.disp_ch_list += [ disparity_change[i+1] ]


class KITTI(FlowDataset):
        def __init__(self, aug_params=None, split='training', root='/ds-av/public_datasets/kitti2015/raw', val_split = 10, validation=False, train_im_file = 'Kitti_sf.txt', override_split = True):
            super(KITTI, self).__init__(aug_params, sparse=True)
            if split == 'testing':
                self.is_test = True

            root = osp.join(root, split)
            print(f'root path {root}')
            images_ref = sorted(glob(osp.join(root, 'image_2/*_10.png')))

            if(exists(train_im_file)):
                pass
            else:
                f = open(train_im_file, 'wb')
                f.close()

            if(os.stat(train_im_file).st_size == 0 or override_split):
                file_names = set([f[-9:-7] for f in images_ref])    
                file_names_val = set(random.sample(file_names, int(len(file_names)*(val_split/100))))
    
                file_names_train = file_names - file_names_val
                file_names_train = np.array(list(file_names_train))
                with open(train_im_file, 'wb') as wf:
                    np.save(wf, file_names_train)

            with open(train_im_file, 'rb') as rf:
                train_files = np.load(rf)
            #print(train_files)
            images_l1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images_l2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            images_r1 = sorted(glob(osp.join(root, 'image_3/*_10.png')))
            images_r2 = sorted(glob(osp.join(root, 'image_3/*_11.png')))
            if(split=='training'):
                if(validation):
                    images_l1 = [i for i in images_l1 if i[-9:-7] not in train_files]
                    images_l2 = [i for i in images_l2 if i[-9:-7] not in train_files]
                    images_r1 = [i for i in images_r1 if i[-9:-7] not in train_files]
                    images_r2 = [i for i in images_r2 if i[-9:-7] not in train_files]
                
                else:
                    images_l1 = [i for i in images_l1 if i[-9:-7] in train_files]
                    #print(f'len(images_l1) {len(images_l1)}')
                    images_l2 = [i for i in images_l2 if i[-9:-7] in train_files]
                    #print(f'len(images_l2) {len(images_l2)}')
                    images_r1 = [i for i in images_r1 if i[-9:-7] in train_files]
                    #print(f'len(images_r1) {len(images_r1)}')
                    images_r2 = [i for i in images_r2 if i[-9:-7] in train_files]
                    #print(f'len(images_r2) {len(images_r2)}')
                    #print('\n')

            for img_l1, img_l2, img_r1, img_r2 in zip(images_l1, images_l2, images_r1, images_r2):
                frame_id = img_l1.split('/')[-1]
                #print(frame_id)
                self.extra_info += [ [frame_id] ]
                self.l_image_list += [ [img_l1, img_l2] ]
                self.r_image_list += [ [img_r1, img_r2] ]

            if split == 'training':
                self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
                self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
                self.disp_ch_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

                if(validation):
                    self.flow_list = [i for i in self.flow_list if i[-9:-7] not in train_files]
                    self.disp_list  = [i for i in  self.disp_list  if i[-9:-7] not in train_files]
                    self.disp_ch_list = [i for i in self.disp_ch_list if i[-9:-7] not in train_files]
                else:
                    self.flow_list = [i for i in self.flow_list if i[-9:-7] in train_files]
                    self.disp_list = [i for i in self.disp_list if i[-9:-7] in train_files]
                    self.disp_ch_list = [i for i in self.disp_ch_list if i[-9:-7] in train_files]
            
            elif split == 'testing':
                self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
                self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
                self.disp_ch_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))



class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)
    #train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        #pin_memory=False, shuffle=True, drop_last=True)
    #train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


'''class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disp_list = []
        self.disp_ch_list = []
        self.l_image_list = []
        self.r_image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img_l1 = frame_utils.read_gen(self.l_image_list[index][0])
            img_l2 = frame_utils.read_gen(self.l_image_list[index][1])
            img_r1 = frame_utils.read_gen(self.r_image_list[index][0])
            img_r2 = frame_utils.read_gen(self.r_image_list[index][1])

            img_l1 = np.array(img_l1).astype(np.uint8)[..., :3]
            img_l2 = np.array(img_l2).astype(np.uint8)[..., :3]
            img_r1 = np.array(img_r1).astype(np.uint8)[..., :3]
            img_r2 = np.array(img_r2).astype(np.uint8)[..., :3]

            img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1).float()
            img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1).float()
            img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1).float()
            img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1).float()
            return img_l1, img_l2, img_r1, img_r2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.l_image_list)
        valid = None
        valid_d = None
        valid_dc = None
        #disp = None
        #disp_ch = None
        dataset=''
        if self.sparse:
            dataset='Kitti'
            flow,  valid = frame_utils.readFlowKITTI(self.flow_list[index])
            disparity, valid_d = frame_utils.readDispKITTI(self.disp_list[index])
            disparity_change, valid_dc = frame_utils.readDispKITTI(self.disp_ch_list[index])

        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            disparity = frame_utils.read_gen(self.disp_list[index])
            disparity_change = frame_utils.read_gen(self.disp_ch_list[index])

        img_l1 = frame_utils.read_gen(self.l_image_list[index][0])
        img_l2 = frame_utils.read_gen(self.l_image_list[index][1])
        img_r1 = frame_utils.read_gen(self.r_image_list[index][0])
        img_r2 = frame_utils.read_gen(self.r_image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        disparity = np.array(disparity).astype(np.float32)
        disparity_change = np.array(disparity_change).astype(np.float32)
        img_l1 = np.array(img_l1).astype(np.uint8)
        img_l2 = np.array(img_l2).astype(np.uint8)
        img_r1 = np.array(img_r1).astype(np.uint8)
        img_r2 = np.array(img_r2).astype(np.uint8)

        # grayscale images
        if(len(img_l1.shape) == 2 and len(img_r1.shape) == 2):
            img_l1 = np.tile(img_l1[...,None], (1, 1, 3))
            img_l2 = np.tile(img_l2[...,None], (1, 1, 3))
            img_r1 = np.tile(img_r1[...,None], (1, 1, 3))
            img_r2 = np.tile(img_r2[...,None], (1, 1, 3))
        else:
            img_l1 = img_l1[..., :3]
            img_l2 = img_l2[..., :3]
            img_r1 = img_r1[..., :3]
            img_r2 = img_r2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img_l1, img_l2, img_r1, img_r2, flow, valid, disparity, valid_d, disparity_change, valid_dc = self.augmentor(img_l1, img_l2,img_r1, img_r2, flow, disparity, disparity_change, valid, valid_d, valid_dc)
            else:
                #print("Augmentor Block")
                img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change = self.augmentor(img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change)

        if(len(disparity.shape) < 3):
            disparity = np.reshape(disparity, (disparity.shape[0], disparity.shape[1], 1))
        if(len(disparity_change.shape) < 3):
            disparity_change = np.reshape(disparity_change, (disparity_change.shape[0], disparity_change.shape[1], 1))

        #print('disparity vals')
        #print(disparity[:, :, 0])

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1).float()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1).float()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1).float()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1).float()

        #print(f'flow.shape {flow.shape}')
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        #print(f'flow.shape {flow.shape}')
        disparity = torch.from_numpy(disparity).permute(2, 0, 1).float()
        disparity_change = torch.from_numpy(disparity_change).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if valid_d is not None:
            valid_d = torch.from_numpy(valid_d)
        else:
            valid_d = (disparity[0].abs() < 1000) & (disparity[1].abs() < 1000)

        if valid_dc is not None:
            valid_dc = torch.from_numpy(valid_dc)
        else:
            valid_dc = (disparity_change[0].abs() < 1000) & (disparity_change[1].abs() < 1000)
        
        if dataset=='Kitti':
            #print(index)
            #print(f'flow.shape, {flow.shape}, disparity_shape {disparity.shape}, disparity change shape {disparity_change.shape}')
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity_change[0,:,:]], dim = 0)
            #print(index)
            #print("Done SF")
            #print(f'valid.shape, {valid.shape}, valid_d shape {valid_d.shape}, valid_dc shape {valid_dc.shape}')
            #print('\n')
            valid = torch.stack([valid.float(), valid_d.float(), valid_dc.float()], dim =0)
            #print("Done VAlid")
            #print("\n")
        else:
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity[0,:,:]+disparity_change[0,:,:]], dim=0)
            valid = torch.stack([valid, valid_d, valid_dc], dim =0)

        return img_l1, img_l2, img_r1, img_r2, scene_flow, valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.l_image_list = v * self.l_image_list
        self.r_image_list = v * self.r_image_list
        return self
        
    def __len__(self):
        return len(self.l_image_list)


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw', dstype='frames_cleanpass', val_split = 5, validation=False, train_im_file = 'FT3d_sf.txt', override_split = True): # original root='datasets/FlyingThings3D' reaplced: root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw'
        super(FlyingThings3D, self).__init__(aug_params)
        image_dirs = sorted(glob(osp.join(root, dstype, f'TRAIN/*/*')))
        
        if(exists(train_im_file)):
            pass
        else:
            f = open(train_im_file, 'wb')
            f.close()
        
        if(os.stat(train_im_file).st_size == 0 or override_split):
            file_names = set([f[-4:len(f)] for f in image_dirs])
            image_dirs_val = random.sample(image_dirs, int(len(image_dirs)*(val_split/300)))
            file_names_val = set([f[-4:len(f)] for f in image_dirs_val])
        
            file_names = set(file_names)
            file_names_val = set(file_names_val)
            file_names_train = file_names - file_names_val
            file_names_train = np.array(list(file_names_train))
            with open(train_im_file, 'wb') as wf:
                np.save(wf, file_names_train)

        with open(train_im_file, 'rb') as rf:
            train_files = np.load(rf)
        
        for cam in ('left', 'right'):
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'TRAIN/*/*')))
                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/TRAIN/*/*')))
                disp_dirs = sorted(glob(osp.join(root, f'disparity/TRAIN/*/*')))
                disp_ch_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))

                if(validation):
                    image_dirs = [i for i in image_dirs if i[-4:len(i)] not in train_files]
                    flow_dirs = [f for f in flow_dirs if f[-4:len(f)] not in train_files]
                    disp_dirs = [d for d in disp_dirs if d[-4:len(d)] not in train_files]
                    disp_ch_dirs = [dc for dc in disp_ch_dirs if dc[-4:len(dc)] not in train_files]
                else:
                    image_dirs = [i for i in image_dirs if i[-4:len(i)] in train_files]
                    flow_dirs = [f for f in flow_dirs if f[-4:len(f)] in train_files]
                    disp_dirs = [d for d in disp_dirs if d[-4:len(d)] in train_files]
                    disp_ch_dirs = [dc for dc in disp_ch_dirs if dc[-4:len(dc)] in train_files]
                
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                disp_dirs = sorted([osp.join(f, cam) for f in disp_dirs])
                disp_ch_dirs = sorted([osp.join(f, direction, cam) for f in disp_ch_dirs])
                    
                for idir, fdir, ddir, dch_dir in zip(image_dirs, flow_dirs, disp_dirs, disp_ch_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    disparities = sorted(glob(osp.join(ddir, '*.pfm')) )
                    disparity_change = sorted(glob(osp.join(dch_dir, '*.pfm')) )

                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i], images[i+1]] ]
                            else:
                                self.r_image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.disp_list += [ disparities[i] ] 
                            self.disp_ch_list += [ disparity_change[i] ]

                        elif direction == 'into_past':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i+1], images[i]] ]
                            else:
                                self.r_image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.disp_list += [ disparities[i+1] ] 
                            self.disp_ch_list += [ disparity_change[i+1] ]


class KITTI(FlowDataset):
        def __init__(self, aug_params=None, split='training', root='/ds-av/public_datasets/kitti2015/raw', val_split = 10, validation=False, train_im_file = 'Kitti_sf.txt', override_split = True):
            super(KITTI, self).__init__(aug_params, sparse=True)
            if split == 'testing':
                self.is_test = True

            root = osp.join(root, split)
            images_ref = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        
            if(exists(train_im_file)):
                pass
            else:
                f = open(train_im_file, 'wb')
                f.close()

            if(os.stat(train_im_file).st_size == 0 or override_split):
                file_names = set([f[-9:-7] for f in images_ref])    
                file_names_val = set(random.sample(file_names, int(len(file_names)*(val_split/100))))
    
                file_names_train = file_names - file_names_val
                file_names_train = np.array(list(file_names_train))
                with open(train_im_file, 'wb') as wf:
                    np.save(wf, file_names_train)

            with open(train_im_file, 'rb') as rf:
                train_files = np.load(rf)
            #print(train_files)
            images_l1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
            images_l2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
            images_r1 = sorted(glob(osp.join(root, 'image_3/*_10.png')))
            images_r2 = sorted(glob(osp.join(root, 'image_3/*_11.png')))
            if(validation):
                images_l1 = [i for i in images_l1 if i[-9:-7] not in train_files]
                
                images_l2 = [i for i in images_l2 if i[-9:-7] not in train_files]
                images_r1 = [i for i in images_r1 if i[-9:-7] not in train_files]
                images_r2 = [i for i in images_r2 if i[-9:-7] not in train_files]
                
            else:
                images_l1 = [i for i in images_l1 if i[-9:-7] in train_files]
                #print(f'len(images_l1) {len(images_l1)}')
                images_l2 = [i for i in images_l2 if i[-9:-7] in train_files]
                #print(f'len(images_l2) {len(images_l2)}')
                images_r1 = [i for i in images_r1 if i[-9:-7] in train_files]
                #print(f'len(images_r1) {len(images_r1)}')
                images_r2 = [i for i in images_r2 if i[-9:-7] in train_files]
                #print(f'len(images_r2) {len(images_r2)}')
                #print('\n')

            for img_l1, img_l2, img_r1, img_r2 in zip(images_l1, images_l2, images_r1, images_r2):
                frame_id = img_l1.split('/')[-1]
                #print(frame_id)
                self.extra_info += [ [frame_id] ]
                self.l_image_list += [ [img_l1, img_l2] ]
                self.r_image_list += [ [img_r1, img_r2] ]

            if split == 'training':
                self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
                self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
                self.disp_ch_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))

                if(validation):
                    self.flow_list = [i for i in self.flow_list if i[-9:-7] not in train_files]
                    self.disp_list  = [i for i in  self.disp_list  if i[-9:-7] not in train_files]
                    self.disp_ch_list = [i for i in self.disp_ch_list if i[-9:-7] not in train_files]
                else:
                    self.flow_list = [i for i in self.flow_list if i[-9:-7] in train_files]
                    self.disp_list = [i for i in self.disp_list if i[-9:-7] in train_files]
                    self.disp_ch_list = [i for i in self.disp_ch_list if i[-9:-7] in train_files]


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    #train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader'''

