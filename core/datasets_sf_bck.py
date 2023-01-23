# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import sys
sys.path.append('./core/')
from os.path import exists
from utils import frame_utils
from utils.augmentor_sf import FlowAugmentor, SparseFlowAugmentor


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
                img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc = self.augmentor(img_l1, img_l2,img_r1, img_r2, flow, disparity, disparity_change, valid, valid_d, valid_dc)
            else:
                img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch = self.augmentor(img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change)

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1).float()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1).float()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1).float()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1).float()

        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        disparity = torch.from_numpy(disp).permute(2, 0, 1).float()
        disparity_change = torch.from_numpy(disp_ch).permute(2, 0, 1).float()

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

        #print(f'dataset {dataset}')
        if dataset=='Kitti':
            #print('dataset {dataset}')
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity_change[0,:,:]], dim = 0)
            #print(index)
            #print("Done SF")
            valid = torch.stack([valid.float(), valid_d.float(), valid_dc.float()], dim =0)
            #print("Done VAlid")
            #print("\n")
        else:
            scene_flow = torch.stack([flow[0,:,:], flow[1,:,:], disparity[0,:,:], disparity[0,:,:]+disparity_change[0,:,:]], dim=0)
            valid = torch.stack([valid, valid_d, valid_dc], dim =0)
        return img_l1, img_l2, img_r1, img_r2, scene_flow, valid
        #return img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change, valid.float(), valid_d.float(), valid_dc.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.l_image_list = v * self.l_image_list
        self.r_image_list = v * self.r_image_list
        return self
        
    def __len__(self):
        return len(self.l_image_list)


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw', dstype='frames_cleanpass'): # original root='datasets/FlyingThings3D' reaplced: root='/ds-av/public_datasets/freiburg_sceneflow_flyingthings3d/raw'
        super(FlyingThings3D, self).__init__(aug_params)
        print("In method")
        for cam in ('left', 'right'):
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                #print(f'image_dirs {len(image_dirs)}')
                
                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                #print(f'flow_dirs {len(flow_dirs)}')

                disp_dirs = sorted(glob(osp.join(root, 'disparity/TRAIN/*/*')))
                disp_dirs = sorted([osp.join(f, cam) for f in disp_dirs])
                #print(f'disp_dirs {len(disp_dirs)}')

                disp_ch_dirs = sorted(glob(osp.join(root, 'disparity_change/TRAIN/*/*')))
                disp_ch_dirs = sorted([osp.join(f, direction, cam) for f in disp_ch_dirs])
                #print(f'disp_ch_dirs {len(disp_ch_dirs)}')

                for idir, fdir, ddir, dch_dir in zip(image_dirs, flow_dirs, disp_dirs, disp_ch_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    disparities = sorted(glob(osp.join(ddir, '*.pfm')) )
                    disparity_change = sorted(glob(osp.join(dch_dir, '*.pfm')) )
                    #print(f'ddir {ddir}')
                    #print(f'ddir {dch_dir}')
                    #print(f'image_len {len(images)}')
                    #print(f'flows_len {len(flows)}')
                    #print(f'disparities len {len(disparities)}')
                    #print(f'disparity_change len {len(disparity_change)}')

                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i], images[i+1]] ]
                            else:
                                self.r_image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                            self.disp_list += [ disparities[i] ] 
                            self.disp_ch_list += [ disparity_change[i] ]
                            #print([ disparity_change[i] ])

                        elif direction == 'into_past':
                            if(cam == 'left'):
                                self.l_image_list += [ [images[i+1], images[i]] ]
                            else:
                                self.r_image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
                            self.disp_list += [ disparities[i+1] ] 
                            self.disp_ch_list += [ disparity_change[i+1] ]
        
      

'''class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/ds-av/public_datasets/kitti2015/raw'): #original - root='datasets/KITTI' reaplced: root='/ds-av/public_datasets/kitti_tracking/raw'
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True
            
        root = osp.join(root, split)
        images_l1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images_l2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        images_r1 = sorted(glob(osp.join(root, 'image_3/*_10.png')))
        images_r2 = sorted(glob(osp.join(root, 'image_3/*_11.png')))

        for img_l1, img_l2, img_r1, img_r2 in zip(images_l1, images_l2, images_r1, images_r2):
            frame_id = img_l1.split('/')[-1]
            #print(frame_id)
            self.extra_info += [ [frame_id] ]
            self.l_image_list += [ [img_l1, img_l2] ]
            self.r_image_list += [ [img_r1, img_r2] ]
            print(img_l1)

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
            self.disp_ch_list = sorted(glob(osp.join(root, 'disp_occ_1/*_10.png')))'''

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/ds-av/public_datasets/kitti2015/raw', val_split = 5, validation=False, train_im_file = 'kitti_sf.txt', override_split = True): #original - root='datasets/KITTI' reaplced: root='/ds-av/public_datasets/kitti_tracking/raw'
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
            images_l2 = [i for i in images_l2 if i[-9:-7] in train_files]
            images_r1 = [i for i in images_r1 if i[-9:-7] in train_files]
            images_r2 = [i for i in images_r2 if i[-9:-7] in train_files]
        

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

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

