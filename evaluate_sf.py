from re import S
import sys
from typing import overload
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets_sf as datasets
from utils import flow_viz
from utils import frame_utils

from raft_sf import RAFT
from utils.utils import InputPadder, forward_interpolate
import torch.utils.data as data
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from skimage.transform import resize

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

@torch.no_grad()
def create_kitti_submission_sf(model, iters=24, output_path='kitti_submission_sf'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    print(f'total test files {len(test_dataset)}')
    cwd = os.getcwd()
    output_path = os.path.join(cwd, output_path)
    flow_path = os.path.join(output_path, 'flow')
    disp_path = os.path.join(output_path, 'disp_0')
    future_disp_path = os.path.join(output_path, 'disp_1')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if not os.path.exists(flow_path):
            os.makedirs(flow_path)
        if not os.path.exists(disp_path):
            os.makedirs(disp_path)
        if not os.path.exists(future_disp_path):
            os.makedirs(future_disp_path)

    for test_id in range(len(test_dataset)):
        img_l1, img_l2, img_r1, img_r2, (frame_id, ) = test_dataset[test_id]
    
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        
        #img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        #img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        #img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        #img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))

        img_l1 = img_l1.transpose(1, 2, 0)
        img_l2 = img_l2.transpose(1, 2, 0)
        img_r1 = img_r1.transpose(1, 2, 0)
        img_r2 = img_r2.transpose(1, 2, 0)
        
        ht, wt = img_l1.shape[0], img_l1.shape[1]
                
        ht_8 = generate_dim_d8(img_l1.shape[0])
        wt_8 = generate_dim_d8(img_l1.shape[1])

        scale_x = wt / wt_8
        scale_y = ht / ht_8
        #print(f'img_l1.shape {img_l1.shape}')
        #print(f'ht, wt {(ht,wt)}')
        img_l1 = resize(img_l1, (ht_8, wt_8))
        img_r1 = resize(img_r1, (ht_8, wt_8))
        img_l2 = resize(img_l2, (ht_8, wt_8))
        img_r2 = resize(img_r2, (ht_8, wt_8))

        #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        #val = (valid_gt[0][0].view(-1) >= 0.5) & (valid_gt[0][1].view(-1) >= 0.5) & (valid_gt[0][2].view(-1) >= 0.5)
        #in_val = (valid_gt[0][0].view(-1) < 0.5) & (valid_gt[0][1].view(-1) < 0.5) & (valid_gt[0][2].view(-1) < 0.5)
        #scene_flow = padder.unpad(flow_pr[0]).cpu()
        scene_flow = flow_pr
        scene_flow = flow_pr[0].cpu()
                #print(f'scene_flow.shape {scene_flow.shape}')
        #scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
        scene_flow = scene_flow.permute(1, 2, 0)
                #print(f'scene_flow.shape {scene_flow.shape}')
        scene_flow = resize(scene_flow.detach().numpy(), (ht, wt))
        #scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))
        scene_flow = scene_flow.transpose(2, 0, 1)

        scene_flow[0, :, :] = scene_flow[0, :, :] * scale_x
        scene_flow[1, :, :] = scene_flow[1, :, :] * scale_y
        scene_flow[2, :, :] = scene_flow[2, :, :] * scale_x
        scene_flow[3, :, :] = scene_flow[3, :, :] * scale_x

        #print(f'scene_flow.dtype {scene_flow.dtype}')
        #print(f'scene_flow.shape {scene_flow.shape}')
        #print('\n')

        scene_flow = scene_flow.transpose(1,2,0)
        #output_filename = os.path.join(output_path, frame_id)
        output_filename_of = os.path.join(flow_path, frame_id)
        output_filename_d = os.path.join(disp_path, frame_id)
        output_filename_fd = os.path.join(future_disp_path, frame_id)

        frame_utils.writeFlowKITTI(output_filename_of, scene_flow[:,:, 0:2])
        frame_utils.writeDispKITTI(output_filename_d, scene_flow[:,:, 2])
        frame_utils.writeDispKITTI(output_filename_fd, scene_flow[:,:, 3])

def generate_dim_d8(number):
    div = number - ((number // 8) * 8)

    if(div == 0):
        return number
    return number + 8 - div


@torch.no_grad()
def validate_things_seperate(args, model, iters=12):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size)
        epe_list_of = []
        epe_list_d = []
        epe_list_fd = []
        epe_list = []
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            #print(f'batch number {i_batch}')
            #print(f'scene_flow_gt_b.shape {scene_flow_gt_b.shape}')
            for val_id in range(img_l1_b.shape[0]):
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                #print(f'scene_flow_gt.shape, {scene_flow_gt.shape}')
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                img_l1 = img_l1.cpu().detach().numpy().astype('float32')
                img_l2 = img_l2.cpu().detach().numpy().astype('float32')
                img_r1 = img_r1.cpu().detach().numpy().astype('float32')
                img_r2 = img_r2.cpu().detach().numpy().astype('float32')
                #scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
                #scene_flow_gt = np.reshape(scene_flow_gt, (1, scene_flow_gt.shape[0], scene_flow_gt.shape[1], scene_flow_gt.shape[2]))
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                # Change accordingly to reverse to single channel disp and disp change

                #scene_flow_gt = np.stack([scene_flow_gt[0,:,:], scene_flow_gt[1,:,:], scene_flow_gt[2,:,:], scene_flow_gt[4,:,:]], axis=0)
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')

                #img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
                #img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
                #img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
                #img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))

                img_l1 = img_l1.transpose(1, 2, 0)
                img_l2 = img_l2.transpose(1, 2, 0)
                img_r1 = img_r1.transpose(1, 2, 0)
                img_r2 = img_r2.transpose(1, 2, 0)
                #scene_flow_gt = np.reshape(scene_flow_gt, (-1, scene_flow_gt.shape[0], scene_flow_gt.shape[1], scene_flow_gt.shape[2]))
                
                #img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
                #img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
                #img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
                #img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
                
                ht, wt = img_l1.shape[0], img_l1.shape[1]
                
                ht_8 = generate_dim_d8(img_l1.shape[0])
                wt_8 = generate_dim_d8(img_l1.shape[1])

                scale_x = wt / wt_8
                scale_y = ht / ht_8
                #print(f'img_l1.shape {img_l1.shape}')
                #print(f'ht, wt {(ht,wt)}')
                img_l1 = resize(img_l1, (ht_8, wt_8))
                img_r1 = resize(img_r1, (ht_8, wt_8))
                img_l2 = resize(img_l2, (ht_8, wt_8))
                img_r2 = resize(img_r2, (ht_8, wt_8))
                
                #scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))

                #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
                #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
                #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
                #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()

                img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
                img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
                img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
                img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()
                #scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
                #print(f' unpadded scene_flow_pr.shape {scene_flow_pr.shape}')
                scene_flow = scene_flow_pr[0].cpu()
                #print(f'scene_flow.shape {scene_flow.shape}')
                #scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
                scene_flow = scene_flow.permute(1, 2, 0)
                #
                #print(f'type(scene_flow) {type(scene_flow)}')
                scene_flow = resize(scene_flow, (ht, wt))
                scene_flow = scene_flow.transpose(2, 0, 1)
                #scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))
                #print(f'scene_flow.shape {scene_flow.shape}')
                scene_flow[0, :, :] = scene_flow[0, :, :] * scale_x
                scene_flow[1, :, :] = scene_flow[1, :, :] * scale_y
                scene_flow[2, :, :] = scene_flow[2, :, :] * scale_x
                scene_flow[3, :, :] = scene_flow[3, :, :] * scale_x
                #print(f'scene_flow.shape {scene_flow.shape}')
                #break
                
                #img_l1 = img_l1[None].cuda()
                #img_l2 = img_l2[None].cuda()
                #img_r1 = img_r1[None].cuda()
                #img_r2 = img_r2[None].cuda()
                #padder = InputPadder(img_l1.shape, mode='things')
                #img_l1_p, img_l2_p = padder.pad(img_l1, img_l2)
                #img_r1_p, img_r2_p = padder.pad(img_r1, img_r2)
                #scene_flow_low, scene_flow_pr = model(img_l1_p, img_l2_p, img_r1_p, img_r2_p, iters=iters, test_mode=True)
                #scene_flow = padder.unpad(scene_flow_pr[0]).cpu()
                #print(f' padded scene_flow_pr.shape {scene_flow_pr.shape}')
                
                #print('\n')
                #scene_flow = scene_flow_pr

                #epe_all = torch.sum((scene_flow.cpu() - scene_flow_gt.cpu())**2, dim=0).sqrt()
                #epe_of = torch.sum((scene_flow.cpu()[:,0:2,:,:] - scene_flow_gt.cpu()[:,0:2,:,:])**2, dim=0).sqrt()
                #epe_d = torch.sum((scene_flow.cpu()[:,2,:,:] - scene_flow_gt.cpu()[:,2,:,:])**2, dim=0).sqrt()
                #epe_fd = torch.sum((scene_flow.cpu()[:,3,:,:] - scene_flow_gt.cpu()[:,3,:,:])**2, dim=0).sqrt()
                #print(f'torch.from_numpy(scene_flow).cpu().shape {torch.from_numpy(scene_flow).cpu().shape}')
                #print(f'torch.from_numpy(scene_flow_gt).cpu().shape {torch.from_numpy(scene_flow_gt).cpu().shape}')
                
                #print(f'type(scene_flow_gt) {scene_flow_gt.shape}')
                scene_flow = torch.from_numpy(scene_flow).cpu()
                scene_flow_gt = scene_flow_gt.cpu()
                epe_all = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
                epe_of = torch.sum((scene_flow[0:2,:,:] - scene_flow_gt[0:2,:,:])**2, dim=0).sqrt()
                epe_d = ((scene_flow[2,:,:] - scene_flow_gt[2,:,:])**2).sqrt()
                epe_fd = ((scene_flow[3,:,:] - scene_flow_gt[3,:,:])**2).sqrt()
                #print("Validation (%s) EPE Overall: %f , EPE of: %f, EPE d: %f, EPE fd: %f" % (dstype, epe_all,  epe_of, epe_d, epe_fd))
                #print(f'epe_of {epe_all}')
                #print(f'epe_d {epe_all}')
                #print(f'epe_fd {epe_all}')
                #print('\n')
                epe_list.append(epe_all.view(-1).numpy())
                epe_list_of.append(epe_of.view(-1).numpy())
                epe_list_d.append(epe_d.view(-1).numpy())
                epe_list_fd.append(epe_fd.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe_of_l = np.concatenate(epe_list_of)
        epe_d_l = np.concatenate(epe_list_d)
        epe_fd_l = np.concatenate(epe_list_fd)

        epe = np.nanmean(epe_all)
        epe_of = np.nanmean(epe_of_l)
        epe_d = np.nanmean(epe_d_l)
        epe_fd = np.nanmean(epe_fd_l)
        
        px1_any = np.nanmean(((epe_of_l<1) | (epe_d_l<1) | (epe_fd_l<1)))
        px3_any = np.nanmean(((epe_of_l<3) | (epe_d_l<3) | (epe_fd_l<3)))
        px5_any = np.nanmean(((epe_of_l<5) | (epe_d_l<1) | (epe_fd_l<5))) 

        px1_sf = np.nanmean(epe_all<1)
        px3_sf = np.nanmean(epe_all<3)
        px5_sf = np.nanmean(epe_all<5)

        px1_of = np.nanmean(epe_of_l<1)
        px3_of = np.nanmean(epe_of_l<3)
        px5_of = np.nanmean(epe_of_l<5)

        px1_d = np.nanmean(epe_d_l<1)
        px3_d = np.nanmean(epe_d_l<3)
        px5_d = np.nanmean(epe_d_l<5)

        px1_fd = np.nanmean(epe_fd_l<1)
        px3_fd = np.nanmean(epe_fd_l<3)
        px5_fd = np.nanmean(epe_fd_l<5)

        #print("Validation (%s) EPE Overall: %f , EPE of: %f, EPE d: %f, EPE fd: %f" % (dstype, epe,  epe_of, epe_d, epe_fd))
        results[str(dstype)+' epe overall val'] = epe
        results[str(dstype)+' epe of val'] = epe_of
        results[str(dstype)+' epe d val'] = epe_d
        results[str(dstype)+' epe fd val'] = epe_fd

        results[str(dstype)+' outlier sf 1px'] = px1_sf
        results[str(dstype)+' outlier of 1px'] = px1_of
        results[str(dstype)+' outlier d 1px'] = px1_d
        results[str(dstype)+' outlier fd 1px'] = px1_fd

        results[str(dstype)+' outlier sf 3px'] = px3_sf
        results[str(dstype)+' outlier of 3px'] = px3_of
        results[str(dstype)+' outlier d 3px'] = px3_d
        results[str(dstype)+' outlier fd 3px'] = px3_fd

        results[str(dstype)+' outlier sf 5px'] = px5_sf
        results[str(dstype)+' outlier of 5px'] = px5_of
        results[str(dstype)+' outlier d 5px'] = px5_d
        results[str(dstype)+' outlier fd 5px'] = px5_fd
        
        results[str(dstype)+' outlier any 1px'] = px1_any
        results[str(dstype)+' outlier any 3px'] = px3_any
        results[str(dstype)+' outlier any 5px'] = px5_any
        print(results)
        #break

    return results

@torch.no_grad()
def validate_kitti_seperate(args,model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True)
    print(f'Total Files {len(val_dataset)}')
    out_sf_any_list ,out_list, out_of_list, out_d_list, out_fd_list, epe_list, epe_of_list, epe_d_list, epe_fd_list = [], [], [], [], [], [], [], [], []

    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        
        #scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
        #valid_gt = valid_gt.cpu().detach().numpy().astype('float32')
        

        #img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        #img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        #img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        #img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
        #scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
        #valid_gt = np.reshape(valid_gt, (valid_gt.shape[1], valid_gt.shape[2], valid_gt.shape[0]))
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        img_l1 = img_l1.transpose(1, 2, 0)
        img_l2 = img_l2.transpose(1, 2, 0)
        img_r1 = img_r1.transpose(1, 2, 0)
        img_r2 = img_r2.transpose(1, 2, 0)
        #scene_flow_gt = scene_flow_gt.transpose(1, 2, 0)
        #valid_gt = valid_gt.transpose(1, 2, 0)
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        
        ht, wt = img_l1.shape[0], img_l1.shape[1]
                
        ht_8 = generate_dim_d8(img_l1.shape[0])
        wt_8 = generate_dim_d8(img_l1.shape[1])

        scale_x = wt / wt_8
        scale_y = ht / ht_8
        #print(f'img_l1.shape {img_l1.shape}')
        #print(f'ht, wt {(ht,wt)}')
        img_l1 = resize(img_l1, (ht_8, wt_8))
        img_r1 = resize(img_r1, (ht_8, wt_8))
        img_l2 = resize(img_l2, (ht_8, wt_8))
        img_r2 = resize(img_r2, (ht_8, wt_8))

        #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).permute(2, 0, 1)[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).permute(2, 0, 1)[None].cuda()
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        #print('\n')
        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        #scene_flow = padder.unpad(flow_pr[0]).cpu()
        scene_flow = flow_pr
        scene_flow = flow_pr[0].cpu()
                #print(f'scene_flow.shape {scene_flow.shape}')
        #scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
        scene_flow = scene_flow.permute(1, 2, 0)
                #print(f'scene_flow.shape {scene_flow.shape}')
        scene_flow = resize(scene_flow, (ht, wt))
        #scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))
        scene_flow = scene_flow.transpose(2, 0, 1)

        #scene_flow[0,0, :, :] = scene_flow[0,0, :, :] * scale_x
        #scene_flow[0,1, :, :] = scene_flow[0,1, :, :] * scale_y
        #scene_flow[0,2, :, :] = scene_flow[0,2, :, :] * scale_x
        #scene_flow[0,3, :, :] = scene_flow[0,3, :, :] * scale_x

        scene_flow[0, :, :] = scene_flow[0, :, :] * scale_x
        scene_flow[1, :, :] = scene_flow[1, :, :] * scale_y
        scene_flow[2, :, :] = scene_flow[2, :, :] * scale_x
        scene_flow[3, :, :] = scene_flow[3, :, :] * scale_x

        
        #epe = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
        #scene_flow_gt = scene_flow_gt[0]
        #val = (valid_gt[0][0].view(-1) >= 0.5) & (valid_gt[0][1].view(-1) >= 0.5) & (valid_gt[0][2].view(-1) >= 0.5)
        val = (valid_gt[0].view(-1) >= 0.5) & (valid_gt[1].view(-1) >= 0.5) & (valid_gt[2].view(-1) >= 0.5)

        epe = torch.sum((torch.from_numpy(scene_flow).cpu() - scene_flow_gt.cpu())**2, dim=0).sqrt()
        mag = torch.sum(scene_flow_gt**2, dim=0).sqrt().cpu()
        epe = epe.view(-1)
        mag = mag.view(-1)
        
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        
        epe_of = torch.sum((torch.from_numpy(scene_flow[0:2,:,:]).cpu() - scene_flow_gt[0:2, :, :].cpu())**2, dim=0).sqrt()
        mag_of = torch.sum(scene_flow_gt[0:2,:,:]**2, dim=0).sqrt().cpu()
        epe_of = epe_of.view(-1)
        mag_of = mag_of.view(-1)
        
        out_of = ((epe_of > 3.0) & ((epe_of/mag_of) > 0.05)).float()


        epe_d = ((torch.from_numpy(scene_flow[2,:,:]).cpu() - scene_flow_gt[2, :, :].cpu())**2).sqrt()
        mag_d = (scene_flow_gt[2,:,:]**2).sqrt().cpu()
        epe_d = epe_d.view(-1)
        mag_d = mag_d.view(-1)
        
        out_d = ((epe_d > 3.0) & ((epe_d/mag_d) > 0.05)).float()

        epe_fd = ((torch.from_numpy(scene_flow[3,:,:]).cpu() - scene_flow_gt[3, :, :].cpu())**2).sqrt()
        mag_fd = (scene_flow_gt[3,:,:]**2).sqrt().cpu()
        epe_fd = epe_fd.view(-1)
        mag_fd = mag_fd.view(-1)

        out_fd = ((epe_fd > 3.0) & ((epe_fd/mag_fd) > 0.05)).float()
        out_sf_any = (((epe_of > 3.0) & ((epe_of/mag_of) > 0.05)) | ((epe_d > 3.0) & ((epe_d/mag_d) > 0.05)) | (epe_fd > 3.0) & ((epe_fd/mag_fd) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        epe_of_list.append(epe_of[val].mean().item())
        epe_d_list.append(epe_d[val].mean().item())
        epe_fd_list.append(epe_fd[val].mean().item())

        out_list.append(out[val].cpu().numpy())
        out_of_list.append(out_of[val].cpu().numpy())
        out_d_list.append(out_d[val].cpu().numpy())
        out_fd_list.append(out_fd[val].cpu().numpy())
        out_sf_any_list.append(out_sf_any[val].cpu().numpy())


    epe_list = np.array(epe_list)
    epe_of_list = np.array(epe_of_list)
    epe_d_list = np.array(epe_d_list)
    epe_fd_list = np.array(epe_fd_list)

    out_list = np.concatenate(out_list)
    out_of_list = np.concatenate(out_of_list)
    out_d_list = np.concatenate(out_d_list)
    out_fd_list = np.concatenate(out_fd_list)
    out_sf_any_list = np.concatenate(out_sf_any_list)

    epe = np.mean(epe_list)
    epe_of = np.mean(epe_of_list)
    epe_d = np.mean(epe_d_list)
    epe_fd = np.mean(epe_fd_list)

    out_fl = 100 * np.mean(out_list)
    out_of = 100 * np.mean(out_of_list)
    out_d = 100 * np.mean(out_d_list)
    out_fd = 100 * np.mean(out_fd_list)
    out_sf_any = 100 * np.mean(out_sf_any_list)

    #print("Validation KITTI: %f, %f" % (epe, f1))
    #print(f'Validation Kitti : epe:{epe}, epe_of:{epe_of}, epe_d:{epe_d}, epe_fd:{epe_fd}')
    
    print("Validation EPE overall: %f , EPE of: %f, EPE d: %f, EPE fd: %f" % (epe,  epe_of, epe_d, epe_fd))
    print("Validation Outlier rate SF overall: %f , OFL of: %f, OD: %f, OFD: %f, OSF_any: %f" % (out_fl,  out_of, out_d, out_fd, out_sf_any))
    return {'kitti-epe': epe, 'kitti-epe-of': epe_of, 'kitti-epe-d': epe_d, 'kitti-epe-fd': epe_fd, 'kitti-out-fl': out_fl,
    'kitti-f1-of': out_of, 'kitti-f1-d': out_d, 'kitti-f1-fd': out_fd}
    #return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_kitti_seperate_test(args, model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True, override_split = False)
    print(f'Total Files {len(val_dataset)}')
    epe_list, epe_of_list, epe_d_list, epe_fd_list = [], [], [], []
    out_3_list, out_of_3_list, out_d_3_list, out_fd_3_list = [], [], [], []
    out_1_list, out_of_1_list, out_d_1_list, out_fd_1_list = [], [], [], []
    out_5_list, out_of_5_list, out_d_5_list, out_fd_5_list = [], [], [], []

    out_sf_3_any_list, out_sf_1_any_list, out_sf_5_any_list = [], [], []

    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        #scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
        #valid_gt = valid_gt.cpu().detach().numpy().astype('float32')
        

        #img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        #img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        #img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        #img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
        #scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
        #valid_gt = np.reshape(valid_gt, (valid_gt.shape[1], valid_gt.shape[2], valid_gt.shape[0]))
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        img_l1 = img_l1.transpose(1, 2, 0)
        img_l2 = img_l2.transpose(1, 2, 0)
        img_r1 = img_r1.transpose(1, 2, 0)
        img_r2 = img_r2.transpose(1, 2, 0)
        #scene_flow_gt = scene_flow_gt.transpose(1, 2, 0)
        #valid_gt = valid_gt.transpose(1, 2, 0)
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        
        ht, wt = img_l1.shape[0], img_l1.shape[1]
                
        ht_8 = generate_dim_d8(img_l1.shape[0])
        wt_8 = generate_dim_d8(img_l1.shape[1])

        scale_x = wt / wt_8
        scale_y = ht / ht_8
        #print(f'img_l1.shape {img_l1.shape}')
        #print(f'ht, wt {(ht,wt)}')
        img_l1 = resize(img_l1, (ht_8, wt_8))
        img_r1 = resize(img_r1, (ht_8, wt_8))
        img_l2 = resize(img_l2, (ht_8, wt_8))
        img_r2 = resize(img_r2, (ht_8, wt_8))

        #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()

        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).permute(2, 0, 1)[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).permute(2, 0, 1)[None].cuda()
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        #print('\n')
        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        #scene_flow = padder.unpad(flow_pr[0]).cpu()
        scene_flow = flow_pr
        scene_flow = flow_pr[0].cpu()
                #print(f'scene_flow.shape {scene_flow.shape}')
        #scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
        scene_flow = scene_flow.permute(1, 2, 0)
                #print(f'scene_flow.shape {scene_flow.shape}')
        scene_flow = resize(scene_flow, (ht, wt))
        #scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))
        scene_flow = scene_flow.transpose(2, 0, 1)

        #scene_flow[0,0, :, :] = scene_flow[0,0, :, :] * scale_x
        #scene_flow[0,1, :, :] = scene_flow[0,1, :, :] * scale_y
        #scene_flow[0,2, :, :] = scene_flow[0,2, :, :] * scale_x
        #scene_flow[0,3, :, :] = scene_flow[0,3, :, :] * scale_x

        scene_flow[0, :, :] = scene_flow[0, :, :] * scale_x
        scene_flow[1, :, :] = scene_flow[1, :, :] * scale_y
        scene_flow[2, :, :] = scene_flow[2, :, :] * scale_x
        scene_flow[3, :, :] = scene_flow[3, :, :] * scale_x

        
        #epe = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
        #scene_flow_gt = scene_flow_gt[0]
        #val = (valid_gt[0][0].view(-1) >= 0.5) & (valid_gt[0][1].view(-1) >= 0.5) & (valid_gt[0][2].view(-1) >= 0.5)
        val = (valid_gt[0].view(-1) >= 0.5) & (valid_gt[1].view(-1) >= 0.5) & (valid_gt[2].view(-1) >= 0.5)

        epe = torch.sum((torch.from_numpy(scene_flow).cpu() - scene_flow_gt.cpu())**2, dim=0).sqrt()
        mag = torch.sum(scene_flow_gt**2, dim=0).sqrt().cpu()
        epe = epe.view(-1)
        mag = mag.view(-1)
        
        out_3 = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        out_1 = ((epe > 1.0) & ((epe/mag) > 0.05)).float()
        out_5 = ((epe > 5.0) & ((epe/mag) > 0.05)).float()
        
        epe_of = torch.sum((torch.from_numpy(scene_flow[0:2,:,:]).cpu() - scene_flow_gt[0:2, :, :].cpu())**2, dim=0).sqrt()
        mag_of = torch.sum(scene_flow_gt[0:2,:,:]**2, dim=0).sqrt().cpu()
        epe_of = epe_of.view(-1)
        mag_of = mag_of.view(-1)
        
        out_of_3 = ((epe_of > 3.0) & ((epe_of/mag_of) > 0.05)).float()
        out_of_1 = ((epe_of > 1.0) & ((epe_of/mag_of) > 0.05)).float()
        out_of_5 = ((epe_of > 5.0) & ((epe_of/mag_of) > 0.05)).float()


        epe_d = ((torch.from_numpy(scene_flow[2,:,:]).cpu() - scene_flow_gt[2, :, :].cpu())**2).sqrt()
        mag_d = (scene_flow_gt[2,:,:]**2).sqrt().cpu()
        epe_d = epe_d.view(-1)
        mag_d = mag_d.view(-1)
        
        out_d_3 = ((epe_d > 3.0) & ((epe_d/mag_d) > 0.05)).float()
        out_d_1 = ((epe_d > 1.0) & ((epe_d/mag_d) > 0.05)).float()
        out_d_5 = ((epe_d > 5.0) & ((epe_d/mag_d) > 0.05)).float()

        epe_fd = ((torch.from_numpy(scene_flow[3,:,:]).cpu() - scene_flow_gt[3, :, :].cpu())**2).sqrt()
        mag_fd = (scene_flow_gt[3,:,:]**2).sqrt().cpu()
        epe_fd = epe_fd.view(-1)
        mag_fd = mag_fd.view(-1)

        out_fd_3 = ((epe_fd > 3.0) & ((epe_fd/mag_fd) > 0.05)).float()
        out_fd_1 = ((epe_fd > 1.0) & ((epe_fd/mag_fd) > 0.05)).float()
        out_fd_5 = ((epe_fd > 5.0) & ((epe_fd/mag_fd) > 0.05)).float()

        out_sf_any_3 = (((epe_of > 3.0) & ((epe_of/mag_of) > 0.05)) | ((epe_d > 3.0) & ((epe_d/mag_d) > 0.05)) | (epe_fd > 3.0) & ((epe_fd/mag_fd) > 0.05)).float()
        out_sf_any_1 = (((epe_of > 1.0) & ((epe_of/mag_of) > 0.05)) | ((epe_d > 1.0) & ((epe_d/mag_d) > 0.05)) | (epe_fd > 1.0) & ((epe_fd/mag_fd) > 0.05)).float()
        out_sf_any_5 = (((epe_of > 5.0) & ((epe_of/mag_of) > 0.05)) | ((epe_d > 5.0) & ((epe_d/mag_d) > 0.05)) | (epe_fd > 5.0) & ((epe_fd/mag_fd) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        epe_of_list.append(epe_of[val].mean().item())
        epe_d_list.append(epe_d[val].mean().item())
        epe_fd_list.append(epe_fd[val].mean().item())

        out_3_list.append(out_3[val].cpu().numpy())
        out_of_3_list.append(out_of_3[val].cpu().numpy())
        out_d_3_list.append(out_d_3[val].cpu().numpy())
        out_fd_3_list.append(out_fd_3[val].cpu().numpy())
        out_sf_3_any_list.append(out_sf_any_3[val].cpu().numpy())

        out_1_list.append(out_1[val].cpu().numpy())
        out_of_1_list.append(out_of_1[val].cpu().numpy())
        out_d_1_list.append(out_d_1[val].cpu().numpy())
        out_fd_1_list.append(out_fd_1[val].cpu().numpy())
        out_sf_1_any_list.append(out_sf_any_1[val].cpu().numpy())

        out_5_list.append(out_5[val].cpu().numpy())
        out_of_5_list.append(out_of_5[val].cpu().numpy())
        out_d_5_list.append(out_d_5[val].cpu().numpy())
        out_fd_5_list.append(out_fd_5[val].cpu().numpy())
        out_sf_5_any_list.append(out_sf_any_5[val].cpu().numpy())


    epe_list = np.array(epe_list)
    epe_of_list = np.array(epe_of_list)
    epe_d_list = np.array(epe_d_list)
    epe_fd_list = np.array(epe_fd_list)

    out_3_list = np.concatenate(out_3_list)
    out_of_3_list = np.concatenate(out_of_3_list)
    out_d_3_list = np.concatenate(out_d_3_list)
    out_fd_3_list = np.concatenate(out_fd_3_list)
    out_sf_3_any_list = np.concatenate(out_sf_3_any_list)

    out_1_list = np.concatenate(out_1_list)
    out_of_1_list = np.concatenate(out_of_1_list)
    out_d_1_list = np.concatenate(out_d_1_list)
    out_fd_1_list = np.concatenate(out_fd_1_list)
    out_sf_1_any_list = np.concatenate(out_sf_1_any_list)

    out_5_list = np.concatenate(out_5_list)
    out_of_5_list = np.concatenate(out_of_5_list)
    out_d_5_list = np.concatenate(out_d_5_list)
    out_fd_5_list = np.concatenate(out_fd_5_list)
    out_sf_5_any_list = np.concatenate(out_sf_5_any_list)

    epe = np.mean(epe_list)
    epe_of = np.mean(epe_of_list)
    epe_d = np.mean(epe_d_list)
    epe_fd = np.mean(epe_fd_list)

    out_fl_3 = 100 * np.mean(out_3_list)
    out_of_3 = 100 * np.mean(out_of_3_list)
    out_d_3 = 100 * np.mean(out_d_3_list)
    out_fd_3 = 100 * np.mean(out_fd_3_list)
    out_sf_any_3 = 100 * np.mean(out_sf_3_any_list)

    out_fl_1 = 100 * np.mean(out_1_list)
    out_of_1 = 100 * np.mean(out_of_1_list)
    out_d_1 = 100 * np.mean(out_d_1_list)
    out_fd_1 = 100 * np.mean(out_fd_1_list)
    out_sf_any_1 = 100 * np.mean(out_sf_1_any_list)

    out_fl_5 = 100 * np.mean(out_5_list)
    out_of_5 = 100 * np.mean(out_of_5_list)
    out_d_5 = 100 * np.mean(out_d_5_list)
    out_fd_5 = 100 * np.mean(out_fd_5_list)
    out_sf_any_5 = 100 * np.mean(out_sf_5_any_list)

    #print("Validation KITTI: %f, %f" % (epe, f1))
    #print(f'Validation Kitti : epe:{epe}, epe_of:{epe_of}, epe_d:{epe_d}, epe_fd:{epe_fd}')
    
    #print("Validation EPE overall: %f , EPE of: %f, EPE d: %f, EPE fd: %f" % (epe,  epe_of, epe_d, epe_fd))
    #print("Validation Outlier rate SF overall: %f , OFL of: %f, OD: %f, OFD: %f, OSF_any: %f" % (out_fl_3,  out_of_3, out_d_3, out_fd_3, out_sf_any_3))

    results = {'kitti-epe': epe, 'kitti-epe-of': epe_of, 'kitti-epe-d': epe_d, 'kitti-epe-fd': epe_fd, 'kitti-out-sf-3': out_fl_3,
    'kitti-out-of-3': out_of_3, 'kitti-out-d-3': out_d_3, 'kitti-out-fd-3': out_fd_3 , 'kitti-out-sf-1': out_fl_1,
    'kitti-out-of-1': out_of_1, 'kitti-out-d-1': out_d_1, 'kitti-out-fd-1': out_fd_1, 'kitti-out-sf-5': out_fl_5,
    'kitti-out-of-5': out_of_5, 'kitti-out-d-5': out_d_5, 'kitti-out-fd-5': out_fd_5}
    print(results)
    return results
    #return {'kitti-epe': epe, 'kitti-f1': f1}

'''@torch.no_grad()
def validate_things_seperate(args, model, iters=12):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size)
        epe_list_of = []
        epe_list_d = []
        epe_list_fd = []
        epe_list = []
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            #print(f'batch number {i_batch}')
            #print(f'scene_flow_gt_b.shape {scene_flow_gt_b.shape}')
            for val_id in range(img_l1_b.shape[0]):
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                #print(f'scene_flow_gt.shape, {scene_flow_gt.shape}')
                img_l1 = img_l1.cpu().detach().numpy().astype('float32')
                img_l2 = img_l2.cpu().detach().numpy().astype('float32')
                img_r1 = img_r1.cpu().detach().numpy().astype('float32')
                img_r2 = img_r2.cpu().detach().numpy().astype('float32')
                scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')

                # Change accordingly to reverse to single channel disp and disp change

                #scene_flow_gt = np.stack([scene_flow_gt[0,:,:], scene_flow_gt[1,:,:], scene_flow_gt[2,:,:], scene_flow_gt[4,:,:]], axis=0)
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')

                img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
                img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
                img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
                img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
                scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
                
                img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
                img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
                img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
                img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
                scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))

                img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
                img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
                img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
                img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
                scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
                
                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
                #print(f' unpadded scene_flow_pr.shape {scene_flow_pr.shape}')
                scene_flow = scene_flow_pr
                
                #img_l1 = img_l1[None].cuda()
                #img_l2 = img_l2[None].cuda()
                #img_r1 = img_r1[None].cuda()
                #img_r2 = img_r2[None].cuda()
                #padder = InputPadder(img_l1.shape, mode='things')
                #img_l1_p, img_l2_p = padder.pad(img_l1, img_l2)
                #img_r1_p, img_r2_p = padder.pad(img_r1, img_r2)
                #scene_flow_low, scene_flow_pr = model(img_l1_p, img_l2_p, img_r1_p, img_r2_p, iters=iters, test_mode=True)
                #scene_flow = padder.unpad(scene_flow_pr[0]).cpu()
                #print(f' padded scene_flow_pr.shape {scene_flow_pr.shape}')
                
                #print('\n')
                #scene_flow = scene_flow_pr

                epe_all = torch.sum((scene_flow.cpu() - scene_flow_gt.cpu())**2, dim=0).sqrt()
                epe_of = torch.sum((scene_flow.cpu()[:,0:2,:,:] - scene_flow_gt.cpu()[:,0:2,:,:])**2, dim=0).sqrt()
                epe_d = torch.sum((scene_flow.cpu()[:,2,:,:] - scene_flow_gt.cpu()[:,2,:,:])**2, dim=0).sqrt()
                epe_fd = torch.sum((scene_flow.cpu()[:,3,:,:] - scene_flow_gt.cpu()[:,3,:,:])**2, dim=0).sqrt()
                #print(f'epe_all {epe_all.shape}')
                #print(f'epe_of {epe_of.shape}')
                #print(f'epe_d {epe_d.shape}')
                #print(f'epe_fd {epe_fd.shape}')
                #print('\n')

                epe_list.append(epe_all.view(-1).numpy())
                epe_list_of.append(epe_of.view(-1).numpy())
                epe_list_d.append(epe_d.view(-1).numpy())
                epe_list_fd.append(epe_fd.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe_of = np.concatenate(epe_list_of)
        epe_d = np.concatenate(epe_list_d)
        epe_fd = np.concatenate(epe_list_fd)

        epe = np.nanmean(epe_all)
        epe_of = np.nanmean(epe_of)
        epe_d = np.nanmean(epe_d)
        epe_fd = np.nanmean(epe_fd)
        #px1 = np.mean(epe_all<1)
        #px3 = np.mean(epe_all<3)
        #px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE Overall: %f , EPE of: %f, EPE d: %f, EPE fd: %f" % (dstype, epe,  epe_of, epe_d, epe_fd))
        results[str(dstype)+' overall'] = np.nanmean(epe_list)
        results[str(dstype)+' of'] = np.nanmean(epe_list_of)
        results[str(dstype)+' d'] = np.nanmean(epe_list_d)
        results[str(dstype)+' fd'] = np.nanmean(epe_list_fd)

    return results'''

@torch.no_grad()
def validate_things(args, model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size)
        epe_list = []
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            #print(f'batch number {i_batch}')
            #print(f'scene_flow_gt_b.shape {scene_flow_gt_b.shape}')
            for val_id in range(img_l1_b.shape[0]):
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                #print(f'scene_flow_gt.shape, {scene_flow_gt.shape}')
                img_l1 = img_l1.cpu().detach().numpy().astype('float32')
                img_l2 = img_l2.cpu().detach().numpy().astype('float32')
                img_r1 = img_r1.cpu().detach().numpy().astype('float32')
                img_r2 = img_r2.cpu().detach().numpy().astype('float32')
                scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')

                img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
                img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
                img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
                img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
                scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
                
                img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
                img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
                img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
                img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
                scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))

                img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
                img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
                img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
                img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
                scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
                #print(f'img_l1.shape, {img_l1.shape}')
                #print(f'img_l2.shape, {img_l1.shape}')
                #print(f'img_r1.shape, {img_l1.shape}')
                #print(f'img_r2.shape, {img_l1.shape}')
                #print('\n')
                
                #print(f'img_l1.shape, {img_l1.shape}')
                #print(f'img_l2.shape, {img_l1.shape}')
                #print(f'img_r1.shape, {img_l1.shape}')
                #print(f'img_r2.shape, {img_l1.shape}')
                #print('\n')
                
                #print(f'img_l1.shape, {img_l1.shape}')
                #print(f'img_l2.shape, {img_l1.shape}')
                #print(f'img_r1.shape, {img_l1.shape}')
                #print(f'img_r2.shape, {img_l1.shape}')
                #print('\n')
                
                #print(f'args.image_size {(int(args.image_size[0]), int(args.image_size[1]))}')
                
                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
                #print(f' unpadded scene_flow_pr.shape {scene_flow_pr.shape}')
                scene_flow = scene_flow_pr
                
                '''img_l1 = img_l1[None].cuda()
                img_l2 = img_l2[None].cuda()
                img_r1 = img_r1[None].cuda()
                img_r2 = img_r2[None].cuda()
                padder = InputPadder(img_l1.shape, mode='things')
                img_l1_p, img_l2_p = padder.pad(img_l1, img_l2)
                img_r1_p, img_r2_p = padder.pad(img_r1, img_r2)
                scene_flow_low, scene_flow_pr = model(img_l1_p, img_l2_p, img_r1_p, img_r2_p, iters=iters, test_mode=True)
                scene_flow = padder.unpad(scene_flow_pr[0]).cpu()
                print(f' padded scene_flow_pr.shape {scene_flow_pr.shape}')
                
                print('\n')'''
                #scene_flow = scene_flow_pr


                epe = torch.sum((scene_flow.cpu() - scene_flow_gt.cpu())**2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())
            

        epe_all = np.concatenate(epe_list)
        epe = np.nanmean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

'''@torch.no_grad()
def validate_things(args, model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size)
        epe_list = []
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            print(f'batch number {i_batch}')
            #print(f'scene_flow_gt_b.shape {scene_flow_gt_b.shape}')
            for val_id in range(img_l1_b.shape[0]):
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                print(f'img_l1.shape, {img_l1.shape}')
                print(f'img_l2.shape, {img_l1.shape}')
                print(f'img_r1.shape, {img_l1.shape}')
                print(f'img_r2.shape, {img_l1.shape}')
                print('\n')
                img_l1 = img_l1[None].cuda()
                img_l2 = img_l2[None].cuda()
                img_r1 = img_r1[None].cuda()
                img_r2 = img_r2[None].cuda()
                #print(f'img_l1.shape, {img_l1.shape}')
                #print(f'img_l2.shape, {img_l1.shape}')
                #print(f'img_r1.shape, {img_l1.shape}')
                #print(f'img_r2.shape, {img_l1.shape}')
                #print('\n')
                padder = InputPadder(img_l1.shape, mode='things')
                img_l1, img_l2 = padder.pad(img_l1, img_l2)
                img_r1, img_r2 = padder.pad(img_r1, img_r2)
                print(f'img_l1.shape, {img_l1.shape}')
                print(f'img_l2.shape, {img_l1.shape}')
                print(f'img_r1.shape, {img_l1.shape}')
                print(f'img_r2.shape, {img_l1.shape}')
                print('\n')

                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
                scene_flow = padder.unpad(scene_flow_pr[0]).cpu()

                epe = torch.sum((scene_flow - scene_flow_gt.cpu())**2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())
            break

        epe_all = np.concatenate(epe_list)
        epe = np.nanmean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results'''



@torch.no_grad()
def validate_kitti(args,model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
        valid_gt = valid_gt.cpu().detach().numpy().astype('float32')

        img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
        scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
        valid_gt = np.reshape(valid_gt, (valid_gt.shape[1], valid_gt.shape[2], valid_gt.shape[0]))
                
        img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
        img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
        img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
        img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
        scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))
        valid_gt = resize(valid_gt, (args.image_size[0], args.image_size[1]))

        img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()

        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        #scene_flow = padder.unpad(flow_pr[0]).cpu()
        scene_flow = flow_pr
        epe = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(scene_flow_gt**2, dim=0).sqrt()

        val = (valid_gt[0][0].view(-1) >= 0.5) & (valid_gt[0][1].view(-1) >= 0.5) & (valid_gt[0][2].view(-1) >= 0.5)
        
        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


'''@torch.no_grad()
def validate_kitti(args,model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        img_l1 = img_l1[None].cuda()
        img_l2 = img_l2[None].cuda()
        img_r1 = img_r1[None].cuda()
        img_r2 = img_r2[None].cuda()

        padder = InputPadder(img_l1.shape, mode='kitti')
        img_l1, img_l2 = padder.pad(img_l1, img_l2)
        img_r1, img_r2 = padder.pad(img_r1, img_r2)

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        scene_flow = padder.unpad(flow_pr[0]).cpu()
        #print(f'scene_flow.shape {scene_flow.shape}')
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        #print(f'valid_gt.shape {valid_gt.shape}')
        #print('\n')
        epe = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(scene_flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = (valid_gt[0].view(-1) >= 0.5) & (valid_gt[1].view(-1) >= 0.5) & (valid_gt[2].view(-1) >= 0.5)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}'''

''''@torch.no_grad()
def validate_things(args, model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    val_dataloader = datasets.fetch_dataloader(args)
    epe_list = []

    for i_batch, data_blob in enumerate(val_dataloader):
            image1, image2, flow_gt, _ = [x.cuda() for x in data_blob]
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % ('frames_cleanpass', epe, px1, px3, px5))
    results['frames_cleanpass'] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    
    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, flow, disparity, disparity_change, valid, valid_d, valid_dc = val_dataset[val_id]
        
        #print(f'flow.shape {flow.shape}')
        #print(f'disparity.shape {disparity.shape}')
        #print(f'disparity_change.shape {disparity_change.shape}')
        scene_flow_gt = torch.cat((flow, disparity, disparity_change), 0)
        img_l1 = img_l1[None].cuda()
        img_l2 = img_l2[None].cuda()
        img_r1 = img_r1[None].cuda()
        img_r2 = img_r2[None].cuda()

        padder = InputPadder(img_l1.shape, mode='kitti')
        img_l1, img_l2 = padder.pad(img_l1, img_l2)
        img_r1, img_r2 = padder.pad(img_r1, img_r2)

        scene_flow_low, scene_flow_pr = model( img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        scene_flow = padder.unpad(scene_flow_pr[0]).cpu()
        #print(f'scene_flow.shape {scene_flow.shape}')
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        
        epe = torch.sum((scene_flow - scene_flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(scene_flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = (valid.view(-1) >= 0.5) & (valid_d.view(-1) >= 0.5) & (valid_dc.view(-1) >= 0.5)

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}'''

def evaluate_all_ckp_kitti(args, root_dir = 'checkpoints_f1', file_kitti = 'kitti_val.txt'):
    checkpoints = ['5000_raft-kitti_sf.pth', '10000_raft-kitti_sf.pth', '15000_raft-kitti_sf.pth', '20000_raft-kitti_sf.pth', '25000_raft-kitti_sf.pth']
    validation_scores = []
    for ckp in checkpoints:
        ckp_dir = os.path.join(root_dir, ckp)
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(ckp_dir))
        validation_scores.append(validate_kitti_seperate_test(args, model.module))
    with open(file_kitti, 'wb') as wf:
        np.save(wf, np.array(validation_scores))

def evaluate_all_ckp_things(args, root_dir = 'checkpoints_f1', file_things = 'things_val.txt'):
    checkpoints = np.arange(5000, 200000, 5000)
    validation_scores = []
    for i in checkpoints:
        ckp = str(i)+'_raft-things_sf.pth'
        ckp_dir = os.path.join(root_dir, ckp)
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(ckp_dir))
        validation_scores.append(validate_things_seperate(args, model.module))
    with open(file_things, 'wb') as wf:
        np.save(wf, np.array(validation_scores))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    #create_sintel_submission(model.module, warm_start=True)
    #create_kitti_submission_sf(model.module)

    with torch.no_grad():

        if args.dataset == 'kitti':
            #validate_kitti_seperate_test(args, model.module)
            evaluate_all_ckp_kitti(args)
        elif args.dataset == 'things':
            validate_things(args, model.module)

        elif args.dataset == 'things_seperate':
            validate_things_seperate(args, model.module)
            #evaluate_all_ckp_things(args)


