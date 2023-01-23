import sys
from turtle import color

from matplotlib.colors import Colormap
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft_sf import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import matplotlib.pyplot as plt
import datasets_sf as datasets
import torch.utils.data as data
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing

from skimage.transform import resize
from evaluate_sf import generate_dim_d8

DEVICE = 'cuda'
import io

DPI = 180

def get_img_from_fig(fig, dpi=DPI):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def visualize_sf_plt(scene_flow_gt, scene_flow_est, index, invalid_map=None):
    
    optical_flow_gt = scene_flow_gt[0:2, : , :]
    disparity_gt = scene_flow_gt[2, : , :]
    future_disparity_gt = scene_flow_gt[3, : , :]

    optical_flow_est = scene_flow_est[0:2, : , :]
    disparity_est = scene_flow_est[2, : , :]
    future_disparity_est = scene_flow_est[3, : , :]
    #fig.pcolormesh(disparity_gt, cmap='jet')
    #fig.colorbar()
    #fig.savefig('color_bar_test.jpeg')

    #fig = plt.figure()
    #fig.plot(optical_flow_gt)
    #fig.colorbar()
    #
    #optical_flow_gt = optical_flow_gt.permute(1,2,0).detach().numpy()
    optical_flow_gt = optical_flow_gt.permute(1,2,0).cpu().detach().numpy()
    optical_flow_gt = flow_viz.flow_to_image(optical_flow_gt)
    if(invalid_map is not None):
        optical_flow_gt[:, :, 0][invalid_map[0,0,:,:]] = 0
        optical_flow_gt[:, :, 1][invalid_map[0,0,:,:]] = 0
        optical_flow_gt[:, :, 2][invalid_map[0,0,:,:]] = 0
    fig, ax = plt.subplots(2,3, figsize=(25, 8), dpi = DPI)

    of_gt = ax[1,0].imshow(optical_flow_gt, aspect='auto')
    #plt.colorbar(of_gt, ax = ax[1, 0])
    ax[1,0].set_title('Optical flow Ground truth')
    
    optical_flow_est = optical_flow_est.permute(1,2,0).detach().numpy()
    #print(f'optical_flow_est.shape {optical_flow_est.shape}')
    optical_flow_est = flow_viz.flow_to_image(optical_flow_est)
    #print(f'optical_flow_est.shape {optical_flow_est.shape}')
    if(invalid_map is not None):
        optical_flow_est[:, :, 0][invalid_map[0,0,:,:]] = 0
        optical_flow_est[:, :, 1][invalid_map[0,0,:,:]] = 0
        optical_flow_est[:, :, 2][invalid_map[0,0,:,:]] = 0

    #color_wheel = flow_viz.make_colorwheel()
    of_est = ax[0, 0].imshow(optical_flow_est, aspect='auto')
    #plt.colorbar(of_est, ax = ax[0, 0])
    ax[0, 0].set_title('Optical flow Estimated')

    disparity_gt = disparity_gt.cpu().detach().numpy() 
    disparity_gt = disparity_gt.astype(np.uint8)
    #disparity_gt = cv2.applyColorMap(disparity_gt, cv2.COLORMAP_JET)
    #print(f'disparity_gt.shape {disparity_gt.shape}')
    if(invalid_map is not None):
        disparity_gt[invalid_map[0,1,:,:]] = 0
    disp_gt = ax[1, 1].imshow(disparity_gt, cmap='jet', aspect='auto')
    plt.colorbar(disp_gt, ax = ax[1, 1])
    ax[1, 1].set_title('disparity groundtruth')


    disparity_est = disparity_est.cpu().numpy() 
    disparity_est = disparity_est.astype(np.uint8)
    if(invalid_map is not None):
        disparity_est[invalid_map[0,1,:,:]] = 0
    #disparity_est = cv2.applyColorMap(disparity_est, cv2.COLORMAP_JET)
    disp_est = ax[0, 1].imshow(disparity_est, cmap='jet', aspect='auto')
    plt.colorbar(disp_gt, ax = ax[0, 1])
    ax[0, 1].set_title('disparity estimated')

    future_disparity_gt = future_disparity_gt.cpu().detach().numpy().astype(np.uint8) 
    if(invalid_map is not None):
        future_disparity_gt[invalid_map[0,2,:,:]] = 0
    #future_disparity_gt = cv2.applyColorMap(future_disparity_gt, cv2.COLORMAP_JET)
    fdisp_gt = ax[1, 2].imshow(future_disparity_gt, cmap='jet', aspect='auto')
    
    plt.colorbar(fdisp_gt, ax = ax[1, 2])
    ax[1, 2].set_title('future disparity ground truth')

    future_disparity_est = future_disparity_est.cpu().numpy().astype(np.uint8) 
    #future_disparity_est = cv2.applyColorMap(future_disparity_est, cv2.COLORMAP_JET)
    if(invalid_map is not None):
        future_disparity_est[invalid_map[0,2,:,:]] = 0
    fdisp_est = ax[0, 2].imshow(future_disparity_est, cmap='jet', aspect='auto')
    plt.colorbar(fdisp_est, ax=ax[0, 2])
    ax[0, 2].set_title('future disparity estimted')

    #img_sf_gt = np.concatenate([optical_flow_gt, disparity_gt, future_disparity_gt], axis=1)
    #img_sf_est = np.concatenate([optical_flow_est, disparity_est, future_disparity_est], axis=1)
    #img_sf_comp = np.concatenate([img_sf_gt, img_sf_est], axis = 0) 
    #if not cv2.imwrite(f'image_disp_{index}.jpeg', img_sf_comp ):
        #raise Exception('Could not write image')
    #fig.savefig('sf_test_plt.jpeg')
    img = get_img_from_fig(fig)
    return img

def visualize_sf_cv(scene_flow_gt, scene_flow_est, index):
    
    optical_flow_gt = scene_flow_gt[0:2, : , :]
    disparity_gt = scene_flow_gt[2, : , :]
    future_disparity_gt = scene_flow_gt[3, : , :]

    optical_flow_est = scene_flow_est[0:2, : , :]
    disparity_est = scene_flow_est[2, : , :]
    future_disparity_est = scene_flow_est[3, : , :]
    
    #optical_flow_gt = optical_flow_gt.permute(1,2,0).detach().numpy()
    optical_flow_gt = optical_flow_gt.permute(1,2,0).cpu().detach().numpy()
    optical_flow_gt = flow_viz.flow_to_image(optical_flow_gt)
    optical_flow_est = optical_flow_est.permute(1,2,0).detach().numpy()
    optical_flow_est = flow_viz.flow_to_image(optical_flow_est)

    disparity_gt = disparity_gt.cpu().detach().numpy() 
    disparity_gt = disparity_gt.astype(np.uint8)
    disparity_gt = cv2.applyColorMap(disparity_gt, cv2.COLORMAP_JET)
    disparity_est = disparity_est.cpu().numpy() 
    disparity_est = disparity_est.astype(np.uint8)
    disparity_est = cv2.applyColorMap(disparity_est, cv2.COLORMAP_JET)

    future_disparity_gt = future_disparity_gt.cpu().detach().numpy().astype(np.uint8) 
    future_disparity_gt = cv2.applyColorMap(future_disparity_gt, cv2.COLORMAP_JET)
    future_disparity_est = future_disparity_est.cpu().numpy().astype(np.uint8) 
    future_disparity_est = cv2.applyColorMap(future_disparity_est, cv2.COLORMAP_JET)

    img_sf_gt = np.concatenate([optical_flow_gt, disparity_gt, future_disparity_gt], axis=1)
    img_sf_est = np.concatenate([optical_flow_est, disparity_est, future_disparity_est], axis=1)
    img_sf_comp = np.concatenate([img_sf_gt, img_sf_est], axis = 0) 
    #if not cv2.imwrite(f'image_disp_{index}.jpeg', img_sf_comp ):
        #raise Exception('Could not write image')

    return img_sf_comp
    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    #img_disp = np.concatenate([img, disp], axis=0)
    
    '''img = img.permute(1,2,0).cpu().numpy()
    disp = disp.cpu().numpy().astype(np.uint8)
    #disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    # map flow to rgb image
    #flo = flow_viz.flow_to_image(flo)
    img_disp = np.concatenate([img, disp], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    #cv2.imshow('image', img_disp[:, :, [2,1,0]]/255.0)

    if not cv2.imwrite(f'image_disp_{index}.jpeg', img_disp):
        raise Exception('Could not write image')'''
    #cv2.waitKey()

def demo_gt_kitti(model, image_size = None): #args
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True)
    out_list, out_of_list, out_d_list, out_fd_list, epe_list, epe_of_list, epe_d_list, epe_fd_list = [], [], [], [], [], [], [], []
    plots_dict = {}
    plots=[]
    iters = 12
    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
        valid_gt = valid_gt.cpu().detach().numpy().astype('int32')
        #print(valid_gt)
        invalid_gt = np.invert(valid_gt.astype(np.bool))

        img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
        scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
        valid_gt = np.reshape(valid_gt, (valid_gt.shape[1], valid_gt.shape[2], valid_gt.shape[0]))
        invalid_gt = np.reshape(invalid_gt, (invalid_gt.shape[1], invalid_gt.shape[2], invalid_gt.shape[0]))

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


        #img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
        #img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
        #img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
        #img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
        #scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))
        #valid_gt = resize(valid_gt, (args.image_size[0], args.image_size[1]))

        img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()
        invalid_gt = torch.from_numpy(invalid_gt).view(invalid_gt.shape[2], invalid_gt.shape[0], invalid_gt.shape[1])[None].cuda()
        valid_gt = valid_gt.cpu().detach().numpy()
        invalid_gt = invalid_gt.cpu().detach().numpy()
        
        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

        flow_low, flow_pr = model(img_l1, img_l2, img_r1, img_r2, iters=iters, test_mode=True)
        #val = (valid_gt[0][0].view(-1) >= 0.5) & (valid_gt[0][1].view(-1) >= 0.5) & (valid_gt[0][2].view(-1) >= 0.5)
        #in_val = (valid_gt[0][0].view(-1) < 0.5) & (valid_gt[0][1].view(-1) < 0.5) & (valid_gt[0][2].view(-1) < 0.5)
        #scene_flow = padder.unpad(flow_pr[0]).cpu()
        scene_flow = flow_pr
        scene_flow = flow_pr[0].cpu()
                #print(f'scene_flow.shape {scene_flow.shape}')
        scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
                #print(f'scene_flow.shape {scene_flow.shape}')
        scene_flow = resize(scene_flow.detach().numpy(), (ht, wt))
        scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))

        scene_flow[0,0, :, :] = scene_flow[0,0, :, :] * scale_x
        scene_flow[0,1, :, :] = scene_flow[0,1, :, :] * scale_y
        scene_flow[0,2, :, :] = scene_flow[0,2, :, :] * scale_x
        scene_flow[0,3, :, :] = scene_flow[0,3, :, :] * scale_x

        #scene_flow_0 = np.reshape(scene_flow[0,0, :, :], -1) 
        #scene_flow_1 = np.reshape(scene_flow[0,1, :, :], -1) 
        #scene_flow_2 = np.reshape(scene_flow[0,2, :, :], -1)
        #scene_flow_3 = np.reshape(scene_flow[0,3, :, :], -1)

        #print(f'scene_flow_0.shape {scene_flow_0.shape}')
        #print(f'valid_gt.shape {valid_gt.shape}')
        scene_flow[0,0, :, :][invalid_gt[0,0,:,:]] = 0
        scene_flow[0,1, :, :][invalid_gt[0,0,:,:]] = 0
        scene_flow[0,2, :, :][invalid_gt[0,1,:,:]] = 0
        scene_flow[0,3, :, :][invalid_gt[0,2,:,:]] = 0

        #print(f'scene_flow.shape {scene_flow.shape}')
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        plots.append(visualize_sf_plt(scene_flow_gt[0].cpu(), torch.from_numpy(scene_flow[0]), val_id))
        plots_dict['Kitti'] = np.array(plots)
        break
    return plots_dict


def demo_gt_kitti_test(args, output_path='results_kitti_val'): #args
    """ Peform validation using the KITTI-2015 (train) split """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    val_dataset = datasets.KITTI(split='training', validation=True)
    out_list, out_of_list, out_d_list, out_fd_list, epe_list, epe_of_list, epe_d_list, epe_fd_list = [], [], [], [], [], [], [], []
    plots_dict = {}
    plots=[]
    iters = 12
    cwd = os.getcwd()
    output_path = os.path.join(cwd, output_path)
    print(f'validation files {len(val_dataset)}')
    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, scene_flow_gt, valid_gt = val_dataset[val_id]
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
        scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
        valid_gt = valid_gt.cpu().detach().numpy().astype('int32')
        #print(valid_gt)
        invalid_gt = np.invert(valid_gt.astype(np.bool))

        #img_l1 = np.reshape(img_l1, (img_l1.shape[1], img_l1.shape[2], img_l1.shape[0]))
        #img_l2 = np.reshape(img_l2, (img_l2.shape[1], img_l2.shape[2], img_l2.shape[0]))
        #img_r1 = np.reshape(img_r1, (img_r1.shape[1], img_r1.shape[2], img_r1.shape[0]))
        #img_r2 = np.reshape(img_r2, (img_r2.shape[1], img_r2.shape[2], img_r2.shape[0]))
        #scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[1], scene_flow_gt.shape[2], scene_flow_gt.shape[0]))
        #valid_gt = np.reshape(valid_gt, (valid_gt.shape[1], valid_gt.shape[2], valid_gt.shape[0]))
        #invalid_gt = np.reshape(invalid_gt, (invalid_gt.shape[1], invalid_gt.shape[2], invalid_gt.shape[0]))


        img_l1 = img_l1.transpose(1, 2, 0)
        img_l2 = img_l2.transpose(1, 2, 0)
        img_r1 = img_r1.transpose(1, 2, 0)
        img_r2 = img_r2.transpose(1, 2, 0)
        scene_flow_gt = scene_flow_gt.transpose(1, 2, 0)
        valid_gt = valid_gt.transpose(1, 2, 0)
        invalid_gt = invalid_gt.transpose(1, 2, 0)

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


        #img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
        #img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
        #img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
        #img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
        #scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))
        #valid_gt = resize(valid_gt, (args.image_size[0], args.image_size[1]))

        #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()
        #invalid_gt = torch.from_numpy(invalid_gt).view(invalid_gt.shape[2], invalid_gt.shape[0], invalid_gt.shape[1])[None].cuda()
        #valid_gt = valid_gt.cpu().detach().numpy()
        #invalid_gt = invalid_gt.cpu().detach().numpy()
        
        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()
        scene_flow_gt = torch.from_numpy(scene_flow_gt).permute(2, 0, 1)[None].cuda()
        valid_gt = torch.from_numpy(valid_gt).permute(2, 0, 1)[None].cuda()
        invalid_gt = torch.from_numpy(invalid_gt).permute(2, 0, 1)[None].cuda()
        valid_gt = valid_gt.cpu().detach().numpy()
        invalid_gt = invalid_gt.cpu().detach().numpy()

        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

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

        #scene_flow_0 = np.reshape(scene_flow[0,0, :, :], -1) 
        #scene_flow_1 = np.reshape(scene_flow[0,1, :, :], -1) 
        #scene_flow_2 = np.reshape(scene_flow[0,2, :, :], -1)
        #scene_flow_3 = np.reshape(scene_flow[0,3, :, :], -1)

        #print(f'scene_flow_0.shape {scene_flow_0.shape}')
        #print(f'valid_gt.shape {valid_gt.shape}')
        #print(f'invalid_gt.shape {invalid_gt.shape}')
        #scene_flow[0, :, :][invalid_gt[0,0,:,:]] = 0
        #scene_flow[1, :, :][invalid_gt[0,0,:,:]] = 0
        #scene_flow[2, :, :][invalid_gt[0,1,:,:]] = 0
        #scene_flow[3, :, :][invalid_gt[0,2,:,:]] = 0

        #print(f'scene_flow.shape {scene_flow.shape}')
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        #plots.append()
        img = visualize_sf_plt(scene_flow_gt[0].cpu(), torch.from_numpy(scene_flow), val_id, invalid_map=invalid_gt)
        #img = visualize_sf_plt(scene_flow_gt[0].cpu(), torch.from_numpy(scene_flow), val_id)
        if not cv2.imwrite(f'{output_path}/image_disp_kitti_{val_id}.jpeg', img):
            raise Exception('Could not write image')
    #return plots_dict

def demo_gt_kitti_test_data(args): #args
    """ Peform validation using the KITTI-2015 (train) split """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    val_dataset = datasets.KITTI(split='testing', aug_params=None)
    out_list, out_of_list, out_d_list, out_fd_list, epe_list, epe_of_list, epe_d_list, epe_fd_list = [], [], [], [], [], [], [], []
    plots_dict = {}
    plots=[]
    iters = 12
    print(f'validation files {len(val_dataset)}')
    for val_id in range(len(val_dataset)):
        img_l1, img_l2, img_r1, img_r2, (frame_id, ) = val_dataset[val_id]
        img_l1 = img_l1.cpu().detach().numpy().astype('float32')
        img_l2 = img_l2.cpu().detach().numpy().astype('float32')
        img_r1 = img_r1.cpu().detach().numpy().astype('float32')
        img_r2 = img_r2.cpu().detach().numpy().astype('float32')
    
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


        #img_l1 = resize(img_l1, (args.image_size[0], args.image_size[1]))
        #img_r1 = resize(img_r1, (args.image_size[0], args.image_size[1]))
        #img_l2 = resize(img_l2, (args.image_size[0], args.image_size[1]))
        #img_r2 = resize(img_r2, (args.image_size[0], args.image_size[1]))
        #scene_flow_gt = resize(scene_flow_gt, (args.image_size[0], args.image_size[1]))
        #valid_gt = resize(valid_gt, (args.image_size[0], args.image_size[1]))

        #img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
        #img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
        #img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
        #img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
        #scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])[None].cuda()
        #valid_gt = torch.from_numpy(valid_gt).view(valid_gt.shape[2], valid_gt.shape[0], valid_gt.shape[1])[None].cuda()
        #invalid_gt = torch.from_numpy(invalid_gt).view(invalid_gt.shape[2], invalid_gt.shape[0], invalid_gt.shape[1])[None].cuda()
        #valid_gt = valid_gt.cpu().detach().numpy()
        #invalid_gt = invalid_gt.cpu().detach().numpy()
        
        img_l1 = torch.from_numpy(img_l1).permute(2, 0, 1)[None].cuda()
        img_r1 = torch.from_numpy(img_r1).permute(2, 0, 1)[None].cuda()
        img_l2 = torch.from_numpy(img_l2).permute(2, 0, 1)[None].cuda()
        img_r2 = torch.from_numpy(img_r2).permute(2, 0, 1)[None].cuda()

        #padder = InputPadder(img_l1.shape, mode='kitti')
        #img_l1, img_l2 = padder.pad(img_l1, img_l2)
        #img_r1, img_r2 = padder.pad(img_r1, img_r2)

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

        #scene_flow_0 = np.reshape(scene_flow[0,0, :, :], -1) 
        #scene_flow_1 = np.reshape(scene_flow[0,1, :, :], -1) 
        #scene_flow_2 = np.reshape(scene_flow[0,2, :, :], -1)
        #scene_flow_3 = np.reshape(scene_flow[0,3, :, :], -1)

        #print(f'scene_flow_0.shape {scene_flow_0.shape}')
        #print(f'valid_gt.shape {valid_gt.shape}')
        #print(f'invalid_gt.shape {invalid_gt.shape}')

        #print(f'scene_flow.shape {scene_flow.shape}')
        #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
        plots.append(visualize_sf_plt(torch.from_numpy(scene_flow), torch.from_numpy(scene_flow), val_id))
        plots_dict['Kitti'] = np.array(plots)
        break
    return plots_dict


def demo_gt_things(model, image_size = None): #args
    #model = torch.nn.DataParallel(RAFT(args))
    #model.load_state_dict(torch.load(args.model))

    #model = model.module
    #model.to(DEVICE)
    model.eval()
    iters = 12
    plots_dict = {}
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        #print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=2)
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            plots=[]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            for val_id in range(img_l1_b.shape[0]):
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                #print(f'img_l1.shape {img_l1.shape}')
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
                
                #img_l1 = resize(img_l1, (image_size[0], image_size[1]))
                #img_r1 = resize(img_r1, (image_size[0], image_size[1]))
                #img_l2 = resize(img_l2, (image_size[0], image_size[1]))
                #img_r2 = resize(img_r2, (image_size[0], image_size[1]))
                #scene_flow_gt = resize(scene_flow_gt, (image_size[0], image_size[1]))

                img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
                img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
                img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
                img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
                scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])
                #scene_flow_gt = torch.tensor(scene_flow_gt, requires_grad = False)
                #scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1]))
                
                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, test_mode=True)
                #print(f' unpadded scene_flow_pr.shape {scene_flow_pr.shape}')
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                scene_flow = scene_flow_pr[0].cpu()
                scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
                scene_flow = resize(scene_flow.detach().numpy(), (ht, wt))
                scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))

                scene_flow[0,0, :, :] = scene_flow[0,0, :, :] * scale_x
                scene_flow[0,1, :, :] = scene_flow[0,1, :, :] * scale_y
                scene_flow[0,2, :, :] = scene_flow[0,2, :, :] * scale_x
                scene_flow[0,3, :, :] = scene_flow[0,3, :, :] * scale_x

                #scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
               # scene_flow_est = scene_flow_est.cpu().detach().numpy().astype('float32')
                plots.append(visualize_sf_plt(scene_flow_gt, torch.from_numpy(scene_flow[0]), val_id))
            break
        plots_dict[dstype] = np.array(plots)
    return plots_dict

def demo_gt_things_test(args, output_path='results_things_val'): #args
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    iters = 12
    plots_dict = {}
    cwd = os.getcwd()
    output_path = os.path.join(cwd, output_path) 
    print('until here')
    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        count = 0
        val_dataset = datasets.FlyingThings3D(dstype=dstype, validation = True)
        image_dir = os.path.join(output_path, dstype)
        print(f'total_files {len(val_dataset)} ')
        val_dataloader = data.DataLoader(val_dataset, batch_size=1)
        for i_batch, data_blob in enumerate(val_dataloader):
            #image1_b, image2_b, flow_gt_b, _ = [x.cuda() for x in data_blob]
            plots=[]
            img_l1_b, img_l2_b, img_r1_b, img_r2_b, scene_flow_gt_b, _ = [x.cuda() for x in data_blob]
            for val_id in range(img_l1_b.shape[0]):
                count+=1
                img_l1, img_l2, img_r1, img_r2, scene_flow_gt, _ = img_l1_b[val_id], img_l2_b[val_id], img_r1_b[val_id], img_r2_b[val_id], scene_flow_gt_b[val_id], _
                #print(f'img_l1.shape {img_l1.shape}')
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
                
                #img_l1 = resize(img_l1, (image_size[0], image_size[1]))
                #img_r1 = resize(img_r1, (image_size[0], image_size[1]))
                #img_l2 = resize(img_l2, (image_size[0], image_size[1]))
                #img_r2 = resize(img_r2, (image_size[0], image_size[1]))
                #scene_flow_gt = resize(scene_flow_gt, (image_size[0], image_size[1]))

                img_l1 = torch.from_numpy(img_l1).view(img_l1.shape[2], img_l1.shape[0], img_l1.shape[1])[None].cuda()
                img_r1 = torch.from_numpy(img_r1).view(img_r1.shape[2], img_r1.shape[0], img_r1.shape[1])[None].cuda()
                img_l2 = torch.from_numpy(img_l2).view(img_l2.shape[2], img_l2.shape[0], img_l2.shape[1])[None].cuda()
                img_r2 = torch.from_numpy(img_r2).view(img_r2.shape[2], img_r2.shape[0], img_r2.shape[1])[None].cuda()
                scene_flow_gt = torch.from_numpy(scene_flow_gt).view(scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1])
                #scene_flow_gt = torch.tensor(scene_flow_gt, requires_grad = False)
                #scene_flow_gt = np.reshape(scene_flow_gt, (scene_flow_gt.shape[2], scene_flow_gt.shape[0], scene_flow_gt.shape[1]))
                
                scene_flow_low, scene_flow_pr = model(img_l1, img_l2, img_r1, img_r2, test_mode=True)
                #print(f' unpadded scene_flow_pr.shape {scene_flow_pr.shape}')
                #print(f'scene_flow_gt.shape {scene_flow_gt.shape}')
                scene_flow = scene_flow_pr[0].cpu()
                scene_flow = scene_flow.view(scene_flow.shape[1], scene_flow.shape[2], scene_flow.shape[0])
                scene_flow = resize(scene_flow.detach().numpy(), (ht, wt))
                scene_flow = np.reshape(scene_flow, (1, scene_flow.shape[2], scene_flow.shape[0], scene_flow.shape[1]))

                scene_flow[0,0, :, :] = scene_flow[0,0, :, :] * scale_x
                scene_flow[0,1, :, :] = scene_flow[0,1, :, :] * scale_y
                scene_flow[0,2, :, :] = scene_flow[0,2, :, :] * scale_x
                scene_flow[0,3, :, :] = scene_flow[0,3, :, :] * scale_x

                #scene_flow_gt = scene_flow_gt.cpu().detach().numpy().astype('float32')
               # scene_flow_est = scene_flow_est.cpu().detach().numpy().astype('float32')
                print(f'count {count}')
                img = visualize_sf_plt(scene_flow_gt, torch.from_numpy(scene_flow[0]), val_id)
                if not cv2.imwrite(f'{image_dir}/image_disp_things_{count}.jpeg', img):
                    raise Exception('Could not write image')
            if(count>100):
                break
        
        #plots_dict[dstype] = np.array(plots)
    #return plots_dict
    '''with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)'''

def visualize_things_sf(model, iters=24, output_path='results_things_val'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    print(f'total test files {len(test_dataset)}')
    cwd = os.getcwd()
    output_path = os.path.join(cwd, output_path)
    flow_path = os.path.join(output_path, 'flow')
    disp_path = os.path.join(output_path, 'disp_0')
    future_disp_path = os.path.join(output_path, 'disp_1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    args = parser.parse_args()

    demo_gt_kitti_test(args)

    #img = demo_gt_kitti_test_data(args)
    #plt.plot(img['Kitti'][0])
    #print(img['Kitti'].shape)
    #if not cv2.imwrite(f'image_disp.jpeg', img['Kitti'][0]):
    #   raise Exception('Could not write image')
    #plt.savefig('sf_test.jpeg')
    #plots = demo_gt_things(args)
    #print(plots['frames_cleanpass'].shape)
#demo_gt_things()

