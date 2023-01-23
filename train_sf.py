from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft_sf import RAFT
import evaluate_sf as evaluate
import datasets_sf
torch.cuda.empty_cache()
from vis_sf import demo_gt_kitti, demo_gt_kitti_test 
from vis_sf import demo_gt_things 

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss_2c(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    #print(n_predictions)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    #print(f'flow_gt.shape {flow_gt.shape}')
    #mag_f = torch.sum(flow_gt[:, 0:2, :, :]**2, dim=1).sqrt()
    
    
    #mag_f_v = torch.sum(flow_gt[:, 1, :, :]**2, dim=1).sqrt()
    #print(f'mag_f.shape {mag_f.shape}')
    #print(f'mag_f.shape {mag_f_v.shape}')
    #print(f'valid.shape {valid.shape}')
    #print(f'flow_gt[:, 2, :, :]**2.shape {(flow_gt[:, 2, :, :]**2).shape}')
   # print(f'(flow_gt[:, 0:2, :, :]**2).shape {(flow_gt[:, 0:2, :, :]**2).shape}')
    #mag_d0 = (flow_gt[:, 2, :, :]**2).sqrt()
    #mag_d1 = (flow_gt[:, 4, :, :]**2).sqrt()

    #mag_d0 = torch.sum(flow_gt[:, 2:4, :, :]**2, dim=1).sqrt()
    #mag_d1 = torch.sum(flow_gt[:, 4:6, :, :]**2, dim=1).sqrt()
    
    #print(f'mag_d0.shape {mag_d0.shape}')
    #valid = (valid[:,0,:,:] >= 0.5) & (valid[:,1, :, :] >= 0.5) & (valid[:,2, :, :] >= 0.5) & (mag_f < max_flow) & (mag_d0 < max_flow) & (mag_d1 < max_flow)

    #valid = (valid[:,0,:,:] >= 0.5) & (mag_f < max_flow) 
    
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        i_loss = (flow_preds[i] - flow_gt).abs()
        #flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        
        flow_loss += i_weight * (i_loss).mean()
        #print(f'flow_loss {flow_loss}')


    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    #epe = epe.view(-1)[valid.view(-1)]
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    #print(n_predictions)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    #print(f'flow_gt.shape {flow_gt.shape}')
    mag_f = torch.sum(flow_gt[:, 0:2, :, :]**2, dim=1).sqrt()
    #mag_f_v = torch.sum(flow_gt[:, 1, :, :]**2, dim=1).sqrt()
    #print(f'mag_f.shape {mag_f.shape}')
    #print(f'mag_f.shape {mag_f_v.shape}')
    #print(f'valid.shape {valid.shape}')
    #print(f'flow_gt[:, 2, :, :]**2.shape {(flow_gt[:, 2, :, :]**2).shape}')
   # print(f'(flow_gt[:, 0:2, :, :]**2).shape {(flow_gt[:, 0:2, :, :]**2).shape}')
    mag_d0 = (flow_gt[:, 2, :, :]**2).sqrt()
    mag_d1 = (flow_gt[:, 3, :, :]**2).sqrt()
    
    #print(f'mag_d0.shape {mag_d0.shape}')
    valid = (valid[:,0,:,:] >= 0.5) & (valid[:,1, :, :] >= 0.5) & (valid[:,2, :, :] >= 0.5) & (mag_f < max_flow) & (mag_d0 < max_flow) & (mag_d1 < max_flow)
    
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        #print(f'i_weight {i_weight}')
        i_loss = (flow_preds[i] - flow_gt).abs()
        #print(f'i_loss {i_weight}')
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe_all = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe_all = epe_all.view(-1)[valid.view(-1)]

    epe_of = torch.sum((flow_preds[-1][:,0:2,:,:] - flow_gt[:,0:2,:,:])**2, dim=1).sqrt()
    epe_of = epe_of.view(-1)[valid.view(-1)]

    disparity_gt = flow_gt[:,2,:,:]
    disparity_gt = disparity_gt.view(1, disparity_gt.shape[0], disparity_gt.shape[1], disparity_gt.shape[2])

    disparity_est = flow_preds[-1][:,2,:,:]
    disparity_est = disparity_est.view(1, disparity_est.shape[0], disparity_est.shape[1], disparity_est.shape[2])

    epe_d = torch.sum((disparity_est - disparity_gt)**2, dim=0).sqrt()
    epe_d = epe_d.view(-1)[valid.view(-1)]

    fdisparity_gt = flow_gt[:,3,:,:]
    fdisparity_gt = fdisparity_gt.view(1, fdisparity_gt.shape[0], fdisparity_gt.shape[1], fdisparity_gt.shape[2])

    fdisparity_est = flow_preds[-1][:,3,:,:]
    fdisparity_est = fdisparity_est.view(1, fdisparity_est.shape[0], fdisparity_est.shape[1], fdisparity_est.shape[2])

    epe_fd = torch.sum((fdisparity_est - fdisparity_gt)**2, dim=0).sqrt()
    epe_fd = epe_fd.view(-1)[valid.view(-1)]

    metrics = {
        'epe_all': epe_all.mean().item(),
        'epe_of': epe_of.mean().item(),
        'epe_d': epe_d.mean().item(),
        'epe_fd': epe_fd.mean().item(),
        '1px': (epe_all < 1).float().mean().item(),
        '3px': (epe_all < 3).float().mean().item(),
        '5px': (epe_all < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [str(k) + ' : ' + "{:10.4f}".format(self.running_loss[k]/SUM_FREQ) + ', ' for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        #metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        metrics_str = ''.join([i for i in metrics_data])[:-2]
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
    
    def write_image(self, plots, step, dataset_type):
        if self.writer is None:
            self.writer = SummaryWriter()
        if(dataset_type == 'things'):
            for key in ['frames_cleanpass', 'frames_finalpass']:
                for i in plots[key]:
                    self.writer.add_image(f'{key} {step} OF, disp, future disp : 1st row prediction, 2nd row ground truth', i, 0, dataformats='HWC')
                    break
        else:
            for i in plots['kitti']:
                self.writer.add_image(f'Kitti {step} OF, disp, future disp : 1st row prediction, 2nd row ground truth', i, 0, dataformats='HWC')
                break
        # Don't forget to reshape.
    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets_sf.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            img_l1, img_l2, img_r1, img_r2, scene_flow, valid = [x.cuda() for x in data_blob]
            
            #print(i_batch)
            #print(f'scene flow gt shape {scene_flow.shape}')
            #print(f'scene valid gt shape {valid.shape}')
            
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                img_l1 = (img_l1 + stdv * torch.randn(*img_l1.shape).cuda()).clamp(0.0, 255.0)
                img_l2 = (img_l2 + stdv * torch.randn(*img_l2.shape).cuda()).clamp(0.0, 255.0)
                img_r1 = (img_r1 + stdv * torch.randn(*img_r1.shape).cuda()).clamp(0.0, 255.0)
                img_r2 = (img_r2 + stdv * torch.randn(*img_r2.shape).cuda()).clamp(0.0, 255.0)

            scene_flow_predictions = model(img_l1, img_l2,img_r1, img_r2, iters=args.iters)
            scene_flow_predictions = scene_flow_predictions.view(args.iters, -1, scene_flow_predictions.shape[2], scene_flow_predictions.shape[3], scene_flow_predictions.shape[4])             
            #print(f'scene_flow_predictions.shape {scene_flow_predictions.shape}')
            #loss, metrics = sequence_loss(scene_flow_predictions, scene_flow, valid, args.gamma)
            loss, metrics = sequence_loss(scene_flow_predictions, scene_flow, valid, args.gamma)
            #print(f'loss {loss}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            '''if total_steps % 10 == 10 - 1:
                plots = demo_gt_things(model.module, image_size = args.image_size)
                #logger.write_dict(results)
                logger.write_image(plots, total_steps+1, 'things')'''

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
            #if total_steps % 100 == 100 - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                plot_dataset = 'things'
                for val_dataset in args.validation:
                    if val_dataset == 'things':
                        results.update(evaluate.validate_things_seperate(args, model.module))
                        #plots = demo_gt_things(model.module, image_size = args.image_size)
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti_seperate(args, model.module))
                        #plots = demo_gt_kitti(model.module, image_size = args.image_size)
                        #plot_dataset = 'kitti'
                
               # plots = demo_gt(args)
                logger.write_dict(results)
                #logger.write_image(plots, total_steps+1)

                #logger.write_dict(results)
                #logger.write_image(plots, total_steps+1, val_dataset)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)