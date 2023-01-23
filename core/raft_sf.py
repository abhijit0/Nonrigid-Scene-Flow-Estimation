import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import AlternateCorrBlockDisparity, CorrBlock, AlternateCorrBlock, CorrBlockDisparity
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def initialize_disparity(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords0[:,1,:,:] = 0
        coords1[:,1,:,:] = 0
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image_l1, image_l2, image_r1, image_r2, iters=12, flow_init_l1_l2=None, flow_init_l1_r1=None, flow_init_l1_r2=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image_l1 = 2 * (image_l1 / 255.0) - 1.0
        image_l2 = 2 * (image_l2 / 255.0) - 1.0
        image_r1 = 2 * (image_r1 / 255.0) - 1.0
        image_r2 = 2 * (image_r2 / 255.0) - 1.0

        image_l1 = image_l1.contiguous()
        image_l2 = image_l2.contiguous()
        image_r1 = image_r1.contiguous()
        image_r2 = image_r2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap_l1, fmap_l2 = self.fnet([image_l1, image_l2])
            fmap_r1, fmap_r2= self.fnet([image_r1, image_r2])        
        
        fmap_l1 = fmap_l1.float()
        fmap_l2 = fmap_l2.float()
        fmap_r1 = fmap_r1.float()
        fmap_r2 = fmap_r2.float()

        #print(f'fmap_l1.shape {fmap_l1.shape}')
        #print(f'fmap_r1.shape {fmap_r1.shape}')
        #print('\n')
        if self.args.alternate_corr:
            corr_fn_l1_l2 = AlternateCorrBlock(fmap_l1, fmap_l2, radius=self.args.corr_radius)
            #corr_fn_l1_r1 = AlternateCorrBlockDisparity(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            #corr_fn_l1_r2 = AlternateCorrBlockDisparity(fmap_l1, fmap_r2, radius=self.args.corr_radius)
            #corr_fn_l1_r1 = AlternateCorrBlock(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            #corr_fn_l1_r2 = AlternateCorrBlock(fmap_l1, fmap_r2, radius=self.args.corr_radius)
        else:
            corr_fn_l1_l2 = CorrBlock(fmap_l1, fmap_l2, radius=self.args.corr_radius)
            #corr_fn_l1_r1 = CorrBlockDisparity(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            #corr_fn_l1_r2 = CorrBlockDisparity(fmap_l1, fmap_r2, radius=self.args.corr_radius)
            corr_fn_l1_r1 = CorrBlock(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            corr_fn_l1_r2 = CorrBlock(fmap_l1, fmap_r2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image_l1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Coords wrt L1 last char indicating the coord index. l2 , r2 .. etc indicate the image
        coords_l20, coords_l21 = self.initialize_flow(image_l1)
        
        coords_r10, coords_r11 = self.initialize_flow(image_l1)
        coords_r20, coords_r21 = self.initialize_flow(image_l1)

        #coords_r10, coords_r11 = self.initialize_disparity(image_l1)
        #coords_r20, coords_r21 = self.initialize_disparity(image_l1)
        #print(f'coords_r10.shape {coords_r10[:,0,:,:]}')

        if flow_init_l1_l2 is not None:
            coords_l21 = coords_l21 + flow_init_l1_l2
        
        if flow_init_l1_r1 is not None:
            coords_r11 = coords_r11 + flow_init_l1_r1
        
        if flow_init_l1_r2 is not None:
            coords_r21 = coords_r21 + flow_init_l1_r2
            


        scene_flow_predictions = []
        net_flow = net
        net_disp = net
        net_fdisp = net
        for itr in range(iters):
            coords_l21 = coords_l21.detach()
            coords_r11 = coords_r11.detach()
            coords_r21 = coords_r21.detach()
            #print(f'coords_r11.shape {coords_r11.shape}')

            #coords_r11, coords_r21 = coords_r11[:,0,:,:].view(coords_r11.shape[0], -1, coords_r11.shape[2], coords_r11.shape[3]), coords_r21[:,0,:,].view(coords_r21.shape[0], -1, coords_r21.shape[2], coords_r21.shape[3]) 

            corr_l1_l2 = corr_fn_l1_l2(coords_l21) # index correlation volume
            corr_l1_r1 = corr_fn_l1_r1(coords_r11)
            corr_l1_r2 = corr_fn_l1_r2(coords_r21)
            #print(f'coords_r11.shape {coords_r11.shape}')
            #print('\n')
            flow_l1_l2 = coords_l21 - coords_l20
            flow_l1_r1 = coords_r11 - coords_r10
            flow_l1_r2 = coords_r21 - coords_r20

            with autocast(enabled=self.args.mixed_precision):
                net_flow, up_mask_l1_l2, delta_flow_l1_l2 = self.update_block(net_flow, inp, corr_l1_l2, flow_l1_l2)
                net_disp, up_mask_l1_r1, delta_flow_l1_r1 = self.update_block(net_disp, inp, corr_l1_r1, flow_l1_r1) #Try with removing net
                net_fdisp, up_mask_l1_r2, delta_flow_l1_r2 = self.update_block(net_fdisp, inp, corr_l1_r2, flow_l1_r2)
            

            # F(t+1) = F(t) + \Delta(t)
            coords_l21 = coords_l21 + delta_flow_l1_l2
            coords_r11 = coords_r11 + delta_flow_l1_r1
            coords_r21 = coords_r21 + delta_flow_l1_r2
            #coords_r11[:,0,:,:] = coords_r11[:,0,:,:] + delta_flow_l1_r1[:,0,:,:]
            #coords_r21[:,0,:,:] = coords_r21[:,0,:,:] + delta_flow_l1_r2[:,0,:,:]
            #print(f'coords_r11.shape {coords_r11.shape}')
            # upsample predictions
            if ((up_mask_l1_l2 is None) and (up_mask_l1_r1 is None) and (up_mask_l1_r2 is None)):
                flow_up_l1_l2 = upflow8(coords_l21 - coords_l20)
                flow_up_l1_r1 = upflow8(coords_r11 - coords_r10)
                flow_up_l1_r2 = upflow8(coords_r21 - coords_r20)
            else:
                flow_up_l1_l2 = self.upsample_flow(coords_l21 - coords_l20, up_mask_l1_l2)
                flow_up_l1_r1 = self.upsample_flow(coords_r11 - coords_r10, up_mask_l1_r1)
                flow_up_l1_r2 = self.upsample_flow(coords_r21 - coords_r20, up_mask_l1_r2)
            
            #print(f'flow_up.shape {flow_up_l1_l2.shape}')
            #scene_flow = np.concatenate([flow_up_l1_l2, flow_up_l1_r1, flow_up_l1_r2], axis = 1)
            #print(f'flow_up_l1_l2[:,0,:,:].shape {flow_up_l1_l2[:,0,:,:].shape}')
            
            #flow_l1_l2_x = flow_up_l1_l2[:,0,:,:].view(flow_up_l1_l2.shape[0], 1, flow_up_l1_l2.shape[2], flow_up_l1_l2.shape[3])
            #flow_l1_l2_x2c = torch.cat((flow_l1_l2_x, flow_l1_l2_x), 1)
            
            scene_flow_prediction = torch.stack([flow_up_l1_l2[:,0,:,:], flow_up_l1_l2[:,1,:,:], flow_up_l1_r1[:,0,:,:], flow_up_l1_r2[:,0,:,:] - flow_up_l1_l2[:,0,:,:]], dim = 1)
            #scene_flow_prediction = torch.stack([flow_up_l1_l2[:,0,:,:], flow_up_l1_l2[:,1,:,:], flow_up_l1_r1[:,0,:,:], flow_up_l1_r1[:,1,:,:], flow_up_l1_r2[:,0,:,:] - flow_l1_l2_x2c[:,0,:,:], flow_up_l1_r2[:,1,:,:] - flow_l1_l2_x2c[:,1,:,:]], dim = 1)

            #flow_predictions_l1_l2.append(flow_up_l1_l2)
            #flow_predictions_l1_r1.append(flow_up_l1_r1)
            #flow_predictions_l1_r2.append(flow_up_l1_r2)
            scene_flow_predictions.append(scene_flow_prediction)
            #break

        if test_mode:
            coords = torch.cat((coords_l21 - coords_l20, coords_r11 - coords_r10, coords_r21 - coords_r20), 1)
            scene_flow = torch.stack([flow_up_l1_l2[:,0,:,:], flow_up_l1_l2[:,1,:,:], flow_up_l1_r1[:,0,:,:], flow_up_l1_r2[:,0,:,:] - flow_up_l1_l2[:,0,:,:]], dim = 1)
            return coords, scene_flow

        #print(f'flow_predictions_l1_l2 shape {np.array(flow_predictions_l1_l2).shape}')    
        
        #flow_predictions_l1_l2 = torch.stack(flow_predictions_l1_l2)
        #flow_predictions_l1_r1 = torch.stack(flow_predictions_l1_r1)
        #flow_predictions_l1_r2 = torch.stack(flow_predictions_l1_r2)
        #scene_flow_predictions = np.concatenate([np.array(flow_predictions_l1_l2, dtype= np.float32).reshape(-1,1), np.array(flow_predictions_l1_r1, dtype= np.float32).reshape(-1,1), np.array(flow_predictions_l1_r2, dtype= np.float32).reshape(-1,1)], axis = 1)  
        #scene_flow_predictions = torch.from_numpy(scene_flow_predictions)
        #scene_flow_predictions = torch.cat((flow_predictions_l1_l2, flow_predictions_l1_r1, flow_predictions_l1_r2), 2)
        
        scene_flow_predictions = torch.stack(scene_flow_predictions)
        
        #print(f'scene_flow_predictions.shape {scene_flow_predictions.shape}')
        return scene_flow_predictions



'''class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image_l1, image_l2, image_r1, image_r2, iters=12, flow_init_l1_l2=None, flow_init_l1_r1=None, flow_init_l1_r2=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image_l1 = 2 * (image_l1 / 255.0) - 1.0
        image_l2 = 2 * (image_l2 / 255.0) - 1.0
        image_r1 = 2 * (image_r1 / 255.0) - 1.0
        image_r2 = 2 * (image_r2 / 255.0) - 1.0

        image_l1 = image_l1.contiguous()
        image_l2 = image_l2.contiguous()
        image_r1 = image_r1.contiguous()
        image_r2 = image_r2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap_l1, fmap_l2 = self.fnet([image_l1, image_l2])
            fmap_r1, fmap_r2= self.fnet([image_r1, image_r2])        
        
        fmap_l1 = fmap_l1.float()
        fmap_l2 = fmap_l2.float()
        fmap_r1 = fmap_r1.float()
        fmap_r2 = fmap_r2.float()
        if self.args.alternate_corr:
            corr_fn_l1_l2 = AlternateCorrBlock(fmap_l1, fmap_l2, radius=self.args.corr_radius)
            corr_fn_l1_r1 = AlternateCorrBlock(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            corr_fn_l1_r2 = AlternateCorrBlock(fmap_l1, fmap_r2, radius=self.args.corr_radius)
        else:
            corr_fn_l1_l2 = CorrBlock(fmap_l1, fmap_l2, radius=self.args.corr_radius)
            corr_fn_l1_r1 = CorrBlock(fmap_l1, fmap_r1, radius=self.args.corr_radius)
            corr_fn_l1_r2 = CorrBlock(fmap_l1, fmap_r2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image_l1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Coords wrt L1 last char indicating the coord index. l2 , r2 .. etc indicate the image
        coords_l20, coords_l21 = self.initialize_flow(image_l1)
        coords_r10, coords_r11 = self.initialize_flow(image_l1)
        coords_r20, coords_r21 = self.initialize_flow(image_l1)
        

        if flow_init_l1_l2 is not None:
            coords_l21 = coords_l21 + flow_init_l1_l2
        
        if flow_init_l1_r1 is not None:
            coords_r11 = coords_r11 + flow_init_l1_r1
        
        if flow_init_l1_r2 is not None:
            coords_r21 = coords_r21 + flow_init_l1_r2
            


        scene_flow_predictions = []
        for itr in range(iters):
            coords_l21 = coords_l21.detach()
            coords_r11 = coords_r11.detach()
            coords_r21 = coords_r21.detach()

            corr_l1_l2 = corr_fn_l1_l2(coords_l21) # index correlation volume
            corr_l1_r1 = corr_fn_l1_r1(coords_r11)
            corr_l1_r2 = corr_fn_l1_r2(coords_r21)

            flow_l1_l2 = coords_l21 - coords_l20
            flow_l1_r1 = coords_r11 - coords_r10
            flow_l1_r2 = coords_r21 - coords_r20

            with autocast(enabled=self.args.mixed_precision):
                net, up_mask_l1_l2, delta_flow_l1_l2 = self.update_block(net, inp, corr_l1_l2, flow_l1_l2)
                net, up_mask_l1_r1, delta_flow_l1_r1 = self.update_block(net, inp, corr_l1_r1, flow_l1_r1) #Try with removing net
                net, up_mask_l1_r2, delta_flow_l1_r2 = self.update_block(net, inp, corr_l1_r2, flow_l1_r2)
            

            # F(t+1) = F(t) + \Delta(t)
            coords_l21 = coords_l21 + delta_flow_l1_l2
            coords_r11 = coords_r11 + delta_flow_l1_r1
            coords_r21 = coords_r21 + delta_flow_l1_r2

            # upsample predictions
            if ((up_mask_l1_l2 is None) and (up_mask_l1_r1 is None) and (up_mask_l1_r2 is None)):
                flow_up_l1_l2 = upflow8(coords_l21 - coords_l20)
                flow_up_l1_r1 = upflow8(coords_r11 - coords_r10)
                flow_up_l1_r2 = upflow8(coords_r21 - coords_r20)
            else:
                flow_up_l1_l2 = self.upsample_flow(coords_l21 - coords_l20, up_mask_l1_l2)
                flow_up_l1_r1 = self.upsample_flow(coords_r11 - coords_r10, up_mask_l1_r1)
                flow_up_l1_r2 = self.upsample_flow(coords_r21 - coords_r20, up_mask_l1_r2)
            
            #print(f'flow_up.shape {flow_up_l1_l2.shape}')
            #scene_flow = np.concatenate([flow_up_l1_l2, flow_up_l1_r1, flow_up_l1_r2], axis = 1)
            scene_flow_prediction = torch.stack([flow_up_l1_l2[:,0,:,:], flow_up_l1_l2[:,1,:,:], flow_up_l1_r1[:,0,:,:], flow_up_l1_r2[:,0,:,:] - flow_up_l1_l2[:,0,:,:]], dim = 1)
            #flow_predictions_l1_l2.append(flow_up_l1_l2)
            #flow_predictions_l1_r1.append(flow_up_l1_r1)
            #flow_predictions_l1_r2.append(flow_up_l1_r2)
            scene_flow_predictions.append(scene_flow_prediction)

        if test_mode:
            coords = torch.cat((coords_l21 - coords_l20, coords_r11 - coords_r10, coords_r21 - coords_r20), 1)
            scene_flow = torch.stack([flow_up_l1_l2[:,0,:,:], flow_up_l1_l2[:,1,:,:], flow_up_l1_r1[:,0,:,:], flow_up_l1_r2[:,0,:,:] - flow_up_l1_l2[:,0,:,:]], dim = 1)
            return coords, scene_flow

        #print(f'flow_predictions_l1_l2 shape {np.array(flow_predictions_l1_l2).shape}')    
        
        #flow_predictions_l1_l2 = torch.stack(flow_predictions_l1_l2)
        #flow_predictions_l1_r1 = torch.stack(flow_predictions_l1_r1)
        #flow_predictions_l1_r2 = torch.stack(flow_predictions_l1_r2)
        #scene_flow_predictions = np.concatenate([np.array(flow_predictions_l1_l2, dtype= np.float32).reshape(-1,1), np.array(flow_predictions_l1_r1, dtype= np.float32).reshape(-1,1), np.array(flow_predictions_l1_r2, dtype= np.float32).reshape(-1,1)], axis = 1)  
        #scene_flow_predictions = torch.from_numpy(scene_flow_predictions)
        #scene_flow_predictions = torch.cat((flow_predictions_l1_l2, flow_predictions_l1_r1, flow_predictions_l1_r2), 2)
        
        scene_flow_predictions = torch.stack(scene_flow_predictions)
        
        #print(f'scene_flow_predictions.shape {scene_flow_predictions.shape}')
        return scene_flow_predictions'''
