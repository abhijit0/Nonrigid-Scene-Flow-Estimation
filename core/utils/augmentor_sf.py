import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor_double:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img_l1, img_l2, img_r1, img_r2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img_l1 = np.array(self.photo_aug(Image.fromarray(img_l1)), dtype=np.uint8)
            img_l2 = np.array(self.photo_aug(Image.fromarray(img_l2)), dtype=np.uint8)
            img_r1 = np.array(self.photo_aug(Image.fromarray(img_r1)), dtype=np.uint8)
            img_r2 = np.array(self.photo_aug(Image.fromarray(img_r2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img_l1, img_l2, img_r1, img_r2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img_l1, img_l2, img_r1, img_r2 = np.split(image_stack, 4, axis=0)

        return img_l1, img_l2, img_r1, img_r2

    def eraser_transform(self, img_l1, img_l2, img_r1, img_r2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img_l1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_l2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_l2[y0:y0+dy, x0:x0+dx, :] = mean_color


        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r1.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_r1[y0:y0+dy, x0:x0+dx, :] = mean_color

        
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_r2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_l1, img_l2, img_r1, img_r2

    def spatial_transform(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        # randomly sample scale

        ht, wd = img_l1.shape[:2]
        
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))
        #print(f'min_scale {min_scale}')
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        
        #print(f'scale {scale}')
        scale_x = scale
        scale_y = scale
        #print(f' np.random.rand() {np.random.rand()} , self.stretch_prob {self.stretch_prob} ')
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        #print(f'scales before clip {scale_x, scale_y}')

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        #print(f'scale_x {scale_x}')
        #print(f'scale_y {scale_y}')
        #print('\n')
        #print(f'scales {scale_x, scale_y}')
        #print(f'scales after clip {scale_x, scale_y}')
        #print('\n')
        
        disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
        disp_ch = np.reshape(disp_ch, (disp_ch.shape[0], disp_ch.shape[1], -1))
        disp = disp * [1.0, 1.0]
        disp_ch = disp_ch * [1.0, 1.0]
        
        #print(f'flow.shape {flow.shape}')
        '''if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            #print(f'img_l1.shape before resize {img_l1.shape}')
            img_l1 = cv2.resize(img_l1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #print(f'img_l1.shape after resize {img_l1.shape}')
            #print('\n')
            img_l2 = cv2.resize(img_l2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r1 = cv2.resize(img_r1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r2 = cv2.resize(img_r2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #print(f'scale_x {scale_x}')
            #print(f'scale_y {scale_y}')
            #print(f'flow.shape {flow.shape}')
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #print(f'disp.shape {disp.shape}')
            #print('\n')
            disp_ch = cv2.resize(disp_ch, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            flow = flow * [scale_x, scale_y]
            #print(f'flow.shape {flow.shape}')
            #print('\n')
            disp = disp * [scale_x]
            disp_ch = disp_ch * [scale_x]
            
            disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
            disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))'''

        y0 = np.random.randint(0, img_r1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img_r1.shape[1] - self.crop_size[1])

        img_l1 = img_l1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_l2 = img_l2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r1 = img_r1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r2 = img_r2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp_ch = disp_ch[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        #print(f'flow.shape {flow.shape}')
        #print(f'disp.shape {disp.shape}')
        #print(f'disp_ch.shape {disp_ch.shape}')
        
        #print(f'arra equal check disp {np.array_equal(disp[:,:,0], disp[:,:,1])}')
        #print(f'arra equal check disp_ch {np.array_equal(disp_ch[:,:,0], disp_ch[:,:,1])}')
        #print('\n')
        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

    def __call__(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        img_l1, img_l2, img_r1, img_r2 = self.color_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2 = self.eraser_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch = self.spatial_transform(img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch)

        img_l1 = np.ascontiguousarray(img_l1)
        img_l2 = np.ascontiguousarray(img_l2)
        img_r1 = np.ascontiguousarray(img_r1)
        img_r2 = np.ascontiguousarray(img_r2)
        flow = np.ascontiguousarray(flow)
        disp = np.ascontiguousarray(disp)
        disp_ch = np.ascontiguousarray(disp_ch)

        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img_l1, img_l2, img_r1, img_r2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img_l1 = np.array(self.photo_aug(Image.fromarray(img_l1)), dtype=np.uint8)
            img_l2 = np.array(self.photo_aug(Image.fromarray(img_l2)), dtype=np.uint8)
            img_r1 = np.array(self.photo_aug(Image.fromarray(img_r1)), dtype=np.uint8)
            img_r2 = np.array(self.photo_aug(Image.fromarray(img_r2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img_l1, img_l2, img_r1, img_r2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img_l1, img_l2, img_r1, img_r2 = np.split(image_stack, 4, axis=0)

        return img_l1, img_l2, img_r1, img_r2

    def eraser_transform(self, img_l1, img_l2, img_r1, img_r2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img_l1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_l2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_l2[y0:y0+dy, x0:x0+dx, :] = mean_color


        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r1.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_r1[y0:y0+dy, x0:x0+dx, :] = mean_color

        
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_r2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_l1, img_l2, img_r1, img_r2

    def spatial_transform(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        # randomly sample scale
        
        ht, wd = img_l1.shape[:2]
        
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))
        #print(f'min_scale {min_scale}')
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        
        scale_x = scale
        scale_y = scale
        #print(f' np.random.rand() {np.random.rand()} , self.stretch_prob {self.stretch_prob} ')
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        #print(f'scales before clip {scale_x, scale_y}')

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        #print(f'scales {scale_x, scale_y}')
        #print(f'scales after clip {scale_x, scale_y}')
        #print('\n')
        #disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
        #disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            #print(f'img_l1.shape before resize {img_l1.shape}')
            img_l1 = cv2.resize(img_l1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_l2 = cv2.resize(img_l2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            
            flow = flow * [scale_x, scale_y]
            
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
                disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))
                
                img_l1 = img_l1[:, ::-1]
                img_l2 = img_l2[:, ::-1]
                
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img_l1 = img_l1[::-1, :]
                img_l2 = img_l2[::-1, :]
                
                flow = flow[::-1, :] * [1.0, -1.0]                

        y0_r = np.random.randint(0, img_r1.shape[0] - self.crop_size[0])
        x0_r = np.random.randint(0, img_r1.shape[1] - self.crop_size[1])
        y0_l = np.random.randint(0, img_l1.shape[0] - self.crop_size[0])
        x0_l = np.random.randint(0, img_l1.shape[1] - self.crop_size[1])

        img_l1 = img_l1[y0_l:y0_l+self.crop_size[0], x0_l:x0_l+self.crop_size[1]]
        img_l2 = img_l2[y0_l:y0_l+self.crop_size[0], x0_l:x0_l+self.crop_size[1]]
        img_r1 = img_r1[y0_r:y0_r+self.crop_size[0], x0_r:x0_r+self.crop_size[1]]
        img_r2 = img_r2[y0_r:y0_r+self.crop_size[0], x0_r:x0_r+self.crop_size[1]]
        flow = flow[y0_l:y0_l+self.crop_size[0], x0_l:x0_l+self.crop_size[1]]
        disp = disp[y0_r:y0_r+self.crop_size[0], x0_r:x0_r+self.crop_size[1]]
        disp_ch = disp_ch[y0_r:y0_r+self.crop_size[0], x0_r:x0_r+self.crop_size[1]]
        #print(f'flow.shape {flow.shape}')
        #print(f'disp.shape {disp.shape}')
        #print(f'disp_ch.shape {disp_ch.shape}')
        #print('\n')
        #print(f'arra equal check disp {np.array_equal(disp[:,:,0], disp[:,:,1])}')
        #print(f'arra equal check disp_ch {np.array_equal(disp_ch[:,:,0], disp_ch[:,:,1])}')
        #print('\n')
        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

    def __call__(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        img_l1, img_l2, img_r1, img_r2 = self.color_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2 = self.eraser_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch = self.spatial_transform(img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch)

        img_l1 = np.ascontiguousarray(img_l1)
        img_l2 = np.ascontiguousarray(img_l2)
        img_r1 = np.ascontiguousarray(img_r1)
        img_r2 = np.ascontiguousarray(img_r2)
        flow = np.ascontiguousarray(flow)
        disp = np.ascontiguousarray(disp)
        disp_ch = np.ascontiguousarray(disp_ch)

        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img_l1, img_l2, img_r1, img_r2):
        image_stack = np.concatenate([img_l1, img_l2, img_r1, img_r2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img_l1, img_l2, img_r1, img_r2 = np.split(image_stack, 4, axis=0)
        return img_l1, img_l2, img_r1, img_r2

    def eraser_transform(self, img_l1, img_l2, img_r1, img_r2):
        ht, wd = img_l1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_l2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_l2[y0:y0+dy, x0:x0+dx, :] = mean_color

        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r1.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_r1[y0:y0+dy, x0:x0+dx, :] = mean_color

        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_r2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_l1, img_l2, img_r1, img_r2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def resize_sparse_disparity_map(self, disp, valid, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        disp = disp.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        disp0 = disp[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp1 = disp0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp1 = disp1[v]

        disp_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_img[yy, xx] = disp1
        valid_img[yy, xx] = 1

        return disp_img, valid_img
    
    def resize_sparse_disp_change_map(self, disp_ch, valid, fx=1.0, fy=1.0):
        ht, wd = disp_ch.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        disp_ch = disp_ch.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        disp_ch0 = disp_ch[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp_ch1 = disp_ch0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp_ch1 = disp_ch1[v]

        disp_ch_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_ch_img[yy, xx] = disp_ch1
        valid_img[yy, xx] = 1

        return disp_ch_img, valid_img

    def spatial_transform(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc):
        # randomly sample scale

        ht, wd = img_l1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img_l1 = cv2.resize(img_l1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_l2 = cv2.resize(img_l2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r1 = cv2.resize(img_r1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r2 = cv2.resize(img_r2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            disp, valid_d = self.resize_sparse_disparity_map(disp, valid_d, fx=scale_x, fy=scale_y)
            disp_ch, valid_dc = self.resize_sparse_disp_change_map(disp_ch, valid_dc, fx=scale_x, fy=scale_y)


        '''if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img_l1 = img_l1[:, ::-1]
                img_l2 = img_l2[:, ::-1]
                img_r1 = img_r1[:, ::-1]
                img_r2 = img_r2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                disp = disp[:, ::-1] * [-1.0, 1.0]
                disp_ch = disp_ch[:, ::-1] * [-1.0, 1.0]

                valid = valid[:, ::-1]
                valid_d = valid_d[:, ::-1]
                valid_dc = valid_d[:, ::-1]'''

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img_l1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img_l1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img_l1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img_l1.shape[1] - self.crop_size[1])

        img_l1 = img_l1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_l2 = img_l2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r1 = img_r1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r2 = img_r2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp_ch = disp_ch[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_d = valid_d[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_dc = valid_dc[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc


    def __call__(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc):
        
        img_l1, img_l2, img_r1, img_r2 = self.color_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2 = self.eraser_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc = self.spatial_transform(img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc)

        img_l1 = np.ascontiguousarray(img_l1)
        img_l2 = np.ascontiguousarray(img_l2)
        img_r1 = np.ascontiguousarray(img_r1)
        img_r2 = np.ascontiguousarray(img_r2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        disp = np.ascontiguousarray(disp)
        valid_d = np.ascontiguousarray(valid_d)
        disp_ch = np.ascontiguousarray(disp_ch)
        valid_dc = np.ascontiguousarray(valid_dc)

        return img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc



''''
class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img_l1, img_l2, img_r1, img_r2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img_l1 = np.array(self.photo_aug(Image.fromarray(img_l1)), dtype=np.uint8)
            img_l2 = np.array(self.photo_aug(Image.fromarray(img_l2)), dtype=np.uint8)
            img_r1 = np.array(self.photo_aug(Image.fromarray(img_r1)), dtype=np.uint8)
            img_r2 = np.array(self.photo_aug(Image.fromarray(img_r2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img_l1, img_l2, img_r1, img_r2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img_l1, img_l2, img_r1, img_r2 = np.split(image_stack, 4, axis=0)

        return img_l1, img_l2, img_r1, img_r2

    def eraser_transform(self, img_l1, img_l2, img_r1, img_r2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht_l, wd_l = img_l1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_l2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd_l)
                y0 = np.random.randint(0, ht_l)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_l2[y0:y0+dy, x0:x0+dx, :] = mean_color

        ht_r, wd_r = img_r1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd_r)
                y0 = np.random.randint(0, ht_r)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img_r2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_l1, img_l2, img_r1, img_r2

    def spatial_transform(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        # randomly sample scale
        ht, wd = img_l1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        print(f'scales {scale_x, scale_y}')
        
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img_l1 = cv2.resize(img_l1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_l2 = cv2.resize(img_l2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r1 = cv2.resize(img_r1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r2 = cv2.resize(img_r2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp_ch = cv2.resize(disp_ch, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
            disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))

            flow = flow * [scale_x, scale_y]
            disp = disp * [scale_x, scale_y]
            disp_ch = disp_ch * [scale_x, scale_y]
            

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
                disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))
                
                img_l1 = img_l1[:, ::-1]
                img_l2 = img_l2[:, ::-1]
                img_r1 = img_r1[:, ::-1]
                img_r2 = img_r2[:, ::-1]

                flow = flow[:, ::-1] * [-1.0, 1.0]
                disp = disp[:, ::-1] * [-1.0, 1.0]
                disp_ch = disp_ch[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                disp = np.reshape(disp, ( disp.shape[0], disp.shape[1], -1))
                disp_ch = np.reshape(disp_ch, (disp.shape[0], disp.shape[1], -1))
                img_l1 = img_l1[::-1, :]
                img_l2 = img_l2[::-1, :]
                img_r1 = img_r1[::-1, :]
                img_r2 = img_r2[::-1, :]

                flow = flow[::-1, :] * [1.0, -1.0]
                disp = disp[::-1, :] * [1.0, -1.0]
                disp_ch = disp_ch[::-1, :] * [1.0, -1.0]


        y0 = np.random.randint(0, img_r1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img_r1.shape[1] - self.crop_size[1])
    

        img_l1 = img_l1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_l2 = img_l2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r1 = img_r1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r2 = img_r2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp_ch = disp_ch[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        #print(f'arra equal check disp {np.array_equal(disp[:,:,0], disp[:,:,1])}')
        #print(f'arra equal check disp_ch {np.array_equal(disp_ch[:,:,0], disp_ch[:,:,1])}')
        #print('\n')
        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

    def __call__(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch):
        img_l1, img_l2, img_r1, img_r2 = self.color_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2 = self.eraser_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch = self.spatial_transform(img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch)

        img_l1 = np.ascontiguousarray(img_l1)
        img_l2 = np.ascontiguousarray(img_l2)
        img_r1 = np.ascontiguousarray(img_r1)
        img_r2 = np.ascontiguousarray(img_r2)
        flow = np.ascontiguousarray(flow)
        disp = np.ascontiguousarray(flow)
        disp_ch = np.ascontiguousarray(flow)

        return img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img_l1, img_l2, img_r1, img_r2):
        image_stack = np.concatenate([img_l1, img_l2, img_r1, img_r2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img_l1, img_l2, img_r1, img_r2 = np.split(image_stack, 4, axis=0)
        return img_l1, img_l2, img_r1, img_r2

    def eraser_transform(self, img_l1, img_l2, img_r1, img_r2):
        ht_l, wd_l = img_l1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_l2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd_l)
                y0 = np.random.randint(0, ht_l)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_l2[y0:y0+dy, x0:x0+dx, :] = mean_color

        ht_r, wd_r = img_r1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img_r2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd_r)
                y0 = np.random.randint(0, ht_r)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img_r2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img_l1, img_l2, img_r1, img_r2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def resize_sparse_disparity_map(self, disp, valid, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        disp = disp.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        disp0 = disp[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp1 = disp0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp1 = disp1[v]

        disp_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_img[yy, xx] = disp1
        valid_img[yy, xx] = 1

        return disp_img, valid_img
    
    def resize_sparse_disp_change_map(self, disp_ch, valid, fx=1.0, fy=1.0):
        ht, wd = disp_ch.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        disp_ch = disp_ch.reshape(-1, 2).astype(np.float32)


        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        disp_ch0 = disp_ch[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp_ch1 = disp_ch0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp_ch1 = disp_ch1[v]

        disp_ch_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_ch_img[yy, xx] = disp_ch1
        valid_img[yy, xx] = 1

        return disp_ch_img, valid_img

    def spatial_transform(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc):
        # randomly sample scale

        ht, wd = img_l1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img_l1 = cv2.resize(img_l1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_l2 = cv2.resize(img_l2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r1 = cv2.resize(img_r1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img_r2 = cv2.resize(img_r2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            disp, valid_d = self.resize_sparse_disparity_map(disp, valid_d, fx=scale_x, fy=scale_y)
            disp_ch, valid_dc = self.resize_sparse_disp_change_map(disp_ch, valid_dc, fx=scale_x, fy=scale_y)


        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img_l1 = img_l1[:, ::-1]
                img_l2 = img_l2[:, ::-1]
                img_r1 = img_r1[:, ::-1]
                img_r2 = img_r2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                disp = disp[:, ::-1] * [-1.0, 1.0]
                disp_ch = disp_ch[:, ::-1] * [-1.0, 1.0]

                valid = valid[:, ::-1]
                valid_d = valid_d[:, ::-1]
                valid_dc = valid_d[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img_l1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img_l1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img_l1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img_l1.shape[1] - self.crop_size[1])

        img_l1 = img_l1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_l2 = img_l2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r1 = img_r1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img_r2 = img_r2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp_ch = disp_ch[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]


        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_d = valid_d[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid_dc = valid_dc[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc


    def __call__(self, img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc):
        
        img_l1, img_l2, img_r1, img_r2 = self.color_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2 = self.eraser_transform(img_l1, img_l2, img_r1, img_r2)
        img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc = self.spatial_transform(img_l1, img_l2, img_r1, img_r2, flow, disp, disp_ch, valid, valid_d, valid_dc)

        img_l1 = np.ascontiguousarray(img_l1)
        img_l2 = np.ascontiguousarray(img_l2)
        img_r1 = np.ascontiguousarray(img_r1)
        img_r2 = np.ascontiguousarray(img_r2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        disp = np.ascontiguousarray(disp)
        valid_d = np.ascontiguousarray(valid_d)
        disp_ch = np.ascontiguousarray(disp_ch)
        valid_dc = np.ascontiguousarray(valid_dc)

        return img_l1, img_l2, img_r1, img_r2, flow, valid, disp, valid_d, disp_ch, valid_dc'''
