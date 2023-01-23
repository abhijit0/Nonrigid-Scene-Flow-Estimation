import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
import numpy as np

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap_l1, fmap_l2, fmap_r1, fmap_r2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid_l1_l2 = []
        self.corr_pyramid_l1_r1 = []
        self.corr_pyramid_l1_r2 = []

        # all pairs correlation
        corr_l1_l2, corr_l1_r1, corr_l1_r2 = CorrBlock.corr(fmap_l1, fmap_l2, fmap_r1, fmap_r2)

        batch_l1_l2, h1_l1_l2, w1_l1_l2, dim_l1_l2, h2_l1_l2, w2_l1_l2 = corr_l1_l2.shape
        corr_l1_l2 = corr_l1_l2.reshape(batch_l1_l2*h1_l1_l2*w1_l1_l2, dim_l1_l2, h2_l1_l2, w2_l1_l2)

        batch_l1_r1, h1_l1_r1, w1_l1_r1, dim_l1_r1, h2_l1_r1, w2_l1_r1 = corr_l1_r1.shape
        corr_l1_r1 = corr_l1_r1.reshape(batch_l1_r1*h1_l1_r1*w1_l1_r1, dim_l1_r1, h2_l1_r1, w2_l1_r1)

        batch_l1_r2, h1_l1_r2, w1_l1_r2, dim_l1_r2, h2_l1_r2, w2_l1_r2 = corr_l1_r2.shape
        corr_l1_r2 = corr_l1_r2.reshape(batch_l1_r2*h1_l1_r2*w1_l1_r2, dim_l1_r2, h2_l1_r2, w2_l1_r2)
        
        self.corr_pyramid_l1_l2.append(corr_l1_l2)
        self.corr_pyramid_l1_r1.append(corr_l1_r1)
        self.corr_pyramid_l1_r2.append(corr_l1_r2)

        for i in range(self.num_levels-1):
            corr_l1_l2 = F.avg_pool2d(corr_l1_l2, 2, stride=2)
            self.corr_pyramid_l1_l2.append(corr_l1_l2)

            corr_l1_r1 = F.avg_pool2d(corr_l1_r1, 2, stride=2)
            self.corr_pyramid_l1_r1.append(corr_l1_r1)

            corr_l1_r2 = F.avg_pool2d(corr_l1_r2, 2, stride=2)
            self.corr_pyramid_l1_r2.append(corr_l1_r2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch_l1_l2, h1_l1_l2, w1_l1_l2, _ = coords_l1_l2.shape
        batch, h1, w1, _ = coords.shape
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap_l1, fmap_l2, fmap_r1, fmap_r2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap_l1, fmap_l2, fmap_r1, fmap_r2)]
        for i in range(self.num_levels):
            fmap_l1 = F.avg_pool2d(fmap_l1, 2, stride=2)
            fmap_l2 = F.avg_pool2d(fmap_l2, 2, stride=2)
            fmap_r1 = F.avg_pool2d(fmap_r1, 2, stride=2)
            fmap_r2 = F.avg_pool2d(fmap_r2, 2, stride=2)
            self.pyramid.append((fmap_l1, fmap_l2, fmap_r1, fmap_r2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list_l1_l2 = []
        corr_list_l1_r1 = []
        corr_list_l1_r2 = []
        for i in range(self.num_levels):
            r = self.radius
            fmap_l1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap_l2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()
            fmap_r1_i = self.pyramid[i][2].permute(0, 2, 3, 1).contiguous()
            fmap_r2_i = self.pyramid[i][3].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr_l1_l2, = alt_cuda_corr.forward(fmap_l1_i, fmap_l2_i, coords_i, r)
            corr_l1_r1, = alt_cuda_corr.forward(fmap_l1_i, fmap_r1_i, coords_i, r)
            corr_l1_r2, = alt_cuda_corr.forward(fmap_l1_i, fmap_r2_i, coords_i, r)

            corr_list_l1_l2.append(corr_l1_l2.squeeze(1))
            corr_list_l1_r1.append(corr_l1_r1.squeeze(1))
            corr_list_l1_r2.append(corr_l1_r2.squeeze(1))

        corr_list_l1_l2 = torch.stack(corr_list_l1_l2, dim=1)
        corr_list_l1_l2 = corr_list_l1_l2.reshape(B, -1, H, W)
        
        corr_list_l1_r1 = torch.stack(corr_list_l1_r1, dim=1)
        corr_list_l1_r1 = corr_list_l1_r1.reshape(B, -1, H, W)
        
        corr_list_l1_r2 = torch.stack(corr_list_l1_r2, dim=1)
        corr_list_l1_r2 = corr_list_l1_r2.reshape(B, -1, H, W)

        corr_list_l1_l2 = corr_list_l1_l2 / torch.sqrt(torch.tensor(dim).float())
        corr_list_l1_r1 = corr_list_l1_r1 / torch.sqrt(torch.tensor(dim).float())
        corr_list_l1_r2 = corr_list_l1_r2 / torch.sqrt(torch.tensor(dim).float())

        return corr_list_l1_l2, corr_list_l1_r1, corr_list_l1_r2
