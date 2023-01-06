"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf  # {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1.0]}
        self.data_aug_conf = data_aug_conf  # {'resize_lim': (0.193, 0.225), 'final_dim': (128, 352), 'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600, 'rand_flip': True, 'bot_pct_lim': (0.0, 0.22), 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], 'Ncams': 5}

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)  # 探测范围网格微分量 tensor([ 0.5000,  0.5000, 20.0000])
        self.bx = nn.Parameter(bx, requires_grad=False)  # 探测范围网格起始格子中心位置 tensor([-49.7500, -49.7500,   0.0000])
        self.nx = nn.Parameter(nx, requires_grad=False)  # 探测范围网格数目 tensor([200, 200,   1])

        self.downsample = 16  # 下采样率
        self.camC = 64  # 多相机特征输出通道数总和
        self.frustum = self.create_frustum()  # torch.Size([41, 8, 22, 3])
        self.D, _, _, _ = self.frustum.shape  # 41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        """
        像素坐标下的3D格子
        """
        # 返回一个记录了坐标的4维数组，并且是等间距的，三个元素分别表示该网格的位置：宽-高-深度，x-y-d
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 128, 352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # (8, 22)
        # self.grid_conf['dbound'] = [4.0, 45.0, 1.0] 探测范围4-45米，间隔1米
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # torch.Size([41, 8, 22])
        D, _, _ = ds.shape  # 41
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # torch.Size([41, 8, 22])
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # torch.Size([41, 8, 22])

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # torch.Size([41, 8, 22, 3]) 相当于为相机图像创建了一个带x,y,d坐标的长方体
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        生成像素坐标和车体坐标的映射
        """
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # torch.Size([4, 5, 3])

        # undo post-transformation
        # B x N x D x H x W x 3
        # self.frustum.shape: torch.Size([41, 8, 22, 3])
        # post_trans.shape: torch.Size([4, 5, 3])
        # post_trans.view(B, N, 1, 1, 1, 3).shape: torch.Size([4, 5, 1, 1, 1, 3])
        # 根据resize、crop情况平移坐标
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # torch.Size([4, 5, 41, 8, 22, 3]) 一个形状为[4, 5, 41, 8, 22]的高维体，每个元素有3个元素组成的位置坐标
        # torch.inverse(post_rots).shape: torch.Size([4, 5, 3, 3])
        # torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).shape: torch.Size([4, 5, 1, 1, 1, 3, 3])
        # points.unsqueeze(-1).shape: torch.Size([4, 5, 41, 8, 22, 3, 1])
        # 根据数据增强的随机rotate（本文实现用了一个均值作为定值）、resize、crop情况旋转坐标
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)) 
        # torch.Size([4, 5, 41, 8, 22, 3, 1])

        # points[:, :, :, :, :, :2].shape: torch.Size([4, 5, 41, 8, 22, 2, 1])
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # 前面两位为宽和高。由于当前points中的x,y坐标是像素坐标（深度是相机坐标，单位m），为了转换为相机坐标，故需根据深度来线性缩放x,y。
                            points[:, :, :, :, :, 2:3]  # 深度初始化时就是相机坐标，单位m，故不需要转换
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))  # 外参和内参旋转矩阵进行矩阵乘得到总的旋转矩阵  torch.Size([4, 5, 41, 8, 22, 3, 1])
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # 根据总的旋转矩阵旋转，将相机坐标下的高维立方体（视锥）转换到车体坐标
        points += trans.view(B, N, 1, 1, 1, 3)  # 根据外参平移

        return points  # torch.Size([4, 5, 41, 8, 22, 3])

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        删减特征+图像特征从像素坐标变换到BEV下
        根据相机的内外参等几何信息获得的像素-车体坐标映射，以及相机的图像特征，去除掉一些冗余的图像特征和映射点后，定义一个BEV下的全0特征，将图像特征根据映射放置（赋值）到BEV下
        """
        B, N, D, H, W, C = x.shape  # 4, 5, 41, 8, 22, 64
        Nprime = B*N*D*H*W  # 144320

        # flatten x
        x = x.reshape(Nprime, C)  # torch.Size([144320, 64])

        # flatten indices  # geom_feats 变成以格子索引为坐标，原来是以米为单位，现在为索引值，无单位
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # torch.Size([4, 5, 41, 8, 22, 3])  # 将几何特征的坐标和车体坐标对齐，原来都是正值，现在一半为正
        # / self.dx 则是从meter 变为无量纲，也就是格子（格子大小为0.5m x 0.5m x 20m），.long()使得所有值变为整数
        geom_feats = geom_feats.view(Nprime, 3)  # torch.Size([144320, 3])
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,  # torch.Size([144320, 1])  # 每个样本首尾相接，用0,1,2,3来代表不同样本
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # torch.Size([144320, 4])  # 增加一个通道，用于指示属于哪个样本

        # filter out points that are outside box  # torch.Size([144320])  # 除去探测范围格子以外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])   # self.nx = tensor([200, 200,   1])
        x = x[kept]  # torch.Size([137828, 64])
        geom_feats = geom_feats[kept]  # torch.Size([137828, 4])

        """
        * 100 * 4
        * 4
        * 4
        * 1
        """
        # get tensors from the same voxel next to each other  # torch.Size([137828])  # 按照x、y、z、样本这四个的加权和的小大来定义顺序
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()  # torch.Size([137828])  # 从大到小编号（索引值）
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 根据索引值，重新排列，大的值在前
        # x.shpae: torch.Size([137828, 64]), geom_feats.shape: torch.Size([137828, 4]), ranks.shape: torch.Size([137828])
        # cumsum trick
        if not self.use_quickcumsum:  # 为了中心区域的点或特征不过于密集，去掉一些点
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # torch.Size([41382, 64]), torch.Size([41382, 4])

        # griddify 网格化 (B x C x Z x X x Y)  # 把特征放到BEV坐标下的网格里
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # torch.Size([4, 64, 1, 200, 200])
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # torch.Size([41382, 64])
        # final.shape: torch.Size([4, 64, 1, 200, 200])

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        # final.shape: torch.Size([4, 64, 200, 200])

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 获得几何映射关系
        # geom.shape: torch.Size([4, 5, 41, 8, 22, 3])
        x = self.get_cam_feats(x)  # 获得相机特征
        # x.shape: torch.Size([4, 5, 41, 8, 22, 64])
        x = self.voxel_pooling(geom, x)  # 将几何映射关系和图像特征转换为BEV下的特征
        # x.shape: torch.Size([4, 64, 200, 200])
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        x.shape: torch.Size([4, 5, 3, 128, 352])
        rots.shape: torch.Size([4, 5, 3, 3])
        trans.shape: torch.Size([4, 5, 3])
        intrins.shape: torch.Size([4, 5, 3, 3])
        post_rots.shape: torch.Size([4, 5, 3, 3])
        post_trans.shape: torch.Size([4, 5, 3])
        """
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 获得BEV下的特征
        # x.shape: torch.Size([4, 64, 200, 200])
        x = self.bevencode(x)  # 获得BEV下的预测
        # x.shape: torch.Size([4, 1, 200, 200])
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
