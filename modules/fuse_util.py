# -*-coding:utf-8-*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time



LEAKY_RATE = 0.1
use_bn = False


def Grid_generator(w, h, merged_channels=True):
    """
    Generate the grid for the image plane
    W = 224
    H = 224
    """
    grid_x, grid_y = torch.meshgrid(torch.linspace(0.5, w - 0.5, w),
                                    torch.linspace(0.5, h - 0.5, h))
    if merged_channels:
        xy = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)),
                       1).unsqueeze(0)
    else:
        xy = torch.cat((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)),
                       1).unsqueeze(0).reshape(-1, w, h, 2)

    return xy


class Conv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_leaky=True,
                 bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(
            LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(), relu)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=[1, 1],
                 bn=False,
                 activation_fn=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels)

        if activation_fn:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x (b,n,s,c)
        x = x.permute(0, 3, 2, 1)  #(b,c,s,n)

        outputs = self.conv(x)

        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)

        return outputs


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm?    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists,
                              nsample,
                              dim=-1,
                              largest=False,
                              sorted=False)
    return group_idx

def index_points_group(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, S]
    Return:
        new_points:, indexed points data, [B, N, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def grouping(feature, K, src_xyz, q_xyz, use_xyz=False):
    '''
    Input:
        feature: (batch_size, ndataset, c)
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
    Return:
        grouped_xyz: (batch_size, npoint, K,3)
        xyz_diff: (batch_size, npoint,K, 3)
        new_points: (batch_size, npoint,K, c+3) if use_xyz else (batch_size, npoint,K, c)
        point_indices: (batch_size, npoint, K)
    '''

    q_xyz  = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()
    B, N, C = src_xyz.shape
    if K > N:
        K = N

    point_indices = knn_point(K,src_xyz,q_xyz)   # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz,point_indices)   # (batch_size, npoint, K,3)
    
    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  #  (batch_size, npoint,K, 3)
     
    grouped_feature = index_points_group(feature, point_indices) #(batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  #(batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices


class fusion2to1(nn.Module):
    def __init__(self, appear_len, point_len, out_channels):
        super(fusion2to1, self).__init__()

        self.appear_len = appear_len
        self.point_len = point_len

        self.gate_p = nn.Sequential(
            nn.Conv2d(point_len, point_len, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_i = nn.Sequential(
            nn.Conv2d(appear_len, appear_len, 1, 1),
            nn.Sigmoid(),
        )

        self.input_p = nn.Sequential(nn.Conv2d(point_len, out_channels, 1, 1))
        self.input_i = nn.Sequential(nn.Conv2d(appear_len, out_channels, 1, 1))

    def forward(self, raw, group):
        # raw:      b, npoints, sample, c
        # group:    b, npoints, sample, c

        # conv2d x = x.permute(0,3,2,1) #(b,c,s,n)
        raw = raw.permute(0, 3, 2, 1).contiguous()  #(b,c,s,n)
        group = group.permute(0, 3, 2, 1).contiguous() #(b,c,s,n)

        gate_p = self.gate_p(group)  #(b,c,s,n)
        gate_i = self.gate_i(raw)  #(b,c,s,n)

        obj_fused = gate_p.mul(self.input_p(group)) + gate_i.mul(
            self.input_i(raw))  #(b,c,s,n)
        obj_feats = obj_fused.div(gate_p + gate_i)

        obj_feats = obj_feats.permute(0, 3, 2, 1).contiguous()  # b, n, s, c
        return obj_feats


class CostVolume(nn.Module):
    def __init__(self,
                 radius,
                 nsample,
                 nsample_q,
                 mlp1,
                 mlp2,
                 appear_len,
                 point_len,
                 out_channels,
                 is_training,
                 bn_decay,
                 bn=True,
                 pooling='max',
                 knn=True,
                 corr_func='elementwise_product'):
        super(CostVolume, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = appear_len + point_len + 7
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func

        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_new = nn.ModuleList()

        self.gated_fuse = fusion2to1(appear_len, point_len, mlp1[-1])

        self.pi_encoding = Conv2d(7, mlp1[-1], [1, 1], stride=[1, 1], bn=True)
        self.pc_encoding = Conv2d(7, mlp1[-1], [1,1],stride=[1,1],bn=True)

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(
                Conv2d(self.in_channels,
                       num_out_channel, [1, 1],
                       stride=[1, 1],
                       bn=True))
            self.in_channels = num_out_channel

        self.in_channels = 2 * mlp1[-1] + appear_len
        for j,num_out_channel in enumerate(mlp2):
            self.mlp2_convs_new.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1],bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points):
        '''
            Input:
                warped_xyz: (b,npoint,3)
                warped_points: (b,npoint,c)
                f2_xyz:  (b,ndataset,3)
                f2_points: (b,ndataset,c)

            Output:
                pc_feat1_new: batch_size, npoints, mlp2[-1]
            '''

        qi_xyz_grouped, _, qi_points_grouped, idx = grouping(
            f2_points, self.nsample_q, f2_xyz, warped_xyz
        )  #(b,npoint,nsample_q,2) (b,npoint,nsample_q,2) (b,npoint,nsample_q,c)
        B, N, S, _ = qi_xyz_grouped.shape

        pi_xyz_expanded = (torch.unsqueeze(warped_xyz, 2)).repeat(
            [1, 1, S, 1])  # batch_size, npoints, nsample_q, 2
        pi_points_expanded = (torch.unsqueeze(warped_points, 2)).repeat(
            [1, 1, S, 1])  # batch_size, npoints, nsample, c

        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded  # batch_size, npoints, nsample_q, 2

        pi_euc_diff = torch.sqrt(
            torch.sum(
                torch.mul(pi_xyz_diff, pi_xyz_diff), dim=-1, keepdim=True) +
            1e-20)  # batch_size, npoints, nsample_q, 1

        pi_xyz_diff_concat = torch.cat(
            [pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff],
            dim=3)  # batch_size, npoints, nsample_q,7

        pi_feat1_new = self.gated_fuse(
            pi_points_expanded,
            qi_points_grouped)  # batch_size, npoints, nsample_q, mlp[-1]

        pi_xyz_encoding = self.pi_encoding(
            pi_xyz_diff_concat)  # batch_size, npoints, nsample_q,mlp1[-1]

        pi_concat = torch.cat(
            [pi_xyz_encoding, pi_feat1_new],
            dim=3)  # batch_size, npoints, nsample_q,2*mlp1[-1]

        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(
                pi_concat)  # batch_size, npoints, nsample_q,mlp2[-1]

        WQ = F.softmax(pi_concat, dim=2)

        pi_feat1_new = WQ * pi_feat1_new  #mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2,
                                 keepdim=False)  # batch_size, npoint, mlp1[-1]

        return pi_feat1_new
        '''
        # second attentive flow embedding
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(
            pi_feat1_new, self.nsample, warped_xyz, warped_xyz
        )  #(b,npoint,nsample,3) (b,npoint,nsample,3) (b,npoint,nsample,mlp1[-1])

        B, N, S, _ = pc_xyz_grouped.shape
        pc_xyz_new = (torch.unsqueeze(warped_xyz, dim=2)).repeat(
            [1, 1, S, 1])  # batch_size, npoints, nsample, 3
        pc_points_new = (torch.unsqueeze(warped_points, dim=2)).repeat(
            [1, 1, S, 1])  # batch_size, npoints, nsample, c

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # batch_size, npoints, nsample, 3

        pc_euc_diff = torch.sqrt(
            torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3, keepdim=True)
            + 1e-20)  # batch_size, npoints, nsample, 1

        pc_xyz_diff_concat = torch.cat(
            [pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
            dim=3)  # batch_size, npoints, nsample, 7

        pc_xyz_encoding = self.pc_encoding(
            pc_xyz_diff_concat)  # batch_size, npoints, nsample, mlp1[-1]

        pc_concat = torch.cat(
            [pc_xyz_encoding, pc_points_new, pc_points_grouped],
            dim=-1)  # batch_size, npoints, nsample, mlp[-1]+3+mlp[-1]

        for j, conv in enumerate(self.mlp2_convs_new):
            pc_concat = conv(
                pc_concat)  # batch_size, npoints, nsample, mlp2[-1]

        WP = F.softmax(pc_concat, dim=2)

        pc_feat1_new = WP * pc_points_grouped  # batch_size, npoints, nsample, mlp2[-1]

        pc_feat1_new = torch.sum(
            pc_feat1_new, dim=2,
            keepdim=False)  # batch_size, npoints, mlp2[-1]

        return pc_feat1_new
        '''


class fuse_extract(nn.Module):
    def __init__(self, nsample, nsample_q, mlp1, mlp2, apper_len, point_len,
                 out_channels):
        super(fuse_extract, self).__init__()
        self.cost_points_centered = CostVolume(radius=None,
                                               nsample = nsample,
                                               nsample_q=nsample_q,
                                               mlp1=mlp1,
                                               mlp2=mlp2,
                                               appear_len=apper_len,
                                               point_len=point_len,
                                               out_channels=out_channels,
                                               is_training=True,
                                               bn_decay=None,
                                               bn=True,
                                               pooling='max',
                                               knn=True,
                                               corr_func='concat')
        self.cost_pixels_centered = CostVolume(radius=None,
                                               nsample = nsample,
                                               nsample_q=nsample_q,
                                               mlp1=mlp1,
                                               mlp2=mlp2,
                                               appear_len=apper_len,
                                               point_len=point_len,
                                               out_channels=out_channels,
                                               is_training=True,
                                               bn_decay=None,
                                               bn=True,
                                               pooling='max',
                                               knn=True,
                                               corr_func='concat')


    def points_centered_sample_group(self, projected_xy, points, img,
                                     M_matrix):

        Bi, C, H, W = img.shape # 1 x C x H x W

        img_d = img.permute(0, 2, 3, 1).reshape(Bi, -1, C)  
        #[1, C, H, W] -> [1, H, W, C] -> [1, H*W, C]
        Bp, D, L = points.shape  # [1, D, L]

        z_cord = torch.ones(1, 1, L).cuda()
        #projected_xy:the original postion on the full image plane 1 x 2 x L

        projected_xyz = torch.cat([projected_xy, z_cord], dim=1)  # 1 x 3 x L

        projected_on_img_points = torch.bmm(M_matrix, projected_xyz)  #1 x 3 x L
        projected_on_img_points = projected_on_img_points[:, :2, :]  #1 x 2 x L
        projected_on_img_points = projected_on_img_points.permute(0, 2, 1)  #1 x L x 2

        img_xy = Grid_generator(W, H).cuda()  #[1, W * H, 2]

        points = points.permute(0, 2, 1)  # [1, L, D]

        out = self.cost_points_centered(projected_on_img_points, points, img_xy, img_d)  # [1, L, D]
        out = out.permute(0, 2, 1)  # 1, D, L

        return out

    def pixels_centered_sample_group(self, projected_xy, points, img,
                                     M_matrix):

        Bi, C, H, W = img.shape
        img_d = img.permute(0, 2, 3, 1).reshape(Bi, -1, C)  
        #[1, C, H, W] -> [1, H, W, C] -> [1, H * W, C]
        Bp, D, L = points.shape  # 1xDxL

        z_cord = torch.ones(1, 1, L).cuda()
        projected_xyz = torch.cat([projected_xy, z_cord], dim=1)  #1 x 3 x L

        projected_on_img_points = torch.bmm(M_matrix, projected_xyz)  #1 x 3 x L
        projected_on_img_points = projected_on_img_points[:, :2, :]  # 1 x 2 x L
        projected_on_img_points = projected_on_img_points.permute(0, 2, 1)  #1 x L x 2

        img_xy = Grid_generator(W, H).cuda()  #[1, W * H, 2]
        points = points.permute(0, 2, 1)  #1 x L x D
        out = self.cost_pixels_centered(img_xy, img_d, projected_on_img_points, points) #[1, H * W, C]
        out = out.permute(0, 2, 1) #[1, H * W, C] -> [1, C, H * W]
        out = out.reshape(Bi, -1, H, W)  # [B, C', H, W]

        return out

    def condense_net(self, xy, points, img, M_matrix):

        feat1 = self.points_centered_sample_group(xy, points, img, M_matrix)

        feat2 = self.pixels_centered_sample_group(xy, points, img, M_matrix)

        return feat1, feat2

    def forward(self, xys, imgs, points, points_split, M_matrixes):
        """
        xys: points position on the image plane 1 x 2 x L
        imgs: (N + M) x C x W x H
        M_matrixes: (N + M) x 3 x 3
        """
        # divide the imgs into N+M tensors of the same size(1 x C x W x H)
        imgs_list = list(imgs.split(split_size=1, dim=0))
        # M_matrixes: (N + M) x 3 x 3
        M_matrix_list = list(M_matrixes.split(split_size=1, dim=0))
        points_list = []
        xy_list = []
        for i in range(len(points_split) - 1):
            start = points_split[i].item()
            end = points_split[i + 1].item()
            points_list.append(points[:, :, start:end])
            xy_list.append(xys[:, :, start:end])

        fused_img_list = []
        fused_points_list = []
        for (xy, img, pts, M_matrix) in zip(xy_list, imgs_list, points_list, M_matrix_list):
            fused_points_tensor, fused_img_tensor = self.condense_net(
                xy, pts, img, M_matrix)  #[1,D,L'] [1, C, H, W]
            fused_points_list.append(fused_points_tensor)
            fused_img_list.append(fused_img_tensor)

        fused_imgs = torch.cat(fused_img_list, dim=0)  # [N+M, C, H, W]
        fused_points = torch.cat(fused_points_list, dim=-1)  #[1, D, L]

        return fused_imgs, fused_points
