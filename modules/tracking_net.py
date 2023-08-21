from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_net import *  # noqa
from .gcn import affinity_module
from .new_end import *  # noqa
from .score_net import *  # noqa
from .appear_net_layers import parse_appearance_layers
from .point_net_layers import parse_pointnet_layers
from .fuse_util import fuse_extract


class TrackingNet(nn.Module):
    def __init__(self,
                 seq_len,
                 appear_len=512,
                 appear_skippool=False,
                 appear_fpn=False,
                 score_arch='vgg',
                 score_fusion_arch='C',
                 appear_arch='vgg',
                 point_arch='v1',
                 point_len=512,
                 softmax_mode='single',
                 test_mode=0,
                 affinity_op='multiply',
                 dropblock=5,
                 end_arch='v2',
                 end_mode='avg',
                 without_reflectivity=True,
                 neg_threshold=0,
                 use_dropout=False):
        super(TrackingNet, self).__init__()
        self.seq_len = seq_len
        self.score_arch = score_arch
        self.neg_threshold = neg_threshold
        self.test_mode = test_mode  # 0:image;1:image;2:fusion
        point_in_channels = 4 - int(without_reflectivity)

        if point_len == 0:
            in_channels = appear_len
        else:
            in_channels = point_len

        self.fusion_module = None
        fusion = eval(f"fusion_module_{score_fusion_arch}")
        self.fusion_module = fusion(appear_len,
                                    point_len,
                                    out_channels=point_len)

        if appear_len == 0:
            print('No image appearance used')
            self.appearance_conv = None
            self.appearance_pool = None
        else:
            self.appearance_conv, self.appearance_pool = parse_appearance_layers(
                appear_arch,
                appear_len,
                skippool=appear_skippool,
                fpn=appear_fpn,
                dropblock=dropblock)

        # build new end indicator
        if end_arch in ['v1', 'v2']:
            new_end = partial(eval("NewEndIndicator_%s" % end_arch),
                              kernel_size=5,
                              reduction=4,
                              mode=end_mode)

        # build point net
        if point_len == 0:
            print("No point cloud used")
            self.stn_layers = None
            self.point_net_layers = None
        elif point_arch in ['v1', 'v2']:
            self.stn_layers, self.point_net_layers = parse_pointnet_layers(
                point_in_channels,
                out_channels=point_len,
                use_dropout=use_dropout)
        else:
            print("Not implemented!!")

        # build affinity matrix module
        assert in_channels != 0
        self.w_link = affinity_module(in_channels,
                                      new_end=new_end,
                                      affinity_op=affinity_op)

        # build negative rejection module
        if score_arch in ['branch_cls', 'branch_reg']:
            self.w_det = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1, 1),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels // 2, 1, 1),
                nn.BatchNorm1d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels // 2, 1, 1, 1),
            )
        else:
            print("Not implemented yet")

        self.softmax_mode = softmax_mode

        nsample = 1
        nsample_q = 1
        self.fuse_net1 = fuse_extract(nsample,
                                      nsample_q,
                                      mlp1=[128],
                                      mlp2=[256, 128],
                                      apper_len=128,
                                      point_len=128,
                                      out_channels=128)
        self.fuse_net2 = fuse_extract(nsample,
                                      nsample_q,
                                      mlp1=[256],
                                      mlp2=[512, 256],
                                      apper_len=256,
                                      point_len=256,
                                      out_channels=256)
        self.fuse_net3 = fuse_extract(nsample,
                                      nsample_q,
                                      mlp1=[512],
                                      mlp2=[1024, 512],
                                      apper_len=512,
                                      point_len=512,
                                      out_channels=512)
        self.fuse_net4 = fuse_extract(nsample,
                                      nsample_q,
                                      mlp1=[512],
                                      mlp2=[1024, 512],
                                      apper_len=512,
                                      point_len=512,
                                      out_channels=512)

    def associate(self, objs, dets):
        link_mat, new_score, end_score = self.w_link(objs, dets)

        if self.softmax_mode == 'single':
            link_score = F.softmax(link_mat, dim=-1)
        elif self.softmax_mode == 'dual':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = link_score_prev.mul(link_score_next)
        elif self.softmax_mode == 'dual_add':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = (link_score_prev + link_score_next) / 2
        elif self.softmax_mode == 'dual_max':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = torch.max(link_score_prev, link_score_next)
        else:
            link_score = link_mat

        return link_score, new_score, end_score

    def feature(self, dets, det_info):
        feats = []

        #layers setting
        appearance_conv1 = self.appearance_conv[0]
        appearance_conv2 = self.appearance_conv[1]
        appearance_conv3 = self.appearance_conv[2]
        appearance_conv4 = self.appearance_conv[3]

        stn1 = self.stn_layers[0]
        stn2 = self.stn_layers[1]
        pointnet_sa1 = self.point_net_layers[0]
        pointnet_sa2 = self.point_net_layers[1]
        pointnet_sa3 = self.point_net_layers[2]
        pointnet_sa4 = self.point_net_layers[3]
        point_conv1 = self.point_net_layers[4]
        point_conv2 = self.point_net_layers[5]
        avg_pool = self.point_net_layers[6]

        appear_pool_out = []
        trans = []

        points0 = det_info['points'].transpose(-1, -2)  # 1 x D x L
        points_split0 = det_info['points_split'].long().squeeze(0)
        #point_xy is the position of the points projected on the image plane
        points_xy0 = det_info['points_xy'].transpose(-1, -2)  # 1 x 2 x L
        points_xyz0 = points0[:, :3, :]

        trans1 = stn1(points_xyz0)  #1x3x3
        trans.append(trans1)
        points_xyz0 = points_xyz0.transpose(2, 1)  #
        points_xyz0 = torch.bmm(points_xyz0, trans1)
        points_xyz0 = points_xyz0.transpose(2, 1)  #1x3xL
        points0 = points0.transpose(2, 1)
        points0 = torch.bmm(points0, trans1)
        points0 = points0.transpose(2, 1)

        M_matrixes0 = det_info['M_matrixes'].squeeze(0)  # N+M(6) x 3 x 3

        M_matrixes1 = torch.cat([torch.div(M_matrixes0[:, :2, :], 4), M_matrixes0[:, 2:, :]],dim=1)
        appear1 = appearance_conv1(dets)

        # print("appear1.shape",appear1.shape) 6 x 128 x 56 x 56
        points_xy1, points_xyz1, points1, points_split1 = pointnet_sa1(
            points_xy0, points_xyz0, points0, points_split0)
        appear1, points1 = self.fuse_net1(points_xy1, appear1, points1, points_split1, M_matrixes1)
        appear_pool_out.append(appear1)
        trans2 = stn2(points1)
        trans.append(trans2)
        points1 = points1.transpose(2, 1)
        points1 = torch.bmm(points1, trans2)
        points1 = points1.transpose(2, 1)

        # 2nd layer
        M_matrixes2 = torch.cat([torch.div(M_matrixes1[:, :2, :], 2), M_matrixes1[:, 2:, :]],dim=1)
        appear2 = appearance_conv2(appear1)
        points_xy2, points_xyz2, points2, points_split2 = pointnet_sa2(
            points_xy1, points_xyz1, points1, points_split1)
        appear2, points2 = self.fuse_net2(points_xy2, appear2, points2, points_split2, M_matrixes2)
        appear_pool_out.append(appear2)

        # 3rd layer
        M_matrixes3 = torch.cat([torch.div(M_matrixes2[:, :2, :], 2), M_matrixes2[:, 2:, :]],dim=1)
        appear3 = appearance_conv3(appear2)
        points_xy3, points_xyz3, points3, points_split3 = pointnet_sa3(
            points_xy2, points_xyz2, points2, points_split2)
        appear3, points3 = self.fuse_net3(points_xy3, appear3, points3, points_split3, M_matrixes3)
        appear_pool_out.append(appear3)

        # 4th layer
        M_matrixes4 = torch.cat(
            [torch.div(M_matrixes3[:, :2, :], 2), M_matrixes3[:, 2:, :]],
            dim=1)
        appear4 = appearance_conv4(appear3)
        points_xy4, points_xyz4, points4, points_split4 = pointnet_sa4(
            points_xy3, points_xyz3, points3, points_split3)
        appear4, points4 = self.fuse_net4(points_xy4, appear4, points4, points_split4, M_matrixes4)
        appear_pool_out.append(appear4)

        appear_skippool_out = []
        for layer, feat in zip(self.appearance_pool, appear_pool_out):
            appear_skippool_out.append(layer(feat))

        appear_feats = torch.cat(appear_skippool_out, dim=-1)

        points4 = point_conv1(points4)
        points_out = []
        for i in range(len(points_split4) - 1):
            start = points_split4[i].item()
            end = points_split4[i + 1].item()
            points_pool = avg_pool(points4[:, :, start:end])
            points_out.append(points_pool.view(-1, 512, 1))

        points_out = torch.cat(points_out, dim=-1)
        points_feats = point_conv2(points_out).transpose(-1, -2).squeeze(0)

        # dimesion needs to be considered
        feats = [appear_feats, points_feats]
        feats = torch.cat(feats, dim=-1).t().unsqueeze(0)  # LxD->1xDxL
        if self.fusion_module is not None:
            feats = self.fusion_module(feats)
            return feats, trans

        return feats, trans

    def determine_det(self, dets, feats):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        if not self.training:
            # add mask
            if 'cls' in self.score_arch:
                det_scores = det_scores.sigmoid()

            mask = det_scores.lt(self.neg_threshold)
            det_scores -= mask.float()
        return det_scores

    def forward(self, dets, det_info, dets_split):
        feats, trans = self.feature(dets, det_info)
        det_scores = self.determine_det(dets, feats)

        start = 0
        link_scores = []
        new_scores = []
        end_scores = []
        for i in range(len(dets_split) - 1):
            prev_end = start + dets_split[i].item()
            end = prev_end + dets_split[i + 1].item()
            link_score, new_score, end_score = self.associate(
                feats[:, :, start:prev_end], feats[:, :, prev_end:end])
            link_scores.append(link_score.squeeze(1))
            new_scores.append(new_score)
            end_scores.append(end_score)
            start = prev_end

        if not self.training:
            fake_new = det_scores.new_zeros(
                (det_scores.size(0), link_scores[0].size(-2)))
            fake_end = det_scores.new_zeros(
                (det_scores.size(0), link_scores[-1].size(-1)))
            new_scores = torch.cat([fake_new] + new_scores, dim=1)
            end_scores = torch.cat(end_scores + [fake_end], dim=1)
        else:
            new_scores = torch.cat(new_scores, dim=1)
            end_scores = torch.cat(end_scores, dim=1)
        return det_scores, link_scores, new_scores, end_scores, trans
