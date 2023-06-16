"""
PointConv util functions
Author: Wenxuan Wu
Date: May 2020
"""

import torch 
import torch.nn as nn 

import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
from pointnet2 import pointnet2_utils
import core.registration as GlobalRegistration

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
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
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
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
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def knn_point_nsample(xyz, new_xyz):
    """
    Input:
        xyz: all points, [B, N, nsample, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, 1]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, 1, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

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

def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)
        xyz = xyz.contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx

class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N1

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost

class PointConvCorrespondences(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvCorrespondences, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # self.weightnet1 = WeightNet(3, last_channel)
        # self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1) # B, N1, C
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        feature1 = torch.cat([points1, xyz1], dim=-1) # B, N1, C+D1
        feature2 = torch.cat([points2, xyz2], dim=-1)  # B, N2, C+D2
        corres1 = torch.arange(N1).view(1, N1, 1).repeat(B, 1, 1).permute(0,2,1).to(xyz1.device)
        corres2 = knn_point(1, feature2, feature1) # B, N1, 1
        neighbor_xyz2 = index_points_group(xyz2, corres2) # B, 1, N1, 3
        direction_xyz = neighbor_xyz2 - xyz1.view(B, N1, 1, C) # B, N1, 1, C
        direction_xyz = direction_xyz.squeeze(2).permute(0, 2, 1)
        # print('corres1 {}'.format(corres1.shape))
        # print('corres2 {}'.format(corres2.shape))
        # print('neighbor_xyz2 {}'.format(neighbor_xyz2.shape))
        # print('direction_xyz {}'.format(direction_xyz.shape))
        return corres1, corres2.permute(0,2,1), direction_xyz # B, 1, N1 and B C N

class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2

class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1)
        return dense_flow

class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

class UpSampleFeatureEstimationPointConv(nn.Module):

    def __init__(self, feat_ch, up_feat_ch, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(UpSampleFeatureEstimationPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + up_feat_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out
    def forward(self, xyz, feats):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        '''
        new_points = feats
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        return new_points

class SceneFlowEstimation(nn.Module):

    def __init__(self, feat_ch, up_flow_ch, coarse_flow_ch = 3, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimation, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + coarse_flow_ch + up_flow_ch + 3 # concat [feature, coarse_flow, up_flow, xyz]
        # print('flow last_channel {}'.format(last_channel))
        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel+3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, coarse_flow, up_flow = None):
        '''
        xyz: B C N
        feats: B C1 N
        coarse_flow: B 3 N
        up_flow: B 3 N
        '''
        print('xyz {}'.format(xyz.shape))
        print('feats {}'.format(feats.shape))
        print('coarse_flow {}'.format(coarse_flow.shape))
        if up_flow is None:
            new_points = torch.cat([xyz, feats, coarse_flow], dim = 1)
        else:
            new_points = torch.cat([xyz, feats, coarse_flow, up_flow], dim = 1)
        # print('new_points {}'.format(new_points.shape))
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        residual_flow = self.fc(new_points)
        flow = residual_flow + coarse_flow
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])

class RobustFlowRefinement(nn.Module):

    def __init__(self, static_flow_ch=3, coarse_flow_ch = 3, channels = [32, 64, 128], up_channels = [128, 64, 32],mlp = [32, 16], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(RobustFlowRefinement, self).__init__()
        self.number_list = [2048, 512, 128]
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.upsample = UpsampleFlow()
        last_channel = static_flow_ch + coarse_flow_ch + 3 # concat [static_flow, coarse_flow, xyz]
        # print('flow last_channel {}'.format(last_channel))

        ### Down sample
        ### 8192
        self.level0_0 = Conv1d(last_channel, 32)
        self.level0 = PointConv(neighbors, 32+3, 32, bn=True, use_leaky=True)
        self.level0_1 = Conv1d(32, 64)
        ### 2048
        self.level1 = PointConvD(2048, 16, 64 + 3, 64)
        self.level1_0 = PointConv(neighbors, 64+3, 64, bn=True, use_leaky=True)
        self.level1_1 = Conv1d(64, 128)
        ### 512
        self.level2 = PointConvD(512, 16, 128+3, 128)
        self.level2_0 = PointConv(neighbors, 128+3, 128, bn=True, use_leaky=True)
        self.level2_1 = Conv1d(128, 256)
        ### 128
        self.level3 = PointConvD(128, 16, 256+3, 256)
        self.level3_0 = PointConv(neighbors, 256+3, 256, bn=True, use_leaky=True)

        ### Up sample
        ### 512
        self.up_level2_0 = Conv1d(256 + 256, 128)
        self.up_level2_1 = Conv1d(128, 128)
        ### 2048
        self.up_level1_0 = Conv1d(128 + 128, 64)
        self.up_level1_1 = Conv1d(64, 64)
        ### 8192
        self.up_level0_0 = Conv1d(64+64, 32)
        self.up_level0_1 = Conv1d(32, 32)
        ### Final
        self.final_0 = Conv1d(32, 32)
        self.final_1 = Conv1d(32, 16)
        self.fc = nn.Conv1d(16, 3, 1)

    def forward(self, xyz, static_flow, coarse_flow):
        '''
        xyz: B C N
        static_flow: B 3 N
        coarse_flow: B 3 N
        '''
        new_points = torch.cat([xyz, static_flow, coarse_flow], dim = 1)
        # print('new_points {}'.format(new_points.shape))
        ### 8192
        feat_l0 = self.level0_0(new_points) #32
        feat_l0_0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0_0)

        ### 2048
        pc_l1, feat_l1, fps_pc_l1 = self.level1(xyz, feat_l0_1)
        feat_l1_0 = self.level1_0(pc_l1, feat_l1)
        feat_l1_1 = self.level1_1(feat_l1_0)

        ### 512
        pc_l2, feat_l2, fps_pc_l2 = self.level2(pc_l1, feat_l1_1)
        feat_l2_0 = self.level2_0(pc_l2, feat_l2)
        feat_l2_1 = self.level2_1(feat_l2_0)

        ### 128
        pc_l3, feat_l3, fps_pc_l3 = self.level3(pc_l2, feat_l2_1)
        feat_l3_0 = self.level3_0(pc_l3, feat_l3)

        ### 512
        feat_l3_up2 = self.upsample(pc_l2, pc_l3, feat_l3_0)
        new_feat_l2 = torch.cat([feat_l2_1, feat_l3_up2], dim=1)
        # print('feat_l2_1 {}'.format(feat_l2_1.shape))
        # print('feat_l3_up2 {}'.format(feat_l3_up2.shape))
        new_feat_l2 = self.up_level2_0(new_feat_l2)
        new_feat_l2 = self.level2_0(pc_l2, new_feat_l2)
        new_feat_l2 = self.up_level2_1(new_feat_l2)

        ### 2048
        feat_l2_up1 = self.upsample(pc_l1, pc_l2, new_feat_l2)
        new_feat_l1 = torch.cat([feat_l1_1, feat_l2_up1], dim=1)
        new_feat_l1 = self.up_level1_0(new_feat_l1)
        new_feat_l1 = self.level1_0(pc_l1, new_feat_l1)
        new_feat_l1 = self.up_level1_1(new_feat_l1)

        ### 8192
        feat_l1_up0 = self.upsample(xyz, pc_l1, new_feat_l1)
        new_feat_l0 = torch.cat([feat_l0_1, feat_l1_up0], dim=1)
        new_feat_l0 = self.up_level0_0(new_feat_l0)
        new_feat_l0 = self.level0(xyz, new_feat_l0)
        new_feat_l0 = self.up_level0_1(new_feat_l0)

        new_feat_l0 = self.final_0(new_feat_l0)
        new_feat_l0 = self.final_1(new_feat_l0)
        residual_flow = self.fc(new_feat_l0)
        flow = residual_flow + static_flow
        return flow.clamp(self.clamp[0], self.clamp[1])

class PoseRegression(nn.Module):

    def __init__(self, feat_ch, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(PoseRegression, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 12, 1)

    def forward(self, xyz, feat0):
        '''
        feat0: B C1 N
        corres1: B 1 N
        '''
        # print('feat0 {}'.format(feat0[0,:,:10]))
        # print('coores1 {}'.format(corres1[0, :, :10]))
        new_points = feat0
        # print('new_points {}'.format(new_points.shape))
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        tensor = self.fc(new_points)
        T = torch.reshape(tensor, (-1, 4, 3))
        p = T[:,:3,3]
        translation = T[:,3:,3]
        return p, translation

class InlierPredictionPointConv(nn.Module):

    def __init__(self, feat_ch, channels = [128, 128], mlp = [128, 64], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(InlierPredictionPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch * 2

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn = True, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 1, 1)

    def forward(self, xyz, feat0, feat1, corres1):
        '''
        feat0: B C1 N
        corres1: B 1 N
        '''
        # print('feat0 {}'.format(feat0[0,:,:10]))
        # print('coores1 {}'.format(corres1[0, :, :10]))
        B = feat1.shape[0]
        new_feat1_list = []
        for i in range(B):
            new_feat1_list.append(feat1[i, :, corres1[i, 0, :]])
        new_feat1 = torch.stack(new_feat1_list, dim=0)
        new_points = torch.cat([feat0, new_feat1], dim = 1)
        # print('new_points {}'.format(new_points.shape))
        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        logit = self.fc(new_points)
        return new_points, logit.clamp(self.clamp[0], self.clamp[1])

class WeightedProcrustes(nn.Module):
    def decompose_by_length(self, tensor, reference_tensors):
        decomposed_tensors = []
        start_ind = 0
        for r in reference_tensors:
            N = len(r)
            decomposed_tensors.append(tensor[start_ind:start_ind + N])
            start_ind += N
        return decomposed_tensors

    def forward(self, xyzs1, xyzs2, pred_pairs, weights):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # decomposed_weights = self.decompose_by_length(weights, pred_pairs)
        # xyzs1 = xyzs1.permute(0, 2, 1)
        # xyzs2 = xyzs2.permute(0, 2, 1)
        # pred_pairs = pred_pairs.permute(0, 2, 1)
        weights = weights.permute(0, 2, 1)
        RT = []
        ws = []
        # print('xyz1s {}'.format(xyzs1.shape))
        # print('pred_pairs {}'.format(pred_pairs.shape))
        # print('weights {}'.format(weights.shape))
        for xyz1, xyz2, pred_pair, w in zip(xyzs1, xyzs2, pred_pairs, weights):

            # print('xyz1 {}'.format(xyz1.shape))
            # print('pred_pair {}'.format(pred_pair.shape))
            # print('w {}'.format(w.shape))
            xyz1.requires_grad = False
            xyz2.requires_grad = False
            ws.append(w.sum().item())
            predT = GlobalRegistration.weighted_procrustes(
                X=xyz1[pred_pair[:, 0]].to(device),
                Y=xyz2[pred_pair[:, 1]].to(device),
                w=w,
                eps=np.finfo(np.float32).eps)
            RT.append(predT)

        Rs, ts = list(zip(*RT))
        Rs = torch.stack(Rs, 0)
        ts = torch.stack(ts, 0)
        ws = torch.Tensor(ws)
        return Rs, ts, ws