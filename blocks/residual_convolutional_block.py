
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors import KernelDensity
from pointnet2 import pointnet2_utils
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from pointconv_util import SceneFlowEstimatorPointConv
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance

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

class Conv1d_woBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d_woBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x

class Residual_Convolutional_Block(nn.Module):
    def __init__(self,nsample, in_channel, out_channel = 256, middle_channel = 64, weightnet = 16):
        super(Residual_Convolutional_Block, self).__init__()
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.conv1d_1 = Conv1d_woBN(in_channel, out_channel)
        self.linear_1 = nn.Linear(weightnet * (out_channel + 3), middle_channel)
        self.linear_2 = nn.Linear(weightnet * (middle_channel + 3), out_channel)
        self.bn_linear_1 = nn.BatchNorm1d(middle_channel)
        self.bn_linear_2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """

        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, C, N]
        Return:
            new_points_concat:  [B, out_channel, N]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1) # B, N, C
        points = points.permute(0, 2, 1) # B, N, C

        # print(points.shape)

        residual_points = self.conv1d_1(points.permute(0, 2, 1)) # intput: B, C, N   output: B, 256, N
        # print(residual_points.shape)
        new_points, grouped_xyz_norm = group(self.nsample, xyz, residual_points.permute(0, 2, 1)) # output: B, N, nsample, C + 3

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # print(new_points.shape)
        # print(weights.shape)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,N,-1)
        # print('##########')
        # print(new_points.shape)

        new_points = self.linear_1(new_points)
        new_points = self.bn_linear_1(new_points.permute(0, 2, 1)) # B, C, N
        new_points = self.relu(new_points)# B, C, N

        new_points, grouped_xyz_norm = group(self.nsample, xyz, new_points.permute(0, 2, 1))
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,N,-1)
        # print(new_points.shape)
        new_points = self.linear_2(new_points)
        new_points = self.bn_linear_2(new_points.permute(0, 2, 1) + residual_points)# B, C, N
        new_points = self.relu(new_points)

        return new_points


class Aligned_Feature_Aggregation(nn.Module):
    def __init__(self, nsample, in_channel, out_channel = 256, weightnet=16):
        super(Aligned_Feature_Aggregation, self).__init__()
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.upsample = UpsampleFlow()

        self.conv1d = Conv1d_woBN(2 * in_channel, out_channel) # out_channel : 256
        self.linear = nn.Linear(weightnet * out_channel, 6)
        self.fc = nn.Conv1d(weightnet * (out_channel + 3), 6, 1)
        self.bn_linear = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz_1, xyz_2, feature_1, feature_2):
        """

        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz_1: input points position data, [B, 3, N]
            xyz_2: input points position data, [B, 3, N]
            feature_1: input feature of xyz_1, [B, C, N]
            feature_2: input feature of xyz_2, [B, C, N]
        Return:
            new_feature: [B, C, N]
        """
        B = feature_2.shape[0]
        N = feature_2.shape[2]
        C = feature_2.shape[1]

        # xyz_2 = xyz_2.permute(0, 2, 1) # B, N, 3
        # points = points.permute(0, 2, 1) # B, N, D

        upsampled_feature_1 = self.upsample(xyz_2, xyz_1, feature_1)

        ## Get offset

        new_features = torch.cat([upsampled_feature_1, feature_2], dim=1) # B, 2 * C, N
        new_features = self.conv1d(new_features) # input:  B, 2 * C, N
        new_features = self.bn_linear(new_features)
        new_features = self.relu(new_features)
        new_features, grouped_xyz_norm = group(self.nsample, xyz_2.permute(0, 2, 1), new_features.permute(0, 2, 1)) # input: B, N, C output: B, 1, N, 3+256

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_features = torch.matmul(input=new_features.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N,-1)
        # new_features = self.linear(new_features)
        # print(new_features.shape)
        concat_offset = self.fc(new_features.permute(0, 2, 1))
        # print(concat_offset.shape)
        # print(xyz_2.shape)

        offsets_1 = concat_offset[:, :3, :]
        offsets_2 = concat_offset[:, 3:, :]

        warped_xyz2_1 = offsets_1 + xyz_2
        warped_xyz2_2 = offsets_2 + xyz_2
        # print(warped_xyz2_1.shape)
        knn_idx_1 = knn_point(1, warped_xyz2_1.permute(0, 2, 1), xyz_2.permute(0, 2, 1))
        knn_idx_2 = knn_point(1, warped_xyz2_2.permute(0, 2, 1), xyz_2.permute(0, 2, 1)) # B, N, 1
        # print(knn_idx_1.shape)
        warped_upsampled_feature_1 = index_points_group(upsampled_feature_1.permute(0, 2, 1), knn_idx_1).view(B, N, C)
        warped_feature_2 = index_points_group(feature_2.permute(0, 2, 1), knn_idx_2).view(B, N, C)

        new_feature = warped_upsampled_feature_1 + warped_feature_2
        return new_feature # B, N, 256