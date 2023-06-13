
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
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = True, use_relu = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        if use_relu:
            self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)


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

        if self.relu:
            new_points = self.relu(new_points)
        else:
            new_points = new_points
        return new_points

class PointConvDS(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_relu = True):
        super(PointConvDS, self).__init__()
        self.use_relu = use_relu
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        if use_relu:
            self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, new_xyz, xyz, points):
        """
        PointConv with downsampling.
        Input:
            new_xyz: input points position data, [B, 3, N/4]
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, D', N/4]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        new_N = new_xyz.shape[2]
        new_xyz = new_xyz.permute(0,2,1)
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)
        # xyz = xyz.contiguous()
        # fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        # new_xyz = index_points_gather(xyz, fps_idx) #input: B, N, 3

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, new_N, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        # print(self.use_relu)

        if self.use_relu:
            new_points = self.relu(new_points)
        else:
            new_points = new_points

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

class SimpleBottleneck(nn.Module):
    def __init__(self, nsample, in_channel, out_channel,base_width=64, groups=1, weightnet = 16):
        super(SimpleBottleneck, self).__init__()
        width = int(out_channel * (base_width / 64.)) * groups
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.conv1 = nn.Conv1d(in_channel, width, 1)
        self.bn1 = nn.BatchNorm1d(width)

        self.linear2 = nn.Linear(weightnet * (width + 3), width) # input: B, N, in_channels
        self.bn2 = nn.BatchNorm1d(width)

        self.conv3 = nn.Conv1d(width, out_channel, 1)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, features):

        B = xyz.shape[0]
        N = xyz.shape[2]
        # xyz_tmp = xyz.permute(0, 2, 1)  # B, N, in_channel
        # features_tmp = features.permute(0, 2, 1)  # B, N, in_channel

        out = self.conv1(features) # intput: B, in_channel, N   output: B, width, N
        out = self.bn1(out)
        out = self.relu(out) # output: B,width,N

        new_out, grouped_xyz_norm = group(self.nsample, xyz.permute(0, 2, 1), out.permute(0, 2, 1)) # input: B, N, width     output: B, N, nsample, width + 3
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        out = torch.matmul(input=new_out.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N,-1) # B, N, weightnet * (width + 3)
        out = self.linear2(out) # input: B,N,weightnet * (width + 3)   output: B,N,width
        out = self.relu(out.permute(0,2,1)) #output: B,width,N

        out = self.conv3(out) #output: B,out_channels,N
        out = self.bn1(out) #output: B,out_channels,N
        out += features # if in_channels == out_channels
        out = self.relu(out)

        return out

class AdaptiveAggregationModule(nn.Module):
    def __init__(self, nsample, num_scales, num_output_branches, cost_volume_channel_list = [32, 64, 128], num_blocks=1, weightnet = 16):
        super(AdaptiveAggregationModule, self).__init__()
        self.num_samples = nsample
        self.num_scales = num_scales
        self.cost_volume_channel_list = cost_volume_channel_list
        # self.num_points_list = [8192, 2048, 512]
        self.num_output_branches = num_output_branches
        self.num_blocks = num_blocks

        self.upsample = UpsampleFlow()

        self.relu = nn.LeakyReLU(LEAKY_RATE, inplace=True)
        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = cost_volume_channel_list[i]
            branch = nn.ModuleList()
            for j in range(num_blocks):
                branch.append(SimpleBottleneck(nsample=16, in_channel=num_candidates, out_channel=num_candidates, base_width=64, groups=1, weightnet=16))

            self.branches.append(nn.Sequential(*branch))

        # Adaptive cross-scale aggregation
        # For each output branch
        self.fuse_layers = nn.ModuleList()

        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv1d(self.cost_volume_channel_list[j], self.cost_volume_channel_list[i], 1),
                                      nn.BatchNorm1d(self.cost_volume_channel_list[i]),
                                      )) ## no activation function
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(PointConvDS(nsample=self.num_samples, in_channel=self.cost_volume_channel_list[k]+3, out_channel=self.cost_volume_channel_list[k+1], weightnet=weightnet, bn=True, use_relu=True))

                    layers.append(PointConvDS(nsample=self.num_samples, in_channel=self.cost_volume_channel_list[i-1]+3, out_channel=self.cost_volume_channel_list[i], weightnet=weightnet, bn=True, use_relu=False))
                    # self.fuse_layers[-1].append(nn.Sequential(*layers))
                    self.fuse_layers[-1].append(layers)




    def forward(self, xyz_list, points_list):
        """

        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz_list: input points position data, [scales, B, 3, N]
            points_list: input points data, [scales, B, C, N]
        Return:
            new_points_list:  [scales, B, C, N]
        """
        # B = xyz_list[0].shape[0]
        # N = xyz_list[0].shape[2]
        # xyz = xyz.permute(0, 2, 1) # B, N, C
        # points = points.permute(0, 2, 1) # B, N, C

        new_points_list = []

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                new_points_list.append(dconv(xyz_list[i], points_list[i]))

        points_fused_list = []

        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    if i == 0 :
                        points_fused_list.append(self.fuse_layers[i][j](new_points_list[0]))
                    elif i == 1:
                        points_fused_list.append(self.fuse_layers[i][j][0](xyz_list[i], xyz_list[j], new_points_list[j]))
                    else:
                        new_points = self.fuse_layers[i][j][0](xyz_list[j+1], xyz_list[j], new_points_list[j])
                        points_fused_list.append(self.fuse_layers[i][j][1](xyz_list[i], xyz_list[j+1], new_points))

                elif j == 1:
                    if i == 0:
                        new_points = self.upsample(xyz_list[i], xyz_list[j], new_points_list[j])

                        # print(points_fused_list[i].shape)
                        # print(self.fuse_layers[i][j](new_points).shape)

                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j](new_points)
                        # points_fused_list.append(self.fuse_layers[i][j](new_points))
                    elif i == 1:
                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j](new_points_list[j])
                        # points_fused_list.append(self.fuse_layers[i][j](new_points_list[j]))
                    else:
                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j][0](xyz_list[j+1], xyz_list[j], new_points_list[j])
                        # points_fused_list.append(self.fuse_layers[i][j][0](xyz_list[j+1], xyz_list[j], new_points_list[j]))
                else:
                    if i == 0:
                        new_points = self.upsample(xyz_list[i], xyz_list[j], new_points_list[j])
                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j](new_points)
                    elif i == 1:
                        new_points = self.upsample(xyz_list[i], xyz_list[j], new_points_list[j])
                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j](new_points)
                    else:
                        points_fused_list[i] = points_fused_list[i] + self.fuse_layers[i][j](new_points_list[j])

        for i in range(len(points_fused_list)):
            points_fused_list[i] = self.relu(points_fused_list[i])

        return points_fused_list

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