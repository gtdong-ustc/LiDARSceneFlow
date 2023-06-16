import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from flownet3d_util import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv
from sklearn.neighbors import NearestNeighbors
from pointconv_util import PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from pointconv_util import SceneFlowEstimatorPointConv
from pointconv_util import index_points_gather as index_points, Conv1d
from sklearn.neighbors import NearestNeighbors
from pointconv_util import knn_point, index_points_group


class FlowNet3D(nn.Module):
    def __init__(self, args):
        super(FlowNet3D, self).__init__()
        self.sample_mode = args.sample_mode
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.5, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat', knn=True)

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256, 256], knn=True)
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel=128 + 128, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256], knn=True)
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel=64, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256], knn=True)
        self.fp = PointNetFeaturePropogation(in_channel=256 + 3, mlp=[256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 3, kernel_size=1, bias=True)


    def forward(self, pc1, pc2, feature1, feature2):

        pc1 = pc1.permute(0, 2, 1).contiguous()
        pc2 = pc2.permute(0, 2, 1).contiguous()
        feature1 = feature1.permute(0, 2, 1).contiguous()  # B 3 N
        feature2 = feature2.permute(0, 2, 1).contiguous()  # B 3 N

        l1_pc1, l1_feature1 = self.sa1(pc1, feature1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)

        l1_pc2, l1_feature2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        return sf

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def  compute_loss_nearest_neighbor(warped_pc1, pc2):
    '''

    :param warped_pc1 (B,C,N):
    :param pc2 (B,C,N):
    :return (B,C,N):
    '''
    b = pc2.shape[0]

    warped_pc1_numpy = warped_pc1.detach().cpu().numpy()
    pc2_numpy = pc2.detach().cpu().numpy()
    indices_list = []
    for i in range(b):
        distances, indices = nearest_neighbor(np.transpose(warped_pc1_numpy[i], (1, 0)),
                                                   np.transpose(pc2_numpy[i], (1, 0)))
        indices_list.append(indices)
    indices_final = np.asarray(indices_list)
    indices_final = torch.from_numpy(indices_final)

    indices_tensor = indices_final.to(pc2.device)
    indices_tensor = torch.unsqueeze(indices_tensor, dim=1)
    indices_tensor = indices_tensor.repeat(1, 3, 1)
    anchored_pc2 = torch.gather(pc2, 2, indices_tensor).to(pc2.device)

    loss = torch.abs(warped_pc1 - anchored_pc2).mean()
    # loss = (torch.sum(torch.abs(warped_pc1 - anchored_pc2), dim=1)).mean()
    # loss = (torch.sum((warped_pc1 - anchored_pc2) * (warped_pc1 - anchored_pc2), dim=1) / 2.0).mean()
    # loss = (torch.sqrt(torch.sum((warped_pc1 - anchored_pc2) * (warped_pc1 - anchored_pc2), dim=1))).mean()
    # loss = L2Loss(warped_pc1, anchored_pc2)
    return loss, anchored_pc2

def anchored_reverse_flow(warped_pc1, pc2, indices):
    indices_tensor = indices.to(pc2.device)
    indices_tensor = torch.unsqueeze(indices_tensor, dim=1)
    indices_tensor = indices_tensor.repeat(1, 3, 1)

    anchored_pc2 = torch.gather(pc2, 2, indices_tensor)
    anchored_warped_pc1 = 0.5 * anchored_pc2 + 0.5 * warped_pc1
    return anchored_warped_pc1

def compute_loss_cycle_consistency(pc1, reprojected_pc1):
    loss = (torch.sum((pc1 - reprojected_pc1) * (pc1 - reprojected_pc1), dim=1) / 2.0).mean()
    return loss

def L2Loss(pred, gt):

    '''
    L2 Loss function
    :param pred :(B,N,C)
    :param gt: (B,N,C)
    :return: (B,N,C)
    '''

    diff_flow = pred - gt
    loss = torch.norm(diff_flow, dim=2).sum(dim=1).mean()
    return loss

def meanL2Loss(pred, gt):

    '''
    mean L2 Loss function
    :param pred :(B,N,C)
    :param gt: (B,N,C)
    :return: (B,N,C)
    '''

    diff_flow = pred - gt
    loss = torch.norm(diff_flow, dim=2).mean()
    return loss

def GetNearestNeighborPointCloud(pc1, pc2, flow):

    '''
    Get the nearest neighbor of pc1 in pc2
    :param pc1: (B,N,C)
    :param pc2:(B,N,C)
    :param flow: (B,N,C)
    :return: nearest_neighbor_of_pc1_in_pc2 (B,N,3)
    '''
    pc1_warped = pc1 + flow
    pc1_warped_t = pc1_warped.clone().detach()
    pc2_t = pc2.clone().detach()

    # print(pc1_warped_t.shape)
    # print(pc2_t.shape)
    idxs_t = knn_point(1, pc1_warped_t, pc2_t)  # idxs of nearest neighbors of pred in target
    # print(idxs_t.shape)
    nearest_neighbor = index_points_group(pc2_t, idxs_t).squeeze(2)  # retrieve nearest neighbors [B, N, 3]
    return nearest_neighbor

def GetAnchoredPointCloud(pc1, pc2, flow):

    '''
    Get the anchored warped pc1
    :param pc1: (B,N,C)
    :param pc2:(B,N,C)
    :param flow:(B,N,C)
    :return: anchored_warped_pc1 (B,N,C)
    '''

    # print(pc1.shape)
    # print(pc2.shape)
    # print(flow.shape)

    pc1_warped = pc1 + flow
    pc1_warped_t = pc1_warped.clone().detach()
    idxs_t = knn_point(1, pc2, pc1_warped_t)  # idxs of nearest neighbors of pred in target
    # print(idxs_t.max())
    nearest_neighbor = index_points_group(pc2, idxs_t).squeeze(2)  # retrieve nearest neighbors [B, N, 3]
    anchored_warped_pc1 = nearest_neighbor

    return anchored_warped_pc1