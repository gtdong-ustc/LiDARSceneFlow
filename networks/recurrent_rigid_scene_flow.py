import torch
import torch.nn as nn
import torch.nn.functional as F
from flownet3d_util import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv
from sklearn.cluster import DBSCAN
from lib.utils import transform_point_cloud, kabsch_transformation_estimation
from pointconv_util import index_points_gather as index_points, index_points_group, square_distance
from collections import defaultdict
import numpy as np
from networks.flownet3d_full_weights import FlowNet3D as FlowNetWeights3D
from torch.nn import L1Loss
from pointconv_util import knn_point, index_points_group

_EPS = 1e-6



class DriftBrake(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(DriftBrake, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)

        self.conv1 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, bias=True)

    def forward(self, pc, feature):
        _, feat_l1 = self.sa1(pc, feature)
        _, feat_l2 = self.sa2(pc, feat_l1)

        x = F.relu(self.bn1(self.conv1(feat_l2)))
        query = self.conv2(x).sum(dim=2, keepdim=True) / 8192.0  # 8192
        return query  # B, 256, 1

    def get_nearest_neighbor_error(self, warped_pc2, pc2, weights):
        warped_pc2 = warped_pc2.permute(0, 2, 1)
        pc2 = pc2.permute(0, 2, 1)

        # warped_pc2_t = warped_pc2.clone().detach().contiguous()
        idx = knn_point(1, pc2, warped_pc2)

        nn_pc2 = index_points_group(pc2, idx).squeeze(2).contiguous()  # retrieve nearest neighbors [B, N, 3]
        error_map = (warped_pc2 - nn_pc2).permute(0, 2, 1) * weights  # B, 3, N
        # error_map = torch.abs(warped_pc2 - nn_pc2).permute(0,2,1) * weights  # B, 3, N
        # error_map = error_map.contiguous()
        return error_map

class H0Net(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(H0Net, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=8, in_channel=128,
                                          mlp=[128, 128, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=8, in_channel=128, mlp=[128],
                                          group_all=False, use_act=False, use_instance_norm=use_instance_norm)

    def forward(self, pc, feature):
        _, feat_l1 = self.sa1(pc, feature)
        _, feat_l2 = self.sa2(pc, feat_l1)
        return feat_l2


class GRU(nn.Module):
    def __init__(self, npoint, hidden_dim, input_dim, use_instance_norm):
        super(GRU, self).__init__()
        in_ch = hidden_dim + input_dim
        self.convz = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False,
                                            use_instance_norm=use_instance_norm)
        self.convr = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False,
                                            use_instance_norm=use_instance_norm)
        self.convq = PointNetSetAbstraction(npoint=int(npoint/8), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False,
                                            use_instance_norm=use_instance_norm)

    def forward(self, h, x, pc):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(pc, hx)[1])
        r = torch.sigmoid(self.convr(pc, hx)[1])
        q = torch.tanh(self.convq(pc, torch.cat([r * h, x], dim=1))[1])
        h = (1 - z) * h + z * q
        return h

class EgoMotionHead(nn.Module):
    """
    Class defining EgoMotionHead
    """

    def __init__(self, add_slack=True, sinkhorn_iter=5):
        nn.Module.__init__(self)

        self.slack = add_slack
        self.sinkhorn_iter = sinkhorn_iter

        # Affinity parameters
        self.beta = torch.nn.Parameter(torch.tensor(-5.0))
        self.alpha = torch.nn.Parameter(torch.tensor(-5.0))

        self.softplus = torch.nn.Softplus()
    def compute_rigid_transform(self, xyz_s, xyz_t, weights):
        """Compute rigid transforms between two point sets

        Args:
            a (torch.Tensor): (B, M, 3) points
            b (torch.Tensor): (B, N, 3) points
            weights (torch.Tensor): (B, M)

        Returns:
            Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
        """

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        # weights_normalized = weights
        centroid_s = torch.sum(xyz_s * weights_normalized, dim=1)
        centroid_t = torch.sum(xyz_t * weights_normalized, dim=1)
        s_centered = xyz_s - centroid_s[:, None, :]
        t_centered = xyz_t - centroid_t[:, None, :]
        cov = s_centered.transpose(-2, -1) @ (t_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_s[:, :, None] + centroid_t[:, :, None]

        transform = torch.cat((rot_mat, translation), dim=2)

        return transform
    def sinkhorn(self, log_alpha, n_iters=5, slack=True):
        """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)
        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        """

        # Sinkhorn iterations

        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)


        log_alpha = log_alpha_padded[:, :-1, :-1]

        return log_alpha
    def forward(self, xyz_s, weighted_t, weights, label):
        """
        直接输入cost volume的计算值代替weighted_t 试一下。
        :param weighted_t: B, 3, N
        :param weights: B, N
        :param xyz_t: B, 3, N
        :return:
        """
        background_label = torch.abs(label-1.0)
        xyz_s = xyz_s.permute(0, 2, 1)
        weighted_t = weighted_t.permute(0, 2, 1)
        weights = weights * background_label
        weights = torch.squeeze(weights, 1)

        R_est, t_est, _, _ = kabsch_transformation_estimation(xyz_s,
                                                              weighted_t,
                                                              weights=weights)
        return R_est, t_est

class SequenceWeights(nn.Module):
    def __init__(self, npoint=8192, use_instance_norm=False, loc_flow_nn=32, loc_flow_rad=1.5, k_decay_fact=1.0,
                 **kwargs):
        super(SequenceWeights, self).__init__()
        self.k_decay_fact = k_decay_fact

        # -------------------- Flow0 Backbone -----------------------
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=None, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=None, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=None, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=None, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat', knn=True)

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256, 256],
                                     knn=True)
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel=128 + 128, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256], knn=True)
        self.fp = PointNetFeaturePropogation(in_channel=256 + 64, mlp=[128, 128])
        # -----------------------------------------------------------

        # -------------------- Flow0 Decoder -----------------------
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 3, kernel_size=1, bias=True)
        # ---------------------------------------------------------

        # -------------------- Flow Decoder ----------------------
        self.conv1_update = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1_update = nn.BatchNorm1d(128)
        self.conv2_update = nn.Conv1d(128, 3, kernel_size=1, bias=True)
        # ---------------------------------------------------------

        # -------------------- Weights0 Decoder -----------------------
        self.weights_conv1 = nn.Conv1d(128 + 16 + 3, 128, kernel_size=1, bias=False)
        self.weights_bn1 = nn.BatchNorm1d(128)
        self.weights_conv2 = nn.Conv1d(128, 1, kernel_size=1, bias=True)
        # ---------------------------------------------------------

        # -------------------- Weights Decoder ----------------------
        self.weights_conv1_update = nn.Conv1d(128 + 16 + 3, 128, kernel_size=1, bias=False)
        self.weights_bn1_update = nn.BatchNorm1d(128)
        self.weights_conv2_update = nn.Conv1d(128, 1, kernel_size=1, bias=True)
        # ---------------------------------------------------------

        # ----------------- Update Unit ---------------------------
        self.drift_brake = DriftBrake(npoint, use_instance_norm)
        self.h0_net = H0Net(npoint, use_instance_norm)
        self.gru = GRU(npoint, hidden_dim=128, input_dim=64+128+16+16+16+3+1+3, use_instance_norm=use_instance_norm)
        self.flow_embedding_conv1 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=16,
                                                           in_channel=3,
                                                           mlp=[32, 32, 32],
                                                           group_all=False, use_instance_norm=use_instance_norm)
        self.flow_embedding_conv2 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=8,
                                                           in_channel=32,
                                                           mlp=[16, 16, 16],
                                                           group_all=False, use_instance_norm=use_instance_norm)
        self.weights_embedding_conv1 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=16,
                                                              in_channel=1,
                                                              mlp=[32, 32, 32],
                                                              group_all=False, use_instance_norm=use_instance_norm)
        self.weights_embedding_conv2 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=8,
                                                              in_channel=32,
                                                              mlp=[16, 16, 16],
                                                              group_all=False, use_instance_norm=use_instance_norm)

        self.error_embedding_conv1 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=16,
                                                              in_channel=3,
                                                              mlp=[32, 32, 32],
                                                              group_all=False, use_instance_norm=use_instance_norm)
        self.error_embedding_conv2 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=8,
                                                              in_channel=32,
                                                              mlp=[16, 16, 16],
                                                              group_all=False, use_instance_norm=use_instance_norm)
        # ---------------------------------------------------------

        # ----------------- Ego Unit ---------------------------
        self.add_slack = True
        self.sinkhorn_iter = 3
        self.ego_motion_decoder = EgoMotionHead(add_slack=self.add_slack,sinkhorn_iter=self.sinkhorn_iter)
        # ---------------------------------------------------------

        # --------------- Upsample Module -------------------------
        self.flow_up_sample = PointNetFeaturePropogation(in_channel=3, mlp=[])
        self.weights_up_sample = PointNetFeaturePropogation(in_channel=1, mlp=[])
        # ---------------------------------------------------------

        # ---------------- Foreground Clustering ------------------
        self.min_p_cluster = 30
        self.min_samples_dbscan = 5
        self.cluster_metric = 'euclidean'
        self.eps_dbscan = 0.75
        self.cluster_estimator = DBSCAN(min_samples=self.min_samples_dbscan,
                                        metric=self.cluster_metric, eps=self.eps_dbscan)
        # ---------------------------------------------------------

    def forward(self, pc1, pc2, feature1, feature2, label1, label2, fg_label1, fg_label2, iters=1):
        # ---------------------- init ----------------------------
        flow_predictions = []
        rigid_flow_predictions = []
        weights_predictions = []
        pc1 = pc1.permute(0, 2, 1).contiguous()
        pc2 = pc2.permute(0, 2, 1).contiguous()
        feature1 = feature1.permute(0, 2, 1).contiguous()  # B 3 N
        feature2 = feature2.permute(0, 2, 1).contiguous()  # B 3 N
        label1 = label1.permute(0, 2, 1).contiguous()  # B 1 N
        label2 = label2.permute(0, 2, 1).contiguous()  # B 1 N
        fg_label1 = fg_label1.permute(0, 2, 1).contiguous()  # B 1 N
        fg_label2 = fg_label2.permute(0, 2, 1).contiguous()  # B 1 N
        # --------------------------------------------------------

        # -------------------- Flownet3D backbone ----------------
        l1_pc1, l1_feature1, l1_fps_idx1 = self.sa1(pc1, feature1) # 1024
        l2_pc1, l2_feature1, l2_fps_idx1 = self.sa2(l1_pc1, l1_feature1) # 256
        l1_pc2, l1_feature2, l1_fps_idx2 = self.sa1(pc2, feature2)
        l2_pc2, l2_feature2, l2_fps_idx2 = self.sa2(l1_pc2, l1_feature2)
        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)
        l3_pc1, l3_feature1, l3_fps_idx1 = self.sa3(l2_pc1, l2_feature1_new) # 64
        l4_pc1, l4_feature1, l4_fps_idx1 = self.sa4(l3_pc1, l3_feature1) # 16
        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.fp(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        # --------------------------------------------------------

        # --------------------- Flow0 Head -----------------------
        x = F.relu(self.bn1(self.conv1(l1_fnew1)))
        flow0_lr = self.conv2(x)  # 1024
        flow0 = self.flow_up_sample(pc1, l1_pc1, None, flow0_lr)
        flow_embedding = self.get_flow_embedding(flow0_lr, pc=l1_pc1)
        # --------------------------------------------------------

        # --------------------- Weights0 Head --------------------
        weights_x = F.relu(self.weights_bn1(self.weights_conv1(torch.cat([l1_fnew1, flow_embedding, flow0_lr], dim=1))))
        weights0_lr = self.weights_conv2(weights_x)
        weights0_norm_lr = torch.sigmoid(weights0_lr)
        weights0_norm = self.weights_up_sample(pc1, l1_pc1, None, weights0_norm_lr)

        # --------------------------------------------------------

        weighted_pc2_lr = l1_pc1 + flow0_lr
        weighted_pc2 = pc1 + flow0
        label1_lr = index_points(label1.permute(0, 2, 1), l1_fps_idx1).permute(0, 2, 1) # B, C, N/4

        # ---------------------- Rigid Motion Unit ----------------------
        R_ego0, t_ego0 = self.ego_motion_decoder(l1_pc1, weighted_pc2_lr, weights0_norm_lr, label1_lr)

        clusters_1 = self.get_instance_cluster_gt(fg_label1)
        clusters_2 = self.get_instance_cluster_gt(fg_label2)

        clusters_foreground_label1 = self.get_clusters_fore_label(pc1, clusters_1)
        clusters_foreground_label2 = self.get_clusters_fore_label(pc2, clusters_2)

        clusters_R0, clusters_t0 = self.get_instance_motion(pc1, weighted_pc2, weights0_norm, clusters_1)
        ego_pc1_new, instance_pc1_new = self.get_rigid_warping(pc1, clusters_1, clusters_R0, clusters_t0, R_ego0, t_ego0)

        rigid_pc1_new = ego_pc1_new * torch.abs(label1-1.0) + \
                        instance_pc1_new * clusters_foreground_label1 + \
                        weighted_pc2 * torch.abs(clusters_foreground_label1-1.0) * label1  # Refine the rigid flow with flow

        rigid_pc1_new_lr = index_points(rigid_pc1_new.permute(0, 2, 1), l1_fps_idx1).permute(0, 2, 1)  # B, C, N/4
        rigid_flow0 = rigid_pc1_new - pc1
        # --------------------------------------------------------------

        h = self.calc_h0(l1_fnew1, l1_pc1)

        flow_predictions.append(flow0.permute(0, 2, 1))
        rigid_flow_predictions.append(rigid_flow0.permute(0,2,1))
        weights_predictions.append(weights0_norm_lr.permute(0,2,1))
        weights_new_lr = weights0_lr.detach()
        weights_norm_new_lr = weights0_norm_lr.detach()
        rigid_pc1_new = rigid_pc1_new.detach()
        rigid_pc1_new_lr = rigid_pc1_new_lr.detach()

        for iter in range(iters - 1):
            weights_new_lr = weights_new_lr.detach()
            weights_norm_new_lr = weights_norm_new_lr.detach()
            rigid_pc1_new = rigid_pc1_new.detach().contiguous()
            rigid_pc1_new_lr = rigid_pc1_new_lr.detach().contiguous()
            rigid_flow_lr = rigid_pc1_new_lr - l1_pc1
            rigid_flow = rigid_pc1_new - pc1

            # -------------------- Flownet3D Backbone -----------------
            # l1_pc1_new, l1_feature1_new, _ = self.sa1(rigid_pc1_new, rigid_pc1_new, fps_idx=l1_fps_idx1)  # 1024
            # l2_pc1_new, l2_feature1_new, _ = self.sa2(l1_pc1_new, l1_feature1_new, fps_idx=l2_fps_idx1)  # 256
            # _, l2_correlation_new = self.fe_layer(l2_pc1_new, l2_pc2, l2_feature1_new, l2_feature2)
            # l3_pc1_new, l3_feature1_new, _ = self.sa3(l2_pc1_new, l2_correlation_new, fps_idx=l3_fps_idx1)  # 64
            # l4_pc1_new, l4_feature1_new, _ = self.sa4(l3_pc1_new, l3_feature1_new, fps_idx=l4_fps_idx1)  # 16
            # l3_fnew1 = self.su1(l3_pc1_new, l4_pc1_new, l3_feature1_new, l4_feature1_new)
            # l2_fnew1 = self.su2(l2_pc1_new, l3_pc1_new, torch.cat([l2_feature1_new, l2_correlation_new], dim=1), l3_fnew1)
            # l1_fnew1 = self.fp(l1_pc1_new, l2_pc1_new, l1_feature1_new, l2_fnew1)
            # --------------------------------------------------------

            # ----------------- Drift Brake ------------------------
            error_map = self.drift_brake.get_nearest_neighbor_error(rigid_pc1_new_lr, pc2, weights_norm_new_lr)
            error_embedding = self.get_error_embedding(error_map, pc=l1_pc1)
            weights_embedding = self.get_weights_embedding(weights_norm_new_lr, pc=l1_pc1)
            rigid_flow_embedding = self.get_flow_embedding(rigid_flow_lr, pc=l1_pc1)
            # -------------------------------------------------------

            # -------------------- GRU Unit --------------------------
            x_embedding = torch.cat([l1_feature1, l1_fnew1, rigid_flow_embedding, weights_embedding, error_embedding, rigid_flow_lr, weights_norm_new_lr, error_map], dim=1)  # [64, 256, 16, 16, 16, 3, 1, 3]
            h = self.gru(h=h, x=x_embedding, pc=l1_pc1)
            # --------------------------------------------------------

            # ---------------- Flow Update Unit ----------------------------
            # --------------------- Flow Head -----------------------
            x = F.relu(self.bn1_update(self.conv1_update(h)))
            delta_flow_lr = self.conv2_update(x)  # 1024
            delta_flow = self.flow_up_sample(pc1, l1_pc1, None, delta_flow_lr)
            flow_lr = rigid_flow_lr + delta_flow_lr
            flow = rigid_flow + delta_flow
            flow_embedding = self.get_flow_embedding(flow_lr, pc=l1_pc1)
            # -------------------------------------------------------
            # --------------------- Weights Head -----------------------
            weights_x = F.relu(self.weights_bn1_update(self.weights_conv1_update(torch.cat([h, flow_embedding, flow_lr], dim=1))))
            delta_weights_lr = self.weights_conv2_update(weights_x)
            weights_new_lr = weights_new_lr + delta_weights_lr
            weights_norm_new_lr = torch.sigmoid(weights_new_lr)
            weights_norm_new = self.weights_up_sample(pc1, l1_pc1, None, weights_norm_new_lr)
            # -------------------------------------------------------
            weighted_pc2_lr = rigid_pc1_new_lr + delta_flow_lr
            weighted_pc2 = rigid_pc1_new + delta_flow
            # --------------------------------------------------------------

            # --------------------- Rigid Motion Unit ----------------------
            R_ego, t_ego = self.ego_motion_decoder(rigid_pc1_new_lr, weighted_pc2_lr, weights_norm_new_lr, label1_lr)
            clusters_R, clusters_t = self.get_instance_motion(rigid_pc1_new, weighted_pc2, weights_norm_new, clusters_1)
            # print('clusters_R {}, clusters_t {}'.format(clusters_R, clusters_t))
            ego_pc1_new, instance_pc1_new = self.get_rigid_warping(rigid_pc1_new, clusters_1, clusters_R, clusters_t, R_ego, t_ego)
            rigid_pc1_new = ego_pc1_new * torch.abs(label1 - 1.0) +\
                            instance_pc1_new * clusters_foreground_label1 +\
                            weighted_pc2 * torch.abs(clusters_foreground_label1 - 1.0) * label1
            rigid_pc1_new_lr = index_points(rigid_pc1_new.permute(0, 2, 1), l1_fps_idx1).permute(0, 2, 1)
            rigid_flow = rigid_pc1_new - pc1
            # --------------------------------------------------------------

            weights_predictions.append(weights_norm_new_lr.permute(0, 2, 1))
            flow_predictions.append(flow.permute(0, 2, 1))
            rigid_flow_predictions.append(rigid_flow.permute(0, 2, 1))

        weights_predictions.append(clusters_foreground_label1)
        weights_predictions.append(clusters_foreground_label2)
        return flow_predictions, rigid_flow_predictions, weights_predictions


    def calc_h0(self, feats1_loc, pc):
        h = self.h0_net(pc, feats1_loc)
        h = torch.tanh(h)
        return h

    def get_flow_embedding(self, flow, pc):
        flow = flow.contiguous()

        _, flow_feats = self.flow_embedding_conv1(pc, flow)
        _, flow_feats = self.flow_embedding_conv2(pc, flow_feats)
        return flow_feats

    def get_weights_embedding(self, weights, pc):

        _, weights_feats = self.weights_embedding_conv1(pc, weights)
        _, weights_feats = self.weights_embedding_conv2(pc, weights_feats)
        return weights_feats

    def get_error_embedding(self, error, pc):

        _, error_feats = self.error_embedding_conv1(pc, error)
        _, error_feats = self.error_embedding_conv2(pc, error_feats)
        return error_feats

    def get_instance_cluster(self, pc1, label1):

        """

        :param pcd1: [B, 3, N]
        :param pcd2: [B, 3, N]
        :param label1: [B, 1, N]
        :param label2: [B, 1, N]

        :return: clusters_1, clusters_2
        """

        clusters_1 = defaultdict(list)

        for b_idx in range(pc1.shape[0]):
            pc1_curr = pc1[b_idx, :, :] # 3, N
            label1_curr = label1[b_idx, :, :]

            fg_idx1_curr = torch.where(label1_curr[0] == 1)[0] # N

            fg_pc1_curr = pc1_curr[:, fg_idx1_curr].cpu().numpy()

            if fg_pc1_curr.shape[-1] <= 30:
                continue
            labels1_curr_np = self.cluster_estimator.fit_predict(fg_pc1_curr.transpose(1, 0))

            for class_label in np.unique(labels1_curr_np):
                if class_label != -1 and np.where(labels1_curr_np == class_label)[0].shape[0] >= self.min_p_cluster:
                    clusters_1[str(b_idx)].append(fg_idx1_curr[np.where(labels1_curr_np == class_label)[0]])

        return clusters_1

    def get_instance_cluster_gt(self, label1):

        """
        :param label1: [B, 1, N]

        :return: clusters_1
        """

        clusters_1 = defaultdict(list)
        for b_idx in range(label1.shape[0]):

            label1_curr = label1[b_idx, :, :]

            fg_idx1_curr = torch.where(label1_curr[0] != -1)[0]  # N
            inlier_label1_curr = label1_curr[:, fg_idx1_curr]

            for class_label in torch.unique(inlier_label1_curr):
                if torch.where(inlier_label1_curr[0] == class_label)[0].shape[0] >= self.min_p_cluster:
                    clusters_1[str(b_idx)].append(fg_idx1_curr[torch.where(inlier_label1_curr[0] == class_label)[0]])

        return clusters_1

    def get_clusters_fore_label(self, pc1, clusters_1):
        clusters_foreground_label1_list = []
        for b_idx in range(pc1.shape[0]):
            clusters_foreground_label1_curr = torch.zeros_like(pc1[0, :1, :]).float()
            for c_idx in clusters_1[str(b_idx)]:
                clusters_foreground_label1_curr[:, c_idx] = 1.0
            clusters_foreground_label1_list.append(clusters_foreground_label1_curr)
        clusters_foreground_label1 = torch.stack(clusters_foreground_label1_list, dim=0)
        return clusters_foreground_label1

    def get_final_warped_pc2(self, ego_pc2, rigid_pc2, weighted_pc2, label1, fore_label1):

        final_warped_pc2 = torch.zeros_like(weighted_pc2)


        clusters_foreground_label1_list = []
        for b_idx in range(pc1.shape[0]):
            clusters_foreground_label1_curr = torch.zeros_like(pc1[0, :1, :]).float()
            for c_idx in clusters_1[str(b_idx)]:
                clusters_foreground_label1_curr[:, c_idx] = 1.0
            clusters_foreground_label1_list.append(clusters_foreground_label1_curr)
        clusters_foreground_label1 = torch.stack(clusters_foreground_label1_list, dim=0)
        return clusters_foreground_label1

    def get_instance_cluster_lr(self, clusters_1, fps_idx_1):

        fps_idx_1_cpu = fps_idx_1.cpu()
        clusters_1_lr = defaultdict(list)
        for b_idx in range(fps_idx_1_cpu.shape[0]):
            for c_idx in clusters_1[str(b_idx)]:
                c_idx_cpu = c_idx.cpu()
                intersection_c_idx = torch.from_numpy(np.intersect1d(c_idx_cpu, fps_idx_1_cpu[b_idx])).cuda()
                # intersection_c_idx_lr = torch.zeros_like(intersection_c_idx)
                for i in range(intersection_c_idx.shape[0]):
                    intersection_c_idx[i] = torch.where(fps_idx_1[b_idx] == intersection_c_idx[i])[0]
                clusters_1_lr[str(b_idx)].append(intersection_c_idx)
        return clusters_1_lr

    def get_instance_motion(self, pc1, aligned_pc2, weights, clusters_1):

        clusters_1_rot = defaultdict(list)
        clusters_1_trans = defaultdict(list)

        for b_idx in range(pc1.shape[0]):
            pc1_curr = pc1[b_idx, :, :]  # 3, N
            aligned_pc2_curr = aligned_pc2[b_idx, :, :]
            weights_curr = weights[b_idx, :, :]
            # Estimate the relative transformation parameteres of each cluster
            for c_idx in clusters_1[str(b_idx)]:
                cluster_pc1_curr = torch.unsqueeze(pc1_curr[:, c_idx], 0).permute(0, 2, 1)
                cluster_aligned_pc2_curr = torch.unsqueeze(aligned_pc2_curr[:, c_idx], 0).permute(0, 2, 1)
                cluster_weights_curr = torch.squeeze(weights_curr[:, c_idx], 0)
                cluster_weights_curr = torch.unsqueeze(cluster_weights_curr, 0)

                R_cluster, t_cluster, _, _ = kabsch_transformation_estimation(cluster_pc1_curr,
                                                              cluster_aligned_pc2_curr,
                                                              weights=cluster_weights_curr,
                                                              w_threshold=0.001)

                clusters_1_rot[str(b_idx)].append(R_cluster.squeeze(0))
                clusters_1_trans[str(b_idx)].append(t_cluster.squeeze(0))
        return clusters_1_rot, clusters_1_trans

    def get_rigid_warping_ori(self, pc1, clusters_1, clusters_1_rot, clusters_1_trans, ego_rot, ego_trans):

        ego_warped_pc2 = transform_point_cloud(pc1.permute(0, 2, 1), ego_rot, ego_trans).permute(0, 2, 1)

        for b_idx in range(pc1.shape[0]):
            pc1_curr = pc1[b_idx, :, :]  # 3, N
            for i in range(len(clusters_1[str(b_idx)])):
                c_idx = clusters_1[str(b_idx)][i]
                inst_rot = clusters_1_rot[str(b_idx)][i]
                inst_trans = clusters_1_trans[str(b_idx)][i]
                cluster_pc1_curr = pc1_curr[:, c_idx]
                ego_warped_pc2[b_idx, :, c_idx] = transform_point_cloud(cluster_pc1_curr.permute(1, 0), inst_rot, inst_trans).permute(0, 2, 1).squeeze(0)
        return ego_warped_pc2

    def get_rigid_warping(self, pc1, clusters_1, clusters_1_rot, clusters_1_trans, ego_rot, ego_trans):

        ego_warped_pc2 = transform_point_cloud(pc1.permute(0, 2, 1), ego_rot, ego_trans).permute(0, 2, 1)
        instance_warped_pc2_list = []
        for b_idx in range(pc1.shape[0]):
            pc1_curr = pc1[b_idx, :, :]  # 3, N
            instance_warped_pc1_curr = pc1_curr.clone()
            for i in range(len(clusters_1[str(b_idx)])):
                c_idx = clusters_1[str(b_idx)][i]
                inst_rot = clusters_1_rot[str(b_idx)][i]
                inst_trans = clusters_1_trans[str(b_idx)][i]
                cluster_pc1_curr = pc1_curr[:, c_idx]
                instance_warped_pc1_curr[:, c_idx] = transform_point_cloud(cluster_pc1_curr.permute(1, 0), inst_rot, inst_trans).permute(0, 2, 1).squeeze(0)
            instance_warped_pc2_list.append(instance_warped_pc1_curr)
        return ego_warped_pc2, torch.stack(instance_warped_pc2_list, dim=0)

    def ego_motion(self, pc1, aligned_pc2, weights, label1):

        R_ego, t_ego = self.ego_motion_decoder(pc1, aligned_pc2, weights, label1)
        return R_ego, t_ego

def computeChamfer(pc1, pc2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False) # B, N
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)  # B, M
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2

def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''

    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow

def ChamferSmoothLoss(pc1, pc2, flow, label1, label2):

    f_chamfer = 1.0
    f_smoothness = 1.0

    pc1 = pc1.permute(0, 2, 1)  # : (B,3,N)
    pc2 = pc2.permute(0, 2, 1)  # : (B,3,N)
    flow = flow.permute(0, 2, 1)  # pred_flow: (B,3,N)
    label1 = label1.permute(0, 2, 1) #(B,1,N)
    label2 = label2.permute(0, 2, 1)  # (B,1,N)

    aligned_pc2 = pc1 + flow

    loss = 0.0

    for b_idx in range(pc1.shape[0]):
        pc1_curr = pc1[b_idx, :, :]  # 3, N
        pc2_curr = pc2[b_idx, :, :]
        aligned_pc2_curr = aligned_pc2[b_idx, :, :]
        flow_curr = flow[b_idx, :, :]
        label1_curr = label1[b_idx, :, :]
        label2_curr = label2[b_idx, :, :]

        fg_idx1_curr = torch.where(label1_curr[0] == 1)[0]  # N
        fg_idx2_curr = torch.where(label2_curr[0] == 1)[0]

        fg_pc1_curr = torch.unsqueeze(pc1_curr[:, fg_idx1_curr], 0)
        fg_flow_curr = torch.unsqueeze(flow_curr[:, fg_idx1_curr], 0)
        fg_aligned_pcc_curr = torch.unsqueeze(aligned_pc2_curr[:, fg_idx1_curr], 0)
        fg_pc2_curr = torch.unsqueeze(pc2_curr[:, fg_idx2_curr], 0)

        if fg_pc1_curr.shape[-1] <= 30 or fg_pc2_curr.shape[-1] <= 30:
            continue

        dist1_curr, dist2_curr = computeChamfer(fg_aligned_pcc_curr, fg_pc2_curr)

        chamferLoss_curr = dist1_curr.sum(dim=1).mean() + dist2_curr.sum(dim=1).mean()

        smoothnessLoss_curr = computeSmooth(fg_pc1_curr, fg_flow_curr).sum(dim=1).mean()

        loss_curr = f_chamfer * chamferLoss_curr + f_smoothness * smoothnessLoss_curr

        loss += loss_curr

    return loss

def fore_back_loss(pc1, pc2, flow, rigid_flow, ego_flow_gt, weights, label1, label2, clusters_label1, clusters_label2):

    pc1 = pc1.permute(0, 2, 1)  # : (B,3,N)
    pc2 = pc2.permute(0, 2, 1)  # : (B,3,N)
    flow = flow.permute(0, 2, 1)  # pred_flow: (B,3,N)
    rigid_flow = rigid_flow.permute(0, 2, 1)  # pred_rigid_flow: (B,3,N)
    ego_flow_gt = ego_flow_gt.permute(0, 2, 1)  # ego_flow_gt: (B,3,N)
    weights = weights.permute(0, 2, 1) # weights: (B,1,N)
    label1 = label1.permute(0, 2, 1)  # (B,1,N)
    label2 = label2.permute(0, 2, 1)  # (B,1,N)
    clusters_label1 = clusters_label1.permute(0, 2, 1)  # (B,1,N)
    clusters_label2 = clusters_label2.permute(0, 2, 1)  # (B,1,N)

    trans_loss = 0.0
    # inlier_loss = 0.0
    chamfer_loss = 0.0
    rigid_loss = 0.0
    smoothness_loss = 0.0

    B = pc1.shape[0]
    l1_loss = L1Loss(reduction='sum')

    # ----------------- InlierLoss ------------------------
    inlier_loss = weights.sum(dim=2).mean()
    # -----------------------------------------------------

    for b_idx in range(B):
        pc1_curr = pc1[b_idx, :, :]  # 3, N
        pc2_curr = pc2[b_idx, :, :]
        flow_curr = flow[b_idx, :, :]
        rigid_flow_curr = rigid_flow[b_idx, :, :]
        ego_flow_gt_curr = ego_flow_gt[b_idx, :, :]
        label1_curr = label1[b_idx, :, :]
        # label2_curr = label2[b_idx, :, :]
        clusters_label1_curr = clusters_label1[b_idx, :, :]
        clusters_label2_curr = clusters_label2[b_idx, :, :]

        # fg_idx1_curr = torch.where(label1_curr[0] == 1)[0]  # N
        # fg_idx2_curr = torch.where(label2_curr[0] == 1)[0]

        clusters_fg_idx1_curr = torch.where(clusters_label1_curr[0] == 1)[0]  # N
        clusters_fg_idx2_curr = torch.where(clusters_label2_curr[0] == 1)[0]

        bg_idx1_curr = torch.where(label1_curr[0] == 0)[0]  # N

        fg_pc1_curr = torch.unsqueeze(pc1_curr[:, clusters_fg_idx1_curr], 0)
        fg_flow_curr = torch.unsqueeze(flow_curr[:, clusters_fg_idx1_curr], 0)
        fg_rigid_flow_curr = torch.unsqueeze(rigid_flow_curr[:, clusters_fg_idx1_curr], 0)
        fg_pc2_curr = torch.unsqueeze(pc2_curr[:, clusters_fg_idx2_curr], 0)

        bg_pc1_curr = torch.unsqueeze(pc1_curr[:, bg_idx1_curr], 0)
        bg_ego_flow_gt_curr = torch.unsqueeze(ego_flow_gt_curr[:, bg_idx1_curr], 0)
        bg_rigid_flow_curr = torch.unsqueeze(rigid_flow_curr[:, bg_idx1_curr], 0)

        # -------------- TransformationLoss -------------------------------------------------
        trans_loss += l1_loss(bg_pc1_curr + bg_rigid_flow_curr, bg_pc1_curr + bg_ego_flow_gt_curr)
        # print('c: {}'.format(bg_rigid_flow_curr.sum()))
        # -----------------------------------------------------------------------------------

        if fg_pc1_curr.shape[-1] > 30 and fg_pc2_curr.shape[-1] > 30:

            # --------------- ChamferLoss and SmoothnessLoss-----------------------------
            dist1_curr, dist2_curr = computeChamfer(fg_pc1_curr + fg_flow_curr, fg_pc2_curr)
            # Clamp the distance to prevent outliers (objects that appear and disappear from the scene)
            # print(dist1_curr)
            # dist1_curr = torch.clamp(dist1_curr, max=1.0)
            # dist2_curr = torch.clamp(dist2_curr, max=1.0)
            # print(dist1_curr)
            chamfer_loss += dist1_curr.sum(dim=1).mean() + dist2_curr.sum(dim=1).mean()
            smoothness_loss += computeSmooth(fg_pc1_curr, fg_flow_curr).sum(dim=1).mean()
            # -----------------------------z---------------------------------------------
            # print('a: {}, b: {}'.format((fg_pc1_curr + fg_rigid_flow_curr).sum(), (fg_pc1_curr + fg_flow_curr).sum()))
            rigid_loss_tmp = l1_loss(fg_pc1_curr + fg_rigid_flow_curr, fg_pc1_curr + fg_flow_curr)
            # -------------- RigidLoss -------------------------------------------------
            rigid_loss += rigid_loss_tmp
            # --------------------------------------------------------------------------

        else:
            # print('a: {}, b: {}'.format((fg_pc1_curr + fg_rigid_flow_curr).sum(), (fg_pc1_curr + fg_flow_curr).sum()))
            chamfer_loss += 0.0
            rigid_loss += 0.0
            # print('Fucking Gay')

    chamfer_loss /= B
    rigid_loss /= B
    trans_loss /= B
    smoothness_loss /= B
    return trans_loss, inlier_loss, chamfer_loss, rigid_loss, smoothness_loss

def L2Loss(pre_flow, gt_flow, label1):

    """

    :param pre_flow: B, N, 3
    :param gt_flow: B, N, 3
    :return:
    """
    diff_flow = pre_flow - gt_flow
    diff_flow = diff_flow * label1
    loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return loss

def L1LossLabel(pre_flow, gt_flow, label1):

    """

    :param pre_flow: B, N, 3
    :param gt_flow: B, N, 3
    :return:
    """
    l1_loss = L1Loss()
    loss = l1_loss(pre_flow * label1, gt_flow * label1)

    return loss

def sequence_loss(pos1, pos2, flows_pred, rigid_flows_pred, ego_flow_gt, label1, label2, weights_pred):

    # loss_iters_w = [0.7, 0.4, 0.4, 0.4]
    loss_iters_w = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

    clusters_foreground_label1 = weights_pred[-2].permute(0,2,1) # B, N, 1
    clusters_foreground_label2 = weights_pred[-1].permute(0,2,1)

    loss = torch.zeros(1).cuda()
    trans_loss_sum = torch.zeros(1).cuda()
    inlier_loss_sum = torch.zeros(1).cuda()
    chamfer_loss_sum = torch.zeros(1).cuda()
    rigid_loss_sum = torch.zeros(1).cuda()

    for i in range(len(rigid_flows_pred)):

        w = loss_iters_w[i]

        trans_loss, inlier_loss, chamfer_loss, rigid_loss, smoothness_loss = fore_back_loss(pos1, pos2,
                                                                        flows_pred[i], rigid_flows_pred[i], ego_flow_gt,
                                                                        weights_pred[i], label1, label2,
                                                                        clusters_foreground_label1,
                                                                        clusters_foreground_label2)

        str_ = 'trans_loss: %.5f, ' % trans_loss + 'inlier_loss: %.5f, ' % inlier_loss + \
               'chamfer_loss: %.5f, ' % chamfer_loss + 'smoothness_loss: %.5f, ' % smoothness_loss \
               + 'rigid_loss: %.5f' % rigid_loss

        print(str_)
        loss += w * (trans_loss + 0.005 * inlier_loss + 0.5 * chamfer_loss + rigid_loss)
        trans_loss_sum += trans_loss
        inlier_loss_sum += 0.005 * inlier_loss
        chamfer_loss_sum += 0.5 * chamfer_loss
        rigid_loss_sum += rigid_loss

    return loss, trans_loss_sum, inlier_loss_sum, chamfer_loss_sum, rigid_loss_sum

if __name__ == '__main__':
    import torch

    npoint = 512
    device = torch.device("cuda")
    model = FlowStep3D(npoint)
    model = model.to(device)
    print(f'num_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    input = torch.randn((1, npoint, 3))
    input = input.to(torch.float32).to(device)

    output = model(input, input, input, input, iters=2)
    print(r'outputs shape:')
    for output_i in output:
        print(f'{output_i.shape}')


        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False, use_instance_norm=use_instance_norm)

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256, 256],
                                     knn=True)
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel=128 + 128, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256], knn=True)
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel=64, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256], knn=True)
        self.fp = PointNetFeaturePropogation(in_channel=256 + 3, mlp=[256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 4, kernel_size=1, bias=True)

    def forward(self, pcds, features):
        ## B,3,N ##################
        l2_pc1 = pcds[2]
        l1_pc1 = pcds[1]
        l0_pc1 = pcds[0]

        l2_feature1_new = features[2]
        l2_feature1 = features[1]
        l1_feature1 = features[0]
        ###########################

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(l0_pc1, l1_pc1, feature1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)
        weights = sf[:, :1, :]
        flow = torch.sigmoid(sf[:, 1:, :])
        return flow, weights