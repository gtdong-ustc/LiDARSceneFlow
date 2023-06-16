"""
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import os, sys
import os.path as osp
from collections import defaultdict
import numbers
import math
import numpy as np
import traceback
import time

import torch

import numba
from numba import njit

from . import functional as F
import MinkowskiEngine as ME
from sklearn.cluster import DBSCAN

# ---------- BASIC operations ----------
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            return pic
        else:
            return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ---------- Build permutalhedral lattice ----------
@njit(numba.int64(numba.int64[:], numba.int64, numba.int64[:], numba.int64[:], ))
def key2int(key, dim, key_maxs, key_mins):
    """
    :param key: np.array
    :param dim: int
    :param key_maxs: np.array
    :param key_mins: np.array
    :return:
    """
    tmp_key = key - key_mins
    scales = key_maxs - key_mins + 1
    res = 0
    for idx in range(dim):
        res += tmp_key[idx]
        res *= scales[idx + 1]
    res += tmp_key[dim]
    return res


@njit(numba.int64[:](numba.int64, numba.int64, numba.int64[:], numba.int64[:], ))
def int2key(int_key, dim, key_maxs, key_mins):
    key = np.empty((dim + 1,), dtype=np.int64)
    scales = key_maxs - key_mins + 1
    for idx in range(dim, 0, -1):
        key[idx] = int_key % scales[idx]
        int_key -= key[idx]
        int_key //= scales[idx]
    key[0] = int_key

    key += key_mins
    return key


@njit
def advance_in_dimension(d1, increment, adv_dim, key):
    key_cp = key.copy()

    key_cp -= increment
    key_cp[adv_dim] += increment * d1
    return key_cp


class Traverse:
    def __init__(self, neighborhood_size, d):
        self.neighborhood_size = neighborhood_size
        self.d = d

    def go(self, start_key, hash_table_list):
        walking_keys = np.empty((self.d + 1, self.d + 1), dtype=np.long)
        self.walk_cuboid(start_key, 0, False, walking_keys, hash_table_list)

    def walk_cuboid(self, start_key, d, has_zero, walking_keys, hash_table_list):
        if d <= self.d:
            walking_keys[d] = start_key.copy()

            range_end = self.neighborhood_size + 1 if (has_zero or (d < self.d)) else 1
            for i in range(range_end):
                self.walk_cuboid(walking_keys[d], d + 1, has_zero or (i == 0), walking_keys, hash_table_list)
                walking_keys[d] = advance_in_dimension(self.d + 1, 1, d, walking_keys[d])
        else:
            hash_table_list.append(start_key.copy())


# ---------- MAIN operations ----------
class ProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2, aligned_pc1 = data
        if pc1 is None:
            return None, None, None,

        # print('shape {} {}'.format(pc1.shape, aligned_pc1.shape))

        sf = pc2[:, :3] - pc1[:, :3]
        rf = pc2[:, :3] - aligned_pc1[:, :3] ## residual scene flow = flow - ego motion flow

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        aligned_pc1 = aligned_pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        rf = rf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, aligned_pc1, sf, rf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class ProcessDataNoFlow(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2, sf = data
        if pc1 is None:
            return None, None, None,

        # sf = pc2[:, :3] - pc1[:, :3]

        # if self.DEPTH_THRESHOLD > 0:
        #     near_mask_1 = np.array(pc1[:, 2] < self.DEPTH_THRESHOLD)
        #     near_mask_2 = np.array(pc2[:, 2] < self.DEPTH_THRESHOLD)
        # else:
        near_mask_1 = np.ones(pc1.shape[0], dtype=np.bool)
        near_mask_2= np.ones(pc2.shape[0], dtype=np.bool)
        indices_1 = np.where(near_mask_1)[0]
        indices_2 = np.where(near_mask_2)[0]
        if len(indices_1) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices_1, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices_2, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices_1, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices_2, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices_1
                    sampled_indices2 = indices_2
        else:
            sampled_indices1 = indices_1
            sampled_indices2 = indices_2

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class ProcessDataNoFlow_Seq3(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc0, pc1, pc2, sf = data
        if pc1 is None:
            return None, None, None,

        # sf = pc2[:, :3] - pc1[:, :3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask_0 = np.array(pc0[:, 2] < self.DEPTH_THRESHOLD)
            near_mask_1 = np.array(pc1[:, 2] < self.DEPTH_THRESHOLD)
            near_mask_2 = np.array(pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask_0 = np.ones(pc0.shape[0], dtype=np.bool)
            near_mask_1 = np.ones(pc1.shape[0], dtype=np.bool)
            near_mask_2= np.ones(pc2.shape[0], dtype=np.bool)
        indices_0 = np.where(near_mask_0)[0]
        indices_1 = np.where(near_mask_1)[0]
        indices_2 = np.where(near_mask_2)[0]
        if len(indices_1) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices_1, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices_2, size=self.num_points, replace=False, p=None)
                    sampled_indices0 = np.random.choice(indices_0, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices0 = sampled_indices1
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices_1, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices_2, size=self.num_points, replace=True, p=None)
                        sampled_indices0 = np.random.choice(indices_0, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                        sampled_indices0 = sampled_indices1
                else:
                    sampled_indices0 = indices_0
                    sampled_indices1 = indices_1
                    sampled_indices2 = indices_2
        else:
            sampled_indices0 = indices_0
            sampled_indices1 = indices_1
            sampled_indices2 = indices_2
        pc0 = pc0[sampled_indices0]
        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc0, pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class NoProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2 = data
        if pc1 is None:
            return None, None, None,

        sf = pc2 - pc1
        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class ProcessDataWaymo(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

        # ---------------- Foreground Clustering ------------------
        self.min_p_cluster = 30
        self.min_samples_dbscan = 5
        self.cluster_metric = 'euclidean'
        self.eps_dbscan = 0.75
        self.cluster_estimator = DBSCAN(min_samples=self.min_samples_dbscan,
                                        metric=self.cluster_metric, eps=self.eps_dbscan)
        # ---------------------------------------------------------


    def __call__(self, data):
        pc1, pc2, aligned_pc1, label1, label2 = data
        if pc1 is None:
            return None, None, None,

        if self.DEPTH_THRESHOLD > 0:
            near_mask_1 = pc1[:, 2] < self.DEPTH_THRESHOLD
            near_mask_2 = pc2[:, 2] < self.DEPTH_THRESHOLD
        else:
            near_mask_1 = np.ones(pc1.shape[0], dtype=np.bool)
            near_mask_2 = np.ones(pc2.shape[0], dtype=np.bool)

        pc1 = pc1[near_mask_1, :]
        aligned_pc1 = aligned_pc1[near_mask_1, :]
        label1 = label1[near_mask_1, :]
        label2 = label2[near_mask_2, :]
        pc2 = pc2[near_mask_2, :]

        # -------------- remove outlier -----------
        is_not_ground_s = (pc1[:, 1] > -2.4)
        is_not_ground_t = (pc2[:, 1] > -2.4)
        pc1 = pc1[is_not_ground_s, :]
        aligned_pc1 = aligned_pc1[is_not_ground_s]
        pc2 = pc2[is_not_ground_t, :]
        # ------------------------------------------

        # -------------- remove ground --------------
        is_not_ground_s = (pc1[:, 1] > -1.4)
        is_not_ground_t = (pc2[:, 1] > -1.4)
        pc1 = pc1[is_not_ground_s, :]
        aligned_pc1 = aligned_pc1[is_not_ground_s]
        pc2 = pc2[is_not_ground_t, :]
        # ------------------------------------------

        # -------------------- Grid Sample ------------------------------------
        # _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc1) / 0.1, return_index=True)
        # _, sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(pc2) / 0.1, return_index=True)
        # pc1 = pc1[sel1]
        # aligned_pc1 = aligned_pc1[sel1]
        # label1 = label1[sel1]
        # label2 = label2[sel2]
        # pc2 = pc2[sel2]
        # --------------------------------------------------------------------

        # indices1 = pc1.shape[0]
        # indices2 = pc2.shape[0]

        # if self.num_points < np.min([indices1, indices2]):
        #     sampled_indices1 = np.random.choice(indices1, size=self.num_points, replace=False, p=None)
        #     sampled_indices2 = np.random.choice(indices2, size=self.num_points, replace=False, p=None)
        # else:
        #     sampled_indices1 = np.random.choice(indices1, size=self.num_points, replace=True, p=None)
        #     sampled_indices2 = np.random.choice(indices2, size=self.num_points, replace=True, p=None)

        if pc1.shape[0] > self.num_points:
            sampled_indices1 = np.random.choice(pc1.shape[0], self.num_points, replace=False)
        else:
            sampled_indices1 = np.random.choice(pc1.shape[0], self.num_points, replace=True)

        if pc2.shape[0] > self.num_points:
            sampled_indices2 = np.random.choice(pc2.shape[0], self.num_points, replace=False)
        else:
            sampled_indices2 = np.random.choice(pc2.shape[0], self.num_points, replace=True)

        pc1 = pc1[sampled_indices1]
        aligned_pc1 = aligned_pc1[sampled_indices1]
        label1 = label1[sampled_indices1]
        label2 = label2[sampled_indices2]
        pc2 = pc2[sampled_indices2]

        fg_labels1_full = np.zeros_like(label1) - 1
        fg_idx1 = np.where(label1[:, 0] == 1.0)[0]  # N
        fg_pc1 = pc1[fg_idx1, :]

        fg_labels2_full = np.zeros_like(label2) - 1
        fg_idx2 = np.where(label2[:, 0] == 1.0)[0]  # N
        fg_pc2 = pc2[fg_idx2, :]

        if fg_pc1.shape[0] > 30 and fg_pc2.shape[0] > 30:

            # print('fg_pc1 {}, fg_pc1 {}'.format(fg_pc1.shape, fg_pc2.shape))

            fg_labels1 = self.cluster_estimator.fit_predict(fg_pc1)  # input: N, 3
            fg_labels2 = self.cluster_estimator.fit_predict(fg_pc2)  # input: N, 3

            # print(fg_labels1)

            fg_labels1_full[fg_idx1, 0] = fg_labels1
            fg_labels2_full[fg_idx2, 0] = fg_labels2

        return pc1, pc2, aligned_pc1, label1, label2, fg_labels1_full, fg_labels2_full

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class ProcessDataKITTI(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

        # ---------------- Foreground Clustering ------------------
        self.min_p_cluster = 30
        self.min_samples_dbscan = 5
        self.cluster_metric = 'euclidean'
        self.eps_dbscan = 0.75
        self.cluster_estimator = DBSCAN(min_samples=self.min_samples_dbscan,
                                        metric=self.cluster_metric, eps=self.eps_dbscan)
        # ---------------------------------------------------------

    def __call__(self, data):
        pc1, pc2, aligned_pc1, sf, label1, label2 = data
        if pc1 is None:
            return None, None, None, None, None, None, None, None, None

        # -------------- remove ground --------------
        is_not_ground_s = (pc1[:, 1] > -1.4)
        is_not_ground_t = (pc2[:, 1] > -1.4)
        pc1 = pc1[is_not_ground_s, :]
        aligned_pc1 = aligned_pc1[is_not_ground_s, :]
        sf = sf[is_not_ground_s, :]
        label1 = label1[is_not_ground_s,:]
        pc2 = pc2[is_not_ground_t, :]
        label2 = label2[is_not_ground_t, :]
        # ------------------------------------------

        # print(label1.shape)

        if pc1.shape[0] > pc2.shape[0]:
            sampled_indices1 = np.random.choice(pc1.shape[0], pc2.shape[0], replace=False)
            pc2 = pc2
            pc1 = pc1[sampled_indices1,:]
            label1 = label1[sampled_indices1, :]
            sf = sf[sampled_indices1, :]
            aligned_pc1 = aligned_pc1[sampled_indices1,:]
        elif pc1.shape[0] < pc2.shape[0]:
            sampled_indices2 = np.random.choice(pc2.shape[0], pc1.shape[0], replace=False)
            pc1 = pc1
            pc2 = pc2[sampled_indices2, :]
            label2 = label2[sampled_indices2, :]
        else:
            pc1 = pc1
            pc2 = pc2

        fg_labels1_full = np.zeros_like(label1) - 1
        fg_idx1 = np.where(label1[:, 0] == 1.0)[0]  # N
        fg_pc1 = pc1[fg_idx1, :]

        fg_labels2_full = np.zeros_like(label2) - 1
        fg_idx2 = np.where(label2[:, 0] == 1.0)[0]  # N
        fg_pc2 = pc2[fg_idx2, :]

        if fg_pc1.shape[0] > 30 and fg_pc2.shape[0] > 30:

            fg_labels1 = self.cluster_estimator.fit_predict(fg_pc1)  # input: N, 3
            fg_labels2 = self.cluster_estimator.fit_predict(fg_pc2)  # input: N, 3

            fg_labels1_full[fg_idx1, 0] = fg_labels1
            fg_labels2_full[fg_idx2, 0] = fg_labels2

        return pc1, pc2, aligned_pc1, sf, label1, label2, fg_labels1_full, fg_labels2_full
        # ------ split the high distance ---------
        # if self.DEPTH_THRESHOLD > 0:
        #     near_mask_1 = pc1[:, 2] < self.DEPTH_THRESHOLD
        #     near_mask_2 = pc2[:, 2] < self.DEPTH_THRESHOLD
        # else:
        #     near_mask_1 = np.ones(pc1.shape[0], dtype=np.bool)
        #     near_mask_2 = np.ones(pc2.shape[0], dtype=np.bool)
        #
        # pc1 = pc1[near_mask_1, :]
        # sf = sf[near_mask_1, :]
        # aligned_pc1 = aligned_pc1[near_mask_1]
        # label1 = label1[near_mask_1, :]
        # label2 = label2[near_mask_2, :]
        # pc2 = pc2[near_mask_2, :]
        # ---------------------------------------


        # ------------- remove ground ------------
        # is_not_ground_s = (pc1[:, 1] > -1.4)
        # is_not_ground_t = (pc2[:, 1] > -1.4)
        # pc1 = pc1[is_not_ground_s, :]
        # aligned_pc1 = aligned_pc1[is_not_ground_s]
        # sf = sf[is_not_ground_s, :]
        # pc2 = pc2[is_not_ground_t, :]
        # ---------------------------------------

        # ------------------Voxel------------------------
        # sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc1) / 0.1, return_index=True)
        # sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(pc2) / 0.1, return_index=True)
        # pc1 = pc1[sel1]
        # aligned_pc1 = aligned_pc1[sel1]
        # sf = sf[sel1]
        # label1 = label1[sel1]
        # label2 = label2[sel2]
        # pc2 = pc2[sel2]
        #
        # indices1 = pc1.shape[0]
        # indices2 = pc2.shape[0]
        #
        # if self.num_points < np.min([indices1, indices2]):
        #     sampled_indices1 = np.random.choice(indices1, size=self.num_points, replace=False, p=None)
        #     sampled_indices2 = np.random.choice(indices2, size=self.num_points, replace=False, p=None)
        # else:
        #     # sampled_indices1 = np.arange(indices1)
        #     # sampled_indices2 = np.arange(indices1)
        #     sampled_indices1 = np.random.choice(indices1, size=self.num_points, replace=True, p=None)
        #     sampled_indices2 = np.random.choice(indices2, size=self.num_points, replace=True, p=None)
        #
        # pc1 = pc1[sampled_indices1]
        # aligned_pc1 = aligned_pc1[sampled_indices1]
        # sf = sf[sampled_indices1]
        # label1 = label1[sampled_indices1]
        # label2 = label2[sampled_indices2]
        # pc2 = pc2[sampled_indices2]
        # ---------------------------------------------

        # return pc1, pc2, aligned_pc1, sf, label1, label2

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
class Augmentation(object):
    def __init__(self, aug_together_args, aug_pc2_args, data_process_args, num_points, allow_less_points=False):
        self.together_args = aug_together_args
        self.pc2_args = aug_pc2_args
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2, aligned_pc1 = data
        if pc1 is None:
            return None, None, None

        sf = pc2[:, :3] - pc1[:, :3]
        rf = pc2[:, :3] - aligned_pc1[:, :3]

        min_length = pc1.shape[0] if pc1.shape[0] < pc2.shape[0] else pc2.shape[0]

        pc1 = pc1[:min_length, :]
        pc2 = pc2[:min_length, :]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)

        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        pc2 = pc2[sampled_indices2]
        sf = sf[sampled_indices1]
        aligned_pc1 = aligned_pc1[sampled_indices1]
        rf = rf[sampled_indices1]

        # together, order: scale, rotation, shift, jitter
        # scale
        scale = np.diag(np.random.uniform(self.together_args['scale_low'],
                                          self.together_args['scale_high'],
                                          3).astype(np.float32))
        # rotation
        angle = np.random.uniform(-self.together_args['degree_range'],
                                  self.together_args['degree_range'])
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rot_matrix = np.array([[cosval, 0, sinval],
                               [0, 1, 0],
                               [-sinval, 0, cosval]], dtype=np.float32)
        matrix = scale.dot(rot_matrix.T)

        # shift
        shifts = np.random.uniform(-self.together_args['shift_range'],
                                   self.together_args['shift_range'],
                                   (1, 3)).astype(np.float32)

        # jitter
        jitter = np.clip(self.together_args['jitter_sigma'] * np.random.randn(pc1.shape[0], 3),
                         -self.together_args['jitter_clip'],
                         self.together_args['jitter_clip']).astype(np.float32)
        bias = shifts + jitter

        pc1[:, :3] = pc1[:, :3].dot(matrix) + bias
        pc2[:, :3] = pc2[:, :3].dot(matrix) + bias
        aligned_pc1[:, :3] = aligned_pc1[:, :3].dot(matrix) + bias

        # pc2, order: rotation, shift, jitter
        # rotation
        # angle2 = np.random.uniform(-self.pc2_args['degree_range'],
        #                            self.pc2_args['degree_range'])
        # cosval2 = np.cos(angle2)
        # sinval2 = np.sin(angle2)
        # matrix2 = np.array([[cosval2, 0, sinval2],
        #                     [0, 1, 0],
        #                     [-sinval2, 0, cosval2]], dtype=pc1.dtype)
        # # shift
        # shifts2 = np.random.uniform(-self.pc2_args['shift_range'],
        #                             self.pc2_args['shift_range'],
        #                             (1, 3)).astype(np.float32)
        #
        # pc2[:, :3] = pc2[:, :3].dot(matrix2.T) + shifts2
        #
        # if not self.no_corr:
        #     jitter2 = np.clip(self.pc2_args['jitter_sigma'] * np.random.randn(pc1.shape[0], 3),
        #                       -self.pc2_args['jitter_clip'],
        #                       self.pc2_args['jitter_clip']).astype(np.float32)
        #     pc2[:, :3] += jitter2

        return pc1, pc2, aligned_pc1, sf, rf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(together_args: \n'
        for key in sorted(self.together_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.together_args[key])
        format_string += '\npc2_args: \n'
        for key in sorted(self.pc2_args.keys()):
            format_string += '\t{:10s} {}\n'.format(key, self.pc2_args[key])
        format_string += '\ndata_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string
