import MinkowskiEngine as ME
import numpy as np

class ProcessDataWaymo(object):
    def __init__(self, depth_threshold=35.0, num_points=8192, grid_size=0.1):
        self.DEPTH_THRESHOLD = depth_threshold
        self.num_points = num_points
        self.grid_size = grid_size

    def __call__(self, data):
        pc1, pc2, aligned_pc1, label1, label2 = data
        if pc1 is None:
            return None, None, None,

        # -------------- remove long range -----------
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
        # --------------------------------------------

        # -------------- remove outlier -----------
        is_not_ground_s = (pc1[:, 1] > -2.4)
        is_not_ground_t = (pc2[:, 1] > -2.4)
        pc1 = pc1[is_not_ground_s, :]
        aligned_pc1 = aligned_pc1[is_not_ground_s]
        pc2 = pc2[is_not_ground_t, :]
        # ------------------------------------------

        # -------------- grid voxel ---------------
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(pc1) / self.grid_size, return_index=True)
        _, sel2 = ME.utils.sparse_quantize(np.ascontiguousarray(pc2) / self.grid_size, return_index=True)
        pc1 = pc1[sel1]
        aligned_pc1 = aligned_pc1[sel1]
        label1 = label1[sel1]
        label2 = label2[sel2]
        pc2 = pc2[sel2]
        # ------------------------------------------

        # -------------- random sample ---------------
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
        # ------------------------------------------

        return pc1, pc2, aligned_pc1, label1, label2