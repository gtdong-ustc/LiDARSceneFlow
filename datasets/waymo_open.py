import sys, os
import os.path as osp
import numpy as np
import pptk
import open3d

import torch.utils.data as data
import MinkowskiEngine as ME
import random
__all__ = ['WaymoOpen']


class WaymoOpen(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 full = True):
        self.root = osp.join(data_root, 'waymo_open')
        self.train = train
        self.transform = transform
        self.num_points = num_points

        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, aligned_pc1_loaded, rot_loaded, trans_loaded, label1_loaded, label2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, \
        pc2_transformed, \
        aligned_pc1_transformed, \
        label1_transformed, \
        label2_transformed, \
        fg_labels1_transformed, \
        fg_labels2_transformed = self.transform([pc1_loaded, pc2_loaded, aligned_pc1_loaded, label1_loaded, label2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        #pc1_norm = pptk.estimate_normals(pc1_transformed, k = 16, r = np.inf, verbose = False)
        #pc2_norm = pptk.estimate_normals(pc2_transformed, k = 16, r = np.inf, verbose = False)
        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        aligned_pc1_norm = aligned_pc1_transformed
        #####################################
        sf_transformed = np.zeros(pc1_transformed.shape).astype(np.float32)
        #####################################
        return pc1_transformed, pc2_transformed, aligned_pc1_transformed, \
               pc1_norm, pc2_norm, aligned_pc1_norm, \
                sf_transformed, rot_loaded, trans_loaded, \
               label1_transformed, label2_transformed, \
               fg_labels1_transformed, fg_labels2_transformed, self.samples[index]



    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):

        print('full {}'.format(full))

        root = osp.realpath(osp.expanduser(self.root))

        root = osp.join(root, 'train') if self.train else osp.join(root, 'validation')

        time_dir_list = os.listdir(root)
        random.shuffle(time_dir_list)
        # time_dir_list.sort()
        useful_paths = []
        for i in range(60):
        # for i in range(40):
            time_dir = osp.join(root, time_dir_list[i])
            all_paths = sorted(os.listdir(time_dir))
            useful_paths += [osp.join(time_dir, item) for item in all_paths]

        print('useful_paths {}'.format(len(useful_paths)))

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        print(len(res_paths))

        # print(res_paths[0])
        # print(res_paths[-1])

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        data = np.load(path)

        pc1 = data['pc1'].astype(np.float32)
        pc2 = data['pc2'].astype(np.float32)
        pose1 = data['pose_s'].astype(np.float32)
        pose2 = data['pose_t'].astype(np.float32)

        # mot_label_1 = np.expand_dims(data['mot_label_s'].astype(np.float32), axis=1)
        # mot_label_2 = np.expand_dims(data['mot_label_t'].astype(np.float32), axis=1)

        sem_label_1 = data['sem_label_s'].astype(np.float32)
        sem_label_2 = data['sem_label_t'].astype(np.float32)

        mot_label_1 = np.zeros((sem_label_1.shape[0])).astype(np.float32)
        mot_label_1[((sem_label_1 < 40) | (sem_label_1 > 99))] = 1.0
        mot_label_1 = np.expand_dims(mot_label_1, axis=1)

        mot_label_2 = np.zeros((sem_label_2.shape[0])).astype(np.float32)
        mot_label_2[((sem_label_2 < 40) | (sem_label_2 > 99))] = 1.0
        mot_label_2 = np.expand_dims(mot_label_2, axis=1)

        rel_trans = np.linalg.inv(pose2) @ pose1
        rot = rel_trans[:3, :3]
        trans = np.expand_dims(rel_trans[:3, 3], axis=-1)

        aligned_pc1 = rot @ pc1.transpose([1,0]) + trans
        aligned_pc1 = aligned_pc1.transpose([1,0])

        ########Visualization#######################
        # visual_pos1 = open3d.geometry.PointCloud()
        # visual_pos2 = open3d.geometry.PointCloud()
        # visual_pos3 = open3d.geometry.PointCloud()
        # aligned_pos1 = gt_rots @ pos1 + gt_trans
        # visual_pos1.points = open3d.utility.Vector3dVector(pc1)
        # visual_pos2.points = open3d.utility.Vector3dVector(pc2)
        # visual_pos3.points = open3d.utility.Vector3dVector(aligned_pc1)
        # visual_pos1.paint_uniform_color([0.0, 0.0, 1.0])
        # visual_pos2.paint_uniform_color([1.0, 0.0, 0.0])
        # visual_pos3.paint_uniform_color([0.0, 1.0, 0.0])
        # open3d.visualization.draw_geometries([visual_pos1, visual_pos2, visual_pos3])
        ###########################################

        return pc1, pc2, aligned_pc1, rot, trans, mot_label_1, mot_label_2
