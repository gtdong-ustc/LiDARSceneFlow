import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['KITTI_odom']


class KITTI_odom(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True,
                 full=True):
        self.root = osp.join(data_root, 'sampled_sequences')
        #assert train is False
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        res_paths = []
        root = osp.realpath(osp.expanduser(self.root))
        all_paths = sorted(os.walk(root))
        sequences_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        sequences_filenames = [item[2] for item in all_paths if len(item[1]) == 0]

        # train_data_number = len(sequences_paths)
        train_data_number = 6 # the train data of deep global registration

        for i in range(train_data_number):
            for j in range(5, len(sequences_filenames[i]), 5):
                res_paths.append(osp.join(sequences_paths[i], sequences_filenames[i][j]))
        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        stride = 1

        number = int(path[-10:].rstrip('.bin'))
        number_str = '%06d' % (number + stride)
        path_2 = path[:-10] + number_str + '.bin'
        if not osp.exists(path_2):
            path_2 = path
        # print(path)
        # print(path_2)
        pc1 = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        pc2 = np.fromfile(path_2, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]

        return pc1, pc2
