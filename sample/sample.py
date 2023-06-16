# import MinkowskiEngine as ME
import numpy as np
import torch

def random_voxel_sample(xyz, npoints=1024):

    """

    :param xyz [B, N, C]:
    :param npoints int:
    :return:
    """
    device = xyz.device
    xyz = xyz.cpu()
    B = xyz.shape[0]

    stride = 0.02
    idx_list = []
    for i in range(B):
        voxel_size = 0.3
        input_tensor = xyz[i]
        sel = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)
        if sel.shape[0] > npoints:
            sel_list = []
            while sel.shape[0] > npoints:
                voxel_size += stride
                sel = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)
                sel_list.append(sel)
            sel = sel_list[len(sel_list)-1]
        else:
            while sel.shape[0] < npoints:
                voxel_size -= stride
                sel = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)


        sel = np.random.choice(sel, size=npoints, p=None)
        idx_list.append(sel)

    idx = torch.from_numpy(np.stack(idx_list)).int().to(device) #[B, npoints]

    return idx