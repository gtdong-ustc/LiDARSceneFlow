import MinkowskiEngine as ME
import numpy as np
def run(input):

    B = input.shape[0]
    voxel_size = 0.3
    stride = 0.02
    scale_256 = 2.4
    scale_64 = 6.0
    scale_16 = 18.0
    idx_1024_list = []
    idx_256_list = []
    idx_64_list = []
    idx_16_list = []
    for i in range(B):
        input_tensor = input[i]

        sel1 = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)
        if sel1.shape[0] > 1024:
            sel1_list = []
            while sel1.shape[0] > 1024:
                voxel_size+=stride
                sel1 = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)
                sel1_list.append(sel1)
            sel1 = sel1_list[len(sel1_list)-1]
        else:
            while sel1.shape[0] < 1024:
                voxel_size-=stride
                sel1 = ME.utils.sparse_quantize(input_tensor / voxel_size, return_index=True)


        sel1 = np.random.choice(sel1, size=1024, p=None)
        idx_1024_list.append(sel1)

        sel2 = ME.utils.sparse_quantize(input_tensor / (voxel_size * scale_256), return_index=True)
        sel2 = np.random.choice(sel2, size=256, replace=True, p=None)
        idx_256_list.append(sel2)

        sel3 = ME.utils.sparse_quantize(input_tensor / (voxel_size * scale_64), return_index=True)
        sel3 = np.random.choice(sel3, size=64, replace=True, p=None)
        idx_64_list.append(sel3)

        sel4 = ME.utils.sparse_quantize(input_tensor / (voxel_size * scale_16), return_index=True)
        sel4 = np.random.choice(sel4, size=16, replace=True, p=None)
        idx_16_list.append(sel4)
    dict = {1024:idx_1024_list, 256:idx_256_list, 64:idx_64_list, 16:idx_16_list}
    return dict