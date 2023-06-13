import sys, os
import os.path as osp
import numpy as np
import open3d as o3d

def pc_loader(path):
    """
    Args:
        path:
    Returns:
        pc1: ndarray (N1, 3) np.float32
        pc2: ndarray (N2, 3) np.float32
        gt: ndarray (N1, 3) np.float32
    """
    data = np.load(path)

    pc1 = data['pc1'].astype(np.float32)
    pc2 = data['pc2'].astype(np.float32)
    flow = data['flow'].astype(np.float32)
    rel_trans = data['pose_s'].astype(np.float32)

    seg_label_1 = data['sem_label_s']
    seg_label_2 = data['sem_label_t']
    # mot_label_1 = np.zeros((seg_label_1.shape[0])).astype(np.float32)
    # mot_label_1[((seg_label_1 < 40) | (seg_label_1 > 99))] = 1.0
    # mot_label_1 = np.expand_dims(mot_label_1, axis=1)
    # mot_label_2 = np.ones(pc2.shape).astype(np.float32)

    mot_label_1 = np.expand_dims(seg_label_1, axis=1)
    mot_label_2 = np.expand_dims(seg_label_2, axis=1)

    rot = rel_trans[:3, :3]
    trans = np.expand_dims(rel_trans[:3, 3], axis=-1)

    aligned_pc1 = rot @ pc1.transpose([1, 0]) + trans
    aligned_pc1 = aligned_pc1.transpose([1, 0])

    return pc1, pc2, aligned_pc1, flow, rot, trans, mot_label_1, mot_label_2

root_dir = '/home/gtdong/Documents/Dataset/WSSF/lidar_kitti_with_ground'

file_list = os.listdir(root_dir)
fg_EPE3D_sum = 0
bg_EPE3D_sum = 0
for file_name in file_list:
    file_path = osp.join(root_dir, file_name)

    pc1, pc2, aligned_pc1, flow, rot, trans, mot_label_1, mot_label_2 = pc_loader(file_path)

    init_T = np.eye(4, dtype=np.float)

    init_T[0:3, 0:3] = rot
    init_T[0:3, 3:4] = trans
    pcd_s = o3d.geometry.PointCloud()
    pcd_t = o3d.geometry.PointCloud()
    pcd_s_t = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(pc1)
    pcd_t.points = o3d.utility.Vector3dVector(pc2)
    pcd_s.paint_uniform_color([1.0, 0.0, 0.0])
    pcd_t.paint_uniform_color([0.0, 0.0, 1.0])

    # o3d.visualization.draw_geometries([pcd_s, pcd_t])

    trans_T = o3d.registration.registration_icp(pcd_s, pcd_t,
                                              max_correspondence_distance=0.15, init=init_T,
                                              criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=300))

    R_ref = trans_T.transformation[0:3, 0:3].astype(np.float32)
    t_ref = trans_T.transformation[0:3, 3:4].astype(np.float32)

    aligned_pc1 = R_ref @ pc1.transpose([1, 0]) + t_ref
    aligned_pc1 = aligned_pc1.transpose([1, 0])

    ego_flow = aligned_pc1 - pc1

    process_flow = flow * mot_label_1 + ego_flow * (1 - mot_label_1)

    pcd_s_t.points = o3d.utility.Vector3dVector(pc1+flow)
    pcd_s_t.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([pcd_s_t, pcd_t])


    fg_idx1 = np.where(mot_label_1[:, 0] == 1.0)[0]
    bg_idx1 = np.where(mot_label_1[:, 0] == 0.0)[0]
    fg_flow = flow[fg_idx1, :]
    fg_ego_flow = ego_flow[fg_idx1, :]

    bg_flow = flow[bg_idx1, :]
    bg_ego_flow = ego_flow[bg_idx1, :]


    bg_l2_norm = np.linalg.norm(bg_flow - bg_ego_flow, axis=-1)
    bg_EPE3D_sum += bg_l2_norm.mean()
    if fg_flow.size != 0:
        fg_l2_norm = np.linalg.norm(fg_flow - fg_ego_flow, axis=-1)
        fg_EPE3D_sum += fg_l2_norm.mean()
    else:
        fg_EPE3D_sum += 0.0

    print(file_path)



bg_EPE3D = bg_EPE3D_sum / float(len(file_list)+1)
fg_EPE3D = fg_EPE3D_sum / float(len(file_list)+1)
print(fg_EPE3D)
print(bg_EPE3D)