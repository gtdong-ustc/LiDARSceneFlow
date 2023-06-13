"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging
import open3d as o3d

from tqdm import tqdm 
from networks.pointpwc import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from networks.pointpwc import multiScaleLoss
from pathlib import Path
from collections import defaultdict
from networks import get_model


import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d
# from networks.pointpwc import multiScaleLoss, multiScaleSelfSupervisedLoss, NormalLoss, SmothessLoss
from networks.pointpwc import multiScaleLoss
from networks.flownet3d import L2Loss, meanL2Loss, compute_loss_nearest_neighbor, GetAnchoredPointCloud, GetNearestNeighborPointCloud


def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args
    print(sys.argv[1])
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('/output/Evaluate_experiment/')
    # experiment_dir = Path('/home/ashesknight/Documents/result/PointPWC')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    visualization_dir = file_dir.joinpath('visualization/')
    visualization_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()
    # model = get_model(args.model_name)()
    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        # transform=transforms.ProcessDataNoFlow(args.data_process,
        #                                  args.num_points,
        #                                  args.allow_less_points),
        num_points=args.num_points,
        data_root = args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    pretrain = args.ckpt_dir + args.pretrain
    data = torch.load(pretrain)
    print("################################")
    print("Renamed_dict:")
    for param_tensor in data:
        print(param_tensor, "\t", data[param_tensor].size())
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        # pos1, pos2, norm1, norm2, flow, path = data
        pos1, pos2, aligned_pos1, norm1, norm2, aligned_norm1, flow, gt_rot, gt_trans, gt_label1, gt_label2, _ = data
        #move to cuda 
        ego_flow = aligned_pos1 - pos1
        pos1 = pos1.cuda() # (B,N,C)
        pos2 = pos2.cuda() 
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
        ego_flow = ego_flow.cuda()
        aligned_pos1 = aligned_pos1.cuda()
        aligned_norm1 = aligned_norm1.cuda()
        # residual_flow = residual_flow.cuda()
        # print('flow {}'.format(flow.shape))

        model = model.eval()
        with torch.no_grad():
            full_flow = 0.0
            loss = 0.0
            if args.model_name == 'PointPWC':
                pred_flows, fps_pc1_idxs, _, pc1s, pc2s = model(pos1, pos2, norm1, norm2)
                loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
                # pred_flows, fps_pc1_idxs, _, pc1s, pc2s = model(aligned_pos1, pos2, aligned_norm1, norm2)
                # loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                full_flow = pred_flows[0].permute(0, 2, 1)

                flow_1 = pred_flows[1].permute(0, 2, 1)
                flow_2 = pred_flows[2].permute(0, 2, 1)
                flow_3 = pred_flows[3].permute(0, 2, 1)

                pc1_1 = pc1s[1].permute(0, 2, 1)
                pc1_2 = pc1s[2].permute(0, 2, 1)
                pc1_3 = pc1s[3].permute(0, 2, 1)

                pc2_1 = pc2s[1].permute(0, 2, 1)
                pc2_2 = pc2s[2].permute(0, 2, 1)
                pc2_3 = pc2s[3].permute(0, 2, 1)

            elif args.model_name == 'FlowNet3D':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,C,N)
                full_flow = pred_flow.permute(0, 2, 1) # (B,N,C)
                loss = L2Loss(full_flow, flow)
            elif args.model_name == 'SequenceWeights':
                pre_aligned_pos1, pred_flows, _, _, pc1, pc2 = model(pos1, pos2, norm1, norm2)
                trans_pc2 = pos1 + pred_flows[0].permute(0, 2, 1)

                root_dir = '/output/results'
                if not os.path.exists(root_dir):
                    os.mkdir(root_dir)
                for i in range(pos1.shape[0]):
                    batchname = '%04d_' % batch_id
                    filename = '%04d.npy' % i
                    filename = batchname+filename
                    pc1_path = os.path.join(root_dir, 'pc1_'+filename)
                    pc2_path = os.path.join(root_dir, 'pc2_'+filename)
                    trans_pc2_path = os.path.join(root_dir, 'trans_pc2_'+filename)
                    pc1_np = pos1[i].cpu().numpy()
                    pc2_np = pos2[i].cpu().numpy()
                    flow_np = flow[i].cpu().numpy()
                    trans_pc2_np = trans_pc2[i].cpu().numpy()

                    np.save(pc1_path, pc1_np)
                    np.save(pc2_path, pc2_np)
                    np.save(trans_pc2_path, trans_pc2_np)

            else:
                print('Train: Model Name {} Error\n'.format(args.model_name))
            # epe3d = torch.norm((full_flow + ego_flow) - flow, dim = 2).mean()
            # epe3d = torch.norm(full_flow - flow, dim=2).mean()

        # total_loss += loss.cpu().data * args.batch_size
        # total_epe += epe3d.cpu().data * args.batch_size
        # total_seen += args.batch_size
        #
        # pc1_np = pos1.cpu().numpy() # (B,N,C) and B = 1
        # pc2_np = pos2.cpu().numpy()
        # sf_np = flow.cpu().numpy()
        # pred_sf = full_flow.cpu().numpy()
        #
        #
        #
        # pred_sf_1 = flow_1.cpu().numpy()
        # pred_sf_2 = flow_2.cpu().numpy()
        # pred_sf_3 = flow_3.cpu().numpy()
        #
        # pc1_np_1 = pc1_1.cpu().numpy()
        # pc1_np_2 = pc1_2.cpu().numpy()
        # pc1_np_3 = pc1_3.cpu().numpy()
        #
        # pc2_np_1 = pc2_1.cpu().numpy()
        # pc2_np_2 = pc2_2.cpu().numpy()
        # pc2_np_3 = pc2_3.cpu().numpy()
        #
        # ego_sf = ego_flow.cpu().numpy()
        #
        # pos1 = o3d.geometry.PointCloud()
        # pos2 = o3d.geometry.PointCloud()
        # pos3 = o3d.geometry.PointCloud()
        #
        # # pos1.points = o3d.utility.Vector3dVector(pc1_np_3.squeeze(axis=(0,)))
        # # pos2.points = o3d.utility.Vector3dVector(pc2_np_3.squeeze(axis=(0,)))
        # # pos3.points = o3d.utility.Vector3dVector(pc1_np_3.squeeze(axis=(0,)) + pred_sf_3.squeeze(axis=(0,)))
        #
        # pos1.points = o3d.utility.Vector3dVector(pc1_np_2.squeeze(axis=(0,)))
        # pos2.points = o3d.utility.Vector3dVector(pc2_np_2.squeeze(axis=(0,)))
        # pos3.points = o3d.utility.Vector3dVector(pc1_np_2.squeeze(axis=(0,)) + pred_sf_2.squeeze(axis=(0,)))
        #
        # pos1.paint_uniform_color([0.0, 0.0, 1.0])
        # pos2.paint_uniform_color([1.0, 0.0, 0.0])
        # pos3.paint_uniform_color([0.0, 1.0, 0.0])
        #
        # o3d.visualization.draw_geometries([pos1, pos3])

        ###  SAVE Visualization ###
        # filename = '%010d.npy' % i
        # pc1_dir = visualization_dir.joinpath('pc1/')
        # pc1_dir.mkdir(exist_ok=True)
        # pc2_dir = visualization_dir.joinpath('pc2/')
        # pc2_dir.mkdir(exist_ok=True)
        # warped_pc1_dir = visualization_dir.joinpath('warped_pc1/')
        # warped_pc1_dir.mkdir(exist_ok=True)
        # pc1_path = os.path.join(pc1_dir, filename)
        # pc2_path = os.path.join(pc2_dir, filename)
        # warped_pc1_path = os.path.join(warped_pc1_dir, filename)
        # np.save(pc1_path, pc1_np.squeeze(axis=(0,)))
        # np.save(pc2_path, pc2_np.squeeze(axis=(0,)))
        # np.save(warped_pc1_path, pc1_np.squeeze(axis=(0,)) + (pred_sf + ego_sf).squeeze(axis=(0,)))
        ###########################

        # EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d((pred_sf + ego_sf), sf_np)
        sf_np = flow_np
        pred_sf = trans_pc2_np - pc1_np
        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        # flow_pred, flow_gt = geometry.get_batch_2d_flow(pc1_np,
        #                                                 pc1_np+sf_np,
        #                                                 pc1_np+ pred_sf + ego_sf,
        #                                                 path)
        # EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

        epe2ds.update(0)
        acc2ds.update(0)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f'%(blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




