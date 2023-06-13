"""
Train on FlyingThings3D
Author: Wenxuan Wu
Date: May 2020
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging
from collections import OrderedDict
import open3d

from tqdm import tqdm 
# from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
# from models import multiScaleLoss
# from networks.pointpwc import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from networks.pointpwc import multiScaleLoss
# from networks.pointpwc_feature_alignment import multiScaleLoss_FA
# from networks.rsfenet import RegistrationLoss
# from networks.pointcorres_v3 import batch_rotation_error, batch_translation_error, decompose_rotation_translation
# from networks.flownet3d_ori import FlowNet3D
# from networks.flownet3d_ori import L1Loss, L2Loss, meanL2Loss, compute_loss_nearest_neighbor, GetAnchoredPointCloud, GetNearestNeighborPointCloud
from networks import get_model
# from networks.pointconv_sf import NormLoss
from networks.flowstep3d import sequence_loss
from pointconv_util import knn_point, index_points_group

from lib.utils import transform_point_cloud, kabsch_transformation_estimation, n_model_parameters
from networks.pointconv_pose import OutlierLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *

from model import load_model
from core.knn import find_knn_batch
from core.correspondence import find_correct_correspondence
from core.loss import UnbalancedLoss, BalancedLoss
from core.metrics import batch_rotation_error, batch_translation_error
import core.registration as GlobalRegistration

from util.timer import Timer, AverageMeter
from util.file import ensure_dir

import MinkowskiEngine as ME

eps = np.finfo(float).eps
np2th = torch.from_numpy

class SceneFlowTrainer:
    def __init__(self, train_loader, val_loader):
        if 'NUMBA_DISABLE_JIT' in os.environ:
            del os.environ['NUMBA_DISABLE_JIT']

        self.args = cmd_args.parse_args_from_yaml(sys.argv[1])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

        '''CREATE DIR'''
        experiment_dir = Path('/output/experiment/')
        experiment_dir.mkdir(exist_ok=True)
        file_dir = Path(str(experiment_dir) + '/PointConv%sFlyingthings3d-' % args.model_name + str(
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
        file_dir.mkdir(exist_ok=True)
        checkpoints_dir = file_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = file_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)
        os.system('cp %s %s' % ('models.py', log_dir))
        os.system('cp %s %s' % ('pointconv_util.py', log_dir))
        os.system('cp %s %s' % ('train.py', log_dir))
        os.system('cp %s %s' % ('config_train.yaml', log_dir))

        '''LOG'''
        logger = logging.getLogger(args.model_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt' % args.model_name)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('----------------------------------------TRAINING----------------------------------')
        logger.info('PARAMETER ...')
        logger.info(args)

        blue = lambda x: '\033[94m' + x + '\033[0m'

        model = get_model(args.model_name)(args)

        feat_model_n_out = 16
        bn_momentum = 0.05
        feat_conv1_kernel_size = 3
        normalize_feature = True

        self.feat_model = get_model(args.feat_model_name)(
            in_channels=6,
            out_channels=feat_model_n_out,
            bn_momentum=bn_momentum,
            conv1_kernel_size=feat_conv1_kernel_size,
            normalize_feature=normalize_feature
        )
        self.flow_model = get_model(args.inlier_model_name)(
            in_channels=6,
            out_channels=3,
            bn_momentum=bn_momentum,
            conv1_kernel_size=feat_conv1_kernel_size,
            normalize_feature=False
        )

    def generate_inlier_input(self, xyz0, xyz1, iC0, iC1, iF0, iF1, len_batch, pos_pairs):
        # pairs consist of (xyz1 index, xyz0 index)
        stime = time.time()
        sinput0 = ME.SparseTensor(iF0, coords=iC0).to(self.device)
        oF0 = self.feat_model(sinput0).F

        sinput1 = ME.SparseTensor(iF1, coords=iC1).to(self.device)
        oF1 = self.feat_model(sinput1).F
        feat_time = time.time() - stime

        stime = time.time()
        pred_pairs = find_pairs(oF0, oF1, len_batch)
        nn_time = time.time() - stime

        is_correct = find_correct_correspondence(pos_pairs, pred_pairs, len_batch=len_batch)

        cat_pred_pairs = []
        start_inds = torch.zeros((1, 2)).long()
        for lens, pred_pair in zip(len_batch, pred_pairs):
            cat_pred_pairs.append(pred_pair + start_inds)
            start_inds += torch.LongTensor(lens)

        cat_pred_pairs = torch.cat(cat_pred_pairs, 0)
        pred_pair_inds0, pred_pair_inds1 = cat_pred_pairs.t()
        reg_coords = torch.cat((iC0[pred_pair_inds0], iC1[pred_pair_inds1, 1:]), 1)
        reg_feats = self.generate_inlier_features(xyz0, xyz1, iC0, iC1, oF0, oF1,
                                                  pred_pair_inds0, pred_pair_inds1).float()

        return reg_coords, reg_feats, pred_pairs, is_correct, feat_time, nn_time

    def find_pairs(self, F0, F1, len_batch):
        nn_batch = find_knn_batch(F0,
                                  F1,
                                  len_batch,
                                  nn_max_n=self.config.nn_max_n,
                                  knn=self.config.inlier_knn,
                                  return_distance=False,
                                  search_method=self.config.knn_search_method)

        pred_pairs = []
        for nns, lens in zip(nn_batch, len_batch):
            pred_pair_ind0, pred_pair_ind1 = torch.arange(
                len(nns)).long()[:, None], nns.long().cpu()
            nn_pairs = []
            for j in range(nns.shape[1]):
                nn_pairs.append(
                    torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))

            pred_pairs.append(torch.cat(nn_pairs, 0))
        return pred_pairs

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    # experiment_dir = Path('/home/ashesknight/Documents/result/RegPointPWC/output')
    experiment_dir = Path('/output')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/PointConv%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('train.py', log_dir))
    os.system('cp %s %s' % ('config_train.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    # model = get_model(args.model_name)(args)
    model = get_model(args.model_name)()
    if args.allow_augmentation == True:
        transform = transforms.Augmentation(args.aug_together,
                                            args.aug_pc2,
                                            args.data_process,
                                            args.num_points)
    else:
        # transform = transforms.ProcessDataNoFlow(args.data_process,
        #                                  args.num_points,
        #                                  args.allow_less_points)
        transform = transforms.ProcessData(args.data_process,
                                           args.num_points,
                                           args.allow_less_points)

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transform,
        num_points=args.num_points,
        data_root = args.data_root,
        full=args.full
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transform,
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

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()
    if args.pretrain is not None:
        renamed_dict = OrderedDict()
        if args.multi_gpu is not None:
            data = torch.load(args.pretrain)
            for k, v in data.items():
                k_split = k.split('.')
                if k_split[0]!='module':
                    name = 'module.'+k
                else:
                    name = k
                renamed_dict[name] = v
            model.load_state_dict(renamed_dict)
        else:
            # pretrained_dict = torch.load(args.pretrain)
            # model_dict = model.state_dict()
            # # 1. filter out unnecessary keys
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # # 2. overwrite entries in the existing state dict
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)

            checkpoint = torch.load(args.pretrain, map_location=lambda storage, loc: storage)
            pretrained_dict = checkpoint['state_dict']
            renamed_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                k_split = k.split('.')
                del(k_split[0])

                new_k = ".".join(k_split)
                renamed_dict[new_k] = v
            # pretrained_dict = {k.lstrip('model.'): v for k, v in pretrained_dict.items()}
            model.load_state_dict(renamed_dict)

        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    if args.allow_self_supervised == True:
        print('Training manner is self_supervised')
        print('Loss function is {}'.format(args.loss_function))
    else:
        print('Loss function is {}'.format(args.loss_function))
    pretrain = args.pretrain
    # init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0
    init_epoch = 0 if args.pretrain is not None else 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)

    # Output number of model parameters
    logger.info("Parameter Count: {:d}".format(n_model_parameters(model)))

    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5, last_epoch = init_epoch - 1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.98, last_epoch=init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()

        print('Epoch {}'.format(epoch))

        if epoch % 2 == 0:

            print("Starting Validation")

            eval_epe3d, eval_loss = eval_sceneflow(model.eval(),val_loader)
            str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
            print(str_out)
            logger.info(str_out)

            if eval_epe3d < best_epe:
                best_epe = eval_epe3d
                if args.multi_gpu is not None:
                    torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                else:
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                logger.info('Save model ...')
                print('Save model ...')
            print('Best epe loss is: %.5f'%(best_epe))
            logger.info('Best epe loss is: %.5f'%(best_epe))

        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, aligned_pos1, norm1, norm2, aligned_norm1, flow, residual_flow, gt_rot, gt_trans, _ = data

            ego_flow = flow -residual_flow

            #move to cuda
            pos1 = pos1.cuda() # (B,N,C)
            pos2 = pos2.cuda()
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda()
            ego_flow = ego_flow.cuda()
            aligned_pos1 = aligned_pos1.cuda()
            aligned_norm1 = aligned_norm1.cuda()
            residual_flow =residual_flow.cuda()
            gt_rot = gt_rot.cuda()
            gt_trans = gt_trans.cuda()
            # pose = pose.cuda()

            model.to(pos1.device)

            model = model.train()
            loss = torch.zeros(1, requires_grad=True)
            loss1 = torch.zeros(1, requires_grad=True)
            loss2 = torch.zeros(1, requires_grad=True)

            optimizer.zero_grad()



            if args.model_name == 'PointPWC':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'FlowNet3D':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
            elif args.model_name == 'FlowNet3DFull':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
            elif args.model_name == 'FlowNet3DFullWeights':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
            elif args.model_name == 'FlowStep3D':
                pred_flows = model(pos1, pos2, norm1, norm2, 4)
                loss = sequence_loss(pos1, pos2, pred_flows, flow)
            elif args.model_name == 'PointPWC_3scales':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'PointPWC_3scales_nounet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'PointPWC_aggregation':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'PointPWC_aggregation_warp':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'PointPWC_aggregation_unet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'PointPWC_nounet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'FPConv':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'FAConv':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                loss = multiScaleLoss_FA(pred_flows, residual_flow, fps_pc1_idxs)
            elif args.model_name == 'RSFENet':
                pred_flows, fps_pc1_idxs, _, _, _, pre_rotation, pre_translation = model(pos1, pos2, norm1, norm2)
                loss1 = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                loss2 = RegistrationLoss(pos1, pos2, pre_rotation, pre_translation, gt_rot, gt_trans)
                loss = loss1 + loss2
            elif args.model_name == 'Simple':
                pre_flow, pre_coarse_flow = model(pos1, pos2, norm1, norm2)
                # loss = L1Loss(pre_flow, flow) + L1Loss(pre_coarse_flow, flow)
                loss = NormLoss(pre_flow, flow)
            elif args.model_name == 'SimplePose':
                R_est, t_est, permutation = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                # pos2_gt = transform_point_cloud(pos1, gt_rot, gt_trans)
                # print((pos2 - pos1).mean())
                # print((pos2_gt - pos1).mean())
                # root_dir = '/output/results'
                # for i in range(pos1.shape[0]):
                #     filename = '%04d.npy' % i
                #     pc1_path = os.path.join(root_dir, 'pc1_' + filename)
                #     pc2_path = os.path.join(root_dir, 'pc2_' + filename)
                #     trans_pc2_path = os.path.join(root_dir, 'trans_pc2_' + filename)
                #     pc1_np = pos1[i].cpu().numpy()
                #     pc2_np = pos2[i].cpu().numpy()
                #     trans_pc2_np = aligned_pos1[i].cpu().numpy()
                #
                #     np.save(pc1_path, pc1_np)
                #     np.save(pc2_path, pc2_np)
                #     np.save(trans_pc2_path, trans_pc2_np)
                loss = NormLoss(pos2_est, aligned_pos1) + OutlierLoss()(permutation) * 0.005
            elif args.model_name == 'SimplePoseFlow':
                R_est, t_est, permutation, pre_flow, pre_coarse_flow = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)

                loss = NormLoss(pos2_est, aligned_pos1) + OutlierLoss()(permutation) * 0.005 + NormLoss(pre_flow, flow)
            elif args.model_name == 'SimplePoseFlow3ScalesNounet':
                R_est, t_est, permutation, pred_flows, fps_pc1_idxs= model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                loss = NormLoss(pos2_est, aligned_pos1) * 0.1 + OutlierLoss()(permutation) * 0.005 + multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            elif args.model_name == 'SimplePoseCostVolume':
                R_est, t_est = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                loss = NormLoss(pos2_est, aligned_pos1)
            elif args.model_name == 'SimplePoseCostVolumeTrue':
                R_est, t_est = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                loss = NormLoss(pos2_est, aligned_pos1)
            elif args.model_name == 'SimplePoseFlowCostVolume':
                R_est, t_est, pred_flows, fps_pc1_idxs = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                loss = NormLoss(pos2_est, aligned_pos1) * 0.1 + multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
            elif args.model_name == 'RecurrentPoseCostVolume':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
            elif args.model_name == 'RecurrentPoseCostVolume3Scales':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
            elif args.model_name == 'RecurrentPoseSceneFlowCostVolume3Scales' or args.model_name == 'RecurrentPoseSceneFlowCostVolume3ScalesV2':
                pred_ego_flows, pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                loss1 = multiScaleLoss(pred_ego_flows, ego_flow, fps_pc1_idxs)
                loss2 = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
                print('loss1: {}, loss2: {}'.format(loss1, loss2))
                loss = loss1 + loss2
            elif args.model_name == 'RecurrentPoseSinkhorn':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
            elif args.model_name == 'RecurrentPoseCostVolumeSingleScale' or args.model_name == 'RecurrentPoseCostVolumeV2' or args.model_name == 'RecurrentPoseCostVolumeV3':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                loss = NormLoss(pos1+pred_flows[0].permute(0,2,1), aligned_pos1)
            else:
                print('Train: Model Name {} Error\n'.format(args.model_name))

            # print('I {}, train loss {}'.format(i, loss.cpu().data))

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d | %s | learning rate:%f | mean loss: %f'%(epoch, 'Train', lr, train_loss)
        print(str_out)
        logger.info(str_out)
        if epoch % 2 == 0:
            eval_epe3d, eval_loss = eval_sceneflow(model.eval(),val_loader)
            str_out = 'EPOCH %d %s mean epe3d: %f  mean eval loss: %f'%(epoch, blue('eval'), eval_epe3d, eval_loss)
            print(str_out)
            logger.info(str_out)

            if eval_epe3d < best_epe:
                best_epe = eval_epe3d
                if args.multi_gpu is not None:
                    torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                else:
                    torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
                logger.info('Save model ...')
                print('Save model ...')
            print('Best epe loss is: %.5f'%(best_epe))
            logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):

        pos1, pos2, aligned_pos1, norm1, norm2, aligned_norm1, flow, residual_flow, gt_rot, gt_trans, _ = data

        ego_flow = flow - residual_flow

        # move to cuda
        pos1 = pos1.cuda()  # (B,N,C)
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
        ego_flow = ego_flow.cuda()
        aligned_pos1 = aligned_pos1.cuda()
        aligned_norm1 = aligned_norm1.cuda()
        residual_flow = residual_flow.cuda()
        gt_rot = gt_rot.cuda()
        gt_trans = gt_trans.cuda()

        with torch.no_grad():

            epe3d = 0.0
            eval_loss = 0.0
            if args.model_name == 'FlowNet3D':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                eval_loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
                epe3d = torch.norm(pred_flow - flow, dim=2).mean()
            elif args.model_name == 'FlowNet3DFull':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                eval_loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
                epe3d = torch.norm(pred_flow - flow, dim=2).mean()
            elif args.model_name == 'FlowNet3DFullWeights':
                pred_flow = model(pos1, pos2, norm1, norm2)  # pred_flow: (B,3,N)
                pred_flow = pred_flow.permute(0, 2, 1)  # pred_flow: (B,N,3)
                diff_flow = pred_flow - flow
                eval_loss = torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
                epe3d = torch.norm(pred_flow - flow, dim=2).mean()
            elif args.model_name == 'FlowStep3D':
                pred_flows = model(pos1, pos2, norm1, norm2, 4)
                eval_loss = sequence_loss(pos1, pos2, pred_flows, flow)
                epe3d = 0.0
                for h in range(len(pred_flows)):
                    cur_epe3d = torch.norm(pred_flows[h] - flow, dim=2).mean()
                    print('{} : {}'.format(h, cur_epe3d))
                    epe3d += cur_epe3d
                epe3d /= 4.0
            elif args.model_name == 'PointPWC':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_3scales':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_3scales_nounet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_aggregation':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_aggregation_warp':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_aggregation_unet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'PointPWC_nounet':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'FPConv':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'FAConv':
                pred_flows, fps_pc1_idxs, _, _, _ = model(aligned_pos1, pos2, aligned_norm1, norm2)
                eval_loss = multiScaleLoss_FA(pred_flows, residual_flow, fps_pc1_idxs)
                epe3d = torch.norm((pred_flows[0].permute(0, 2, 1) + ego_flow) - flow, dim=2).mean()
            elif args.model_name == 'RSFENet':
                pred_flows, fps_pc1_idxs, _, _, _, pre_rotation, pre_translation = model(pos1, pos2, norm1, norm2)
                loss1 = multiScaleLoss(pred_flows, residual_flow, fps_pc1_idxs)
                loss2 = RegistrationLoss(pos1, pos2, pre_rotation, pre_translation, gt_rot, gt_trans)
                eval_loss = loss1 + loss2
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim=2).mean()
            elif args.model_name == 'FlowNet3D':
                pred_flow = model(pos1, pos2, norm1, norm2)
                pred_flow = pred_flow.permute(0, 2, 1)

            elif args.model_name == 'Simple':
                pre_flow, pre_coarse_flow = model(pos1, pos2, norm1, norm2)
                # eval_loss = L1Loss(pre_flow, residual_flow) + L1Loss(pre_coarse_flow, residual_flow)
                eval_loss = NormLoss(pre_flow, flow)
                epe3d = torch.norm(pre_flow - flow, dim=2).mean()
            elif args.model_name == 'SimplePose':
                R_est, t_est, permutation = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                # pos2_gt = transform_point_cloud(pos1, gt_rot, gt_trans)
                eval_loss = NormLoss(pos2_est, aligned_pos1) + OutlierLoss()(permutation) * 0.005
                pre_pose_flow = pos2_est - pos1
                pre_gt_flow = aligned_pos1 - pos1
                epe3d = torch.norm(pre_pose_flow - pre_gt_flow, dim=2).mean()
            elif args.model_name == 'SimplePoseFlow':
                R_est, t_est, permutation, pre_flow, pre_coarse_flow = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                eval_loss = NormLoss(pos2_est, aligned_pos1) + OutlierLoss()(permutation) * 0.005 + NormLoss(pre_flow, flow)
                epe3d = torch.norm(pre_flow - flow, dim=2).mean()

            elif args.model_name == 'SimplePoseFlow3ScalesNounet':
                R_est, t_est, permutation, pred_flows, fps_pc1_idxs= model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                eval_loss = NormLoss(pos2_est, aligned_pos1) * 0.1 + OutlierLoss()(permutation) * 0.005 + multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim=2).mean()

            elif args.model_name == 'SimplePoseCostVolume':
                R_est, t_est = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                eval_loss = NormLoss(pos2_est, aligned_pos1)
                pre_pose_flow = pos2_est - pos1
                pre_gt_flow = aligned_pos1 - pos1
                epe3d = torch.norm(pre_pose_flow - pre_gt_flow, dim=2).mean()
            elif args.model_name == 'SimplePoseCostVolumeTrue':
                R_est, t_est = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                eval_loss = NormLoss(pos2_est, aligned_pos1)
                pre_pose_flow = pos2_est - pos1
                pre_gt_flow = aligned_pos1 - pos1
                epe3d = torch.norm(pre_pose_flow - pre_gt_flow, dim=2).mean()
            elif args.model_name == 'SimplePoseFlowCostVolume':
                R_est, t_est, pred_flows, fps_pc1_idxs = model(pos1, pos2, norm1, norm2)
                pos2_est = transform_point_cloud(pos1, R_est, t_est)
                eval_loss = NormLoss(pos2_est, aligned_pos1) * 0.1 + multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim=2).mean()
            elif args.model_name == 'RecurrentPoseCostVolume':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - ego_flow, dim=2).mean()
            elif args.model_name == 'RecurrentPoseCostVolume3Scales':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - ego_flow, dim=2).mean()
            elif args.model_name == 'RecurrentPoseSceneFlowCostVolume3Scales' or args.model_name == 'RecurrentPoseSceneFlowCostVolume3ScalesV2':
                pred_ego_flows, pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                eval_loss = multiScaleLoss(pred_ego_flows, ego_flow, fps_pc1_idxs) + multiScaleLoss(pred_flows, flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim=2).mean()
            elif args.model_name == 'RecurrentPoseSinkhorn':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                eval_loss = multiScaleLoss(pred_flows, ego_flow, fps_pc1_idxs)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - ego_flow, dim=2).mean()
            elif args.model_name == 'RecurrentPoseCostVolumeSingleScale' or args.model_name == 'RecurrentPoseCostVolumeV2' or args.model_name == 'RecurrentPoseCostVolumeV3':
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                eval_loss = NormLoss(pos1+pred_flows[0].permute(0,2,1), aligned_pos1)
                epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - ego_flow, dim=2).mean()
            else:
                print('Eval: Model Name {} Error\n'.format(args.model_name))


        metrics['epe3d_loss'].append(epe3d.cpu().data.numpy())
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_epe3d = np.mean(metrics['epe3d_loss'])
    mean_eval = np.mean(metrics['eval_loss'])

    return mean_epe3d, mean_eval

if __name__ == '__main__':
    main()




