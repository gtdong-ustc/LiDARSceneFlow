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

from tqdm import tqdm

from networks import get_model
from lib.utils import n_model_parameters
from pathlib import Path
from collections import defaultdict, OrderedDict
from networks.recurrent_rigid_scene_flow_v6_oripc_er_2 import sequence_loss
from networks.flowstep3d import sequence_loss as ori_sequence_loss
from networks.flowstep3d import self_flowstep3d_sequence_loss
from evaluation_utils import evaluate_2d, evaluate_3d
import transforms
import datasets
import cmd_args
from main_utils import *

_ITE_NUM = 4


def get_instance_cluster_gt(label1):
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
            if torch.where(inlier_label1_curr[0] == class_label)[0].shape[0] >= 30:
                clusters_1[str(b_idx)].append(fg_idx1_curr[torch.where(inlier_label1_curr[0] == class_label)[0]])

    return clusters_1


def get_clusters_fore_label(pc1, clusters_1):
    clusters_foreground_label1_list = []
    for b_idx in range(pc1.shape[0]):
        clusters_foreground_label1_curr = torch.zeros_like(pc1[0, :1, :]).float()
        for c_idx in clusters_1[str(b_idx)]:
            clusters_foreground_label1_curr[:, c_idx] = 1.0
        clusters_foreground_label1_list.append(clusters_foreground_label1_curr)
    clusters_foreground_label1 = torch.stack(clusters_foreground_label1_list, dim=0)
    return clusters_foreground_label1


def main():
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

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
    os.system('cp %s %s' % ('_train.py', log_dir))
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

    model = get_model(args.model_name)()

    if args.dataset == 'LidarKITTI':
        transform_function = transforms.ProcessDataKITTI(args.data_process,
                                         args.num_points,
                                         args.allow_less_points)
    elif args.dataset == 'WaymoOpen':
        transform_function = transforms.ProcessDataWaymo(args.data_process,
                                         args.num_points,
                                         args.allow_less_points)
    elif args.dataset == 'SemanticKITTI':
        transform_function = transforms.ProcessDataWaymo(args.data_process,
                                                         args.num_points,
                                                         args.allow_less_points)
    else:
        transform_function = None

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transform_function,
        num_points=args.num_points,
        data_root=args.data_root,
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

    val_dataset = datasets.__dict__['LidarKITTI'](
        train=False,
        transform=transforms.ProcessDataKITTI(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        num_points=args.num_points,
        data_root=args.data_root
    )


    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
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
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    if args.pretrain is not None:

        if args.multi_gpu is not None:
            ####### Only update the Encoder #########
            pretrained_dict = torch.load(args.pretrain)
            renamed_dict = model.state_dict()

            pretrained_temp_dict = OrderedDict()

            for k, v in pretrained_dict.items():
                k_split = k.split('.')
                new_k = ".".join(k_split)
                new_k = 'module.' + new_k
                pretrained_temp_dict[new_k] = v
            pretrained_temp_dict = {k: v for k, v in pretrained_temp_dict.items() if k in renamed_dict}
            renamed_dict.update(pretrained_temp_dict)
            model.load_state_dict(renamed_dict)
        else:
            pretrained_dict = torch.load(args.pretrain)
            renamed_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in renamed_dict}
            for k, v in pretrained_dict.items():
                k_split = k.split('.')
                new_k = ".".join(k_split)
                renamed_dict[new_k] = v
            model.load_state_dict(renamed_dict)


        print('load model %s' % args.pretrain)
        logger.info('load model %s' % args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    pretrain = args.pretrain
    if args.continue_pretrain and args.pretrain is not None:
        init_epoch = int(pretrain[-14:-11])
    else:
        init_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)

    # Output number of model parameters
    logger.info("Parameter Count: {:d}".format(n_model_parameters(model)))

    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        total_trans_loss = 0
        total_inlier_loss = 0
        total_chamfer_loss = 0
        total_rigid_loss = 0
        total_seen = 0
        optimizer.zero_grad()


        ### TRAIN ####
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):

            pos1, pos2, aligned_pos1, \
            norm1, norm2, aligned_norm1, \
            flow, gt_rot, gt_trans, \
            gt_label1, gt_label2, \
            fg_labels1, fg_labels2, _ = data

            ego_flow = aligned_pos1 - pos1

            # move to cuda
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            norm1 = norm1.cuda()
            norm2 = norm2.cuda()
            flow = flow.cuda()  # like as pos1
            ego_flow = ego_flow.cuda()
            gt_label1 = gt_label1.cuda()
            gt_label2 = gt_label2.cuda()

            model = model.train()
            if args.model_name == 'SequenceWeights':
                pred_flows, pred_rigid_flows, pred_weights = model(pos1, pos2,
                                                                   norm1, norm2,
                                                                   gt_label1, gt_label2,
                                                                   fg_labels1, fg_labels2, _ITE_NUM)

                loss,  _, _, _, _ = sequence_loss(pos1, pos2, pred_flows, pred_rigid_flows, ego_flow,gt_label1, gt_label2, pred_weights)
            elif args.model_name == 'FlowRigidStep3D':
                pred_flows, pred_rigid_flows, pred_weights = model(pos1, pos2,
                                                                   norm1, norm2,
                                                                   gt_label1, gt_label2,
                                                                   fg_labels1, fg_labels2, _ITE_NUM)
                loss,  _, _, _, _ = sequence_loss(pos1, pos2, pred_flows, pred_rigid_flows, ego_flow,gt_label1, gt_label2, pred_weights)
            elif args.model_name == 'FlowStep3D':

                pc1 = pos1.permute(0, 2, 1).contiguous()
                pc2 = pos2.permute(0, 2, 1).contiguous()

                fg_label1 = fg_labels1.permute(0, 2, 1).contiguous()  # B 1 N
                fg_label2 = fg_labels2.permute(0, 2, 1).contiguous()  # B 1 N

                clusters_1 = get_instance_cluster_gt(fg_label1)
                clusters_2 = get_instance_cluster_gt(fg_label2)
                clusters_foreground_label1 = get_clusters_fore_label(pc1, clusters_1)
                clusters_foreground_label2 = get_clusters_fore_label(pc2, clusters_2)

                pred_flows = model(pos1, pos2, norm1, norm2, _ITE_NUM)
                loss, _, _, _ = self_flowstep3d_sequence_loss(pos1, pos2, pred_flows, ego_flow, gt_label1, gt_label2, clusters_foreground_label1, clusters_foreground_label2)
            elif args.model_name == 'SequenceWeightsNR':

                pc1 = pos1.permute(0, 2, 1).contiguous()
                pc2 = pos2.permute(0, 2, 1).contiguous()

                fg_label1 = fg_labels1.permute(0, 2, 1).contiguous()  # B 1 N
                fg_label2 = fg_labels2.permute(0, 2, 1).contiguous()  # B 1 N

                clusters_1 = get_instance_cluster_gt(fg_label1)
                clusters_2 = get_instance_cluster_gt(fg_label2)
                clusters_foreground_label1 = get_clusters_fore_label(pc1, clusters_1)
                clusters_foreground_label2 = get_clusters_fore_label(pc2, clusters_2)

                pred_flows,_ = model(pos1, pos2, norm1, norm2, _ITE_NUM)
                loss, _, _, _ = self_flowstep3d_sequence_loss(pos1, pos2, pred_flows, ego_flow, gt_label1, gt_label2, clusters_foreground_label1, clusters_foreground_label2)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        train_trans_loss = total_trans_loss / total_seen
        train_inlier_loss = total_inlier_loss / total_seen
        train_chamfer_loss = total_chamfer_loss / total_seen
        train_rigid_loss = total_rigid_loss / total_seen

        str_out = 'EPOCH %d %s mean loss: %f' % (epoch, 'train', train_loss)
        print(str_out)
        logger.info(str_out)
        str_out1 = 'EPOCH %d %s mean trans_loss: %f' % (epoch, 'train', train_trans_loss)
        str_out2 = 'EPOCH %d %s mean inlier_loss: %f' % (epoch, 'train', train_inlier_loss)
        str_out3 = 'EPOCH %d %s mean chamfer_loss: %f' % (epoch, 'train', train_chamfer_loss)
        str_out4 = 'EPOCH %d %s mean rigid_loss: %f' % (epoch, 'train', train_rigid_loss)

        print(str_out1)
        print(str_out2)
        print(str_out3)
        print(str_out4)
        logger.info(str_out1)
        logger.info(str_out2)
        logger.info(str_out3)
        logger.info(str_out4)

        #### EVAL #####

        eval_epe3d_list, eval_acc3d_strict_list, eval_acc3d_relax_list, eval_outlier_list, eval_loss= eval_sceneflow(model.eval(), val_loader)
        str_out_1 = 'EPOCH %d %s mean eval loss: %f mean epe3d 1 : %f  mean acc3d strict 1 : %f, mean acc3d relax 1 : %f  mean outlier 1 : %f  ' % \
                  (epoch, 'eval', eval_loss, eval_epe3d_list[0], eval_acc3d_strict_list[0], eval_acc3d_relax_list[0], eval_outlier_list[0])
        str_out_2 = 'EPOCH %d %s mean eval loss: %f mean epe3d 2 : %f  mean acc3d strict 2 : %f, mean acc3d relax 2 : %f  mean outlier 2 : %f  ' % \
                  (epoch, 'eval', eval_loss, eval_epe3d_list[1], eval_acc3d_strict_list[1], eval_acc3d_relax_list[1], eval_outlier_list[1])
        str_out_3 = 'EPOCH %d %s mean eval loss: %f mean epe3d 3 : %f  mean acc3d strict 3 : %f, mean acc3d relax 3 : %f  mean outlier 3 : %f  ' % \
                  (epoch, 'eval', eval_loss, eval_epe3d_list[2], eval_acc3d_strict_list[2], eval_acc3d_relax_list[2], eval_outlier_list[2])
        str_out_4 = 'EPOCH %d %s mean eval loss: %f mean epe3d 4 : %f  mean acc3d strict 4 : %f, mean acc3d relax 4 : %f  mean outlier 4 : %f  ' % \
                  (epoch, 'eval', eval_loss, eval_epe3d_list[3], eval_acc3d_strict_list[3], eval_acc3d_relax_list[3], eval_outlier_list[3])
        print(str_out_1)
        logger.info(str_out_1)
        print(str_out_2)
        logger.info(str_out_2)
        print(str_out_3)
        logger.info(str_out_3)
        print(str_out_4)
        logger.info(str_out_4)

        str_out1 = 'EPOCH %d %s mean eval_loss: %f' % (epoch, 'eval', eval_loss)
        print(str_out1)
        logger.info(str_out1)

        if eval_epe3d_list[-1] < best_epe:
            best_epe = eval_epe3d_list[-1]
        torch.save(optimizer.state_dict(), '%s/optimizer.pth' % (checkpoints_dir))
        if args.multi_gpu is not None:
            torch.save(model.module.state_dict(),
                       '%s/%s_%.3d_%.4f.pth' % (checkpoints_dir, args.model_name, epoch, best_epe))
        else:
            torch.save(model.state_dict(),
                       '%s/%s_%.3d_%.4f.pth' % (checkpoints_dir, args.model_name, epoch, best_epe))
        logger.info('Save model ...')
        print('Save model ...')
        print('Best epe loss is: %.5f' % (best_epe))
        logger.info('Best epe loss is: %.5f' % (best_epe))

def eval_sceneflow(model, loader):
    metrics = defaultdict(lambda: list())
    time_mean = 0
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, aligned_pos1, \
        norm1, norm2, aligned_norm1, \
        flow, gt_rot, gt_trans, \
        gt_label1, gt_label2, \
        fg_labels1, fg_labels2, _ = data

        ego_flow = aligned_pos1-pos1
        # move to cuda
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()
        ego_flow = ego_flow.cuda()
        gt_label1 = gt_label1.cuda()
        gt_label2 = gt_label2.cuda()

        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if args.model_name == 'SequenceWeights':

                pred_flows, pred_rigid_flows, pred_weights = model(pos1, pos2,
                                                                   norm1, norm2,
                                                                   gt_label1, gt_label2,
                                                                   fg_labels1, fg_labels2, _ITE_NUM)
                eval_loss, _, _, _, _ = sequence_loss(pos1, pos2, pred_flows, pred_rigid_flows, ego_flow, gt_label1,
                                                      gt_label2, pred_weights)
            elif args.model_name == 'FlowStep3D':
                pred_rigid_flows = model(pos1, pos2, norm1, norm2, _ITE_NUM)
                print(pred_rigid_flows[-1].shape)
                eval_loss = ori_sequence_loss(pos1, pos2, pred_rigid_flows, flow)
            elif args.model_name == 'SequenceWeightsNR':
                pred_rigid_flows, _ = model(pos1, pos2, norm1, norm2, _ITE_NUM)
                print(pred_rigid_flows[-1].shape)
                eval_loss = ori_sequence_loss(pos1, pos2, pred_rigid_flows, flow)
            end.record()
            torch.cuda.synchronize()
            print('Inference time: {}'.format(start.elapsed_time(end)))


            root_dir = '/output/results'
            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            trans_pc2_4 = pos1 + pred_rigid_flows[-1]
            trans_pc2_3 = pos1+pred_rigid_flows[-2]
            trans_pc2_2 = pos1+pred_rigid_flows[-3]
            trans_pc2_1 = pos1+pred_rigid_flows[-4]
            # print(trans_pc2.shape)
            filename = '%04d.npy' % batch_id
            pc1_path = os.path.join(root_dir, 'pc1_'+filename)
            pc2_path = os.path.join(root_dir, 'pc2_'+filename)
            trans_pc2_1_path = os.path.join(root_dir, 'trans_pc2_1_'+filename)
            trans_pc2_2_path = os.path.join(root_dir, 'trans_pc2_2_'+filename)
            trans_pc2_3_path = os.path.join(root_dir, 'trans_pc2_3_'+filename)
            trans_pc2_4_path = os.path.join(root_dir, 'trans_pc2_4_'+filename)
            gt_trans_pc2_path = os.path.join(root_dir, 'gt_trans_pc2_'+filename)
            sf_np = flow[0].cpu().numpy()
            pc1_np = pos1[0].cpu().numpy()
            pc2_np = pos2[0].cpu().numpy()
            trans_pc2_1_np = trans_pc2_1[0].cpu().numpy()
            trans_pc2_2_np = trans_pc2_2[0].cpu().numpy()
            trans_pc2_3_np = trans_pc2_3[0].cpu().numpy()
            trans_pc2_4_np = trans_pc2_4[0].cpu().numpy()
            gt_trans_pc2_np = sf_np + pc1_np

            np.save(pc1_path, pc1_np)
            np.save(pc2_path, pc2_np)
            np.save(trans_pc2_1_path, trans_pc2_1_np)
            np.save(trans_pc2_2_path, trans_pc2_2_np)
            np.save(trans_pc2_3_path, trans_pc2_3_np)
            np.save(trans_pc2_4_path, trans_pc2_4_np)
            np.save(gt_trans_pc2_path, gt_trans_pc2_np)
            pred_sf_1 = trans_pc2_1_np - pc1_np
            pred_sf_2 = trans_pc2_2_np-pc1_np
            pred_sf_3 = trans_pc2_3_np-pc1_np
            pred_sf_4 = trans_pc2_4_np-pc1_np
            epe3d_1, acc3d_strict_1, acc3d_relax_1, outlier_1 = evaluate_3d(pred_sf_1, sf_np)
            epe3d_2, acc3d_strict_2, acc3d_relax_2, outlier_2 = evaluate_3d(pred_sf_2, sf_np)
            epe3d_3, acc3d_strict_3, acc3d_relax_3, outlier_3 = evaluate_3d(pred_sf_3, sf_np)
            epe3d_4, acc3d_strict_4, acc3d_relax_4, outlier_4 = evaluate_3d(pred_sf_4, sf_np)

        metrics['epe3d_1_loss'].append(epe3d_1)
        metrics['epe3d_2_loss'].append(epe3d_2)
        metrics['epe3d_3_loss'].append(epe3d_3)
        metrics['epe3d_4_loss'].append(epe3d_4)
        metrics['acc3d_strict_1'].append(acc3d_strict_1)
        metrics['acc3d_strict_2'].append(acc3d_strict_2)
        metrics['acc3d_strict_3'].append(acc3d_strict_3)
        metrics['acc3d_strict_4'].append(acc3d_strict_4)
        metrics['acc3d_relax_1'].append(acc3d_relax_1)
        metrics['acc3d_relax_2'].append(acc3d_relax_2)
        metrics['acc3d_relax_3'].append(acc3d_relax_3)
        metrics['acc3d_relax_4'].append(acc3d_relax_4)
        metrics['outlier_1'].append(outlier_1)
        metrics['outlier_2'].append(outlier_2)
        metrics['outlier_3'].append(outlier_3)
        metrics['outlier_4'].append(outlier_4)
        metrics['eval_loss'].append(eval_loss.cpu().data.numpy())

    mean_eval = np.mean(metrics['eval_loss'])
    mean_epe3d_1 = np.mean(metrics['epe3d_1_loss'])
    mean_acc3d_strict_1 = np.mean(metrics['acc3d_strict_1'])
    mean_acc3d_relax_1 = np.mean(metrics['acc3d_relax_1'])
    mean_outlier_1 = np.mean(metrics['outlier_1'])

    mean_epe3d_2 = np.mean(metrics['epe3d_2_loss'])
    mean_acc3d_strict_2 = np.mean(metrics['acc3d_strict_2'])
    mean_acc3d_relax_2 = np.mean(metrics['acc3d_relax_2'])
    mean_outlier_2 = np.mean(metrics['outlier_2'])

    mean_epe3d_3 = np.mean(metrics['epe3d_3_loss'])
    mean_acc3d_strict_3 = np.mean(metrics['acc3d_strict_3'])
    mean_acc3d_relax_3 = np.mean(metrics['acc3d_relax_3'])
    mean_outlier_3 = np.mean(metrics['outlier_3'])

    mean_epe3d_4 = np.mean(metrics['epe3d_4_loss'])
    mean_acc3d_strict_4 = np.mean(metrics['acc3d_strict_4'])
    mean_acc3d_relax_4 = np.mean(metrics['acc3d_relax_4'])
    mean_outlier_4 = np.mean(metrics['outlier_4'])
    mean_epe3d_list = [mean_epe3d_1, mean_epe3d_2, mean_epe3d_3, mean_epe3d_4]
    mean_acc3d_strict_list = [mean_acc3d_strict_1, mean_acc3d_strict_2, mean_acc3d_strict_3, mean_acc3d_strict_4]
    mean_acc3d_relax_list = [mean_acc3d_relax_1, mean_acc3d_relax_2, mean_acc3d_relax_3, mean_acc3d_relax_4]
    mean_outlier_list = [mean_outlier_1, mean_outlier_2, mean_outlier_3, mean_outlier_4]
    return mean_epe3d_list, mean_acc3d_strict_list, mean_acc3d_relax_list, mean_outlier_list, mean_eval


if __name__ == '__main__':
    main()
