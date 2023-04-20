import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange

from common.opt import posegraphnet_opts
import logging
from common.h36m_dataset import Human36mDataset
from common.load_data_h36m import Fusion as Fusion_h36m
from common.utils import *
from model.block.graph import Graph
from model.block.spatial_posegraphnet_encoder import SpatialPoseGraphNet

def main(opt):
    manualSeed = 457

    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    opt.manualSeed = manualSeed

    print('Creating log file........' , os.path.join(opt.checkpoint, 'train.log'))

    if opt.train:
        set_logger(os.path.join(opt.checkpoint, 'train.log'))

    logging.info('==> Using settings {}'.format(opt))

    logging.info('==> Loading dataset...')
    dataset_dir = os.path.join(opt.root_path, opt.dataset)
    if opt.dataset == 'h36m':
        
        dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
        dataset = Human36mDataset(dataset_path, opt)

        if opt.train:
            train_data = Fusion_h36m(opt=opt, train=True, dataset=dataset, root_path=dataset_dir, keypoints=opt.keypoints)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)

        test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir, keypoints=opt.keypoints)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=8, pin_memory=True)
            
    else:
        raise KeyError('Invalid dataset')

    actions = define_actions(opt.actions, opt.dataset)

    if opt.occlusion_augmentation_train or opt.occlusion_augmentation_test:
        print(f'INFO: Occluded: Joint {opt.occluded_joint} in {opt.num_occluded_f} frames')

    # Create model
    logging.info("==> Creating model...")
    a = Graph(layout='hm36_gt', strategy='spatial', with_hip=True)
    adj = torch.from_numpy(a.A).float().cuda()
    model =  SpatialPoseGraphNet(2, [256, 256, 256, 256], 3, 0.20, opt.n_joints, adj, 3, opt.pose_embed_dim, pretrain=True).cuda()

    logging.info("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.resume or opt.evaluate:
        ckpt_path = (opt.resume if opt.resume else opt.evaluate)

        if os.path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
            opt.lr_now = optimizer.param_groups[0]['lr']
            opt.previous_best_threshold, opt.step = ckpt['extra']
            model.load_state_dict(ckpt['model_pos'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, opt.previous_best_threshold))

            opt.checkpoint = os.path.dirname(ckpt_path)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 1
        opt.lr_now = opt.lr
        opt.step = 0

    for epoch in range(start_epoch, opt.nepoch):

        # Reset seed to get the same occluded val dataset per epoch
        if opt.occlusion_augmentation_test:
            test_data.reset_seed(201)

        if opt.train:
            print('\nEpoch: %d | LR: %.8f' % (epoch, opt.lr_now))
            # Train for one epoch
            loss = train(opt, actions, train_dataloader, model, criterion, optimizer)

        # Evaluate
        p1, p2 = evaluate(opt, actions, test_dataloader, model, criterion)

        if opt.train:
            # Save checkpoint
            if opt.previous_best_threshold > p1:
                opt.previous_best_threshold = p1
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, optimizer, model, 'best',
                                               extra=(opt.previous_best_threshold, opt.step))

            if (epoch + 1) % opt.snapshot == 0:
                save_model(None, opt.checkpoint, epoch, p1, optimizer, model, 'snapshot',
                           extra=(opt.previous_best_threshold, opt.step))

        if not opt.train:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.6f, p1: %.2f, p2: %.2f' % (
                epoch, opt.lr_now, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.6f, p1: %.2f, p2: %.2f' % (
                epoch, opt.lr_now, loss, p1, p2))

    return


def train(opt, actions, train_loader, model, criterion, optimizer):
    return step('train', opt, actions, train_loader, model, criterion, optimizer)


def evaluate(opt, actions, val_loader, model, criterion):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, criterion)


def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):
    if split == 'train':
        model.train()
    else:
        model.eval()    

    epoch_loss_3d = AccumLoss()
    action_error_sum = define_error_mpjpe_list(actions)
    # error_sum_joints = define_error_joints_mpjpe_list(opt, actions)

    for i, data in enumerate(tqdm(dataLoader)):
        # batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, dist, scale, bb_box, extra = data
        batch_cam, gt_3d, gt_2d, inputs_2d, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [inputs_2d, gt_2d, gt_3d, batch_cam, scale, bb_box] = \
            get_variable('test', [inputs_2d, gt_2d, gt_3d, batch_cam, scale, bb_box])

        target = gt_3d.clone() # B x F x J x 3
        target[:, :, 0] = 0 
        b, f = inputs_2d.shape[0], inputs_2d.shape[1]
        inputs_2d = rearrange(inputs_2d, 'b f j c  -> (b f) j c', ) # B x 1 x J x 2 -> B x J x 2

        prediction = model(inputs_2d) # B x J x 3
        prediction = rearrange(prediction, '(b f) j c -> b f j c', f=f).contiguous() # B x 1 x J x 3

        if split == 'train':
            N = inputs_2d.size(0)

            opt.step += 1
            if opt.step % opt.lr_decay == 0 or opt.step == 1:
                opt.lr_now = lr_decay(opt.step, opt.lr, opt.lr_decay, opt.lr_gamma)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr_now

            loss_3d = criterion(prediction, target)

            optimizer.zero_grad()
            loss_3d.backward()
            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss_3d.update(loss_3d.detach().cpu().numpy() * N, N)

        elif split == 'test':
            prediction[:, :, 0] = 0
            action_error_sum = test_calculation_mpjpe(prediction, target, action, action_error_sum)
            # error_sum_joints = test_calculation_joints_mpjpe(inputs_2d, gt_2d, action, error_sum_joints, vis)

    if split == 'train':
        return epoch_loss_3d.avg
    elif split == 'test':
        p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum, opt.train)
        print('p1: %.2f, p2: %.2f' % (p1, p2))
        # p1_joints = print_error_mpjpe_joint(error_sum_joints)
        return p1, p2
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = posegraphnet_opts(parser).get_posegraphnet_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)