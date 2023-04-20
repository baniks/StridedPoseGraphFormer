import random
import glob
import logging
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_h36m import Fusion as Fusion_h36m
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.strided_posegraphnet import Model as StridedPoseGraphNet

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
print(opt)


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']

    if split == 'train':
        model_trans.train()
        if opt.freeze_spatial_module and opt.set_spatial_module_eval_mode:
            model_trans.set_spatial_module_eval_mode()
        model_refine.train()
    else:
        model_trans.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss(), 'loss_single': AccumLoss(), 'loss_VTE': AccumLoss(), 'loss_vis': AccumLoss()}
    action_error_sum = define_error_mpjpe_list(actions)
    action_error_sum_refine = define_error_mpjpe_list(actions)

    # error_sum_joints = define_error_joints_mpjpe_list(opt, actions)

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, gt_2D, input_2D, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [input_2D, gt_3D, gt_2D, batch_cam, scale, bb_box] = get_variable(split,
                                                                               [input_2D, gt_3D, gt_2D, batch_cam,
                                                                                scale, bb_box])

        if opt.use_2d_gt:
            input_2D = gt_2D

        if split == 'train':
            output_3D, output_3D_VTE = model_trans(input_2D)
        else:
            input_2D, output_3D, output_3D_VTE = input_augmentation(input_2D, model_trans)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.refine:
            pred_uv = input_2D[:, opt.pad, :, :].unsqueeze(1)
            uvd = torch.cat((pred_uv, output_3D[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0
            output_3D = model_refine(output_3D, xyz)

        N, F = input_2D.size(0), input_2D.size(1)
        if split == 'train':
            if opt.refine:
                loss = mpjpe_cal(output_3D, out_target_single)
            else:
                loss_VTE = mpjpe_cal(output_3D_VTE, out_target)
                loss_all['loss_VTE'].update(loss_VTE.detach().cpu().numpy() * N, N)
                loss_single = mpjpe_cal(output_3D, out_target_single)
                loss_all['loss_single'].update(loss_single.detach().cpu().numpy() * N, N)

                loss = loss_VTE + loss_single

            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation_mpjpe(output_3D, out_target, action, action_error_sum)

            if opt.refine:
                action_error_sum_refine = test_calculation_mpjpe(output_3D, out_target, action, action_error_sum_refine)

    if split == 'train':
        return loss_all['loss'].avg, loss_all['loss_single'].avg, loss_all['loss_VTE'].avg
    elif split == 'test':
        if opt.refine:
            p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum_refine, opt.train)
        else:
            p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum, opt.train)

        return p1, p2


def input_augmentation(input_2D, model_trans):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip, output_3D_non_flip_VTE = model_trans(input_2D_non_flip)
    output_3D_flip, output_3D_flip_VTE = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, :, :, 0] *= -1
    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D, output_3D_VTE


def print_layers(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            psize_list = list(p.size())
            psize_str = [str(x) for x in psize_list]
            psize_str = ",".join(psize_str)
            print(name + "\t" + psize_str)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    opt.manualSeed = 0

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path

    print('Loading dataset...')
    dataset_dir = os.path.join(root_path, opt.dataset)

    if opt.dataset == 'h36m':
        dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
        dataset = Human36mDataset(dataset_path, opt)

        if opt.train:
            train_data = Fusion_h36m(opt=opt, train=True, dataset=dataset, root_path=dataset_dir)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)

        test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
        opt.out_joints = dataset.skeleton().num_joints()
    else:
        raise KeyError('Invalid dataset')

    actions = define_actions(opt.actions, opt.dataset)

    if opt.occlusion_augmentation_train or opt.occlusion_augmentation_test:
        print(f'INFO: Occluded: Joint {opt.occluded_joint} in {opt.num_occluded_f} frames')

    model = {}
    model_trans = StridedPoseGraphNet(opt).cuda()

    # Use pretrained weights for spatial part without pose regression head
    if opt.pretrained_spatial_module_init:
        filename = os.path.join(opt.pretrained_spatial_module_dir, opt.pretrained_spatial_module)
        pretrained_dict = torch.load(filename)['state_dict']

        model_trans.Transformer.load_state_dict(pretrained_dict, strict=False)
        opt.freeze_spatial_module = True
        model_trans.freeze_spatial_module()

    model['trans'] = model_trans
    model['refine'] = refine(opt).cuda()

    model_dict = model['trans'].state_dict()

    all_param = []
    lr_spatial = opt.spatial_module_lr
    lr = opt.lr
    lr_refine = opt.lr_refine
    for i_model in model:
        all_param += list(model[i_model].parameters())

    optimizer_all = optim.Adam([
        {"params": model['trans'].Transformer.parameters(), "lr": opt.spatial_module_lr},
        {"params": model['trans'].Transformer_reduce.parameters()},
        {"params": model['trans'].Transformer_full.parameters()},
        {"params": model['trans'].fcn.parameters()},
        {"params": model['trans'].head.parameters()},
        {"params": model['refine'].parameters(), "lr": opt.lr_refine},
    ], lr=opt.lr, amsgrad=True)

    epoch_start = 1
    if opt.reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        no_refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'n' and 'best' in path:
                no_refine_path = path
                print(no_refine_path)
                break

        pre_dict = torch.load(no_refine_path)
        pre_dict_model = pre_dict['model_pos']
        for name, key in model_dict.items():
            model_dict[name] = pre_dict_model[name]
        model['trans'].load_state_dict(model_dict)

        if opt.freeze_spatial_module:
            model['trans'].freeze_spatial_module()

        if opt.freeze_trans_module:
            model['trans'].freeze()

        if opt.train and opt.resume:
            optimizer_all.load_state_dict(pre_dict['optimizer'])
            epoch_start = pre_dict['epoch'] + 1
            if pre_dict['epoch'] % opt.large_decay_epoch == 0:
                for param_group in optimizer_all.param_groups:
                    param_group['lr'] *= opt.lr_decay_large
                lr_spatial = optimizer_all.param_groups[0]['lr']
                lr = optimizer_all.param_groups[1]['lr']
                lr_refine = optimizer_all.param_groups[5]['lr']
            else:
                for param_group in optimizer_all.param_groups:
                    param_group['lr'] *= opt.lr_decay
                lr_spatial = optimizer_all.param_groups[0]['lr']
                lr = optimizer_all.param_groups[1]['lr']
                lr_refine = optimizer_all.param_groups[5]['lr']

    refine_dict = model['refine'].state_dict()

    if opt.refine_reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'r' and 'best' in path:
                refine_path = path
                print(refine_path)
                break

        pre_dict_refine = torch.load(refine_path)
        pre_dict_refine_model = pre_dict_refine['model_pos']
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine_model[name]
        model['refine'].load_state_dict(refine_dict)

        if opt.train and opt.resume:
            optimizer_all.load_state_dict(pre_dict_refine['optimizer'])
            epoch_start = pre_dict_refine['epoch'] + 1
            if pre_dict_refine['epoch'] % opt.large_decay_epoch == 0:
                for param_group in optimizer_all.param_groups:
                    param_group['lr'] *= opt.lr_decay_large
                lr_spatial = optimizer_all.param_groups[0]['lr']
                lr = optimizer_all.param_groups[1]['lr']
                lr_refine = optimizer_all.param_groups[5]['lr']
            else:
                for param_group in optimizer_all.param_groups:
                    param_group['lr'] *= opt.lr_decay
                lr_spatial = optimizer_all.param_groups[0]['lr']
                lr = optimizer_all.param_groups[1]['lr']
                lr_refine = optimizer_all.param_groups[5]['lr']

    count_model_params = sum(p.numel() for p in all_param)
    print('INFO: Parameter count:', count_model_params)
    count_trainable_model_params = sum(p.numel() for p in all_param if p.requires_grad)
    print('INFO: Trainable parameter count:', count_trainable_model_params)

    for epoch in range(epoch_start, opt.nepoch):
        print('Epoch: ' + str(epoch))
        print('LR spatial: ' + str(lr_spatial))
        print('LR: ' + str(lr))
        if opt.refine:
            print('LR refine: ' + str(lr_refine))

        if opt.train:
            # Reset seed to get the same occluded train dataset per epoch
            # if opt.occlusion_augmentation_train:
            # train_data.reset_seed(200)
            loss, loss_single, loss_VTE = train(opt, actions, train_dataloader, model, optimizer_all, epoch)

        # Reset seed to get the same occluded val dataset per epoch
        if opt.occlusion_augmentation_test:
            test_data.reset_seed(201)

        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train:
            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, optimizer_all,
                                               model['trans'], 'no_refine_best')
                if opt.refine:
                    opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                          p1, optimizer_all, model['refine'], 'refine_best')

                opt.previous_best_threshold = p1

            if epoch % opt.save_ckpt_intervall == 0:
                save_model(None, opt.checkpoint, epoch, p1, optimizer_all, model['trans'], 'no_refine')

                if opt.refine:
                    save_model(None, opt.checkpoint, epoch, p1, optimizer_all, model['refine'], 'refine')

        if not opt.train:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:

            logging.info('epoch: %d, lr: %.7f, loss: %.4f, loss_single: %.4f, loss_VTE: %.4f, p1: %.2f, p2: %.2f' % (
                epoch, lr, loss, loss_single, loss_VTE, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, loss_single: %.4f, loss_VTE: %.4f, p1: %.2f, p2: %.2f' % (
                epoch, lr, loss, loss_single, loss_VTE, p1, p2))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
            lr_spatial *= opt.lr_decay_large
            lr *= opt.lr_decay_large
            lr_refine *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
            lr_spatial *= opt.lr_decay
            lr *= opt.lr_decay
            lr_refine *= opt.lr_decay
