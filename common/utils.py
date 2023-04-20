import zipfile

import torch
import numpy as np
import hashlib

from torch.autograd import Variable
import os
import logging

actions_h36m = ["Directions", "Discussion", "Eating", "Greeting",
                "Phoning", "Photo", "Posing", "Purchases",
                "Sitting", "SittingDown", "Smoking", "Waiting",
                "WalkDog", "Walking", "WalkTogether"]

actions_unrealcv = ['Basketball', 'Dance', 'Dance2', 'Martial', 'Soccer', 'Boxing', 'Exercise']


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def mpjpe_by_joint_p1(predicted, target, action, joint_error_sum, head_dist):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    f = predicted.size(1)
    assert f == 1
    J = predicted.size(2)
    dist = torch.norm(predicted - target, dim=len(target.shape) - 1).squeeze()
    # dist = head_dist.squeeze()

    for i in range(num):
        end_index = action[i].find(' ')
        if end_index != -1:
            action_name = action[i][:end_index]
        else:
            action_name = action[i]
        for j in range(J):
            joint_error_sum[action_name][j].update(dist[i, j].item() * f, f)
    return joint_error_sum


def test_calculation_joints_mpjpe(predicted, target, action, error_sum, dist):
    error_sum = mpjpe_by_joint_p1(predicted, target, action, error_sum, dist)
    return error_sum


def test_calculation_mpjpe(predicted, target, action, error_sum, dataset='h36m'):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum, dataset)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum, dataset)

    return error_sum


def test_calculation_mpjpe_sb(predicted, target, action, error_sum, dataset='h36m'):
    error_sum = mpjpe_p1(predicted, target, error_sum)
    error_sum = mpjpe_p2(predicted, target, error_sum)

    return error_sum


def mpjpe_p1(predicted, target, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    batch_size = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    action_error_sum['p1'].update(dist.item(), 1)
    return action_error_sum


def mpjpe_p2(predicted, target, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    action_error_sum['p2'].update(np.mean(dist), 1)

    return action_error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum, dataset):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    f = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item() * f, f)

    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum, dataset):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def get_action_from_idx(action_idx, dataset):
    actions = actions_unrealcv if dataset == 'unrealcv' else actions_h36m
    return actions[action_idx]


def define_actions(action, dataset='h36m'):
    actions = actions_unrealcv if dataset == 'unrealcv' else actions_h36m

    if action == "All" or action == "all" or action == '*':
        return actions

    if action not in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_error_joints_mpjpe_list(opt, actions):
    error_sum = {}
    for i in range(len(actions)):
        error_sum[actions[i]] = {}
        for j in range(opt.n_joints):
            error_sum[actions[i]][j] = AccumLoss()
    return error_sum


def define_error_mpjpe_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1': AccumLoss(), 'p2': AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error_mpjpe(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_mpjpe_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2


def print_error_mpjpe_joint(joint_error_sum):
    mean_error_each_action = define_error_mpjpe_list(list(joint_error_sum.keys()))
    mean_error_each_joint = {}
    mean_error_each_joint.update({i: {'p1': AccumLoss()} for i in range(17)})
    mean_error_all = {'p1': AccumLoss()}
    mean_error_each = {'p1': AccumLoss()}

    print("{0:=^12} {1:=^12} {2:=^10}".format("Action", "Joint", "p#1 mm"))

    for action in joint_error_sum.keys():
        for joint, value in joint_error_sum[action].items():
            mean_error_each['p1'] = joint_error_sum[action][joint].avg * 1000.0
            mean_error_each_action[action]['p1'].update(mean_error_each['p1'], 1)
            mean_error_each_joint[joint]['p1'].update(mean_error_each['p1'], 1)
            print("{0:<12} ".format(str(action)), end="")
            print("{0:<12} ".format(str(joint)), end="")

            print("{0:>6.4f}".format(mean_error_each['p1']))

        mean_error_each['p1'] = mean_error_each_action[action]['p1'].avg
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        print("{0:<12} ".format(str(action)), end="")
        print("{0:>6.4f}".format(mean_error_each['p1']))

    for joint in range(17):
        mean_error_each['p1'] = mean_error_each_joint[joint]['p1'].avg
        print("{0:<12} ".format(str(joint)), end="")
        print("{0:>6.4f}".format(mean_error_each['p1']))

    print("{0:<12} {1:>6.4f}".format("Average", mean_error_all['p1'].avg))

    return mean_error_all['p1'].avg


def print_mpjpe(action_error_sum):
    p1_err = action_error_sum['p1']
    p2_err = action_error_sum['p2']
    return p1_err.avg, p2_err.avg


def print_error_mpjpe_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg,
                                                    mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, optimizer, model, model_name, extra=None):
    if previous_name is not None and os.path.exists(previous_name):
        os.remove(previous_name)

    chk_path = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)

    torch.save({
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'model_pos': model.state_dict(),
        'extra': extra,
    }, chk_path)

    return chk_path


def lr_decay(step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    return lr


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    returns mean error across all data points
    and mean per joint error 17 x 1
    """
    assert predicted.shape == target.shape
    err = torch.norm(predicted - target, dim=len(target.shape) - 1)  # num_batch x num_joint
    return torch.mean(err)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
