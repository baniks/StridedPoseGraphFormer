import numpy as np
import torch
# Source: StridedTransformer-Pose3D and https://github.com/facebookresearch/VideoPose3D/blob/main/common/camera.py


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2

    return f*XXX + c


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def get_uvd2xyz(uvd, gt_3D, cam):
    N, T, V, _ = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()

    cam_f_all = cam[..., :2].view(-1, 1, 1, 2).repeat(1, T, V, 1)
    cam_c_all = cam[..., 2:4].view(-1, 1, 1, 2).repeat(1, T, V, 1)

    z_global = dec_out_all[:, :, :, 2]
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]
    z_global = z_global.unsqueeze(-1)

    uv = enc_in_all - cam_c_all
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all
    xyz_global = torch.cat((xy, z_global), -1)
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))

    return xyz_offset
