from common.generator import ChunkedGenerator


class GeneratorH36M(ChunkedGenerator):
    def __init__(self, batch_size, cameras, poses_3d, poses_2d_keypoints, poses_2d_gt, chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234, augment=False, reverse_aug=False, kps_left=None, kps_right=None,
                 joints_left=None, joints_right=None, endless=False, out_all=False):
        super().__init__(batch_size, cameras, poses_3d, poses_2d_keypoints, poses_2d_gt,
                         chunk_length, pad, causal_shift,
                         shuffle, random_seed, augment, reverse_aug, kps_left, kps_right, joints_left, joints_right,
                         endless, out_all, dataset='h36m')

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        extra = action, subject, int(cam_index)

        super().augment_batch(seq_name, start_3d, end_3d, flip, reverse)

        return self.batch_cam, self.batch_3d.copy(), self.batch_2d_gt.copy(), self.batch_2d.copy(), extra

