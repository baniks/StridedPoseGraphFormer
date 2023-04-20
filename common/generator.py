import numpy as np


class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d_keypoints, poses_2d_gt, chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234, augment=False, reverse_aug=False, kps_left=None, kps_right=None,
                 joints_left=None, joints_right=None, endless=False, out_all=False, dataset='h36m'):
        assert poses_3d is None or len(poses_3d) == len(poses_2d_keypoints), (len(poses_3d), len(poses_2d_keypoints))
        assert cameras is None or len(cameras) == len(poses_2d_keypoints)
        assert poses_2d_gt is None or len(poses_2d_gt) == len(poses_2d_keypoints)

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d_keypoints.keys():
            assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d_keypoints[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d_keypoints[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            key_size = len(key)
            keys = np.tile(np.array(key).reshape([1, key_size]), (len(bounds - 1), 1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector,~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index,end_index]
            start_index = start_index + poses_3d[key].shape[0]

        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))

        if poses_2d_gt is not None:
            example_key = next(iter(poses_2d_gt))
            self.batch_2d_gt = np.empty(
                (batch_size, chunk_length + 2 * pad, poses_2d_gt[example_key].shape[-2], poses_2d_gt[example_key].shape[-1]))

        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d_keypoints[key].shape[-2], poses_2d_keypoints[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d_keypoints
        self.poses_2d_gt = poses_2d_gt

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all
        self.dataset = dataset

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def augment_seq(self, batch, poses, seq_name, start_3d, end_3d, flip, reverse, left, right, out_all=None):
        if out_all is None or out_all:
            start = start_3d - self.pad - self.causal_shift
            end = end_3d + self.pad - self.causal_shift
        else:
            start = start_3d
            end = end_3d

        seq = poses[seq_name].copy()
        low = max(start, 0)
        high = min(end, seq.shape[0])
        pad_left = low - start
        pad_right = end - high
        if pad_left != 0 or pad_right != 0:
            batch = np.pad(seq[low:high], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
        else:
            batch = seq[low:high]

        if flip:
            if len(batch.shape) > 2 and batch.shape[2] != 1:
                batch[:, :, 0] *= -1
            batch[:, left + right] = batch[:, right + left]
        if reverse:
            batch = batch[::-1].copy()
        return batch

    def augment_batch(self, seq_name, start_3d, end_3d, flip, reverse):
        self.batch_2d = self.augment_seq(self.batch_2d, self.poses_2d, seq_name, start_3d, end_3d, flip, reverse,
                                         self.kps_left, self.kps_right)

        if self.poses_3d is not None:
            self.batch_3d = self.augment_seq(self.batch_3d, self.poses_3d, seq_name, start_3d, end_3d, flip, reverse,
                                             self.joints_left, self.joints_right, out_all=self.out_all)

        if self.poses_2d_gt is not None:
            self.batch_2d_gt = self.augment_seq(self.batch_2d_gt, self.poses_2d_gt, seq_name, start_3d, end_3d, flip, reverse,
                                                self.kps_left, self.kps_right)
        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[ 2] *= -1
                self.batch_cam[ 7] *= -1

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        raise NotImplementedError
