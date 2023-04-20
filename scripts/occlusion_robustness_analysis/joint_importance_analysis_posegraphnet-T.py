import argparse
import subprocess


def run(args):
    batch_size = args.batch_size
    frames = args.frames
    num_occluded_f = 30
    print(num_occluded_f)
    if args.previous_dir is None:
        args.previous_dir = f'../../checkpoint/pretrained/{frames}'

    default_command = ['python', '../../main.py',
                       '--test', '--refine', '--refine_reload', '--reload',
                       '--occlusion_augmentation_test',
                       '--occlusion_augmentation_train',
                       '--frames', str(frames),
                       '--previous_dir', args.previous_dir,
                       '--root_path', args.root_path,
                       '--batch_size', str(batch_size)]

    for joint_idx in range(0, 17):
        joint_specific_command = default_command + ['--occluded_joint', str(joint_idx)]

        if not args.consecutive_frames:
            filename = f'occlusion_joint{joint_idx}.txt'
        else:
            filename = f'occlusion_joint{joint_idx}_consecutive{args.subset_size}.txt'
            joint_specific_command = joint_specific_command + ['--consecutive_frames',
                                                               '--subset_size', str(args.subset_size)]

        # Occlude part of sequence, random frames
        with open(f'part_{filename}', 'w+') as outfile:
            frame_specific_command = joint_specific_command + ['--num_occluded_f', str(num_occluded_f)]

            subprocess.call(frame_specific_command, stdout=outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=351)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--previous_dir', type=str, default=None)
    parser.add_argument('--root_path', type=str, default='../../dataset/')
    parser.add_argument('--consecutive_frames', action='store_true')
    parser.add_argument('--subset_size', type=int, default=None)

    args = parser.parse_args()
    run(args)
