# Exploiting Spatial-Temporal Relationships for Occlusion-Robust 3D Human Pose Estimation

## Dependencies

- Cuda 11.6
- Python 3.10.4
- Pytorch 1.12.1

## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing). 

```bash
${POSE_ROOT}/
|-- dataset
|   |--h36m
|   |  |-- data_3d_h36m.npz
|   |  |-- data_2d_h36m_gt.npz
|   |  |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Test the models

To test on pretrained models on Human3.6M:

```bash
python main.py --test --reload --refine_reload --refine --frames 81 --previous_dir checkpoint/posegraphnet-T-data-aug/0118_1113_07_81
```

## Train the models

To train on Human3.6M:

### Pre-Training PoseGraphNet (single-frame 3D HPE):
```bash
python pretrain_posegraphnet.py --nepoch 20 --frames 1
```

Alternatively, you can download the pretrained PoseGraphNet [here](https://github.com/baniks/PoseGraphNet/tree/main/models/icip_v2/run7_PoseGraphNetV1).

### Transfer Learning (Phase 1):

```bash
python main.py --frames 81 --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --pretrained_spatial_module_init --pretrained_spatial_module_dir [your pre-trained PoseGraphNet directory path] --pretrained_spatial_module [your pre-trained PoseGraphNet file name inside directory] 
```

### Fine-Tuning (Phase 2):

```bash
python main.py --frames 81 --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --reload --spatial_module_lr 1e-3 --previous_dir [your phase-1 model saved directory path]
```

### Pose Refinement (Phase 3):

```bash
python main.py --frames 81 --reload --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --spatial_module_lr 1e-3 --refine --lr_refine 1e-3 --previous_dir [your phase-2 model saved directory path]
```

## Occlusion Robustness Analysis 

To test a model's robustness against missing keypoints (occluding a specific joint [0-16] across 30 frames):

```bash
cd scripts/occlusion_robustness_analysis

python joint_importance_analysis_posegraphnet-T.py --frames 81 --previous_dir ../../checkpoint/PATH/TO/MODEL_DIR --root_path ../../dataset
```

## Acknowledgement

The code is built on top of [StridedTransformer](https://github.com/Vegetebird/StridedTransformer-Pose3D).

