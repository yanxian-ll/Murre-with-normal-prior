# Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
### [Project Page](https://zju3dv.github.io/murre) | [Paper](https://arxiv.org/pdf/2503.14483)

![teaser](./assets/teaser.jpg)

> [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://zju3dv.github.io/murre)  
> [Haoyu Guo](https://github.com/ghy0324)<sup>\*</sup>, [He Zhu](https://ada4321.github.io/)<sup>\*</sup>, [Sida Peng](https://pengsida.net), [Haotong Lin](https://haotongl.github.io/), [Yunzhi Yan](https://yunzhiy.github.io/), [Tao Xie](https://github.com/xbillowy), [Wenguan Wang](https://sites.google.com/view/wenguanwang), [Xiaowei Zhou](https://xzhou.me), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)  
> CVPR 2025 Oral

## Murre with normal prior

This fork extends Murre with a three-channel surface-normal condition.

The original U-Net input is 13 channels:

```text
RGB latent (4) + interpolated SfM depth latent (4) + distance map (1) + noisy depth latent (4)
```

The modified input is 16 channels:

```text
RGB latent (4) + interpolated SfM depth latent (4) + distance map (1) + noisy depth latent (4) + normal prior (3)
```

The normal channels are appended after all original Murre channels. When loading an original 13-channel Murre checkpoint, the original weights are copied exactly and the new three input-channel weights are initialized to zero. This makes the modified network start from the original Murre prediction before normal-guided fine-tuning.

`murre/util/normal_util.py` additionally provides a differentiable neighborhood `depth_to_normal` implementation and `normal_consistency_loss`. The loss supervises all pixels where input SfM depth is missing. In regions with valid SfM depth, it keeps the lowest-loss 90% of pixels by default and removes the highest-loss 10% independently for each image.

## Installation

### Clone this repository

```bash
git clone https://github.com/yanxian-ll/Murre-with-normal-prior.git
cd Murre-with-normal-prior
```

### Create the environment

```bash
conda create -n murre python=3.10
conda activate murre
```

### Installing dependencies

```bash
conda install cudatoolkit=11.8 pytorch==2.0.1 torchvision=0.15.2 torchtriton=2.0.0 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Checkpoint

The original pretrained Murre model weights can be downloaded from [here](https://drive.google.com/file/d/1gcThkgOQRmjAxhGJRV7SwzwXKBWP1cDa/view?usp=sharing).

The modified pipeline can load the original checkpoint directly. The U-Net input convolution is expanded from 13 to 16 channels automatically.

## Inference

### Parse SfM output

```bash
cd sfm_depth
python get_sfm_depth.py --input_sfm_dir ${your_input_path} --output_sfm_dir ${your_output_path} --processing_res ${your_desired_resolution}
```

The parsed sparse depth maps, camera intrinsics, and camera poses will be stored in `${your_output_path}/sparse_depth`, `${your_output_path}/intrinsic`, and `${your_output_path}/pose` respectively.

### Normal prior format

Normal files are matched to RGB and sparse-depth files by filename stem. Supported formats are `.npy`, `.npz`, `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`.

Array normals can be either `H x W x 3` or `3 x H x W`. The pipeline accepts normals encoded as `[-1, 1]`, `[0, 1]`, or image-style `[0, 255]` and renormalizes them to unit vectors.

### SfM- and normal-guided monocular depth estimation

```bash
python run.py \
  --checkpoint ${your_ckpt_path} \
  --input_rgb_dir ${your_rgb_path} \
  --input_sdpt_dir ${your_sparse_depth_path} \
  --input_normal_dir ${your_normal_path} \
  --output_dir ${your_output_path} \
  --denoise_steps 10 \
  --ensemble_size 5 \
  --processing_res ${your_desired_resolution} \
  --max_depth 10.0
```

For indoor scenes, the original Murre recommendation is `--max_depth=10.0`. For outdoor scenes, increase it according to the scene scale.

To filter unreliable SfM depth estimates, use:

```text
--err_thr=${your_error_thresh}
--nviews_thr=${your_nviews_thresh}
```

Make sure the RGB/SfM processing resolution and the normal prior spatial alignment are consistent.

## Normal supervision for fine-tuning

The reusable loss is:

```python
from murre.util.normal_util import normal_consistency_loss

loss_normal = normal_consistency_loss(
    pred_depth,
    normal_prior,
    sparse_depth,
    keep_ratio=0.9,
)
```

The mask policy is:

```text
missing SfM depth: keep 100% of valid normal pixels
valid SfM depth:   keep the lowest normal-loss 90%, discard the highest 10%
```

`pred_depth` is converted to normals directly from neighboring depth values using central differences. No camera intrinsics are required for this loss version.

## TSDF fusion

```bash
python tsdf_fusion.py --image_dir ${your_rgb_path} --depth_dir ${your_depth_path} --intrinsic_dir ${your_intrinsic_path} --pose_dir ${your_pose_path}
```

## Evaluation

Please refer to [EVAL.md](./EVAL.md).

## Citation

If you find the original Murre work useful, please cite:

```bibtex
@inproceedings{guo2025murre,
  title={Multi-view Reconstruction via SfM-guided Monocular Depth Estimation},
  author={Guo, Haoyu and Zhu, He and Peng, Sida and Lin, Haotong and Yan, Yunzhi and Xie, Tao and Wang, Wenguan and Zhou, Xiaowei and Bao, Hujun},
  booktitle={CVPR},
  year={2025},
}
```

## Acknowledgement

We sincerely thank the following excellent projects, from which the original Murre work has benefited.

- [Diffusers](https://huggingface.co/docs/diffusers)
- [Marigold](https://marigoldmonodepth.github.io/)
- [COLMAP](https://colmap.github.io/)
- [Detector-Free SfM](https://zju3dv.github.io/DetectorFreeSfM/)
