# Training Murre with a normal prior

This repository now includes a lightweight Marigold-style fine-tuning loop in `train.py`.
The VAE and CLIP text encoder are frozen and only the Murre U-Net is optimized.

## Training objective

For each image, dense GT depth is encoded with the frozen VAE. A random diffusion timestep is sampled and noise is added with a DDPM scheduler. The 16-channel U-Net input is

```text
RGB latent (4)
+ interpolated SfM-depth latent (4)
+ SfM distance map (1)
+ noisy target-depth latent (4)
+ normal prior (3)
= 16 channels
```

The base diffusion loss follows Marigold:

```text
L = L_diffusion + lambda_normal * L_normal
```

`L_diffusion` is the scheduler-dependent latent MSE (`epsilon`, `v_prediction`, or `sample`).

For `L_normal`, the U-Net output at the sampled timestep is first converted back to a predicted clean latent `x0`. This latent is decoded by the frozen VAE **with autograd enabled on the input**, so the normal loss can back-propagate to the U-Net. The decoded depth is converted to a normal map with differentiable central differences.

Normal supervision follows the requested robust rule:

- pixels without observed SfM depth: keep 100%;
- pixels with observed SfM depth: keep only the lowest-loss 90% by default;
- invalid/zero normal-prior pixels are ignored.

The 90% ratio is controlled by `--normal_keep_ratio`.

## Dataset layout

All four folders are matched by filename stem:

```text
data/
  rgb/
    000001.jpg
    000002.jpg
  gt_depth/
    000001.npy
    000002.npy
  sparse_depth/
    000001.npz
    000002.npz
  normal/
    000001.npy
    000002.npy
```

Recommended formats:

- RGB: ordinary RGB images.
- GT depth: `.npy`, `.npz`, `.png`, `.tif`, `.tiff`, or `.exr`; values should be metric depth after applying `--gt_depth_scale`.
- Sparse depth: Murre SfM `.npz` (`arr_0[...,0] = depth`, `[...,1] = reprojection error`, `[...,2] = visible-view count`). A plain 2D `.npy` sparse-depth map is also accepted.
- Normal prior: `HxWx3` or `3xHxW`, encoded as `[-1,1]`, `[0,1]`, or `[0,255]`.

GT depth and SfM depth are resized with nearest-neighbor interpolation. The per-image target normalization range is derived from the filtered sparse SfM depth, matching Murre inference.

## Example

```bash
python train.py \
  --checkpoint /path/to/original_murre_ckpt \
  --rgb_dir /path/to/data/rgb \
  --gt_depth_dir /path/to/data/gt_depth \
  --sparse_depth_dir /path/to/data/sparse_depth \
  --normal_dir /path/to/data/normal \
  --output_dir output/murre_normal \
  --height 512 \
  --width 768 \
  --max_depth 80 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 10000 \
  --learning_rate 3e-5 \
  --normal_weight 0.1 \
  --normal_keep_ratio 0.9 \
  --precision fp16 \
  --gradient_checkpointing \
  --xformers \
  --random_flip
```

If GT depth is stored in millimeters, add:

```bash
--gt_depth_scale 0.001
```

Existing Murre SfM filtering is also available during training:

```bash
--err_thr 2.0 --nviews_thr 2
```

## Checkpoints and resume

A complete Diffusers/Murre checkpoint and optimizer state are written every `--save_every` optimizer steps:

```text
output/murre_normal/checkpoint-0001000/
  ... pipeline files ...
  trainer_state.pt
```

Resume with:

```bash
python train.py \
  --resume output/murre_normal/checkpoint-0001000 \
  --rgb_dir ... \
  --gt_depth_dir ... \
  --sparse_depth_dir ... \
  --normal_dir ... \
  --output_dir output/murre_normal \
  --max_steps 10000
```

`--max_steps` counts optimizer updates, not micro-batches. The effective batch size is

```text
batch_size * gradient_accumulation_steps
```

The final pipeline is saved to `OUTPUT_DIR/final` and can be passed directly to the modified `run.py` together with `--input_normal_dir`.

## Suggested first ablation

Start with the same data and schedule and change only `--normal_weight`:

```text
0.0, 0.01, 0.05, 0.1, 0.2
```

Keep `--normal_keep_ratio 0.9` fixed for the first experiment. This separates the effect of the normal constraint from the robust masking rule.
