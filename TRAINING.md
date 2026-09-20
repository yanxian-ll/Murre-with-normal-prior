# Training Murre with a normal prior

`train.py` implements a Marigold-style Murre fine-tuning loop. The VAE and CLIP text encoder are frozen and only the Murre U-Net is optimized.

## Current training setup

Training data is hierarchical: multiple datasets, each containing many scenes. Every scene only needs RGB images and dense depth.

```text
DATASET_A/
  scene_000/
    images/
      000001.jpg
      000002.jpg
    depth/
      000001.exr
      000002.exr
  scene_001/
    images/
    depth/

DATASET_B/
  scene_000/
    images/
    depth/
  ...
```

Image/depth files are paired by filename stem. Scene folders may be nested one or more levels under a dataset root as long as each scene contains `images/` and `depth/`.

The default depth format is EXR, but `.npy`, `.npz`, `.png`, `.tif`, and `.tiff` are also accepted.

## Hierarchical random sampling

Every training sample is drawn as

```text
uniform dataset -> uniform scene -> uniform image
```

This is intentional. A dataset or scene containing many more images does not automatically dominate training.

With multiple DataLoader workers, every worker receives an independent Python/NumPy random stream.

## Crop and resize augmentation

RGB and depth receive the same random crop. The crop keeps the final training aspect ratio and uses a random scale in

```text
[crop_scale_min, crop_scale_max]
```

before being resized to `height x width`.

Default:

```text
crop_scale_min = 0.6
crop_scale_max = 1.0
height = 512
width = 768
```

Optional horizontal flipping can be enabled with `--random_flip`.

## Simulated incomplete depth input

The dense GT depth is used as the complete target. A copy is made and random parts of the image borders are removed before it is used as the Murre depth condition.

By default:

```text
occlusion_probability = 1.0
occlusion_min_ratio = 0.05
occlusion_max_ratio = 0.25
occlusion_min_sides = 1
occlusion_max_sides = 4
```

For every sample, 1-4 sides are randomly selected and an independently sampled border width is masked to zero.

The incomplete input depth then goes through Murre's original depth normalization, interpolation, and distance-map construction. `d_min` and `d_max` are estimated only from the depth that remains visible after masking. The complete GT depth is normalized with the same range and remains the diffusion target.

This creates the desired training situation:

```text
complete RGB
+ incomplete / border-occluded depth
+ normal structural prior
-> complete depth
```

## Temporary normal prior

No normal files are required for training right now.

The current implementation computes the normal prior online from the **complete GT depth** after crop/resize. The depth is first normalized to the same `[0,1]` convention used by the prediction, then the normal is computed by central differences:

```text
normal = normalize([-dD/dx, -dD/dy, 1])
```

This intentionally matches `murre/util/normal_util.py`, which computes the normal from predicted depth in the same way.

The trainer itself only consumes a tensor called `normal`, so this depth-derived prior can later be replaced by DSINE or another normal-estimation model without changing the diffusion training logic.

## Model input

The modified Murre U-Net receives 16 channels:

```text
RGB latent                    4
interpolated input-depth      4
input-depth distance map      1
noisy target-depth latent     4
normal prior                  3
--------------------------------
total                        16
```

The original 13 Murre channels remain in their original positions. The 3 normal channels are appended at the end. When an original Murre checkpoint is loaded, the original weights are copied exactly and the new normal-channel weights are initialized to zero.

## Training objective

The diffusion part follows Marigold: dense target depth is encoded with the frozen VAE, a random DDPM timestep is sampled, noise is added, and the U-Net predicts the scheduler target (`epsilon`, `v_prediction`, or `sample`).

```text
L = L_diffusion + lambda_normal * L_normal
```

For the normal term, the U-Net output is converted to predicted clean latent `x0`, decoded by the frozen VAE with autograd still enabled on the latent, and converted from depth to normal.

Normal supervision follows the requested robust rule:

- input-depth missing / border-masked pixels: **100% normal supervision**;
- pixels where input depth remains visible: retain only the lowest normal-loss **90%** by default;
- invalid GT/normal pixels are ignored.

The retained fraction is controlled with `--normal_keep_ratio`.

## Example

```bash
python train.py \
  --checkpoint /path/to/original_murre_ckpt \
  --dataset_roots \
      /data/UAV_dataset_A \
      /data/UAV_dataset_B \
      /data/UAV_dataset_C \
  --output_dir output/murre_normal \
  --height 512 \
  --width 768 \
  --max_depth 120 \
  --crop_scale_min 0.6 \
  --crop_scale_max 1.0 \
  --occlusion_probability 1.0 \
  --occlusion_min_ratio 0.05 \
  --occlusion_max_ratio 0.25 \
  --occlusion_min_sides 1 \
  --occlusion_max_sides 4 \
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

If stored depth requires a unit conversion, use `--depth_scale`. For example, millimeters to meters:

```bash
--depth_scale 0.001
```

If scene folders use different subdirectory names:

```bash
--images_subdir rgb --depth_subdir depths
```

## Checkpoints and resume

A complete Murre/Diffusers checkpoint and optimizer state are saved every `--save_every` optimizer updates:

```text
output/murre_normal/checkpoint-0001000/
  ... pipeline files ...
  trainer_state.pt
```

Resume with:

```bash
python train.py \
  --resume output/murre_normal/checkpoint-0001000 \
  --dataset_roots /data/UAV_dataset_A /data/UAV_dataset_B \
  --output_dir output/murre_normal \
  --max_steps 10000
```

`--max_steps` counts optimizer updates. Effective batch size is

```text
batch_size * gradient_accumulation_steps
```

The final trained pipeline is saved in `OUTPUT_DIR/final`.

## First ablations

For the normal term, first keep all other settings fixed and test

```text
normal_weight = 0, 0.01, 0.05, 0.1, 0.2
```

with

```text
normal_keep_ratio = 0.9
```

For simulated missing depth, useful first comparisons are

```text
max border ratio = 0.10 / 0.20 / 0.30
selected sides    = 1 / 1-2 / 1-4
```

The key evaluation split should separately report errors inside the synthetically masked region and the still-observed region.
