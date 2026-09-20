# Training Murre with a normal prior

`train.py` implements a Marigold-style Murre fine-tuning loop. The VAE and CLIP text encoder are frozen and only the Murre U-Net is optimized.

## 本项目运行

```bash
# 查看参数（不构建索引、不训练）
bash Murre-with-normal-prior/train_murre-normal-prior.sh --dry-run
# 自动构建 TXT 索引，使用本地 Murre + Metric3D 权重训练
bash Murre-with-normal-prior/train_murre-normal-prior.sh
# 可调整大小、有效 batch size、步数和保存路径
RESOLUTIONS="128x192 192x256" GRAD_ACCUM=4 MAX_STEPS=10000 \
OUTPUT_DIR=output/murre-normal-prior-exp1 \
bash Murre-with-normal-prior/train_murre-normal-prior.sh
```

默认 Python 为 `.tools/murre-env/bin/python`；可用 `PYTHON_BIN` 覆盖。
Metric3D 额外依赖见 `requirements-metric3d.txt`，当前 `.tools/murre-env` 已安装并验证。
索引 `dataset/murre_normal_training_pairs.txt` 为制表符分隔的
`dataset / scene / rgb / depth / camera`，另有 `.summary.json`。
默认包含 `/mnt/nas/Mapanything_dataset/datasets_processed` 下全部数据集，
包括名称含 testsplit 的目录；需要独立测试集时，用索引生成器的 `--datasets` 选择训练数据。
设置 `REBUILD_INDEX=1` 可重建。索引阶段不解码 depth，加载失败或有效像素不足时
自动换样本；连续 `MAX_RETRIES=1000` 次失败才退出，防止坏数据导致无限等待。

默认 `MAX_DEPTH=0`，按每帧有效深度分位数确定范围，不固定截断到 80 米。
需要统一单位时传 `--depth_scales 数据集名=缩放系数`。
`METRIC3D_DEVICE=cpu` 可节省显存但在线推理较慢。
恢复示例：`RESUME=output/murre-normal-prior/checkpoint-0001000 bash Murre-with-normal-prior/train_murre-normal-prior.sh`。
短测可传 `MAX_STEPS=1 GRAD_ACCUM=1 SAVE_EVERY=0` 和 `--skip_final_save`；正式训练不要跳过保存。

## Current training setup

Training data is hierarchical: multiple datasets, each containing many scenes. Each scene needs RGB images, camera-Z depth, and matching `cams/<stem>.txt` pixel intrinsics.

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

before being resized to a randomly selected `--resolutions` size. All samples in one batch share the same size, including with multiple workers.

Default:

```text
crop_scale_min = 0.9
crop_scale_max = 1.0
resolutions = 128x192 192x256 256x384
```

Optional horizontal flipping can be enabled with `--random_flip`.

## Simulated incomplete depth input

The dense GT depth is used as the complete target. A copy is made and random regions connected to the image borders are removed before it is used as the Murre depth condition.

The occluder is no longer restricted to axis-aligned rectangles. Every selected side is represented by a polygon whose inward boundary can be horizontal/vertical or oblique. With several selected sides, the resulting visible region can become a trapezoid, triangular wedge, or irregular polygon.

Default:

```text
occlusion_probability = 1.0
occlusion_min_ratio = 0.05
occlusion_max_ratio = 0.25
occlusion_min_sides = 1
occlusion_max_sides = 4
occlusion_slant_probability = 0.8
occlusion_slant_max_delta = 0.25
occlusion_min_visible_ratio = 0.15
```

`occlusion_slant_probability` controls how often a selected border uses an oblique cutting line. Setting it to `0` exactly recovers straight border masks.

`occlusion_slant_max_delta` controls the maximum difference between the two endpoints of the cutting line. For a top/bottom border this is measured as a fraction of image height; for a left/right border it is measured as a fraction of image width. Larger values create stronger diagonal wedges.

`occlusion_min_visible_ratio` prevents combinations of several border masks from removing almost all valid depth. Aggressive masks are resampled internally before the sample is returned.

The incomplete input depth then goes through Murre's original depth normalization, interpolation, and distance-map construction. `d_min` and `d_max` are estimated only from the depth that remains visible after masking. The complete GT depth is normalized with the same range and remains the diffusion target.

This creates the desired training situation:

```text
complete RGB
+ incomplete / polygon-border-occluded depth
+ normal structural prior
-> complete depth
```

## Online Metric3D normal prior

The default prior is predicted from augmented RGB by the frozen local Metric3D model
(`checkpoints/Metric3D/metric_depth_vit_large_800k.pth`). No normal files or downloads
are needed. The main training process runs Metric3D after crop/resize/flip; data
workers only read and augment RGB/depth/camera data. The normal loss back-projects
predicted camera-Z depth using the correspondingly transformed camera intrinsics.
`--normal_source gt_depth` is an explicit ablation that leaks GT geometry; it is not
the training default.

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

- input-depth missing / masked pixels: **100% normal supervision**;
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
  --resolutions 128x192 192x256 256x384 \
  --max_depth 120 \
  --crop_scale_min 0.6 \
  --crop_scale_max 1.0 \
  --occlusion_probability 1.0 \
  --occlusion_min_ratio 0.05 \
  --occlusion_max_ratio 0.25 \
  --occlusion_min_sides 1 \
  --occlusion_max_sides 4 \
  --occlusion_slant_probability 0.8 \
  --occlusion_slant_max_delta 0.25 \
  --occlusion_min_visible_ratio 0.15 \
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
max border ratio        = 0.10 / 0.20 / 0.30
selected sides          = 1 / 1-2 / 1-4
slant probability       = 0.0 / 0.5 / 0.8 / 1.0
slant max delta         = 0.0 / 0.10 / 0.25 / 0.40
```

The key evaluation split should separately report errors inside the synthetically masked region and the still-observed region.

## 本机验证

已使用本地 Murre/Metric3D 权重和 NAS EXR 完成 CUDA 短测：
128×192 一步、默认变分辨率配合 2 个 worker 三步、256×384 一步；
包含在线 normal、diffusion/normal loss、反向传播和 AdamW 更新，损失均有限。
这些短测使用 `--skip_final_save`，没有生成正式训练模型，也不代表收敛验证。
数据测试：`.tools/murre-env/bin/python -m unittest discover -s Murre-with-normal-prior/tests -v`。

## TensorBoard 和文本日志

默认保存到 `OUTPUT_DIR/tensorboard` 和 `OUTPUT_DIR/train.log`。记录 total、diffusion、
normal、加权 normal loss、学习率和当前图像尺寸。loss 是上次记录以来所有 micro-batch 的均值，
横轴为 optimizer step；默认第 1 步、每 20 步和最后一步记录，`LOG_EVERY=1` 可每步记录。
`TENSORBOARD_DIR` 可指定事件文件目录。恢复训练继续原步数，并清除该目录中恢复点之后的旧事件显示。

在项目根目录运行：
```bash
LOGDIR=output/murre-normal-prior/tensorboard bash Murre-with-normal-prior/tensorboard.sh
```
浏览器打开 `http://localhost:6006`。训练正在运行时也可查看。

## 验证预览

默认 `VAL_EVERY=500 VAL_SAMPLES=3 VAL_RESOLUTION=192x256 VAL_DENOISING_STEPS=4`。
每隔指定步数以及训练最后一步，用固定样本、固定遮挡和随机种子进行完整去噪推理。
`VAL_EVERY=0` 关闭。推理时切换到 eval/no-grad，结束后恢复训练模式和随机状态。

保存到 `OUTPUT_DIR/validation/step-0000500/sample-00.png` 等路径，
每张图从左到右是 RGB、输入 depth、Metric3D normal、预测 depth、预测 depth 转换的 normal、GT depth。
三个 depth 面板共用 GT 有效深度 2%–98% 分位数色标，黑色表示无效像素。
预测不使用 GT latent，不进行 GT 对齐。每次的数值误差保存为 `metrics.json`；
RMSE 使用原始存储单位，仅逐样本记录，混合单位数据集不汇总 RMSE。
固定样本和 normal 缓存在 `validation/samples.pt`，恢复时复用。

默认是训练样本的定性预览，TensorBoard 标签为 `train_preview/*`，并非独立验证成绩。
如需独立验证，先按场景拆分训练/验证 TXT，传 `VAL_INDEX=/path/to/val.txt`；
代码拒绝训练/验证场景重叠。标签为 `validation/*`。验证还会记录平均 AbsRel 和对比图。
修改样本数量、尺寸或索引时，请使用新输出目录，或移走旧 `validation/samples.pt` 重新选样。
当前在跑的进程不会热加载这些改动，需要从完整 checkpoint 恢复后生效。

## TensorBoard 环境兼容

`3dgs` 中 TensorBoard 2.14 + protobuf 5.29 会触发
`MessageToJson(... including_default_value_fields ...)` 错误。
本机已将该环境 protobuf 调整为 4.25.8；以后重新安装时可执行：
`python -m pip install protobuf==4.25.8`。无需安装 TensorFlow。
也可直接使用独立启动脚本（调用训练 Python 的 TensorBoard 模块）：

```bash
LOGDIR=output/murre-normal-prior-b16/tensorboard \
bash Murre-with-normal-prior/tensorboard.sh
```

已运行的 TensorBoard 服务需要停止后重新启动。验证 PNG 也可直接打开，不依赖 TensorBoard。

## 保留较大视野的缩放裁剪

默认顺序：原始 RGB/depth → 等比例缩放（最长边 384，不放大小图）→
按目标宽高比随机裁剪（最大可用裁剪框边长的 90%–100%）→ 调整到该 batch 的训练尺寸。
相机内参同步缩放并减去裁剪偏移，在线 Metric3D 使用最终 RGB。
宽高比接近时约保留原图 81%–100% 面积；宽高比差异仍会损失一部分视野。
缩放本身不扩大视野，提高裁剪比例才会减少局部取景。

```bash
WORK_RESOLUTION=384 CROP_SCALE_MIN=0.9 CROP_SCALE_MAX=1.0 \
bash Murre-with-normal-prior/train_murre-normal-prior.sh
```

`CROP_SCALE_MIN=1 CROP_SCALE_MAX=1` 保留目标宽高比允许的最大裁剪框。
当前进程需要重启/恢复才使用新设置。验证已经缓存的样本不会自动改变；如需重新取样，
先移走 `OUTPUT_DIR/validation/samples.pt`，避免拿不同视野的预览作直接对比。

## 自动续训和启动验证

脚本默认 `AUTO_RESUME=1 VAL_ON_START=1`，查找输出目录中保存完成的 checkpoint
（包括 final），按训练状态文件保存时间选择最近一个。显式 `RESUME` 优先。
找不到任何 checkpoint 时从预训练模型开始；存在不完整保存记录时直接提示错误。
启动后先在恢复步数执行一次预览，再训练；随后仍按 `VAL_EVERY` 运行。
`VAL_ON_START=0` 跳过启动预览，`VAL_EVERY=0` 关闭全部预览。
从头训练请使用新的 `OUTPUT_DIR` 并设置 `AUTO_RESUME=0`。
自动恢复不读取旧 train_args.json，当前脚本设置仍决定训练数据和目标总步数。

预测 normal 使用 camera-Z 反投影及同步变换后的内参，与训练使用相同的叉乘顺序；
normal RGB 编码为 `(n+1)/2`，与 Metric3D normal 面板一致。无效预测深度、
无效四邻域和图像最外圈显示为黑色。PNG 和 TensorBoard 对比图均包含此面板。

## Metric3D normal 输入尺度修复

Metric3D ViT-L 默认使用官方 616×1064 画布：等比例缩放 RGB（小图也会上采样）、
均值补边、归一化、预测后去补边，再将 normal 调回 Murre 的训练尺寸并归一化。
`METRIC3D_MAX_EDGE=1064` 控制画布长边，与 `RESOLUTIONS` 是独立参数。
先前直接输入约 192×256 的图像会使某些 normal 退化到近似 (0,0,-1)；
不要继续用 392 作为正式训练默认。修复了模型输入，不只是调整显示色彩。

验证缓存新增 `normal_metadata.json`，旧缓存或预处理设置变化时只重新计算 normal，
保留原 RGB、GT、遮挡和相机内参。加载权重时也会检查漏加载参数，
仅允许未使用的 encoder.mask_token 缺失。

此修复同时影响训练条件和 normal loss 的监督目标。已经训练的 checkpoint 受旧先验影响，
可以继续作为诊断实验，但正式对比建议用新的输出目录从原始 Murre 重新训练：

```bash
AUTO_RESUME=0 OUTPUT_DIR=output/murre-normal-prior-metric3d-fixed \
bash Murre-with-normal-prior/train_murre-normal-prior.sh
```

Metric3D 使用较大画布后计算量、显存需求增加；显存不足可设 `METRIC3D_DEVICE=cpu`，
在线预测会更慢。当前运行进程不会热加载修复，需要重启。
