import argparse
import json
import logging
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Allow running this script from any working directory (python Murre-with-normal-prior/train.py).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from murre.pipeline import MurrePipeline
from murre.training_dataset import MurreNormalTrainingDataset, ResolutionBatchSampler
from murre.util.metric3d_normal import Metric3DNormalEstimator
from murre.util.normal_util import camera_normal_consistency_loss
from murre.validation import ValidationPreview


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int):
    """Give each DataLoader worker an independent Python/NumPy RNG stream."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def encode_depth(pipe: MurrePipeline, depth: torch.Tensor) -> torch.Tensor:
    """Encode [B,1,H,W] depth in [-1,1] using the frozen RGB VAE encoder."""
    return pipe.encode_rgb(depth.repeat(1, 3, 1, 1))


def valid_latent_mask(valid: torch.Tensor, latent_hw) -> torch.Tensor:
    """Conservative Marigold-style GT validity mask at latent resolution."""
    h, w = valid.shape[-2:]
    lh, lw = latent_hw
    invalid = (~valid).float()
    if h % lh == 0 and w % lw == 0:
        kh, kw = h // lh, w // lw
        invalid_down = F.max_pool2d(invalid, kernel_size=(kh, kw), stride=(kh, kw))
    else:
        invalid_down = F.interpolate(invalid, size=(lh, lw), mode="nearest")
    return (invalid_down < 0.5).repeat(1, 4, 1, 1)


def prediction_to_x0(
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
) -> torch.Tensor:
    """Convert the one-step diffusion prediction to a clean target latent x0."""
    if prediction_type == "sample":
        return model_pred

    alpha_bar = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )[timesteps]
    alpha_bar = alpha_bar.view(-1, 1, 1, 1).clamp(1e-6, 1.0)
    beta_bar = (1.0 - alpha_bar).clamp(0.0, 1.0)

    if prediction_type == "epsilon":
        return (noisy_latents - beta_bar.sqrt() * model_pred) / alpha_bar.sqrt()
    if prediction_type == "v_prediction":
        return alpha_bar.sqrt() * noisy_latents - beta_bar.sqrt() * model_pred
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def make_lr_scheduler(optimizer, max_steps: int, warmup_steps: int, final_ratio: float):
    """Iteration-based warmup + exponential decay, following Marigold's training style."""
    max_steps = max(int(max_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)
    final_ratio = float(final_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)
        denom = max(max_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        return final_ratio**progress

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    pipe: MurrePipeline,
    optimizer,
    lr_scheduler,
    scaler,
    output_dir: str,
    global_step: int,
):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    pipe.save_pretrained(ckpt_dir)
    state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
    with open(os.path.join(output_dir, "latest.txt"), "w") as f:
        f.write(ckpt_dir + "\n")
    logging.info("Saved checkpoint: %s", ckpt_dir)


def parse_depth_scales(specs):
    """Parse ``--depth_scales`` into a dataset -> scale mapping.

    Accepts either ``NAME=SCALE`` entries or a single JSON path whose object maps
    dataset directory names to scales (a wrapping ``{"datasets": {...}}`` is allowed,
    matching dataset/murre_training_pairs.summary.json).
    """
    if not specs:
        return {}
    if len(specs) == 1 and "=" not in specs[0]:
        path = Path(specs[0]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"depth scale table not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "datasets" in payload:
            payload = payload["datasets"]
        if not isinstance(payload, dict):
            raise ValueError(f"depth scale table must map dataset -> scale, got {type(payload)}")
        return {str(key): float(value) for key, value in payload.items()}

    table = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=SCALE or a JSON path, got '{spec}'")
        name, _, value = spec.partition("=")
        table[name.strip()] = float(value)
    return table


def build_parser():
    parser = argparse.ArgumentParser(
        description="Marigold-style Murre fine-tuning with a surface-normal prior."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Original Murre checkpoint.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a saved checkpoint directory.")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help=(
            "TSV index built by dataset/build_murre_training_index.py. Preferred over "
            "--dataset_roots: sampling then does not walk the dataset tree."
        ),
    )
    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more dataset roots. Each root contains many scene folders; "
            "every scene contains images/ and depth/ by default. Used when --index is absent."
        ),
    )
    parser.add_argument('--resolutions', nargs='+', default=['128x192','192x256','256x384'], help='HxW, one size per batch')
    parser.add_argument('--skip_final_save', action='store_true', help='Smoke tests only: skip multi-GB final checkpoint')
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--depth_subdir", type=str, default="depth")
    parser.add_argument("--camera_subdir", type=str, default="cams")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Global depth unit conversion applied to every dataset unless overridden.",
    )
    parser.add_argument(
        "--depth_scales",
        nargs="*",
        default=None,
        help=(
            "Per-dataset depth unit overrides, e.g. --depth_scales whu_whuomvs=0.001 "
            "urbanscene3d=0.01, or a single JSON path mapping dataset -> scale."
        ),
    )

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument(
        "--work_resolution",
        type=int,
        default=384,
        help=(
            "Longest edge of the working image before the random crop; RGB, depth, "
            "intrinsics use this grid before augmentation. Online Metric3D uses the final crop. 0 disables."
        ),
    )
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument(
        "--max_depth_mode",
        choices=["auto", "fixed"],
        default="auto",
        help=(
            "auto: cap each frame at a high quantile of its own valid depth (unit free, "
            "recommended for mixed-unit datasets). fixed: always use --max_depth."
        ),
    )
    parser.add_argument("--max_depth_quantile", type=float, default=99.5)
    parser.add_argument(
        "--max_depth_scale",
        type=float,
        default=1.5,
        help="Multiplier applied to the auto quantile cap.",
    )
    parser.add_argument("--crop_scale_min", type=float, default=0.9)
    parser.add_argument("--crop_scale_max", type=float, default=1.0)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=8,
        help="Draw another sample when a frame cannot be read or is degenerate.",
    )
    parser.add_argument(
        "--min_valid_pixels",
        type=int,
        default=256,
        help="Minimum valid GT-depth pixels for a frame to be usable.",
    )
    parser.add_argument(
        "--min_gt_valid_ratio",
        type=float,
        default=0.9,
        help=(
            "Minimum fraction of finite positive GT-depth pixels. Checked both before "
            "augmentation and after the final crop/resize; samples below it are skipped."
        ),
    )

    parser.add_argument(
        "--normal_source",
        choices=["metric3d", "gt_depth"],
        default="metric3d",
        help=(
            "metric3d: online Metric3D prior from RGB (matches inference). "
            "gt_depth: camera-space prior from GT depth, ablation only (leaks GT)."
        ),
    )
    parser.add_argument("--metric3d_checkpoint", type=str, default=None)
    parser.add_argument("--metric3d_config", type=str, default=None)
    parser.add_argument("--metric3d_root", type=str, default=None)
    parser.add_argument(
        "--metric3d_max_edge",
        type=int,
        default=1064,
        help="Longest edge passed to Metric3D.",
    )
    parser.add_argument(
        "--metric3d_device",
        type=str,
        default=None,
        help="Device for the Metric3D prior; use cpu to keep its VRAM free for the UNet.",
    )

    parser.add_argument(
        "--occlusion_probability",
        type=float,
        default=1.0,
        help="Probability of applying simulated border occlusion to the input depth.",
    )
    parser.add_argument(
        "--occlusion_min_ratio",
        type=float,
        default=0.05,
        help="Minimum mean inward extent of a selected border mask.",
    )
    parser.add_argument(
        "--occlusion_max_ratio",
        type=float,
        default=0.25,
        help="Maximum mean inward extent of a selected border mask.",
    )
    parser.add_argument("--occlusion_min_sides", type=int, default=1)
    parser.add_argument("--occlusion_max_sides", type=int, default=4)
    parser.add_argument(
        "--occlusion_slant_probability",
        type=float,
        default=0.8,
        help=(
            "Probability that each selected border uses an oblique cutting line. "
            "The remaining cases use the old straight horizontal/vertical boundary."
        ),
    )
    parser.add_argument(
        "--occlusion_slant_max_delta",
        type=float,
        default=0.25,
        help=(
            "Maximum difference between the two endpoints of an oblique border line, "
            "as a fraction of image width/height. Larger values give stronger wedges."
        ),
    )
    parser.add_argument(
        "--occlusion_min_visible_ratio",
        type=float,
        default=0.15,
        help="Minimum fraction of originally valid depth kept after masking.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_final_ratio", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument(
        "--normal_weight",
        type=float,
        default=0.1,
        help="lambda_normal in total_loss = diffusion_loss + lambda_normal * normal_loss.",
    )
    parser.add_argument(
        "--normal_keep_ratio",
        type=float,
        default=0.9,
        help=(
            "For pixels where input depth remains visible, retain the lowest-loss "
            "fraction for normal supervision. Border-masked pixels are all supervised."
        ),
    )

    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--tensorboard_dir", default=None, help="Default: OUTPUT_DIR/tensorboard")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_on_start", action="store_true", help="Run previews before the first optimizer update")
    parser.add_argument("--val_every", type=int, default=500, help="0 disables inference previews")
    parser.add_argument("--val_samples", type=int, default=3)
    parser.add_argument("--val_index", default=None, help="Separate index; default is training previews, not held-out validation")
    parser.add_argument("--val_resolution", default="192x256")
    parser.add_argument("--val_denoising_steps", type=int, default=4)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.resume is None and args.checkpoint is None:
        parser.error("Either --checkpoint or --resume is required")
    if args.index is None and not args.dataset_roots:
        parser.error("Either --index or --dataset_roots is required")
    if args.index is not None and args.dataset_roots:
        parser.error("--index and --dataset_roots are mutually exclusive")
    if not (0.0 < args.normal_keep_ratio <= 1.0):
        parser.error("--normal_keep_ratio must be in (0, 1]")
    if not (0.0 < args.min_gt_valid_ratio <= 1.0):
        parser.error("--min_gt_valid_ratio must be in (0, 1]")
    if args.log_every < 1:
        parser.error("--log_every must be >= 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be >= 1")
    if args.metric3d_device is not None and args.normal_source != "metric3d":
        parser.error("--metric3d_device only applies to --normal_source metric3d")
    if args.normal_source == "gt_depth":
        logging.warning(
            "normal_source=gt_depth derives the normal prior from GT depth; this leaks "
            "the training target and is intended for ablations only."
        )

    if args.val_every < 0 or args.val_samples < 1 or args.val_denoising_steps < 1:
        parser.error("Invalid validation frequency, sample count or denoising steps")
    val_size = tuple(map(int,args.val_resolution.lower().split('x')))
    if len(val_size) != 2 or min(val_size) < 32 or any(v%8 for v in val_size):
        parser.error("--val_resolution must be HxW, >=32 and divisible by 8")
    resolutions = [tuple(map(int, item.lower().split('x'))) for item in args.resolutions]
    if any(len(size)!=2 or min(size)<32 or any(v%8 for v in size) for size in resolutions):
        parser.error('--resolutions must be HxW, >=32 and divisible by 8')
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train.log"), mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    seed_everything(args.seed)

    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.precision != "fp32":
        logging.warning("CUDA is unavailable; forcing fp32 training.")
        args.precision = "fp32"
    logging.info("device=%s precision=%s", device, args.precision)

    model_path = args.resume if args.resume is not None else args.checkpoint
    pipe: MurrePipeline = MurrePipeline.from_pretrained(model_path, torch_dtype=torch.float32, local_files_only=True)
    pipe = pipe.to(device)

    # Follow Marigold: freeze VAE + text encoder, optimize only U-Net.
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.train()

    if args.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xFormers memory-efficient attention.")
        except Exception as exc:
            logging.warning("Could not enable xFormers: %s", exc)

    pipe.encode_empty_text()
    empty_text_embed = pipe.empty_text_embed.detach().to(device=device, dtype=torch.float32)

    normal_predictor = None
    if args.normal_source == 'metric3d':
        normal_predictor = Metric3DNormalEstimator(
            checkpoint=args.metric3d_checkpoint, config=args.metric3d_config,
            metric3d_root=args.metric3d_root, device=args.metric3d_device or device.type,
            max_edge=args.metric3d_max_edge)
    logging.info('Local Murre module: %s', __import__('murre.pipeline',fromlist=['']).__file__)
    dataset = MurreNormalTrainingDataset(
        dataset_roots=args.dataset_roots,
        index_path=args.index,
        height=args.height,
        width=args.width,
        images_subdir=args.images_subdir,
        depth_subdir=args.depth_subdir,
        camera_subdir=args.camera_subdir,
        max_depth=args.max_depth,
        max_depth_mode=args.max_depth_mode,
        max_depth_quantile=args.max_depth_quantile,
        max_depth_scale=args.max_depth_scale,
        depth_scales=parse_depth_scales(args.depth_scales),
        default_depth_scale=args.depth_scale,
        work_resolution=args.work_resolution,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        random_flip=args.random_flip,
        occlusion_probability=args.occlusion_probability,
        occlusion_min_ratio=args.occlusion_min_ratio,
        occlusion_max_ratio=args.occlusion_max_ratio,
        occlusion_min_sides=args.occlusion_min_sides,
        occlusion_max_sides=args.occlusion_max_sides,
        occlusion_slant_probability=args.occlusion_slant_probability,
        occlusion_slant_max_delta=args.occlusion_slant_max_delta,
        occlusion_min_visible_ratio=args.occlusion_min_visible_ratio,
        normal_source="deferred_metric3d" if args.normal_source == "metric3d" else args.normal_source,
        metric3d_checkpoint=args.metric3d_checkpoint,
        metric3d_config=args.metric3d_config,
        metric3d_root=args.metric3d_root,
        metric3d_max_edge=args.metric3d_max_edge,
        metric3d_device=args.metric3d_device or device.type,
        max_retries=args.max_retries,
        min_valid_pixels=args.min_valid_pixels,
        min_gt_valid_ratio=args.min_gt_valid_ratio,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=ResolutionBatchSampler(len(dataset),args.batch_size,resolutions,args.seed),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    logging.info(
        "Training hierarchy: %d datasets, %d scenes, %d image/depth pairs (%s)",
        len(dataset.datasets),
        dataset.total_scenes,
        dataset.total_pairs,
        "index=%s" % args.index if args.index else "directory scan",
    )
    logging.info(
        "Sampling policy: uniform dataset -> uniform scene -> uniform image; crop_scale=[%.2f, %.2f]",
        args.crop_scale_min,
        args.crop_scale_max,
    )
    logging.info(
        "GT validity: min_ratio=%.3f min_pixels=%d; invalid GT is masked from diffusion and normal losses",
        args.min_gt_valid_ratio,
        args.min_valid_pixels,
    )
    logging.info(
        "Depth conditioning: work_resolution=%d max_depth_mode=%s quantile=%.2f scale=%.2f fixed_cap=%.2f per_dataset=%s default_scale=%.4g",
        args.work_resolution,
        args.max_depth_mode,
        args.max_depth_quantile,
        args.max_depth_scale,
        args.max_depth,
        dataset.depth_scales or "{}",
        dataset.default_depth_scale,
    )
    logging.info(
        "Occlusion: sides=%d-%d ratio=[%.2f, %.2f] slant_prob=%.2f slant_delta<=%.2f min_visible=%.2f",
        args.occlusion_min_sides,
        args.occlusion_max_sides,
        args.occlusion_min_ratio,
        args.occlusion_max_ratio,
        args.occlusion_slant_probability,
        args.occlusion_slant_max_delta,
        args.occlusion_min_visible_ratio,
    )
    logging.info(
        "Normal prior: source=%s device=%s max_edge=%d keep_ratio=%.2f weight=%.4g",
        args.normal_source,
        args.metric3d_device or device.type,
        args.metric3d_max_edge,
        args.normal_keep_ratio,
        args.normal_weight,
    )

    # Marigold uses DDPM for the training forward diffusion process.
    training_noise_scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    prediction_type = training_noise_scheduler.config.prediction_type
    num_train_timesteps = training_noise_scheduler.config.num_train_timesteps
    logging.info(
        "Training scheduler: prediction_type=%s num_train_timesteps=%d",
        prediction_type,
        num_train_timesteps,
    )

    optimizer = AdamW(
        pipe.unet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = make_lr_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        final_ratio=args.lr_final_ratio,
    )

    scaler = None
    if device.type == "cuda" and args.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    if args.resume is not None:
        state_path = os.path.join(args.resume, "trainer_state.pt")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"Missing resume state: {state_path}")
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("global_step", 0))
        logging.info("Resumed optimizer state at global_step=%d", global_step)

    preview = None
    tb_dir = args.tensorboard_dir or os.path.join(args.output_dir, "tensorboard")
    logging.info("TensorBoard directory: %s", tb_dir)
    # Close/flush on normal exit, exceptions, and Ctrl+C. Remove stale future
    # events when resuming an older checkpoint into the same log directory.
    with SummaryWriter(tb_dir, purge_step=global_step + 1 if args.resume else None) as writer:
        writer.add_text("config", json.dumps(vars(args), indent=2), global_step)
        if args.val_every and args.val_on_start:
            logging.info("Running initial evaluation at step %d", global_step)
            preview = ValidationPreview(dataset, normal_predictor, args.output_dir,
                count=args.val_samples, size=val_size, index=args.val_index)
            preview.run(pipe, writer, global_step, args.val_denoising_steps, args.precision)
        optimizer.zero_grad(set_to_none=True)
        data_iter = iter(dataloader)
        micro_step = 0
        running_total = 0.0
        running_diff = 0.0
        running_normal = 0.0
        running_count = 0

        progress = tqdm(total=args.max_steps, initial=global_step, desc="Training", dynamic_ncols=True)

        while global_step < args.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            rgb = batch["rgb_norm"].to(device, non_blocking=True)
            gt_depth = batch["gt_depth_norm"].to(device, non_blocking=True)
            gt_valid = batch["gt_valid"].to(device, non_blocking=True)
            interp_depth = batch["interp_depth_norm"].to(device, non_blocking=True)
            distance = batch["distance"].to(device, non_blocking=True)
            # Predict after crop/flip so RGB, depth, K and normal share the same grid.
            if normal_predictor is not None:
                normal_list = []
                for image in rgb.detach().cpu():
                    image_np = ((image.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                    normal_np = normal_predictor.predict(image_np)['normal']
                    normal_list.append(torch.from_numpy(np.asarray(normal_np,dtype=np.float32)).permute(2,0,1))
                normal_prior = torch.stack(normal_list).to(device)
            else:
                normal_prior = batch['normal'].to(device, non_blocking=True)
            sparse_observed = batch["sparse_observed"].to(device, non_blocking=True)
            # Always use the dataset-provided GT-geometry mask. For gt_depth this also
            # removes hole-adjacent normals; for Metric3D it prevents prior-only loss
            # supervision where the dataset has no trustworthy GT geometry.
            normal_valid = batch["normal_valid"].to(device, non_blocking=True)
            intrinsics = batch["intrinsics"].to(device, non_blocking=True)
            d_min = batch["d_min"].to(device, non_blocking=True)
            d_max = batch["d_max"].to(device, non_blocking=True)

            batch_size = rgb.shape[0]
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            with get_autocast(device, args.precision):
                # Frozen encoders mirror Marigold: no gradient through RGB/GT/input-depth encodings.
                with torch.no_grad():
                    rgb_latent = pipe.encode_rgb(rgb)
                    gt_latent = encode_depth(pipe, gt_depth)
                    interp_latent = pipe.encode_rgb(interp_depth.repeat(1, 3, 1, 1))

                noise = torch.randn_like(gt_latent)
                noisy_latent = training_noise_scheduler.add_noise(gt_latent, noise, timesteps)

                distance_down = F.interpolate(
                    distance,
                    size=rgb_latent.shape[-2:],
                    mode="nearest",
                )
                normal_down = F.interpolate(
                    normal_prior,
                    size=rgb_latent.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                normal_mag = torch.linalg.vector_norm(normal_down, dim=1, keepdim=True)
                normal_down = torch.where(
                    normal_mag > 1e-6,
                    normal_down / normal_mag.clamp_min(1e-6),
                    torch.zeros_like(normal_down),
                )

                # Original 13 Murre channels + 3 normal-prior channels.
                unet_input = torch.cat(
                    [rgb_latent, interp_latent, distance_down, noisy_latent, normal_down],
                    dim=1,
                )
                text_embed = empty_text_embed.repeat(batch_size, 1, 1)
                model_pred = pipe.unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=text_embed,
                ).sample

                if prediction_type == "sample":
                    diffusion_target = gt_latent
                elif prediction_type == "epsilon":
                    diffusion_target = noise
                elif prediction_type == "v_prediction":
                    diffusion_target = training_noise_scheduler.get_velocity(
                        gt_latent, noise, timesteps
                    )
                else:
                    raise ValueError(f"Unsupported prediction type: {prediction_type}")

                # GT depth holes never contribute to the diffusion loss. The mask is
                # downsampled conservatively: any invalid source pixel invalidates the
                # corresponding latent cell.
                valid_down = valid_latent_mask(gt_valid, gt_latent.shape[-2:])
                latent_sqerr = (model_pred.float() - diffusion_target.float()).pow(2)
                valid_count = valid_down.sum().clamp_min(1)
                diffusion_loss = latent_sqerr[valid_down].sum() / valid_count

                if args.normal_weight > 0:
                    pred_x0 = prediction_to_x0(
                        noisy_latent,
                        model_pred,
                        timesteps,
                        training_noise_scheduler,
                        prediction_type,
                    )
                    # VAE weights are frozen, but decoding keeps autograd on the latent so
                    # the normal loss can back-propagate into the U-Net.
                    pred_depth = pipe.decode_depth(pred_x0)
                    pred_depth_01 = (pred_depth.clamp(-1.0, 1.0) + 1.0) * 0.5
                    # The prior is camera space, so the prediction is mapped back to the
                    # metric range of the input depth first. d_min/d_max come from the
                    # (partially observed) input depth, exactly like at inference time.
                    depth_range = (d_max - d_min).clamp_min(1e-6).view(-1, 1, 1, 1)
                    pred_depth_metric = pred_depth_01 * depth_range + d_min.view(-1, 1, 1, 1)
                    normal_loss = camera_normal_consistency_loss(
                        pred_depth=pred_depth_metric,
                        intrinsics=intrinsics,
                        normal_prior=normal_prior,
                        observed_depth=sparse_observed,
                        keep_ratio=args.normal_keep_ratio,
                        prior_valid=normal_valid,
                    )
                else:
                    normal_loss = diffusion_loss.new_zeros(())

                total_loss = diffusion_loss + args.normal_weight * normal_loss
                backward_loss = total_loss / args.gradient_accumulation_steps

            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite training loss; inspect depth units and precision")
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            micro_step += 1
            running_total += float(total_loss.detach())
            running_diff += float(diffusion_loss.detach())
            running_normal += float(normal_loss.detach())
            running_count += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            if global_step % args.log_every == 0 or global_step == 1 or global_step == args.max_steps:
                denom = max(running_count, 1)
                avg_total = running_total / denom
                avg_diff = running_diff / denom
                avg_normal = running_normal / denom
                lr = lr_scheduler.get_last_lr()[0]
                progress.set_postfix(
                    total=f"{avg_total:.4f}",
                    diff=f"{avg_diff:.4f}",
                    normal=f"{avg_normal:.4f}",
                    lr=f"{lr:.2e}",
                )
                logging.info(
                    "step=%d total=%.6f diffusion=%.6f normal=%.6f lr=%.3e size=%s",
                    global_step,
                    avg_total,
                    avg_diff,
                    avg_normal,
                    lr,
                    tuple(rgb.shape[-2:]),
                )
                writer.add_scalar("loss/total", avg_total, global_step)
                writer.add_scalar("loss/diffusion", avg_diff, global_step)
                writer.add_scalar("loss/normal", avg_normal, global_step)
                writer.add_scalar("loss/normal_weighted", args.normal_weight * avg_normal, global_step)
                writer.add_scalar("train/learning_rate", lr, global_step)
                writer.add_scalar("train/image_height", rgb.shape[-2], global_step)
                writer.add_scalar("train/image_width", rgb.shape[-1], global_step)
                writer.flush()
                running_total = running_diff = running_normal = 0.0
                running_count = 0

            if args.val_every and (global_step % args.val_every == 0 or global_step == args.max_steps):
                if preview is None:
                    preview = ValidationPreview(dataset, normal_predictor, args.output_dir,
                        count=args.val_samples, size=val_size, index=args.val_index)
                preview.run(pipe, writer, global_step, args.val_denoising_steps, args.precision)

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    pipe,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    args.output_dir,
                    global_step,
                )

        progress.close()
        if args.skip_final_save:
            logging.info("Smoke test finished without final checkpoint")
            return
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        pipe.save_pretrained(final_dir)
        torch.save(
            {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            os.path.join(final_dir, "trainer_state.pt"),
        )
        logging.info("Training finished. Final model: %s", final_dir)


if __name__ == "__main__":
    main()
