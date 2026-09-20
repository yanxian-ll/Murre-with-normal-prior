import argparse
import json
import logging
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from murre.pipeline import MurrePipeline
from murre.training_dataset import MurreNormalTrainingDataset
from murre.util.normal_util import normal_consistency_loss


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Marigold-style Murre fine-tuning with depth-derived surface-normal prior."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Original Murre checkpoint.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a saved checkpoint directory.")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--dataset_roots",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more dataset roots. Each root contains many scene folders; "
            "every scene contains images/ and depth/ by default."
        ),
    )
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--depth_subdir", type=str, default="depth")
    parser.add_argument("--depth_scale", type=float, default=1.0)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--crop_scale_min", type=float, default=0.6)
    parser.add_argument("--crop_scale_max", type=float, default=1.0)
    parser.add_argument("--random_flip", action="store_true")

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
        help="Minimum fraction masked from a selected image border.",
    )
    parser.add_argument(
        "--occlusion_max_ratio",
        type=float,
        default=0.25,
        help="Maximum fraction masked from a selected image border.",
    )
    parser.add_argument("--occlusion_min_sides", type=int, default=1)
    parser.add_argument("--occlusion_max_sides", type=int, default=4)

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
    parser.add_argument("--save_every", type=int, default=1000)
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
    if not (0.0 < args.normal_keep_ratio <= 1.0):
        parser.error("--normal_keep_ratio must be in (0, 1]")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.precision != "fp32":
        logging.warning("CUDA is unavailable; forcing fp32 training.")
        args.precision = "fp32"
    logging.info("device=%s precision=%s", device, args.precision)

    model_path = args.resume if args.resume is not None else args.checkpoint
    pipe: MurrePipeline = MurrePipeline.from_pretrained(model_path, torch_dtype=torch.float32)
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

    dataset = MurreNormalTrainingDataset(
        dataset_roots=args.dataset_roots,
        height=args.height,
        width=args.width,
        images_subdir=args.images_subdir,
        depth_subdir=args.depth_subdir,
        max_depth=args.max_depth,
        depth_scale=args.depth_scale,
        crop_scale_min=args.crop_scale_min,
        crop_scale_max=args.crop_scale_max,
        random_flip=args.random_flip,
        occlusion_probability=args.occlusion_probability,
        occlusion_min_ratio=args.occlusion_min_ratio,
        occlusion_max_ratio=args.occlusion_max_ratio,
        occlusion_min_sides=args.occlusion_min_sides,
        occlusion_max_sides=args.occlusion_max_sides,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # __getitem__ performs hierarchical random sampling itself.
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=loader_generator,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    n_scenes = sum(len(d["scenes"]) for d in dataset.datasets)
    logging.info(
        "Training hierarchy: %d datasets, %d scenes, %d matched image/depth pairs",
        len(dataset.datasets),
        n_scenes,
        dataset.total_pairs,
    )
    logging.info(
        "Sampling policy: uniform dataset -> uniform scene -> uniform image; crop_scale=[%.2f, %.2f]",
        args.crop_scale_min,
        args.crop_scale_max,
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
        normal_prior = batch["normal"].to(device, non_blocking=True)
        sparse_observed = batch["sparse_observed"].to(device, non_blocking=True)

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
                normal_loss = normal_consistency_loss(
                    pred_depth=pred_depth_01,
                    normal_prior=normal_prior,
                    sparse_depth=sparse_observed,
                    keep_ratio=args.normal_keep_ratio,
                )
            else:
                normal_loss = diffusion_loss.new_zeros(())

            total_loss = diffusion_loss + args.normal_weight * normal_loss
            backward_loss = total_loss / args.gradient_accumulation_steps

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

        if global_step % args.log_every == 0 or global_step == 1:
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
                "step=%d total=%.6f diffusion=%.6f normal=%.6f lr=%.3e",
                global_step,
                avg_total,
                avg_diff,
                avg_normal,
                lr,
            )
            running_total = running_diff = running_normal = 0.0
            running_count = 0

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
