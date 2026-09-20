import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from .util.depth_util import normalize_depth, interp_depth, renorm_depth, align_depth


class MurreDepthOutput(BaseOutput):
    """
    Output class for Murre monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MurrePipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Murre with a surface-normal prior.

    The original Murre U-Net uses 13 input channels in this order:
        RGB latent (4) + interpolated SfM depth latent (4) + distance map (1)
        + noisy depth latent (4).

    This version appends a 3-channel camera-space/image-space normal prior:
        RGB latent (4) + interpolated SfM depth latent (4) + distance map (1)
        + noisy depth latent (4) + normal prior (3) = 16 channels.

    Appending normals after the original 13 channels is deliberate: when an original
    13-channel Murre checkpoint is loaded, all pretrained convolution weights are
    copied unchanged and the three new normal-channel weights are initialized to zero.
    The model therefore starts from the original Murre behavior and learns to use the
    normal prior only during fine-tuning.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    original_unet_in_channels = 13
    normal_channels = 3
    normal_unet_in_channels = original_unet_in_channels + normal_channels

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()

        # Expand an original Murre U-Net from 13 -> 16 input channels while
        # preserving every pretrained weight in the original 13 channels.
        self._ensure_normal_conditioning_channels(unet)

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @classmethod
    def _ensure_normal_conditioning_channels(cls, unet: UNet2DConditionModel) -> None:
        """Expand the first U-Net convolution from 13 to 16 input channels.

        Existing 13-channel weights are copied exactly. The new normal-channel
        weights are zero initialized so loading an old Murre checkpoint does not
        perturb its initial prediction before fine-tuning.
        """
        conv_in = unet.conv_in
        if conv_in.in_channels == cls.normal_unet_in_channels:
            return

        if conv_in.in_channels != cls.original_unet_in_channels:
            raise ValueError(
                "Unexpected Murre U-Net input channels: "
                f"{conv_in.in_channels}. Expected either "
                f"{cls.original_unet_in_channels} (original Murre) or "
                f"{cls.normal_unet_in_channels} (Murre with normal prior)."
            )

        new_conv = nn.Conv2d(
            in_channels=cls.normal_unet_in_channels,
            out_channels=conv_in.out_channels,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
            dilation=conv_in.dilation,
            groups=conv_in.groups,
            bias=conv_in.bias is not None,
            padding_mode=conv_in.padding_mode,
        ).to(device=conv_in.weight.device, dtype=conv_in.weight.dtype)

        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, : cls.original_unet_in_channels].copy_(conv_in.weight)
            if conv_in.bias is not None:
                new_conv.bias.copy_(conv_in.bias)

        unet.conv_in = new_conv
        unet.register_to_config(in_channels=cls.normal_unet_in_channels)
        logging.info(
            "Expanded Murre U-Net input from 13 to 16 channels; "
            "new normal-channel weights are zero initialized."
        )

    @staticmethod
    def _prepare_normal(
        input_normal: Union[Image.Image, np.ndarray, torch.Tensor],
        target_size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a normal prior to normalized [1, 3, H, W] in [-1, 1]."""
        if isinstance(input_normal, Image.Image):
            normal = pil_to_tensor(input_normal.convert("RGB"))
        elif isinstance(input_normal, np.ndarray):
            normal = torch.from_numpy(input_normal)
        elif isinstance(input_normal, torch.Tensor):
            normal = input_normal
        else:
            raise TypeError(f"Unknown normal input type: {type(input_normal) = }")

        if normal.ndim == 3:
            if normal.shape[0] == 3:
                normal = normal.unsqueeze(0)
            elif normal.shape[-1] == 3:
                normal = normal.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"Normal must have 3 channels, got shape {tuple(normal.shape)}"
                )
        elif normal.ndim == 4:
            if normal.shape[1] == 3:
                pass
            elif normal.shape[-1] == 3:
                normal = normal.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Normal must have 3 channels, got shape {tuple(normal.shape)}"
                )
        else:
            raise ValueError(f"Unsupported normal shape: {tuple(normal.shape)}")

        if normal.shape[0] != 1:
            raise ValueError(
                f"Pipeline expects one normal map per call, got batch {normal.shape[0]}"
            )

        normal = normal.to(torch.float32)
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)

        # Support common normal encodings: uint/float [0,255], float [0,1],
        # or already-decoded float [-1,1].
        n_min = float(normal.min())
        n_max = float(normal.max())
        if n_max > 2.0 or n_min < -1.5:
            normal = normal / 127.5 - 1.0
        elif n_min >= 0.0 and n_max <= 1.0:
            normal = normal * 2.0 - 1.0

        if tuple(normal.shape[-2:]) != tuple(target_size):
            normal = F.interpolate(
                normal,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        magnitude = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        valid = magnitude > 1e-6
        normal = torch.where(
            valid,
            normal / magnitude.clamp_min(1e-6),
            torch.zeros_like(normal),
        )
        return normal.to(dtype)

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        input_sparse_depth: Union[np.ndarray],
        input_normal: Union[Image.Image, np.ndarray, torch.Tensor],
        max_depth: float = 10.0,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        model_dtype=torch.float32,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MurreDepthOutput:
        """Predict metric depth from RGB, sparse SfM depth, and a normal prior."""
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        self._check_inference_step(denoising_steps)
        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")

        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        sdpt = input_sparse_depth

        rgb = resize_max_res(
            rgb,
            max_edge_resolution=processing_res,
            resample_method=resample_method,
        )

        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Normal Preprocess -----------------
        normal = self._prepare_normal(
            input_normal,
            target_size=rgb.shape[2:],
            dtype=self.dtype,
        )

        # ----------------- Sparse Depth Preprocess -----------------
        logging.info(f"sdpt.shape {sdpt.shape} & rgb.shape[2:] {rgb.shape[2:]}")
        assert sdpt.shape == rgb.shape[2:]
        sdpt_norm, d_min, d_max = normalize_depth(sdpt, pre_clip_max=max_depth)

        idpt, dist = interp_depth(sdpt_norm)
        idpt, dist = torch.from_numpy(idpt), torch.from_numpy(dist)
        idpt = idpt * 2.0 - 1.0

        # ----------------- Predicting depth -----------------
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_idpt = idpt.unsqueeze(0).unsqueeze(0).expand(ensemble_size, 3, -1, -1)
        duplicated_dist = dist.unsqueeze(0).unsqueeze(0).expand(ensemble_size, -1, -1, -1)
        duplicated_normal = normal.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(
            duplicated_rgb,
            duplicated_idpt,
            duplicated_dist,
            duplicated_normal,
        )

        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        for batch in iterable:
            (batched_img, batched_idpt, batched_dist, batched_normal) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                idpt_in=batched_idpt,
                dist_in=batched_dist,
                normal_in=batched_normal,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                model_dtype=model_dtype,
            )
            depth_pred_ls.append(depth_pred_raw.detach())

        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()

        if ensemble_size > 1:
            depth_pred = depth_preds.median(dim=0, keepdim=True)[0]
            pred_uncert = None
        else:
            depth_pred = depth_preds
            pred_uncert = None

        depth_pred = depth_pred.squeeze().clip(0, 1)
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        depth_pred_metric = renorm_depth(depth_pred, d_min, d_max)
        depth_pred_metric = align_depth(depth_pred_metric, sdpt)

        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, depth_pred.min(), depth_pred.max(), cmap=color_map
            ).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MurreDepthOutput(
            depth_np=depth_pred_metric.clip(0.0, max_depth),
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """Check if the requested number of denoising steps is reasonable."""
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """Encode text embedding for an empty prompt."""
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        idpt_in: torch.Tensor,
        dist_in: torch.Tensor,
        normal_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        model_dtype=torch.float32,
    ) -> torch.Tensor:
        """Perform an individual depth prediction without ensembling."""
        device = self.device
        rgb_in = rgb_in.to(device).to(model_dtype)
        idpt_in = idpt_in.to(device).to(model_dtype)
        dist_in = dist_in.to(device).to(model_dtype)
        normal_in = normal_in.to(device).to(model_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        rgb_latent = self.encode_rgb(rgb_in)
        ipdt_latent = self.encode_rgb(idpt_in)

        dist_down = F.interpolate(
            dist_in,
            size=(rgb_latent.shape[2], rgb_latent.shape[3]),
            mode="nearest",
        )
        normal_down = F.interpolate(
            normal_in,
            size=(rgb_latent.shape[2], rgb_latent.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        normal_norm = torch.linalg.vector_norm(normal_down, dim=1, keepdim=True)
        normal_down = torch.where(
            normal_norm > 1e-6,
            normal_down / normal_norm.clamp_min(1e-6),
            torch.zeros_like(normal_down),
        )

        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )

        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            # Keep the original 13 Murre channels in exactly the same positions and
            # append the 3 normal channels at the end for checkpoint compatibility.
            unet_input = torch.cat(
                [rgb_latent, ipdt_latent, dist_down, depth_latent, normal_down],
                dim=1,
            )

            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample

            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(depth_latent)
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """Encode a three-channel image into the VAE latent space."""
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """Decode a depth latent into a one-channel depth map."""
        depth_latent = depth_latent / self.depth_latent_scale_factor
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
