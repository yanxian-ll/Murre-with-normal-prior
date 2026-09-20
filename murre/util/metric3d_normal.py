"""Online Metric3D surface-normal prior for Murre training.

The normal prior used during fine-tuning must be available at inference time, so it
is predicted from RGB by a frozen Metric3D model instead of being derived from the
dense ground-truth depth. Metric3D's RAFT depth+normal head returns both a metric
depth map and a camera-space normal map in a single forward pass, which is exactly
the prior consumed by :class:`murre.pipeline.MurrePipeline`.

The module loads the Metric3D copy vendored under ``baselines/DP-GS/Metric3D`` and
the local checkpoint at ``checkpoints/Metric3D``; nothing is downloaded. Importing
Metric3D requires ``mmengine``/``mmcv`` (available in the ``3dgs`` conda
environment). Failures raise an actionable error instead of falling back silently.
"""

from __future__ import annotations

import math
import logging
import cv2
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METRIC3D_ROOT = REPO_ROOT / "baselines" / "DP-GS" / "Metric3D"
DEFAULT_CONFIG = (
    DEFAULT_METRIC3D_ROOT / "mono" / "configs" / "HourglassDecoder" / "vit.raft5.large.py"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "Metric3D" / "metric_depth_vit_large_800k.pth"

# ImageNet statistics used by Metric3D for its input normalization.
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
# DPT/RAFT decoder patch size: both spatial dimensions must be divisible by 14.
_ALIGN = 14


def resolve_metric3d_paths(
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    metric3d_root: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Resolve Metric3D root/config/checkpoint, validating that all exist."""
    root = Path(metric3d_root).expanduser() if metric3d_root else DEFAULT_METRIC3D_ROOT
    root = root.resolve()
    config_path = Path(config).expanduser().resolve() if config else DEFAULT_CONFIG
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else DEFAULT_CHECKPOINT

    if not root.is_dir():
        raise FileNotFoundError(
            f"Metric3D source tree not found: {root}. "
            "Expected the copy vendored under baselines/DP-GS/Metric3D."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Metric3D config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Metric3D checkpoint not found: {checkpoint_path}. Expected "
            "checkpoints/Metric3D/metric_depth_vit_large_800k.pth (or pass --metric3d_checkpoint)."
        )
    return root, config_path, checkpoint_path


def _import_metric3d(metric3d_root: Path):
    """Import Metric3D's model factory with an actionable error on missing deps."""
    root_str = str(metric3d_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from mmengine import Config  # noqa: WPS433 (runtime import)

        from mono.model.monodepth_model import (  # noqa: WPS433
            get_configured_monodepth_model,
        )
        from mono.utils.running import load_ckpt  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Cannot import Metric3D (needs mmengine/mmcv). Use the conda environment that "
            "provides them, e.g. the local '3dgs' environment, or install them with "
            "'pip install mmengine mmcv'. Original error: {0}".format(exc)
        ) from exc
    return Config, get_configured_monodepth_model, load_ckpt


def _align_size(height: int, width: int, max_edge: int, align: int = _ALIGN) -> Tuple[int, int]:
    """Pick an aspect-preserving size that is a multiple of ``align``."""
    scale = 1.0 if max_edge <= 0 else min(1.0, float(max_edge) / float(max(height, width)))
    target_h = max(align, int(round(height * scale / align)) * align)
    target_w = max(align, int(round(width * scale / align)) * align)
    return target_h, target_w


def _register_device_depth_anchor(model, device: torch.device) -> None:
    """Register the depth-bin anchor on ``device``.

    Metric3D's RAFT head builds its depth bins with a hard-coded ``device="cuda"``
    (``RAFTDepthNormalDPTDecoder5.get_bins``), which breaks any non-CUDA device. The
    buffer is normally created lazily inside ``forward``; pre-registering the exact
    same values on the requested device keeps upstream code untouched and makes
    ``device="cpu"`` usable.
    """
    if device.type == "cuda":
        return
    for module in model.modules():
        if not (hasattr(module, "num_depth_regressor_anchor") and hasattr(module, "min_val")):
            continue
        if "depth_expectation_anchor" in getattr(module, "_buffers", {}):
            continue
        bins = torch.exp(
            torch.linspace(
                math.log(float(module.min_val)),
                math.log(float(module.max_val)),
                int(module.num_depth_regressor_anchor),
                device=device,
            )
        )
        module.register_buffer(
            "depth_expectation_anchor", bins.unsqueeze(0), persistent=False
        )


class Metric3DNormalEstimator:
    """Frozen Metric3D depth+normal predictor used as the online normal prior.

    Args:
        checkpoint: Metric3D ``.pth`` checkpoint; defaults to the local copy.
        config: Metric3D config with the RAFT depth+normal head.
        metric3d_root: Directory containing the vendored Metric3D package.
        device: Torch device string; ``cpu`` avoids competing with the UNet for VRAM.
        max_edge: Canvas longest edge; default 1064 gives official 616x1064 resize/pad.
        depth_scale: Multiplies the predicted depth (e.g. ``0.001`` for mm -> m).
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        config: Optional[str] = None,
        metric3d_root: Optional[str] = None,
        device: str = "cuda",
        max_edge: int = 1064,
        depth_scale: float = 1.0,
    ):
        root, config_path, checkpoint_path = resolve_metric3d_paths(
            checkpoint=checkpoint, config=config, metric3d_root=metric3d_root
        )
        Config, build_model, load_ckpt = _import_metric3d(root)

        cfg = Config.fromfile(str(config_path))
        model = build_model(cfg)
        weights = torch.load(str(checkpoint_path), map_location="cpu")["model_state_dict"]
        result = model.load_state_dict(weights, strict=False)
        missing = set(result.missing_keys) - {"depth_model.encoder.mask_token"}
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Metric3D checkpoint mismatch: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
        del weights
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Metric3D normal prior requested device=cuda but CUDA is unavailable."
            )
        if self.device.type == "cpu":
            # Installed xFormers has no CPU attention kernel. Bind the upstream
            # PyTorch fallback only on this predictor's attention instances.
            from types import MethodType
            from mono.model.backbones.ViT_DINO_reg import MemEffAttention, Attention
            for layer in model.modules():
                if isinstance(layer, MemEffAttention):
                    layer.forward = MethodType(Attention.forward, layer)
        model.eval().to(self.device)
        model.requires_grad_(False)
        _register_device_depth_anchor(model, self.device)
        self.model = model
        self.max_edge = int(max_edge)
        self.depth_scale = float(depth_scale)
        self.checkpoint_path = checkpoint_path
        factor = float(max_edge) / 1064 if max_edge > 0 else 1.0
        self.input_size = (max(28, round(616*factor/28)*28), max(28, round(1064*factor/28)*28))
        self.cache_signature = dict(preprocessing="metric3d-letterbox-v2", input_size=list(self.input_size),
            checkpoint=str(checkpoint_path), checkpoint_mtime=checkpoint_path.stat().st_mtime_ns)
        if self.input_size != (616,1064):
            logging.warning("Metric3D input %s differs from official 616x1064; small sizes can collapse normals",self.input_size)

    def _prepare_input(self, rgb: np.ndarray):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}")
        height,width = rgb.shape[:2]
        target_h,target_w = self.input_size
        scale = min(target_h/height,target_w/width)
        new_h,new_w = max(1,int(height*scale)),max(1,int(width*scale))
        resized = cv2.resize(rgb,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
        top,left = (target_h-new_h)//2,(target_w-new_w)//2
        padded = cv2.copyMakeBorder(resized,top,target_h-new_h-top,left,target_w-new_w-left,
            cv2.BORDER_CONSTANT,value=_MEAN)
        tensor = torch.from_numpy(np.ascontiguousarray(padded)).permute(2,0,1).float()
        mean = torch.tensor(_MEAN)[:,None,None]
        std = torch.tensor(_STD)[:,None,None]
        return ((tensor-mean)/std)[None].to(self.device), (height,width), (top,left,new_h,new_w)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict:
        """Predict ``depth`` [H, W] and camera-space ``normal`` [H, W, 3] for one image."""
        tensor, original_hw, (top,left,new_h,new_w) = self._prepare_input(rgb)
        _, _, output = self.model.inference({"input": tensor})

        if "prediction_normal" not in output:
            raise RuntimeError(
                "Metric3D output has no 'prediction_normal'; use a config with the "
                "RAFT depth+normal head (e.g. vit.raft5.large.py)."
            )
        normal = output["prediction_normal"][:, :3].float()
        depth = output["prediction"][:, :1].float()

        normal = normal[:,:,top:top+new_h,left:left+new_w]
        depth = depth[:,:,top:top+new_h,left:left+new_w]
        normal = F.interpolate(normal, size=original_hw, mode="bilinear", align_corners=False)
        depth = F.interpolate(depth, size=original_hw, mode="bilinear", align_corners=False)

        magnitude = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        normal = torch.where(
            magnitude > 1e-6, normal / magnitude.clamp_min(1e-6), torch.zeros_like(normal)
        )
        normal_np = normal[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        depth_np = depth[0, 0].cpu().numpy().astype(np.float32, copy=False) * self.depth_scale
        return {"depth": depth_np, "normal": normal_np}

    def close(self) -> None:
        """Release the frozen model."""
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
