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
        max_edge: Longest edge fed to Metric3D (original aspect ratio preserved).
        depth_scale: Multiplies the predicted depth (e.g. ``0.001`` for mm -> m).
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        config: Optional[str] = None,
        metric3d_root: Optional[str] = None,
        device: str = "cuda",
        max_edge: int = 1024,
        depth_scale: float = 1.0,
    ):
        root, config_path, checkpoint_path = resolve_metric3d_paths(
            checkpoint=checkpoint, config=config, metric3d_root=metric3d_root
        )
        Config, build_model, load_ckpt = _import_metric3d(root)

        cfg = Config.fromfile(str(config_path))
        model = build_model(cfg)
        model, _, _, _ = load_ckpt(str(checkpoint_path), model, strict_match=False)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Metric3D normal prior requested device=cuda but CUDA is unavailable."
            )
        model.eval().to(self.device)
        model.requires_grad_(False)
        _register_device_depth_anchor(model, self.device)
        self.model = model
        self.max_edge = int(max_edge)
        self.depth_scale = float(depth_scale)
        self.checkpoint_path = checkpoint_path

    def _prepare_input(self, rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got shape {tuple(rgb.shape)}")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        target_h, target_w = _align_size(height, width, self.max_edge)
        if (target_h, target_w) != (height, width):
            resized = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
            resized = F.interpolate(
                resized.float(), size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            tensor = resized[0]
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()

        mean = torch.tensor(_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(_STD, dtype=torch.float32)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        return tensor, (height, width), (target_h, target_w)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict:
        """Predict ``depth`` [H, W] and camera-space ``normal`` [H, W, 3] for one image."""
        tensor, original_hw, _ = self._prepare_input(rgb)
        _, _, output = self.model.inference({"input": tensor})

        if "prediction_normal" not in output:
            raise RuntimeError(
                "Metric3D output has no 'prediction_normal'; use a config with the "
                "RAFT depth+normal head (e.g. vit.raft5.large.py)."
            )
        normal = output["prediction_normal"][:, :3].float()
        depth = output["prediction"][:, :1].float()

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
