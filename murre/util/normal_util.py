from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _to_bchw_depth(depth: torch.Tensor) -> torch.Tensor:
    """Convert a depth tensor to [B, 1, H, W]."""
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth.unsqueeze(0)
        else:
            depth = depth.unsqueeze(1)
    elif depth.ndim == 4:
        if depth.shape[1] != 1:
            raise ValueError(f"Expected depth with one channel, got {tuple(depth.shape)}")
    else:
        raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")
    return depth


def _to_bchw_normal(normal: torch.Tensor) -> torch.Tensor:
    """Convert a normal tensor to [B, 3, H, W]."""
    if normal.ndim == 3:
        if normal.shape[0] == 3:
            normal = normal.unsqueeze(0)
        elif normal.shape[-1] == 3:
            normal = normal.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported normal shape: {tuple(normal.shape)}")
    elif normal.ndim == 4:
        if normal.shape[1] == 3:
            pass
        elif normal.shape[-1] == 3:
            normal = normal.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported normal shape: {tuple(normal.shape)}")
    else:
        raise ValueError(f"Unsupported normal shape: {tuple(normal.shape)}")
    return normal


def depth_to_normal(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute differentiable image-space normals from neighboring depth values.

    The depth map is treated as a height field z=D(x, y). Central differences are
    used in the interior and replicate padding is used at the image boundary:

        n = normalize([-dD/dx, -dD/dy, 1]).

    This intentionally does not use camera intrinsics. It is the lightweight
    neighborhood normal requested for Murre fine-tuning.

    Args:
        depth: [H, W], [B, H, W], or [B, 1, H, W].
        eps: Numerical epsilon for normalization.

    Returns:
        Normal tensor with shape [B, 3, H, W].
    """
    depth = _to_bchw_depth(depth)

    padded = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    dzdx = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    dzdy = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])

    normal = torch.cat((-dzdx, -dzdy, torch.ones_like(depth)), dim=1)
    return F.normalize(normal, p=2, dim=1, eps=eps)


def _to_bchw_intrinsics(
    intrinsics: torch.Tensor, batch_size: int, height: int, width: int
) -> torch.Tensor:
    """Normalize intrinsics to [B, 3, 3] and check that they match the depth grid."""
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsics must be [3,3] or [B,3,3], got {tuple(intrinsics.shape)}")
    if intrinsics.shape[0] == 1 and batch_size > 1:
        intrinsics = intrinsics.expand(batch_size, -1, -1)
    if intrinsics.shape[0] != batch_size:
        raise ValueError(
            f"Intrinsics batch {intrinsics.shape[0]} does not match depth batch {batch_size}"
        )
    # Cropping can legitimately move the principal point outside the image.
    if not bool(torch.isfinite(intrinsics).all()):
        raise ValueError("Intrinsics contain non-finite values")
    return intrinsics


def depth_to_camera_normal(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert camera-Z depth to a camera-space surface normal.

    Points are unprojected with the standard pinhole model (OpenCV convention:
    x right, y down, z forward)::

        P(u, v) = ((u - cx) / fx * d, (v - cy) / fy * d, d)

    and the normal follows the cross product of the horizontal and vertical
    derivatives. The sign convention matches what Metric3D predicts (and what
    DP-GS renders): a front-facing surface has a negative z component.

    Args:
        depth: Camera-Z depth, [H, W], [B, H, W] or [B, 1, H, W].
        intrinsics: [3, 3] or [B, 3, 3] pixel intrinsics.
        eps: Numerical epsilon for the final normalization.

    Returns:
        Unit camera-space normals [B, 3, H, W].
    """
    depth = _to_bchw_depth(depth)
    batch_size, _, height, width = depth.shape
    intrinsics = _to_bchw_intrinsics(intrinsics, batch_size, height, width).to(
        device=depth.device, dtype=depth.dtype
    )

    fx = intrinsics[:, 0, 0].view(-1, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(-1, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(-1, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(-1, 1, 1, 1)
    if bool((fx.abs() < eps).any()) or bool((fy.abs() < eps).any()):
        raise ValueError("Intrinsics contain a zero focal length")

    u = torch.arange(width, device=depth.device, dtype=depth.dtype).view(1, 1, 1, width)
    v = torch.arange(height, device=depth.device, dtype=depth.dtype).view(1, 1, height, 1)

    pad = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    d_left = pad[:, :, 1:-1, :-2]
    d_right = pad[:, :, 1:-1, 2:]
    d_up = pad[:, :, :-2, 1:-1]
    d_down = pad[:, :, 2:, 1:-1]

    def unproject(depth_map, uu, vv):
        return torch.cat(
            [
                (uu - cx) / fx * depth_map,
                (vv - cy) / fy * depth_map,
                depth_map,
            ],
            dim=1,
        )

    left = unproject(d_left, u - 1.0, v)
    right = unproject(d_right, u + 1.0, v)
    up = unproject(d_up, u, v - 1.0)
    down = unproject(d_down, u, v + 1.0)

    dP_du = right - left
    bottom_to_top = up - down  # = -dP/dv in the y-down image convention
    normal = torch.cross(dP_du, bottom_to_top, dim=1)
    return F.normalize(normal, p=2, dim=1, eps=eps)


def _select_supervision_mask(
    pixel_loss: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
    keep_ratio: float,
) -> torch.Tensor:
    """Missing region fully supervised; observed region keeps the lowest-loss fraction."""
    missing = (~observed) & valid
    observed = observed & valid

    selected = missing.clone()
    batch_size = pixel_loss.shape[0]
    for b in range(batch_size):
        observed_flat = observed[b, 0].reshape(-1)
        indices = torch.nonzero(observed_flat, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue
        n_keep = max(1, int(indices.numel() * keep_ratio))
        observed_losses = pixel_loss[b, 0].reshape(-1)[indices]
        kept_local = torch.topk(observed_losses, k=n_keep, largest=False, sorted=False).indices
        selected[b, 0].reshape(-1)[indices[kept_local]] = True
    return selected


def camera_normal_consistency_loss(
    pred_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    normal_prior: torch.Tensor,
    observed_depth: torch.Tensor,
    keep_ratio: float = 0.9,
    prior_valid: torch.Tensor | None = None,
    eps: float = 1e-6,
    return_mask: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Normal supervision in camera space against an online Metric3D prior.

    ``pred_depth`` is the predicted camera-Z depth in the same units as the input
    depth. Pixels without input depth are supervised completely; pixels where input
    depth is visible keep only the lowest-error ``keep_ratio`` fraction, so a noisy
    or differently-calibrated prior cannot pull the observed geometry away.

    Args:
        pred_depth: Predicted camera-Z depth, [B,1,H,W].
        intrinsics: [B,3,3] or [3,3] pixel intrinsics for ``pred_depth``.
        normal_prior: Camera-space prior, [B,3,H,W] or [B,H,W,3].
        observed_depth: Input depth used only as an observed/missing indicator.
        keep_ratio: Fraction of observed pixels kept for supervision.
        prior_valid: Optional [B,1,H,W] mask marking valid prior pixels. Pixels with
            an invalid prior (e.g. Metric3D returned a zero vector) are ignored.
        eps: Numerical epsilon.
        return_mask: Also return the supervision mask.

    Returns:
        Scalar cosine normal loss, optionally with the [B,1,H,W] supervision mask.
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    pred_depth = _to_bchw_depth(pred_depth)
    observed_depth = _to_bchw_depth(observed_depth).to(
        device=pred_depth.device, dtype=pred_depth.dtype
    )
    normal_prior = _to_bchw_normal(normal_prior).to(
        device=pred_depth.device, dtype=pred_depth.dtype
    )
    if normal_prior.shape[-2:] != pred_depth.shape[-2:]:
        normal_prior = F.interpolate(
            normal_prior, size=pred_depth.shape[-2:], mode="bilinear", align_corners=False
        )
    if observed_depth.shape[-2:] != pred_depth.shape[-2:]:
        observed_depth = F.interpolate(
            observed_depth, size=pred_depth.shape[-2:], mode="nearest"
        )

    prior_norm = torch.linalg.vector_norm(normal_prior, dim=1, keepdim=True)
    valid_prior = torch.isfinite(normal_prior).all(dim=1, keepdim=True)
    valid_prior = valid_prior & torch.isfinite(prior_norm) & (prior_norm > eps)
    if prior_valid is not None:
        prior_valid = prior_valid.to(device=pred_depth.device)
        if prior_valid.shape[-2:] != pred_depth.shape[-2:]:
            prior_valid = F.interpolate(
                prior_valid.float(), size=pred_depth.shape[-2:], mode="nearest"
            )
        valid_prior = valid_prior & (prior_valid > 0.5)
    normal_prior = F.normalize(normal_prior, p=2, dim=1, eps=eps)

    pred_normal = depth_to_camera_normal(pred_depth, intrinsics, eps=eps)
    cosine = (pred_normal * normal_prior).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    pixel_loss = 1.0 - cosine

    observed = torch.isfinite(observed_depth) & (observed_depth > 0)
    selected = _select_supervision_mask(pixel_loss, observed, valid_prior, keep_ratio)

    selected_count = selected.sum()
    if selected_count == 0:
        loss = pixel_loss.sum() * 0.0
    else:
        loss = pixel_loss[selected].mean()

    if return_mask:
        return loss, selected
    return loss


def normal_consistency_loss(
    pred_depth: torch.Tensor,
    normal_prior: torch.Tensor,
    sparse_depth: torch.Tensor,
    keep_ratio: float = 0.9,
    eps: float = 1e-6,
    return_mask: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Robust normal supervision for Murre.

    Pixels without input sparse/SfM depth are supervised completely. Pixels with
    valid sparse/SfM depth are sorted by their per-pixel normal error, and only
    the lowest-error ``keep_ratio`` fraction is retained. With the default
    ``keep_ratio=0.9``, the highest-loss 10% of observed pixels are ignored.

    The selection is performed independently for every image in a batch.

    Args:
        pred_depth: Predicted depth, [B,1,H,W] (or compatible shape).
        normal_prior: Normal prior, [B,3,H,W] or [B,H,W,3].
        sparse_depth: Input SfM depth. Values <= 0 or non-finite are treated as
            missing observations.
        keep_ratio: Fraction of observed-depth pixels kept for supervision.
        eps: Numerical epsilon.
        return_mask: If True, also return the final selected supervision mask.

    Returns:
        Scalar mean cosine normal loss. If ``return_mask=True``, returns
        ``(loss, mask)`` where mask has shape [B,1,H,W].
    """
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    pred_depth = _to_bchw_depth(pred_depth)
    sparse_depth = _to_bchw_depth(sparse_depth).to(
        device=pred_depth.device, dtype=pred_depth.dtype
    )
    normal_prior = _to_bchw_normal(normal_prior).to(
        device=pred_depth.device, dtype=pred_depth.dtype
    )

    if sparse_depth.shape[-2:] != pred_depth.shape[-2:]:
        sparse_depth = F.interpolate(
            sparse_depth, size=pred_depth.shape[-2:], mode="nearest"
        )
    if normal_prior.shape[-2:] != pred_depth.shape[-2:]:
        normal_prior = F.interpolate(
            normal_prior,
            size=pred_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prior_norm = torch.linalg.vector_norm(normal_prior, dim=1, keepdim=True)
    valid_prior = torch.isfinite(normal_prior).all(dim=1, keepdim=True)
    valid_prior = valid_prior & torch.isfinite(prior_norm) & (prior_norm > eps)
    normal_prior = F.normalize(normal_prior, p=2, dim=1, eps=eps)

    pred_normal = depth_to_normal(pred_depth, eps=eps)
    cosine = (pred_normal * normal_prior).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    pixel_loss = 1.0 - cosine

    observed = torch.isfinite(sparse_depth) & (sparse_depth > 0)
    selected = _select_supervision_mask(pixel_loss, observed, valid_prior, keep_ratio)

    selected_count = selected.sum()
    if selected_count == 0:
        loss = pixel_loss.sum() * 0.0
    else:
        loss = pixel_loss[selected].mean()

    if return_mask:
        return loss, selected
    return loss
