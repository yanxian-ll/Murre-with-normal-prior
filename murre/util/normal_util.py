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
    missing = (~observed) & valid_prior
    observed = observed & valid_prior

    selected = missing.clone()
    batch_size = pred_depth.shape[0]
    for b in range(batch_size):
        observed_flat = observed[b, 0].reshape(-1)
        indices = torch.nonzero(observed_flat, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            continue

        n_keep = max(1, int(indices.numel() * keep_ratio))
        observed_losses = pixel_loss[b, 0].reshape(-1)[indices]
        kept_local = torch.topk(
            observed_losses,
            k=n_keep,
            largest=False,
            sorted=False,
        ).indices
        kept_indices = indices[kept_local]
        selected[b, 0].reshape(-1)[kept_indices] = True

    selected_count = selected.sum()
    if selected_count == 0:
        loss = pixel_loss.sum() * 0.0
    else:
        loss = pixel_loss[selected].mean()

    if return_mask:
        return loss, selected
    return loss
