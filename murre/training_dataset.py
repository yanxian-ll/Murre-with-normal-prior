import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .util.depth_util import interp_depth, normalize_depth


RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTENSIONS = {".npy", ".npz", ".png", ".tif", ".tiff", ".exr"}
NORMAL_EXTENSIONS = DEPTH_EXTENSIONS | {".jpg", ".jpeg", ".bmp"}
SPARSE_EXTENSIONS = {".npz", ".npy"}


def _stem_map(root: str, extensions) -> Dict[str, str]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root)
    mapping = {}
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            # Training folders are expected to contain unique filename stems.
            # This also works for a flat folder, which is the recommended layout.
            if path.stem in mapping:
                raise ValueError(
                    f"Duplicate filename stem '{path.stem}' in {root}. "
                    "Use unique stems for training samples."
                )
            mapping[path.stem] = str(path)
    return mapping


def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path))
    if ext == ".npz":
        pack = np.load(path, allow_pickle=True)
        if "depth" in pack:
            return np.asarray(pack["depth"])
        if "arr_0" in pack:
            return np.asarray(pack["arr_0"])
        return np.asarray(pack[pack.files[0]])
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read array/image: {path}")
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.asarray(arr)


def _load_depth(path: str, scale: float) -> np.ndarray:
    depth = _load_array(path).astype(np.float32)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape} from {path}")
    depth = depth * float(scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def _load_normal(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    # Preserve RGB channel order for encoded normal images. Depth arrays keep
    # using OpenCV because 16-bit/float depth formats need unchanged loading.
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        normal = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    else:
        normal = _load_array(path).astype(np.float32)

    if normal.ndim != 3:
        raise ValueError(f"Normal must be HxWx3 or 3xHxW, got {normal.shape} from {path}")
    if normal.shape[0] == 3 and normal.shape[-1] != 3:
        normal = np.transpose(normal, (1, 2, 0))
    if normal.shape[-1] != 3:
        raise ValueError(f"Normal must have 3 channels, got {normal.shape} from {path}")

    normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    n_min = float(normal.min())
    n_max = float(normal.max())
    if n_max > 2.0 or n_min < -1.5:
        normal = normal / 127.5 - 1.0
    elif n_min >= 0.0 and n_max <= 1.0:
        normal = normal * 2.0 - 1.0

    magnitude = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = magnitude > 1e-6
    normal = np.where(valid, normal / np.maximum(magnitude, 1e-6), 0.0)
    return normal.astype(np.float32)


def _load_sparse_pack(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        pack = np.load(path, allow_pickle=True)
        if "arr_0" in pack:
            arr = np.asarray(pack["arr_0"])
        else:
            arr = np.asarray(pack[pack.files[0]])
    else:
        arr = np.asarray(np.load(path))

    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        depth = arr
        err = np.zeros_like(depth)
        nviews = np.full_like(depth, np.inf)
    elif arr.ndim == 3 and arr.shape[-1] >= 1:
        depth = arr[..., 0]
        err = arr[..., 1] if arr.shape[-1] > 1 else np.zeros_like(depth)
        nviews = arr[..., 2] if arr.shape[-1] > 2 else np.full_like(depth, np.inf)
    else:
        raise ValueError(f"Unsupported sparse-depth shape {arr.shape} from {path}")

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    err = np.nan_to_num(err, nan=np.inf, posinf=np.inf, neginf=np.inf)
    nviews = np.nan_to_num(nviews, nan=0.0, posinf=np.inf, neginf=0.0)
    return depth, err, nviews


class MurreNormalTrainingDataset(Dataset):
    """Folder-based dataset for Murre fine-tuning with a normal prior.

    Files are matched by filename stem across four folders:
      RGB, dense GT depth, SfM sparse depth, and normal prior.

    The sparse depth is normalized and interpolated exactly in the same style as
    Murre inference. The dense GT depth is normalized with the *same* per-image
    SfM-derived range, so the target latent and inference decoding convention match.
    """

    def __init__(
        self,
        rgb_dir: str,
        gt_depth_dir: str,
        sparse_depth_dir: str,
        normal_dir: str,
        height: int,
        width: int,
        max_depth: float = 80.0,
        gt_depth_scale: float = 1.0,
        sparse_depth_scale: float = 1.0,
        err_thr: Optional[float] = None,
        nviews_thr: Optional[int] = None,
        random_flip: bool = False,
        min_sparse_points: int = 3,
    ):
        super().__init__()
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must both be divisible by 8")

        rgb_map = _stem_map(rgb_dir, RGB_EXTENSIONS)
        gt_map = _stem_map(gt_depth_dir, DEPTH_EXTENSIONS)
        sparse_map = _stem_map(sparse_depth_dir, SPARSE_EXTENSIONS)
        normal_map = _stem_map(normal_dir, NORMAL_EXTENSIONS)

        common = sorted(set(rgb_map) & set(gt_map) & set(sparse_map) & set(normal_map))
        if not common:
            raise RuntimeError(
                "No common filename stems across RGB / GT depth / sparse depth / normal folders"
            )

        self.samples: List[Dict[str, str]] = [
            {
                "stem": stem,
                "rgb": rgb_map[stem],
                "gt": gt_map[stem],
                "sparse": sparse_map[stem],
                "normal": normal_map[stem],
            }
            for stem in common
        ]

        self.height = int(height)
        self.width = int(width)
        self.max_depth = float(max_depth)
        self.gt_depth_scale = float(gt_depth_scale)
        self.sparse_depth_scale = float(sparse_depth_scale)
        self.err_thr = err_thr
        self.nviews_thr = nviews_thr
        self.random_flip = bool(random_flip)
        self.min_sparse_points = int(min_sparse_points)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        rgb = np.asarray(Image.open(sample["rgb"]).convert("RGB"))
        gt_depth = _load_depth(sample["gt"], self.gt_depth_scale)
        sparse_depth, err, nviews = _load_sparse_pack(sample["sparse"])
        sparse_depth = sparse_depth * self.sparse_depth_scale
        normal = _load_normal(sample["normal"])

        if self.err_thr is not None:
            sparse_depth[err > self.err_thr] = 0.0
        if self.nviews_thr is not None:
            sparse_depth[nviews <= self.nviews_thr] = 0.0
        if self.max_depth > 0:
            sparse_depth[(sparse_depth > self.max_depth) | (sparse_depth < 0)] = 0.0
            gt_depth[(gt_depth > self.max_depth) | (gt_depth < 0)] = 0.0

        target_size = (self.width, self.height)
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        gt_depth = cv2.resize(gt_depth, target_size, interpolation=cv2.INTER_NEAREST)
        sparse_depth = cv2.resize(sparse_depth, target_size, interpolation=cv2.INTER_NEAREST)
        normal = cv2.resize(normal, target_size, interpolation=cv2.INTER_LINEAR)

        normal_mag = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.where(
            normal_mag > 1e-6,
            normal / np.maximum(normal_mag, 1e-6),
            0.0,
        )

        if self.random_flip and np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            gt_depth = np.ascontiguousarray(gt_depth[:, ::-1])
            sparse_depth = np.ascontiguousarray(sparse_depth[:, ::-1])
            normal = np.ascontiguousarray(normal[:, ::-1])
            # Horizontal image flip reverses camera/image-space normal x.
            normal[..., 0] *= -1.0

        sparse_valid = np.isfinite(sparse_depth) & (sparse_depth > 0)
        if int(sparse_valid.sum()) < self.min_sparse_points:
            raise RuntimeError(
                f"Sample '{sample['stem']}' has only {int(sparse_valid.sum())} valid SfM points; "
                f"at least {self.min_sparse_points} are required for Murre normalization."
            )

        gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)

        # Murre inference derives its normalization range from the sparse SfM depth.
        sparse_norm, d_min, d_max = normalize_depth(
            sparse_depth.copy(), pre_clip_max=self.max_depth
        )
        interp_norm, dist = interp_depth(sparse_norm)

        depth_range = max(float(d_max - d_min), 1e-6)
        gt_norm = (gt_depth - float(d_min)) / depth_range
        gt_norm = np.clip(gt_norm, 0.0, 1.0)

        # VAE inputs follow Marigold/Murre convention: [-1, 1].
        rgb_norm = rgb.astype(np.float32) / 127.5 - 1.0
        interp_norm = interp_norm.astype(np.float32) * 2.0 - 1.0
        gt_norm = gt_norm.astype(np.float32) * 2.0 - 1.0

        return {
            "stem": sample["stem"],
            "rgb_norm": torch.from_numpy(rgb_norm).permute(2, 0, 1).contiguous(),
            "gt_depth_norm": torch.from_numpy(gt_norm).unsqueeze(0).contiguous(),
            "gt_valid": torch.from_numpy(gt_valid).unsqueeze(0).bool(),
            "interp_depth_norm": torch.from_numpy(interp_norm).unsqueeze(0).contiguous(),
            "distance": torch.from_numpy(dist.astype(np.float32)).unsqueeze(0).contiguous(),
            "normal": torch.from_numpy(normal.astype(np.float32)).permute(2, 0, 1).contiguous(),
            # normal_consistency_loss only needs observed (>0) vs missing (<=0).
            "sparse_observed": torch.from_numpy(sparse_valid.astype(np.float32)).unsqueeze(0).contiguous(),
            "d_min": torch.tensor(float(d_min), dtype=torch.float32),
            "d_max": torch.tensor(float(d_max), dtype=torch.float32),
        }
