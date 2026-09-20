import os

# OpenCV disables EXR in some builds unless this flag is set before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import random
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .util.depth_util import interp_depth, normalize_depth


RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTENSIONS = {".exr", ".npy", ".npz", ".png", ".tif", ".tiff"}


def _file_map(root: Path, extensions) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not root.is_dir():
        return mapping
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            mapping[path.stem] = str(path)
    return mapping


def _load_depth(path: str, scale: float = 1.0) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.asarray(np.load(path))
    elif ext == ".npz":
        pack = np.load(path, allow_pickle=True)
        if "depth" in pack:
            depth = np.asarray(pack["depth"])
        elif "arr_0" in pack:
            depth = np.asarray(pack["arr_0"])
        else:
            depth = np.asarray(pack[pack.files[0]])
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(
                f"Failed to read depth '{path}'. For EXR, make sure your OpenCV build has OpenEXR support."
            )

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            # Some EXR writers save depth in one channel of a multi-channel image.
            depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape} from {path}")

    depth = depth * float(scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize depth while preserving invalid (<=0) pixels."""
    valid = np.isfinite(depth) & (depth > 0)
    resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    valid_resized = cv2.resize(
        valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    resized[~valid_resized] = 0.0
    return resized.astype(np.float32)


def _depth_to_normal(depth_01: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Simple image-space normal from neighboring normalized depth values.

    This intentionally matches `murre.util.normal_util.depth_to_normal`:
        n = normalize([-dD/dx, -dD/dy, 1]).

    The prior is generated from full GT depth for now. Later this function can be
    replaced by an external normal estimator without changing the trainer interface.
    """
    depth_01 = np.asarray(depth_01, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)

    padded = np.pad(depth_01, ((1, 1), (1, 1)), mode="edge")
    dzdx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    dzdy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])

    normal = np.stack([-dzdx, -dzdy, np.ones_like(depth_01)], axis=-1)
    magnitude = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(magnitude, 1e-6)

    # A normal is supervised only where the center and four finite-difference
    # neighbours belong to valid GT depth.
    vp = np.pad(valid.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
    normal_valid = (
        vp[1:-1, 1:-1]
        & vp[1:-1, :-2]
        & vp[1:-1, 2:]
        & vp[:-2, 1:-1]
        & vp[2:, 1:-1]
    ).astype(bool)
    normal[~normal_valid] = 0.0
    return normal.astype(np.float32)


def _random_crop_pair(
    rgb: np.ndarray,
    depth: np.ndarray,
    target_height: int,
    target_width: int,
    crop_scale_min: float,
    crop_scale_max: float,
):
    """Random crop at the target aspect ratio, followed by resize."""
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(
            f"RGB/depth resolution mismatch: RGB={rgb.shape[:2]} depth={(h, w)}"
        )

    target_aspect = float(target_width) / float(target_height)
    image_aspect = float(w) / float(h)

    if image_aspect >= target_aspect:
        max_crop_h = h
        max_crop_w = int(round(h * target_aspect))
    else:
        max_crop_w = w
        max_crop_h = int(round(w / target_aspect))

    max_crop_h = max(1, min(max_crop_h, h))
    max_crop_w = max(1, min(max_crop_w, w))

    scale = random.uniform(crop_scale_min, crop_scale_max)
    crop_h = max(1, int(round(max_crop_h * scale)))
    crop_w = max(1, int(round(max_crop_w * scale)))

    # Keep exact target aspect as closely as possible after integer rounding.
    crop_w = min(w, max(1, int(round(crop_h * target_aspect))))
    crop_h = min(h, max(1, int(round(crop_w / target_aspect))))

    top = random.randint(0, max(h - crop_h, 0))
    left = random.randint(0, max(w - crop_w, 0))
    rgb = rgb[top : top + crop_h, left : left + crop_w]
    depth = depth[top : top + crop_h, left : left + crop_w]

    rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    depth = _resize_depth(depth, target_width, target_height)
    return rgb, depth


def _apply_border_occlusion(
    depth: np.ndarray,
    min_ratio: float,
    max_ratio: float,
    min_sides: int,
    max_sides: int,
    probability: float,
) -> np.ndarray:
    """Mask random portions of 1-4 image borders to simulate missing input depth."""
    masked = depth.copy()
    if random.random() > probability:
        return masked

    h, w = masked.shape
    sides = ["top", "bottom", "left", "right"]
    n_sides = random.randint(min_sides, max_sides)
    chosen = random.sample(sides, k=n_sides)

    for side in chosen:
        ratio = random.uniform(min_ratio, max_ratio)
        if side == "top":
            n = min(h, max(1, int(round(h * ratio))))
            masked[:n, :] = 0.0
        elif side == "bottom":
            n = min(h, max(1, int(round(h * ratio))))
            masked[h - n :, :] = 0.0
        elif side == "left":
            n = min(w, max(1, int(round(w * ratio))))
            masked[:, :n] = 0.0
        elif side == "right":
            n = min(w, max(1, int(round(w * ratio))))
            masked[:, w - n :] = 0.0

    return masked


class MurreNormalTrainingDataset(Dataset):
    """Hierarchical random sampler for multi-dataset / multi-scene training.

    Expected layout for every dataset root:

        DATASET_ROOT/
          scene_000/
            images/
              000001.jpg
            depth/
              000001.exr
          scene_001/
            images/
            depth/

    Sampling is deliberately hierarchical and uniform:
        dataset -> scene -> image

    Therefore a very large dataset or scene does not dominate simply because it
    contains more images.

    Full GT depth is used for the diffusion target and to generate the temporary
    normal prior. A copy of GT depth is border-masked to simulate the incomplete
    depth condition given to Murre.
    """

    def __init__(
        self,
        dataset_roots: Sequence[str],
        height: int,
        width: int,
        images_subdir: str = "images",
        depth_subdir: str = "depth",
        max_depth: float = 80.0,
        depth_scale: float = 1.0,
        crop_scale_min: float = 0.6,
        crop_scale_max: float = 1.0,
        random_flip: bool = False,
        occlusion_probability: float = 1.0,
        occlusion_min_ratio: float = 0.05,
        occlusion_max_ratio: float = 0.25,
        occlusion_min_sides: int = 1,
        occlusion_max_sides: int = 4,
    ):
        super().__init__()
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height and width must both be divisible by 8")
        if not 0.0 < crop_scale_min <= crop_scale_max <= 1.0:
            raise ValueError("crop scale must satisfy 0 < min <= max <= 1")
        if not 0.0 <= occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability must be in [0, 1]")
        if not 0.0 <= occlusion_min_ratio <= occlusion_max_ratio < 0.5:
            raise ValueError("occlusion ratios must satisfy 0 <= min <= max < 0.5")
        if not 1 <= occlusion_min_sides <= occlusion_max_sides <= 4:
            raise ValueError("occlusion side counts must satisfy 1 <= min <= max <= 4")

        self.height = int(height)
        self.width = int(width)
        self.images_subdir = images_subdir
        self.depth_subdir = depth_subdir
        self.max_depth = float(max_depth)
        self.depth_scale = float(depth_scale)
        self.crop_scale_min = float(crop_scale_min)
        self.crop_scale_max = float(crop_scale_max)
        self.random_flip = bool(random_flip)
        self.occlusion_probability = float(occlusion_probability)
        self.occlusion_min_ratio = float(occlusion_min_ratio)
        self.occlusion_max_ratio = float(occlusion_max_ratio)
        self.occlusion_min_sides = int(occlusion_min_sides)
        self.occlusion_max_sides = int(occlusion_max_sides)

        self.datasets: List[Dict] = []
        self.total_pairs = 0

        for dataset_root in dataset_roots:
            root = Path(dataset_root)
            if not root.is_dir():
                raise FileNotFoundError(dataset_root)

            scenes = []
            # Search recursively so DATASET_ROOT may contain one extra grouping level.
            for image_dir in root.rglob(images_subdir):
                if not image_dir.is_dir() or image_dir.name != images_subdir:
                    continue
                scene_dir = image_dir.parent
                depth_dir = scene_dir / depth_subdir
                if not depth_dir.is_dir():
                    continue

                image_map = _file_map(image_dir, RGB_EXTENSIONS)
                depth_map = _file_map(depth_dir, DEPTH_EXTENSIONS)
                common = sorted(set(image_map) & set(depth_map))
                if not common:
                    continue

                pairs = [
                    {
                        "stem": stem,
                        "rgb": image_map[stem],
                        "depth": depth_map[stem],
                    }
                    for stem in common
                ]
                scenes.append(
                    {
                        "name": str(scene_dir.relative_to(root)),
                        "pairs": pairs,
                    }
                )
                self.total_pairs += len(pairs)

            if scenes:
                self.datasets.append(
                    {
                        "name": root.name,
                        "root": str(root),
                        "scenes": scenes,
                    }
                )

        if not self.datasets:
            raise RuntimeError(
                "No valid training scenes found. Expected scene folders containing "
                f"'{images_subdir}/' and '{depth_subdir}/' with matching filename stems."
            )

    def __len__(self):
        # Sampling ignores the index, but a meaningful finite length lets DataLoader
        # form epochs. The outer trainer simply starts a new iterator when needed.
        return max(self.total_pairs, 1)

    def _sample_pair(self):
        dataset = random.choice(self.datasets)
        scene = random.choice(dataset["scenes"])
        pair = random.choice(scene["pairs"])
        return dataset, scene, pair

    def __getitem__(self, _index):
        dataset, scene, pair = self._sample_pair()

        rgb = np.asarray(Image.open(pair["rgb"]).convert("RGB"))
        gt_depth = _load_depth(pair["depth"], self.depth_scale)

        if self.max_depth > 0:
            gt_depth[(gt_depth > self.max_depth) | (gt_depth < 0)] = 0.0

        rgb, gt_depth = _random_crop_pair(
            rgb=rgb,
            depth=gt_depth,
            target_height=self.height,
            target_width=self.width,
            crop_scale_min=self.crop_scale_min,
            crop_scale_max=self.crop_scale_max,
        )

        if self.random_flip and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            gt_depth = np.ascontiguousarray(gt_depth[:, ::-1])

        gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
        if int(gt_valid.sum()) < 16:
            # Extremely invalid frames are rare. Resample rather than returning a
            # batch with no meaningful diffusion target.
            return self.__getitem__(0)

        # Simulated incomplete depth condition: start from the complete GT depth,
        # then remove random border regions. The full GT remains the target.
        input_depth = _apply_border_occlusion(
            gt_depth,
            min_ratio=self.occlusion_min_ratio,
            max_ratio=self.occlusion_max_ratio,
            min_sides=self.occlusion_min_sides,
            max_sides=self.occlusion_max_sides,
            probability=self.occlusion_probability,
        )
        observed = np.isfinite(input_depth) & (input_depth > 0)
        if int(observed.sum()) < 3:
            return self.__getitem__(0)

        # Match Murre inference: the normalization range is determined only by the
        # depth that remains visible after masking.
        input_norm, d_min, d_max = normalize_depth(
            input_depth.copy(), pre_clip_max=self.max_depth
        )
        interp_norm, dist = interp_depth(input_norm)

        depth_range = max(float(d_max - d_min), 1e-6)
        gt_01 = np.clip((gt_depth - float(d_min)) / depth_range, 0.0, 1.0)

        # Temporary normal prior: full GT depth -> normalized depth -> neighbour normal.
        # This uses exactly the same normalized-depth convention as the predicted-depth
        # normal loss, avoiding a scale mismatch in the image-space normal formula.
        normal = _depth_to_normal(gt_01, gt_valid)

        rgb_norm = rgb.astype(np.float32) / 127.5 - 1.0
        gt_norm = gt_01.astype(np.float32) * 2.0 - 1.0
        interp_norm = interp_norm.astype(np.float32) * 2.0 - 1.0

        sample_name = f"{dataset['name']}/{scene['name']}/{pair['stem']}"
        return {
            "stem": sample_name,
            "rgb_norm": torch.from_numpy(rgb_norm).permute(2, 0, 1).contiguous(),
            "gt_depth_norm": torch.from_numpy(gt_norm).unsqueeze(0).contiguous(),
            "gt_valid": torch.from_numpy(gt_valid).unsqueeze(0).bool(),
            "interp_depth_norm": torch.from_numpy(interp_norm).unsqueeze(0).contiguous(),
            "distance": torch.from_numpy(dist.astype(np.float32)).unsqueeze(0).contiguous(),
            "normal": torch.from_numpy(normal).permute(2, 0, 1).contiguous(),
            # The normal loss uses this only as an observed/missing indicator:
            # masked borders are fully supervised; observed interior keeps the
            # lowest-loss 90% by default.
            "sparse_observed": torch.from_numpy(observed.astype(np.float32)).unsqueeze(0).contiguous(),
            "d_min": torch.tensor(float(d_min), dtype=torch.float32),
            "d_max": torch.tensor(float(d_max), dtype=torch.float32),
        }
