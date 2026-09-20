import os

# OpenCV disables EXR in some builds unless this flag is set before importing cv2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import logging
import random
import csv
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .util.depth_util import interp_depth, normalize_depth
from .util.normal_util import depth_to_camera_normal


RGB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTENSIONS = {".exr", ".npy", ".npz", ".png", ".tif", ".tiff"}

NORMAL_SOURCES = ("metric3d", "gt_depth", "deferred_metric3d")


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
        depth = np.asarray(np.load(path, allow_pickle=False))
    elif ext == ".npz":
        with np.load(path, allow_pickle=False) as pack:
            if "depth" in pack:
                depth = np.asarray(pack["depth"])
            elif "arr_0" in pack:
                depth = np.asarray(pack["arr_0"])
            elif pack.files:
                depth = np.asarray(pack[pack.files[0]])
            else:
                raise ValueError(f"Empty depth archive: {path}")
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


def _read_intrinsics(path: str) -> np.ndarray:
    """Read the 3x3 pixel intrinsics block written next to every processed image.

    The processed scenes store one `<stem>.txt` per image containing a world-to-camera
    extrinsic block followed by an ``intrinsic: fx fy cx cy (pixel)`` 3x3 block.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("intrinsic"):
            rows = [
                [float(value) for value in lines[index + 1 + row].split()[:3]]
                for row in range(3)
            ]
            intrinsics = np.asarray(rows, dtype=np.float32)
            if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                    or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
                raise ValueError(f"Malformed intrinsics in {path}")
            return intrinsics
    raise ValueError(f"No intrinsic block found in {path}")


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize depth while preserving invalid (<=0) pixels."""
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, 0).astype(np.float32)
    resized = cv2.resize(clean, (width, height), interpolation=cv2.INTER_LINEAR)
    coverage = cv2.resize(valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    valid_resized = coverage > .999
    resized /= np.maximum(coverage, 1e-6)
    resized[~valid_resized] = 0.0
    return resized.astype(np.float32)


def _scale_intrinsics(intrinsics: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.array(intrinsics, dtype=np.float32, copy=True)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _work_resolution_size(
    height: int, width: int, work_resolution: int
) -> Tuple[int, int]:
    """Aspect-preserving working size; never upscales beyond the source."""
    if work_resolution <= 0 or max(height, width) <= work_resolution:
        return height, width
    scale = float(work_resolution) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _sample_crop_box(
    rgb: np.ndarray,
    depth: np.ndarray,
    target_height: int,
    target_width: int,
    crop_scale_min: float,
    crop_scale_max: float,
):
    """Sample a random crop box at the target aspect ratio."""
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
    return (top, left, crop_h, crop_w)


def _sample_border_endpoints(
    min_ratio: float,
    max_ratio: float,
    slant_probability: float,
    slant_max_delta: float,
):
    """Sample the two inward offsets of one border occluder.

    Equal offsets give the old axis-aligned mask. Different offsets produce an
    oblique cutting line. `slant_max_delta` is the maximum difference between
    the two endpoint offsets, expressed as a fraction of image size.
    """
    center = random.uniform(min_ratio, max_ratio)
    if random.random() >= slant_probability or slant_max_delta <= 0:
        return center, center

    delta = random.uniform(-slant_max_delta, slant_max_delta)
    r0 = float(np.clip(center - 0.5 * delta, 0.0, 0.49))
    r1 = float(np.clip(center + 0.5 * delta, 0.0, 0.49))
    return r0, r1


def _draw_border_occluder(
    mask: np.ndarray,
    side: str,
    r0: float,
    r1: float,
):
    """Union one straight/slanted border polygon into a binary mask."""
    h, w = mask.shape
    x1 = max(w - 1, 0)
    y1 = max(h - 1, 0)

    if side == "top":
        y_left = int(round(r0 * h))
        y_right = int(round(r1 * h))
        points = np.array(
            [[0, 0], [x1, 0], [x1, min(y_right, y1)], [0, min(y_left, y1)]],
            dtype=np.int32,
        )
    elif side == "bottom":
        y_left = int(round(r0 * h))
        y_right = int(round(r1 * h))
        points = np.array(
            [
                [0, y1],
                [x1, y1],
                [x1, max(y1 - y_right, 0)],
                [0, max(y1 - y_left, 0)],
            ],
            dtype=np.int32,
        )
    elif side == "left":
        x_top = int(round(r0 * w))
        x_bottom = int(round(r1 * w))
        points = np.array(
            [[0, 0], [min(x_top, x1), 0], [min(x_bottom, x1), y1], [0, y1]],
            dtype=np.int32,
        )
    elif side == "right":
        x_top = int(round(r0 * w))
        x_bottom = int(round(r1 * w))
        points = np.array(
            [
                [x1, 0],
                [x1, y1],
                [max(x1 - x_bottom, 0), y1],
                [max(x1 - x_top, 0), 0],
            ],
            dtype=np.int32,
        )
    else:
        raise ValueError(f"Unknown border side: {side}")

    cv2.fillPoly(mask, [points], 1)


def _apply_border_occlusion(
    depth: np.ndarray,
    min_ratio: float,
    max_ratio: float,
    min_sides: int,
    max_sides: int,
    probability: float,
    slant_probability: float = 0.8,
    slant_max_delta: float = 0.25,
    min_visible_ratio: float = 0.15,
    max_attempts: int = 8,
) -> np.ndarray:
    """Mask random border regions using straight or oblique polygon boundaries.

    Each selected side contributes a polygon connected to that image border. The
    inward boundary may be horizontal/vertical or slanted. Combining several sides
    naturally creates trapezoids, triangular wedges, and irregular convex/non-convex
    visible regions while keeping the missing region attached to the image boundary.
    """
    if random.random() > probability:
        return depth.copy()

    valid_original = np.isfinite(depth) & (depth > 0)
    original_count = int(valid_original.sum())
    min_visible = max(3, int(round(original_count * min_visible_ratio)))

    sides = ["top", "bottom", "left", "right"]
    best = depth.copy()
    best_visible = -1

    for _ in range(max_attempts):
        occlusion_mask = np.zeros(depth.shape, dtype=np.uint8)
        n_sides = random.randint(min_sides, max_sides)
        chosen = random.sample(sides, k=n_sides)

        for side in chosen:
            r0, r1 = _sample_border_endpoints(
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                slant_probability=slant_probability,
                slant_max_delta=slant_max_delta,
            )
            _draw_border_occluder(occlusion_mask, side, r0, r1)

        candidate = depth.copy()
        candidate[occlusion_mask.astype(bool)] = 0.0
        visible_count = int((np.isfinite(candidate) & (candidate > 0)).sum())

        if visible_count >= min_visible:
            return candidate
        if visible_count > best_visible:
            best = candidate
            best_visible = visible_count

    # Extremely aggressive combinations are retried above. If all attempts violate
    # the minimum-visible constraint, use the least destructive attempted candidate.
    if best_visible >= 3:
        return best
    return depth.copy()


_ESTIMATOR_CACHE: Dict[tuple, object] = {}


def _get_normal_estimator(params: dict):
    """Lazily build one Metric3D estimator per DataLoader worker and share it.

    Building Metric3D takes seconds and ~1GB, so it happens on the first sample of
    each worker instead of in the parent process.
    """
    key = (
        params["metric3d_checkpoint"],
        params["metric3d_config"],
        params["metric3d_root"],
        params["metric3d_device"],
        params["metric3d_max_edge"],
    )
    estimator = _ESTIMATOR_CACHE.get(key)
    if estimator is None:
        from .util.metric3d_normal import Metric3DNormalEstimator

        estimator = Metric3DNormalEstimator(
            checkpoint=params["metric3d_checkpoint"],
            config=params["metric3d_config"],
            metric3d_root=params["metric3d_root"],
            device=params["metric3d_device"],
            max_edge=params["metric3d_max_edge"],
        )
        _ESTIMATOR_CACHE[key] = estimator
    return estimator


class MurreNormalTrainingDataset(Dataset):
    """Hierarchical random sampler for multi-dataset / multi-scene training.

    Sampling is deliberately hierarchical and uniform:
        dataset -> scene -> image

    so that a dataset or scene containing many more images does not dominate
    training. Pairs come from a TXT index written by
    ``dataset/build_murre_training_index.py`` when ``index_path`` is given, and from
    a directory walk otherwise.

    Full GT depth is used for the diffusion target. A copy of GT depth is
    border-masked with straight/slanted polygon occluders to simulate an incomplete
    depth condition given to Murre. The surface-normal prior is predicted online by
    Metric3D from RGB (``normal_source="metric3d"``), so training matches inference;
    ``normal_source="gt_depth"`` derives the same camera-space prior from GT depth
    and is intended for ablations only.

    Unreadable or degenerate frames (missing EXR codec, all-zero depth, RGB/depth
    shape mismatch, missing intrinsics, ...) are skipped by drawing another sample
    instead of raising, up to ``max_retries`` attempts.
    """

    def __init__(
        self,
        dataset_roots: Optional[Sequence[str]] = None,
        index_path: Optional[str] = None,
        height: int = 512,
        width: int = 768,
        images_subdir: str = "images",
        depth_subdir: str = "depth",
        camera_subdir: str = "cams",
        max_depth: float = 80.0,
        max_depth_mode: str = "auto",
        max_depth_quantile: float = 99.5,
        max_depth_scale: float = 1.5,
        depth_scales: Optional[dict] = None,
        default_depth_scale: float = 1.0,
        work_resolution: int = 384,
        crop_scale_min: float = 0.9,
        crop_scale_max: float = 1.0,
        random_flip: bool = False,
        occlusion_probability: float = 1.0,
        occlusion_min_ratio: float = 0.05,
        occlusion_max_ratio: float = 0.25,
        occlusion_min_sides: int = 1,
        occlusion_max_sides: int = 4,
        occlusion_slant_probability: float = 0.8,
        occlusion_slant_max_delta: float = 0.25,
        occlusion_min_visible_ratio: float = 0.15,
        normal_source: str = "metric3d",
        metric3d_checkpoint: Optional[str] = None,
        metric3d_config: Optional[str] = None,
        metric3d_root: Optional[str] = None,
        metric3d_max_edge: int = 1064,
        metric3d_device: str = "cuda",
        max_retries: int = 8,
        min_valid_pixels: int = 256,
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
        if not 0.0 <= occlusion_slant_probability <= 1.0:
            raise ValueError("occlusion_slant_probability must be in [0, 1]")
        if not 0.0 <= occlusion_slant_max_delta < 0.5:
            raise ValueError("occlusion_slant_max_delta must be in [0, 0.5)")
        if not 0.0 < occlusion_min_visible_ratio <= 1.0:
            raise ValueError("occlusion_min_visible_ratio must be in (0, 1]")
        if normal_source not in NORMAL_SOURCES:
            raise ValueError(
                f"normal_source must be one of {NORMAL_SOURCES}, got {normal_source}"
            )
        if max_depth_mode not in ("auto", "fixed"):
            raise ValueError(f"Unknown max_depth_mode: {max_depth_mode}")
        if not 0.0 < max_depth_quantile <= 100.0:
            raise ValueError("max_depth_quantile must be in (0, 100]")
        if max_depth_scale <= 0:
            raise ValueError("max_depth_scale must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be >= 1")
        if dataset_roots is None and index_path is None:
            raise ValueError("Provide either dataset_roots or index_path")

        self.height = int(height)
        self.width = int(width)
        self.images_subdir = images_subdir
        self.depth_subdir = depth_subdir
        self.camera_subdir = camera_subdir
        self.max_depth = float(max_depth)
        self.max_depth_mode = max_depth_mode
        self.max_depth_quantile = float(max_depth_quantile)
        self.max_depth_scale = float(max_depth_scale)
        self.depth_scales = self._resolve_depth_scales(depth_scales)
        self.default_depth_scale = float(default_depth_scale)
        self.work_resolution = int(work_resolution)
        self.crop_scale_min = float(crop_scale_min)
        self.crop_scale_max = float(crop_scale_max)
        self.random_flip = bool(random_flip)
        self.occlusion_probability = float(occlusion_probability)
        self.occlusion_min_ratio = float(occlusion_min_ratio)
        self.occlusion_max_ratio = float(occlusion_max_ratio)
        self.occlusion_min_sides = int(occlusion_min_sides)
        self.occlusion_max_sides = int(occlusion_max_sides)
        self.occlusion_slant_probability = float(occlusion_slant_probability)
        self.occlusion_slant_max_delta = float(occlusion_slant_max_delta)
        self.occlusion_min_visible_ratio = float(occlusion_min_visible_ratio)
        self.normal_source = normal_source
        self.max_retries = int(max_retries)
        self.min_valid_pixels = int(min_valid_pixels)
        self.index_path = str(index_path) if index_path is not None else None
        self.metric3d_params = {
            "metric3d_checkpoint": metric3d_checkpoint,
            "metric3d_config": metric3d_config,
            "metric3d_root": metric3d_root,
            "metric3d_max_edge": int(metric3d_max_edge),
            "metric3d_device": metric3d_device,
        }

        self._intrinsics_cache: Dict[str, np.ndarray] = {}
        self.stats = {"ok": 0, "retries": 0, "failures": 0}

        if index_path is not None:
            self.datasets = self._datasets_from_index(index_path)
        else:
            self.datasets = self._datasets_from_roots(dataset_roots)

        if not self.datasets:
            raise RuntimeError(
                "No valid training pairs found. Expected scenes containing "
                f"'{images_subdir}/' and '{depth_subdir}/' with matching filename stems, "
                "or a non-empty index built by dataset/build_murre_training_index.py."
            )
        self.total_pairs = sum(
            len(scene["pairs"]) for dataset in self.datasets for scene in dataset["scenes"]
        )
        self.total_scenes = sum(len(dataset["scenes"]) for dataset in self.datasets)

    @staticmethod
    def _resolve_depth_scales(depth_scales) -> Dict[str, float]:
        """Per-dataset depth unit conversions, given inline or as a JSON path."""
        if depth_scales is None:
            return {}
        if isinstance(depth_scales, (str, Path)):
            path = Path(depth_scales).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"depth scale table not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "datasets" in payload:
                payload = payload["datasets"]
            if not isinstance(payload, dict):
                raise ValueError(
                    f"depth scale table must map dataset -> scale, got {type(payload)}"
                )
            depth_scales = payload
        return {str(key): float(value) for key, value in dict(depth_scales).items()}

    def _datasets_from_index(self, index_path) -> List[Dict]:
        path = Path(index_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Training index not found: {path}")

        order: List[str] = []
        grouped: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            if header[:3] != ["dataset", "scene", "rgb"]:
                raise ValueError(
                    f"Unexpected index header in {path}: {header}. Rebuild it with "
                    "dataset/build_murre_training_index.py"
                )
            for line in handle:
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                dataset, scene, rgb, depth = fields[0], fields[1], fields[2], fields[3]
                camera = fields[4] if len(fields) > 4 else ""
                if dataset not in grouped:
                    grouped[dataset] = {}
                    order.append(dataset)
                grouped[dataset].setdefault(scene, []).append(
                    (Path(rgb).stem, rgb, depth, camera)
                )

        datasets = []
        for name in order:
            scenes = [
                {"name": scene, "pairs": pairs}
                for scene, pairs in sorted(grouped[name].items())
            ]
            if scenes:
                datasets.append({"name": name, "root": None, "scenes": scenes})
        return datasets

    def _datasets_from_roots(self, dataset_roots: Sequence[str]) -> List[Dict]:
        datasets = []
        for dataset_root in dataset_roots:
            root = Path(dataset_root)
            if not root.is_dir():
                raise FileNotFoundError(dataset_root)

            scenes = []
            # Search recursively so DATASET_ROOT may contain one extra grouping level.
            for image_dir in sorted(root.rglob(self.images_subdir)):
                if not image_dir.is_dir() or image_dir.name != self.images_subdir:
                    continue
                scene_dir = image_dir.parent
                depth_dir = scene_dir / self.depth_subdir
                if not depth_dir.is_dir():
                    continue

                image_map = _file_map(image_dir, RGB_EXTENSIONS)
                depth_map = _file_map(depth_dir, DEPTH_EXTENSIONS)
                common = sorted(set(image_map) & set(depth_map))
                if not common:
                    continue

                camera_dir = scene_dir / self.camera_subdir
                pairs = [
                    (
                        stem,
                        image_map[stem],
                        depth_map[stem],
                        str(camera_dir / (stem + ".txt"))
                        if (camera_dir / (stem + ".txt")).is_file()
                        else "",
                    )
                    for stem in common
                ]
                scenes.append({"name": str(scene_dir.relative_to(root)), "pairs": pairs})

            if scenes:
                datasets.append({"name": root.name, "root": str(root), "scenes": scenes})
        return datasets

    def __len__(self):
        # Sampling ignores the index, but a meaningful finite length lets DataLoader
        # form epochs. The outer trainer simply starts a new iterator when needed.
        return max(self.total_pairs, 1)

    def _sample_pair(self):
        dataset = random.choice(self.datasets)
        scene = random.choice(dataset["scenes"])
        pair = random.choice(scene["pairs"])
        return dataset, scene, pair

    def _intrinsics_for(self, camera_path: str) -> Optional[np.ndarray]:
        if not camera_path:
            return None
        cached = self._intrinsics_cache.get(camera_path)
        if cached is None:
            cached = _read_intrinsics(camera_path)
            if len(self._intrinsics_cache) > 512:
                self._intrinsics_cache.clear()
            self._intrinsics_cache[camera_path] = cached
        return cached

    def _clip_max_for(self, depth: np.ndarray) -> float:
        """Per-frame upper clip for depth normalization.

        Dataset depth units differ (meters, centimeters, arbitrary SfM units), so a
        single fixed cap either discards far geometry or keeps outliers that squash
        the normalized target. ``auto`` uses a high quantile of the valid depth of
        this very frame, which is unit free.
        """
        if self.max_depth_mode == "fixed":
            return self.max_depth
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size == 0:
            return 0.0
        cap = float(np.percentile(valid, self.max_depth_quantile)) * self.max_depth_scale
        if self.max_depth > 0:
            cap = min(cap, self.max_depth)
        return cap

    def _load_sample(self, dataset_name: str, pair) -> Optional[Dict]:
        stem, rgb_path, depth_path, camera_path = pair

        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))

        scale = self.depth_scales.get(dataset_name, self.default_depth_scale)
        depth = _load_depth(depth_path, scale=scale)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth resolution mismatch: RGB={rgb.shape[:2]} depth={depth.shape}"
            )
        finite = np.isfinite(depth) & (depth > 0)
        if int(finite.sum()) < self.min_valid_pixels:
            raise ValueError(
                f"Unusable depth: only {int(finite.sum())} valid pixels in {depth_path}"
            )

        intrinsics = self._intrinsics_for(camera_path)
        if intrinsics is None:
            raise ValueError(f"Missing camera intrinsics for {rgb_path}")

        height, width = rgb.shape[:2]
        work_h, work_w = _work_resolution_size(height, width, self.work_resolution)
        if (work_h, work_w) != (height, width):
            rgb = cv2.resize(rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
            depth = _resize_depth(depth, work_w, work_h)
            intrinsics = _scale_intrinsics(
                intrinsics, work_w / float(width), work_h / float(height)
            )

        normal = self._normal_prior(rgb, depth, intrinsics)

        top, left, crop_h, crop_w = _sample_crop_box(
            rgb=rgb,
            depth=depth,
            target_height=self.height,
            target_width=self.width,
            crop_scale_min=self.crop_scale_min,
            crop_scale_max=self.crop_scale_max,
        )
        rgb = rgb[top : top + crop_h, left : left + crop_w]
        depth = depth[top : top + crop_h, left : left + crop_w]
        normal = normal[top : top + crop_h, left : left + crop_w]

        intrinsics = np.array(intrinsics, dtype=np.float32, copy=True)
        intrinsics[0, 2] -= float(left)
        intrinsics[1, 2] -= float(top)
        intrinsics = _scale_intrinsics(
            intrinsics, self.width / float(crop_w), self.height / float(crop_h)
        )

        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        depth = _resize_depth(depth, self.width, self.height)
        normal = cv2.resize(
            normal, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

        if self.random_flip and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])
            normal = np.ascontiguousarray(normal[:, ::-1])
            # A mirrored image is described by a mirrored camera: the principal point
            # flips and the camera x axis reverses, so the prior flips in x too.
            intrinsics[0, 2] = (self.width - 1.0) - intrinsics[0, 2]
            normal = normal.copy()
            normal[..., 0] *= -1.0

        gt_valid = np.isfinite(depth) & (depth > 0)
        if int(gt_valid.sum()) < 16:
            raise ValueError("Too few valid depth pixels after crop/resize")

        # Simulated incomplete depth condition: start from the complete GT depth,
        # then remove random straight/slanted border polygons. The full GT remains
        # the diffusion target.
        input_depth = _apply_border_occlusion(
            depth,
            min_ratio=self.occlusion_min_ratio,
            max_ratio=self.occlusion_max_ratio,
            min_sides=self.occlusion_min_sides,
            max_sides=self.occlusion_max_sides,
            probability=self.occlusion_probability,
            slant_probability=self.occlusion_slant_probability,
            slant_max_delta=self.occlusion_slant_max_delta,
            min_visible_ratio=self.occlusion_min_visible_ratio,
        )
        observed = np.isfinite(input_depth) & (input_depth > 0)
        if int(observed.sum()) < 3:
            raise ValueError("Border occlusion removed every valid depth pixel")

        # Match Murre inference: the normalization range is determined only by the
        # depth that remains visible after masking.
        clip_max = self._clip_max_for(input_depth)
        input_norm, d_min, d_max = normalize_depth(input_depth.copy(), pre_clip_max=clip_max)
        interp_norm, dist = interp_depth(input_norm)

        depth_range = max(float(d_max - d_min), 1e-6)
        gt_01 = np.clip((depth - float(d_min)) / depth_range, 0.0, 1.0)

        normal_mag = np.linalg.norm(normal, axis=-1)
        normal_valid = normal_mag > 1e-6

        rgb_norm = rgb.astype(np.float32) / 127.5 - 1.0
        gt_norm = gt_01.astype(np.float32) * 2.0 - 1.0
        interp_norm = interp_norm.astype(np.float32) * 2.0 - 1.0

        sample_name = f"{dataset_name}/{stem}"
        return {
            "stem": sample_name,
            "rgb_norm": torch.from_numpy(rgb_norm).permute(2, 0, 1).contiguous(),
            "depth_metric": torch.from_numpy(depth.copy()).unsqueeze(0),
            "input_depth_metric": torch.from_numpy(input_depth.copy()).unsqueeze(0),
            "gt_depth_norm": torch.from_numpy(gt_norm).unsqueeze(0).contiguous(),
            "gt_valid": torch.from_numpy(gt_valid).unsqueeze(0).bool(),
            "interp_depth_norm": torch.from_numpy(interp_norm).unsqueeze(0).contiguous(),
            "distance": torch.from_numpy(dist.astype(np.float32)).unsqueeze(0).contiguous(),
            "normal": torch.from_numpy(
                np.ascontiguousarray(normal, dtype=np.float32)
            ).permute(2, 0, 1),
            "normal_valid": torch.from_numpy(normal_valid).unsqueeze(0).bool(),
            "intrinsics": torch.from_numpy(np.ascontiguousarray(intrinsics, dtype=np.float32)),
            # The normal loss uses this only as an observed/missing indicator:
            # masked borders are fully supervised; observed interior keeps the
            # lowest-loss fraction by default.
            "sparse_observed": torch.from_numpy(observed.astype(np.float32))
            .unsqueeze(0)
            .contiguous(),
            "d_min": torch.tensor(float(d_min), dtype=torch.float32),
            "d_max": torch.tensor(float(d_max), dtype=torch.float32),
            "depth_scale": torch.tensor(float(scale), dtype=torch.float32),
        }

    def _normal_prior(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        """Camera-space normal prior for one working-resolution frame."""
        if self.normal_source == "deferred_metric3d":
            return np.zeros((*depth.shape, 3), dtype=np.float32)
        if self.normal_source == "metric3d":
            estimator = _get_normal_estimator(self.metric3d_params)
            prediction = estimator.predict(rgb)
            normal = np.asarray(prediction["normal"], dtype=np.float32)
            if normal.shape[:2] != depth.shape[:2]:
                raise ValueError(
                    f"Metric3D normal grid {normal.shape[:2]} does not match depth {depth.shape[:2]}"
                )
            return normal

        # Ablation only: the same camera-space prior, but from leaked GT depth.
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))
        intrinsics_tensor = torch.from_numpy(np.ascontiguousarray(intrinsics, dtype=np.float32))
        with torch.no_grad():
            normal = depth_to_camera_normal(
                depth_tensor.unsqueeze(0).unsqueeze(0), intrinsics_tensor.unsqueeze(0)
            )
        return normal[0].permute(1, 2, 0).contiguous().numpy()

    def __getitem__(self, _index):
        if isinstance(_index, tuple):
            self.height, self.width = _index[-2:]
        last_error = None
        for attempt in range(self.max_retries):
            dataset, _scene, pair = self._sample_pair()
            try:
                sample = self._load_sample(dataset["name"], pair)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile, cv2.error) as exc:  # unreadable / degenerate frame
                last_error = exc
                self.stats["retries"] += 1
                if attempt == 0:
                    logging.warning(
                        "Skipping unusable sample %s/%s: %s",
                        dataset["name"],
                        pair[0],
                        exc,
                    )
                continue
            self.stats["ok"] += 1
            return sample

        self.stats["failures"] += 1
        raise RuntimeError(
            f"Could not load a usable training sample after {self.max_retries} attempts; "
            f"last error: {last_error}"
        )


class ResolutionBatchSampler(Sampler):
    """A resolution per batch, including with multiple prefetched workers."""
    def __init__(self, length, batch_size, resolutions, seed=42):
        self.length, self.batch_size = length, batch_size
        self.resolutions, self.seed, self.epoch = resolutions, seed, 0
        if batch_size < 1 or not resolutions or any(min(h,w)<32 or h%8 or w%8 for h,w in resolutions):
            raise ValueError('Need positive batch size and HxW resolutions >=32 divisible by 8')
    def __len__(self):
        return max(1, math.ceil(self.length/self.batch_size))
    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        self.epoch += 1
        for batch in range(len(self)):
            h,w = rng.choice(self.resolutions)
            yield [(batch*self.batch_size+i,h,w) for i in range(self.batch_size)]
