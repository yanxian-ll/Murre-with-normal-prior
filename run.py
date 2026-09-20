import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from murre.pipeline import MurrePipeline

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
SDPT_EXTENSION_LIST = [".npz"]
NORMAL_EXTENSION_LIST = [".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def filter_common_files(*file_lists):
    """Keep files whose filename stem exists in every input list."""
    if len(file_lists) == 0:
        return []
    common_stems = set(stem(p) for p in file_lists[0])
    for file_list in file_lists[1:]:
        common_stems &= set(stem(p) for p in file_list)
    return [
        [p for p in file_list if stem(p) in common_stems]
        for file_list in file_lists
    ]


def load_normal(path):
    """Load a normal map as either HxWx3 or 3xHxW float32.

    Supported encodings are .npy/.npz arrays and ordinary three-channel images.
    The pipeline handles conversion from [0,255] or [0,1] to [-1,1].
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        normal = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "normal" in data:
            normal = data["normal"]
        elif "arr_0" in data:
            normal = data["arr_0"]
        else:
            normal = data[data.files[0]]
    else:
        normal = np.asarray(Image.open(path).convert("RGB"))

    normal = np.asarray(normal)
    if normal.ndim != 3:
        raise ValueError(f"Normal map must be 3D, got {normal.shape} from {path}")
    if normal.shape[-1] != 3 and normal.shape[0] != 3:
        raise ValueError(
            f"Normal map must be HxWx3 or 3xHxW, got {normal.shape} from {path}"
        )
    return normal.astype(np.float32)


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run SfM- and normal-guided depth estimation using Murre."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--input_sdpt_dir",
        type=str,
        required=True,
        help="Path to the sparse depth map folder.",
    )
    parser.add_argument(
        "--input_normal_dir",
        type=str,
        required=True,
        help="Path to the input normal-prior folder. Files are matched by filename stem.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps. Original DDIM: 10-50; LCM: 1-4.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to ensemble.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half precision.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum processing resolution. 0 uses input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="RGB resampling method.",
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. None means unseeded inference.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. 0 selects automatically.",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Run on Apple Silicon when available.",
    )
    parser.add_argument(
        "--scale_invariant",
        action="store_true",
        help="Whether the diffusion model outputs scale-invariant depth.",
    )
    parser.add_argument(
        "--shift_invariant",
        action="store_true",
        help="Whether the diffusion model outputs shift-invariant depth.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Maximum depth value.",
    )
    parser.add_argument(
        "--err_thr",
        type=float,
        default=None,
        help="Filter SfM depth values whose reprojection error exceeds this threshold.",
    )
    parser.add_argument(
        "--nviews_thr",
        type=int,
        default=None,
        help="Filter SfM depth values with visible-view count <= this threshold.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    input_sdpt_dir = args.input_sdpt_dir
    input_normal_dir = args.input_normal_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    max_depth = args.max_depth
    err_thr = args.err_thr
    nviews_thr = args.nviews_thr

    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    resample_method = args.resample_method
    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and batch_size == 0:
        batch_size = 1

    scale_invariant = args.scale_invariant
    shift_invariant = args.shift_invariant

    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    rgb_filename_list = sorted(
        f
        for f in glob(os.path.join(input_rgb_dir, "*"))
        if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    )
    sdpt_filename_list = sorted(
        f
        for f in glob(os.path.join(input_sdpt_dir, "*"))
        if os.path.splitext(f)[1].lower() in SDPT_EXTENSION_LIST
    )
    normal_filename_list = sorted(
        f
        for f in glob(os.path.join(input_normal_dir, "*"))
        if os.path.splitext(f)[1].lower() in NORMAL_EXTENSION_LIST
    )

    rgb_filename_list, sdpt_filename_list, normal_filename_list = filter_common_files(
        rgb_filename_list,
        sdpt_filename_list,
        normal_filename_list,
    )
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Matched {n_images} RGB / sparse-depth / normal triplets.")
    else:
        logging.error(
            "No common filename stems found across RGB, sparse depth, and normal directories."
        )
        raise SystemExit(1)

    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
        variant = None

    pipe: MurrePipeline = MurrePipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass

    pipe = pipe.to(device)
    pipe.scale_invariant = scale_invariant
    pipe.shift_invariant = shift_invariant
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; color_map = {color_map}."
    )

    with torch.no_grad():
        for rgb_path, sdpt_path, normal_path in tqdm(
            zip(rgb_filename_list, sdpt_filename_list, normal_filename_list),
            total=n_images,
            desc="Estimating depth",
            leave=True,
        ):
            input_image = Image.open(rgb_path)

            sdpt_pack = np.load(sdpt_path, allow_pickle=True)["arr_0"].astype(np.float32)
            sdpt, err, nviews = sdpt_pack[..., 0], sdpt_pack[..., 1], sdpt_pack[..., 2]
            sdpt = np.nan_to_num(sdpt, nan=0.0, posinf=0.0, neginf=0.0)
            if err_thr is not None:
                sdpt[err > err_thr] = 0.0
            if nviews_thr is not None:
                sdpt[nviews <= nviews_thr] = 0.0
            input_sparse_depth = np.clip(sdpt, 0.0, max_depth)

            input_normal = load_normal(normal_path)

            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            pipe_out = pipe(
                input_image,
                input_sparse_depth,
                input_normal,
                max_depth=max_depth,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                batch_size=batch_size,
                model_dtype=dtype,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored

            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"

            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            if depth_colored is not None:
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                depth_colored.save(colored_save_path)
