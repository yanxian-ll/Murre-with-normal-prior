"""Deterministic, small inference previews without changing training RNG state."""
import copy
import json
import logging
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps

from .util.normal_util import depth_to_camera_normal


@contextmanager
def fixed_random(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def color_depth(depth, low, high, valid=None):
    x = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    rgb = (colormaps['Spectral_r'](np.nan_to_num(x))[..., :3] * 255).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = 0
    return rgb


def comparison(sample, prediction):
    rgb = ((sample['rgb_norm'].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
    gt = sample['depth_metric'].squeeze().numpy()
    prior = sample['input_depth_metric'].squeeze().numpy()
    valid = sample['gt_valid'].squeeze().numpy()
    low, high = np.percentile(gt[valid], [2, 98])
    normal = ((sample['normal'].permute(1,2,0).numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
    # Use camera-Z depth and the crop/resize-adjusted K, exactly as in training.
    depth_tensor = torch.from_numpy(np.asarray(prediction, dtype=np.float32).copy())
    depth_valid = torch.isfinite(depth_tensor) & (depth_tensor > 0)
    clean_depth = torch.where(depth_valid, depth_tensor, torch.zeros_like(depth_tensor))
    pred_normal = depth_to_camera_normal(clean_depth, sample['intrinsics'].cpu().float())[0]
    # Central differences need all four neighbors; exclude outer image borders.
    normal_valid = torch.zeros_like(depth_valid)
    normal_valid[1:-1,1:-1] = (depth_valid[1:-1,1:-1] & depth_valid[1:-1,:-2]
        & depth_valid[1:-1,2:] & depth_valid[:-2,1:-1] & depth_valid[2:,1:-1])
    normal_valid &= torch.linalg.vector_norm(pred_normal,dim=0) > 1e-6
    pred_normal_rgb = ((pred_normal.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    pred_normal_rgb[~normal_valid.numpy()] = 0
    panels = [rgb, color_depth(prior,low,high,prior>0), normal,
              color_depth(prediction,low,high,depth_valid.numpy()), pred_normal_rgb, color_depth(gt,low,high,valid)]
    labels = ['RGB', 'Input depth', 'Metric3D normal', 'Prediction', 'Pred depth normal', 'GT depth']
    h,w = rgb.shape[:2]
    canvas = Image.new('RGB',(w*len(panels),h+44),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(panel,label) in enumerate(zip(panels,labels)):
        canvas.paste(Image.fromarray(panel),(i*w,44))
        draw.text((i*w+5,5),label,fill='black')
    draw.text((5,23),f'Depth color range: {low:.3g} .. {high:.3g} (stored units); invalid=black',fill='black')
    return canvas


class ValidationPreview:
    def __init__(self, dataset, predictor, output_dir, count=3, size=(192,256), seed=2024, index=None):
        self.root = Path(output_dir)/'validation'
        self.root.mkdir(parents=True,exist_ok=True)
        self.seed = seed
        self.kind = 'validation' if index else 'train_preview'
        config = dict(count=count,size=list(size),seed=seed,index=str(Path(index).resolve()) if index else None)
        config_path = self.root/'samples_config.json'
        cache = self.root/'samples.pt'
        if cache.exists():
            if not config_path.exists() or json.loads(config_path.read_text()) != config:
                raise ValueError('Validation sample configuration changed; use a new output directory or remove validation/samples.pt')
            self.samples = torch.load(cache,map_location='cpu',weights_only=False)
            self._refresh_normals(predictor, cache)
            return
        data = copy.copy(dataset)
        data.height,data.width = size
        data.stats = dict(dataset.stats)
        data._intrinsics_cache = {}
        if index:
            data.datasets = data._datasets_from_index(index)
            if not data.datasets:
                raise ValueError('Validation index contains no samples')
            train_scenes = {(d['name'],s['name']) for d in dataset.datasets for s in d['scenes']}
            val_scenes = {(d['name'],s['name']) for d in data.datasets for s in d['scenes']}
            if train_scenes & val_scenes:
                raise ValueError('Validation scenes overlap the training index; split scenes before training')
        with fixed_random(seed):
            self.samples = []
            for i in range(count):
                sample = data[i]
                if predictor is not None:
                    rgb = ((sample['rgb_norm'].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                    sample['normal'] = torch.from_numpy(predictor.predict(rgb)['normal'].copy()).permute(2,0,1)
                self.samples.append(sample)
        torch.save(self.samples,cache)
        if predictor is not None:
            (self.root/'normal_metadata.json').write_text(json.dumps(predictor.cache_signature,indent=2))
        config_path.write_text(json.dumps(config,indent=2))
        (self.root/'samples.json').write_text(json.dumps([s['stem'] for s in self.samples],indent=2))

    def _refresh_normals(self, predictor, cache):
        if predictor is None:
            return
        metadata = self.root/'normal_metadata.json'
        if metadata.exists() and json.loads(metadata.read_text()) == predictor.cache_signature:
            return
        logging.info('Refreshing stale Metric3D normal cache (%d samples)',len(self.samples))
        with fixed_random(self.seed):
            for sample in self.samples:
                rgb = ((sample['rgb_norm'].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                sample['normal'] = torch.from_numpy(predictor.predict(rgb)['normal'].copy()).permute(2,0,1)
        temp = cache.with_suffix('.tmp')
        torch.save(self.samples,temp)
        temp.replace(cache)
        metadata.write_text(json.dumps(predictor.cache_signature,indent=2))

    @torch.no_grad()
    def run(self,pipe,writer,step,denoising_steps=4,precision="fp32"):
        target = self.root/f'step-{step:07d}'
        target.mkdir(exist_ok=True)
        was_training = pipe.unet.training
        pipe.unet.eval()
        metrics = []
        try:
            autocast = (torch.autocast("cuda",dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
                        if pipe.device.type == "cuda" and precision != "fp32" else nullcontext())
            with fixed_random(self.seed), autocast:
                for i,sample in enumerate(self.samples):
                    # Genuine inference: no noisy GT latent and no GT alignment.
                    pred01 = pipe.single_infer(
                        rgb_in=sample['rgb_norm'][None],
                        idpt_in=sample['interp_depth_norm'][None].repeat(1,3,1,1),
                        dist_in=sample['distance'][None],normal_in=sample['normal'][None],
                        num_inference_steps=denoising_steps,show_pbar=False,
                        generator=torch.Generator(device=pipe.device).manual_seed(self.seed+i),
                        model_dtype=pipe.dtype).float().squeeze().cpu().numpy()
                    prediction = pred01 * float(sample['d_max']-sample['d_min']) + float(sample['d_min'])
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError('Non-finite validation prediction')
                    canvas = comparison(sample,prediction)
                    canvas.save(target/f'sample-{i:02d}.png')
                    writer.add_image(f'{self.kind}/sample_{i:02d}',np.asarray(canvas),step,dataformats='HWC')
                    gt = sample['depth_metric'].squeeze().numpy()
                    valid = sample['gt_valid'].squeeze().numpy()
                    diff = prediction[valid]-gt[valid]
                    metrics.append(dict(stem=sample['stem'],abs_rel=float(np.mean(np.abs(diff)/gt[valid])),rmse=float(np.sqrt(np.mean(diff**2)))))
            (target/'metrics.json').write_text(json.dumps(metrics,indent=2))
            for key in ['abs_rel']:
                writer.add_scalar(f'{self.kind}/{key}',float(np.mean([m[key] for m in metrics])),step)
            writer.flush()
            logging.info('Saved %s: %s (%d samples)',self.kind,target,len(metrics))
        finally:
            pipe.unet.train(was_training)
