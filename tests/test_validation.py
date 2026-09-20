import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import test_training_data as fixtures
from murre.validation import ValidationPreview


class ValidationTests(unittest.TestCase):
    setUp = fixtures.TrainingDataTests.setUp
    def test_stale_normal_cache_refreshes_without_changing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = ValidationPreview(self.dataset,None,tmp,count=1,size=(64,96))
            rgb = old.samples[0]['rgb_norm'].clone()
            calls = []
            def predict(image):
                calls.append(1)
                normal = np.zeros_like(image,dtype=np.float32)
                normal[...,2] = -1
                return {'normal':normal}
            teacher = SimpleNamespace(cache_signature={'preprocessing':'test-v2'},predict=predict)
            fresh = ValidationPreview(self.dataset,teacher,tmp,count=1,size=(64,96))
            self.assertTrue(torch.equal(rgb,fresh.samples[0]['rgb_norm']))
            self.assertTrue((fresh.samples[0]['normal'][2] == -1).all())
            ValidationPreview(self.dataset,teacher,tmp,count=1,size=(64,96))
            self.assertEqual(len(calls),1)

    def test_metric3d_upsamples_small_rgb_to_official_canvas(self):
        from murre.util.metric3d_normal import Metric3DNormalEstimator
        teacher = Metric3DNormalEstimator.__new__(Metric3DNormalEstimator)
        teacher.device = torch.device('cpu')
        teacher.input_size = (616,1064)
        image = np.full((192,256,3),128,np.uint8)
        tensor, original, (top,left,h,w) = teacher._prepare_input(image)
        self.assertEqual(tuple(tensor.shape),(1,3,616,1064))
        self.assertEqual(original,(192,256))
        self.assertEqual(h,616)
        self.assertGreater(w,256)
        self.assertGreater(left,0)

    def test_inference_preview_restores_rng_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            py_state = random.getstate()
            th_state = torch.get_rng_state().clone()
            preview = ValidationPreview(self.dataset,None,tmp,count=2,size=(64,96))
            self.assertEqual(random.getstate(),py_state)
            self.assertTrue(torch.equal(th_state,torch.get_rng_state()))
            unet = torch.nn.Linear(1,1).train()
            def infer(**kwargs):
                self.assertFalse(unet.training)
                self.assertNotIn('gt_depth_norm',kwargs)
                return torch.full((1,1,64,96),.5)
            pipe = SimpleNamespace(unet=unet,device=torch.device('cpu'),dtype=torch.float32,single_infer=infer)
            before_run = torch.get_rng_state().clone()
            with SummaryWriter(tmp+'/tb') as writer:
                preview.run(pipe,writer,500)
            self.assertTrue(unet.training)
            self.assertTrue(torch.equal(before_run,torch.get_rng_state()))
            with Image.open(tmp+'/validation/step-0000500/sample-00.png') as image:
                self.assertEqual(image.size,(576,108))
                # Constant positive camera-Z plane faces the camera: n=(0,0,-1).
                self.assertEqual(image.getpixel((4*96+48,44+32)),(127,127,0))
                self.assertEqual(image.getpixel((4*96,44)),(0,0,0))
            events = EventAccumulator(tmp+'/tb').Reload()
            self.assertIn('train_preview/sample_00',events.Tags()['images'])
            metrics = json.loads(Path(tmp+'/validation/step-0000500/metrics.json').read_text())
            self.assertEqual(len(metrics),2)
            cached = ValidationPreview(self.dataset,None,tmp,count=2,size=(64,96))
            self.assertTrue(torch.equal(cached.samples[0]['rgb_norm'],preview.samples[0]['rgb_norm']))
