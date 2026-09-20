import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murre.training_dataset import MurreNormalTrainingDataset, ResolutionBatchSampler
from murre.util.normal_util import depth_to_camera_normal


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        Image.fromarray(np.full((96, 128, 3), 128, np.uint8)).save(root / 'rgb.png')
        np.save(root / 'depth.npy', np.full((96, 128), 10, np.float32))
        (root / 'cam.txt').write_text('intrinsic\n100 0 64\n0 100 48\n0 0 1\n')
        index = root / 'index.txt'
        with index.open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['dataset', 'scene', 'rgb', 'depth', 'camera'])
            writer.writerow(['test', 'scene', root/'rgb.png', root/'depth.npy', root/'cam.txt'])
        self.dataset = MurreNormalTrainingDataset(index_path=index, height=64, width=96,
            normal_source='deferred_metric3d', max_depth=0, max_retries=4)

    def test_bad_depth_resamples(self):
        good = self.dataset._sample_pair()
        dataset, scene, pair = good
        bad = (dataset, scene, (pair[0], pair[1], '/nonexistent/depth.npy', pair[3]))
        with patch.object(self.dataset, '_sample_pair', side_effect=[bad, good]):
            sample = self.dataset[0]
        self.assertEqual(self.dataset.stats['retries'], 1)
        self.assertEqual(tuple(sample['rgb_norm'].shape), (3, 64, 96))
        self.assertTrue(torch.isfinite(sample['gt_depth_norm']).all())

    def test_exhausted_bad_data_is_bounded(self):
        dataset, scene, pair = self.dataset._sample_pair()
        bad = (dataset, scene, (pair[0], pair[1], '/nonexistent/depth.npy', pair[3]))
        with patch.object(self.dataset, '_sample_pair', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'after 4 attempts'):
                self.dataset[0]

    def test_multiworker_batch_resolution(self):
        sizes = [(64, 96), (96, 128)]
        sampler = ResolutionBatchSampler(8, 2, sizes)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=2)
        seen = set()
        for batch in loader:
            shape = tuple(batch['rgb_norm'].shape[-2:])
            seen.add(shape)
            self.assertIn(shape, sizes)
            self.assertEqual(tuple(batch['gt_depth_norm'].shape[-2:]), shape)
        self.assertEqual(seen, set(sizes))

    def test_camera_normal_outside_crop_has_grad(self):
        depth = torch.full((1, 1, 16, 24), 10., requires_grad=True)
        k = torch.tensor([[[100.,0.,-30.],[0.,100.,40.],[0.,0.,1.]]])
        normal = depth_to_camera_normal(depth, k)
        self.assertTrue(torch.allclose(normal[:,2], torch.full_like(normal[:,2], -1.)))
        normal[:,0].sum().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(depth.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
