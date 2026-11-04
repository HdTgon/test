"""DualRAIL reference equations, task replay, checkpoint and CLI routing."""
import contextlib
import copy
import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import main as entry
from dualrail_trainer import Trainer
from models.models import DualRAILCLIP
from utils.config import _C
from utils.rail import gaussian_kernel
from test_rail import tiny_clip, dataset_type


class DualRAILTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(23)

    def test_dual_matches_reference_after_two_tasks(self):
        cfg = SimpleNamespace(gamma=0.1, dual_regularization=0.001,
                              current_class_num=2, trained_class_num=0)
        model = DualRAILCLIP(cfg, tiny_clip())
        x = torch.randn(9, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0, 0])
        model.analytic_adaption(0, cfg, [(x[:5], labels[:5])])
        cfg.current_class_num = 3
        cfg.trained_class_num = 2
        model.analytic_adaption(1, cfg, [(x[5:7], labels[5:7]), (x[7:], labels[7:])])
        # Batch-wise CLIP GEMMs can differ from an all-at-once encoding by
        # float32 rounding. The reference fits the features actually cached.
        features = model.feature_memory
        query_features = model.encode_images(x).float()
        torch.testing.assert_close(features, query_features)
        global_labels = labels.clone()
        global_labels[5:] += 2
        y = F.one_hot(global_labels, 3).float().numpy()
        # Independent translation of the reference utils.py equations.
        kernel = torch.exp(-cfg.gamma * (features[:, None] - features[None, :]).square().sum(-1)).numpy()
        alpha = np.linalg.solve(kernel + cfg.dual_regularization * np.eye(9), y)
        np.testing.assert_allclose(model.alpha.numpy(), alpha, atol=1e-8, rtol=1e-8)
        query_kernel = torch.exp(-cfg.gamma * (query_features[:, None] - features[None, :]).square().sum(-1)).numpy()
        np.testing.assert_allclose(model(x).numpy(), query_kernel @ alpha, atol=2e-6, rtol=2e-6)
        self.assertEqual(model.feature_memory.shape, (9, 8))
        self.assertEqual(model.num_classes, 3)
        self.assertEqual(labels.tolist(), [0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        model.train()
        self.assertFalse(model.clip_model.training)
        model.text_features = F.normalize(torch.randn(3, 8), dim=-1)
        torch.testing.assert_close(model.zero_shot(x), 100.0 * query_features @ model.text_features.t())

    def test_kernel_chunking(self):
        features = F.normalize(torch.randn(67, 8), dim=-1)
        queries = F.normalize(torch.randn(4, 8), dim=-1)
        expected = torch.exp(-0.2 * (queries[:, None] - features[None, :]).square().sum(-1))
        torch.testing.assert_close(gaussian_kernel(queries, features, 0.2), expected)

    def test_original_loaders_and_checkpoint_resume(self):
        base = tiny_clip()
        registry = {'first': dataset_type(['cat', 'dog']), 'second': dataset_type(['bird'])}
        with tempfile.TemporaryDirectory() as directory, \
             patch('scenario_datasets.build_functions.dataset_list', registry), \
             patch('dualrail_trainer.load_clip_to_cpu', side_effect=lambda *args: copy.deepcopy(base)), \
             contextlib.redirect_stdout(io.StringIO()):
            cfg = _C.clone()
            cfg.merge_from_file('configs/model/dualrail_vit_b16.yaml')
            cfg.dataset_sequence = ['first', 'second']
            cfg.dataset = 'first'
            cfg.resolution = 8
            cfg.num_workers = 0
            cfg.batch_size = 2
            cfg.output_dir = cfg.model_dir = directory
            cfg.continue_train_first = True
            first = Trainer(cfg)
            first.train()
            images, _ = next(iter(first.TAIL_test_loader))
            expected = first.model(images)
            cfg.dataset = 'second'
            cfg.continue_train_first = False
            second = Trainer(cfg)
            torch.testing.assert_close(second.model(images), expected)
            second.train()
            self.assertEqual(second.seen_tasks, ['first', 'second'])
            self.assertEqual(second._learned_class_indices(), [0, 1, 2])
            self.assertEqual(second.model.feature_memory.shape[0], 6)
            checkpoint = torch.load(os.path.join(directory, 'checkpoint.pth.tar'), weights_only=True)
            self.assertIsInstance(checkpoint['backbone'], str)
            self.assertNotIn('R', checkpoint)
            restored = Trainer.__new__(Trainer)
            restored.cfg, restored.device = cfg, torch.device('cpu')
            restored.indices, restored.merged_classnames = second.indices, second.merged_classnames
            restored.model = DualRAILCLIP(cfg, copy.deepcopy(base))
            restored.load_model4CL(directory)
            torch.testing.assert_close(restored.model(images), second.model(images))
            torch.testing.assert_close(restored.model.target_memory, second.model.target_memory)
            cfg.gamma = 0.2
            restored.model.gamma = cfg.gamma
            with self.assertRaisesRegex(ValueError, 'gamma'):
                restored.load_model4CL(directory)
            torch.save({'format': 'primal_RAIL_v1'}, os.path.join(directory, 'checkpoint.pth.tar'))
            with self.assertRaisesRegex(ValueError, 'Expected a DualRAIL'):
                restored.load_model4CL(directory)

    def test_dual_explicit_fusion_loops(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(fusion_weight=0.8)
        trainer.device = torch.device('cpu')
        trainer.indices = [0, 2]
        trainer.evaluator = unittest.mock.Mock()
        trainer.model = unittest.mock.Mock()
        logits = torch.tensor([[5., 1., 0.], [0., 1., 5.]])
        trainer.model.zero_shot.return_value = logits
        trainer.model.return_value = torch.tensor([[-0.2, 0.4]])
        trainer._learned_class_indices = lambda: [0, 1]
        trainer.TAIL_test_loader = [(torch.zeros(2, 3, 8, 8), torch.tensor([0, 2]))]
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.test_w_selector()
        expected = logits.softmax(-1)
        expected[0] = 0.2 * expected[0] + 0.8 * torch.tensor([-0.2, 0.4, 0.])
        torch.testing.assert_close(trainer.evaluator.process.call_args.args[0], expected)
        trainer.model.return_value = torch.tensor([[-0.2, 0.4], [0.3, -0.1]])
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.test_wo_selector()
        expected = 0.2 * logits.softmax(-1) + 0.8 * torch.tensor([[-0.2, 0.4, 0.], [0.3, -0.1, 0.]])
        torch.testing.assert_close(trainer.evaluator.process.call_args.args[0], expected)

    def test_main_algorithm_and_zero_shot_routing(self):
        for algorithm, zero_shot, selected in [('RAIL', False, 'rail'), ('DualRAIL', False, 'dual'),
                                                ('dualrail', True, 'rail')]:
            with self.subTest(algorithm=algorithm, zero_shot=zero_shot), \
                 patch.object(entry, 'cfg', _C.clone()), patch.object(entry, 'setup_logger'), \
                 patch('rail_trainer.Trainer') as rail, patch('dualrail_trainer.Trainer') as dual, \
                 contextlib.redirect_stdout(io.StringIO()):
                args = SimpleNamespace(data='TAIL', model='clip_vit_b16',
                                       opts=['algorithm', algorithm, 'zero_shot', str(zero_shot)])
                entry.main(args)
                chosen, other = (rail, dual) if selected == 'rail' else (dual, rail)
                chosen.assert_called_once()
                other.assert_not_called()
                if zero_shot:
                    chosen.return_value.test_zero_shot.assert_called_once()
                    chosen.return_value.train.assert_not_called()
                else:
                    chosen.return_value.train.assert_called_once()


if __name__ == '__main__':
    unittest.main()
