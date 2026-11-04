"""CPU regression tests; no pretrained weights or external datasets required.

Run: python -m unittest discover -s tests -v
"""
import contextlib
import copy
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch.nn import functional as F

from clip.model import CLIP
from models.models import RAILCLIP
from scenario_datasets.utils import Datum
from trainer import Trainer
from utils.config import _C


def tiny_clip():
    return CLIP(embed_dim=8, image_resolution=8, vision_layers=1,
                vision_width=64, vision_patch_size=4, context_length=77,
                vocab_size=49408, transformer_width=64,
                transformer_heads=1, transformer_layers=1)


def dataset_type(names):
    class FixtureDataset:
        def __init__(self, root, num_shots):
            self.classnames = names
            self.train_x = [Datum(Image.new('RGB', (10, 10), (40 + 50 * label, 30, 60)),
                                  label=label, classname=name)
                            for label, name in enumerate(names) for _ in range(2)]
            self.test = self.train_x
    return FixtureDataset


class RAILTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_recursive_fit_matches_joint_ridge(self):
        for expansion in (False, True):
            with self.subTest(expansion=expansion):
                cfg = SimpleNamespace(feature_expansion=expansion, hidden_dim=12,
                                      regularization=0.5, current_class_num=2, trained_class_num=0)
                model = RAILCLIP(cfg, tiny_clip())
                x = torch.randn(10, 3, 8, 8)
                y = torch.tensor([0, 1, 0, 1, 0])
                model.analytic_adaption(0, cfg, [(x[:5], y)])
                cfg.current_class_num = 3
                cfg.trained_class_num = 2
                local = torch.zeros(5, dtype=torch.long)
                model.analytic_adaption(1, cfg, [(x[5:8], local[:3]), (x[8:], local[3:])])
                features = model._analytic_features(x)
                labels = F.one_hot(torch.cat((y, local + 2)), 3).float()
                system = features.t() @ features + 0.5 * torch.eye(features.shape[1])
                expected = torch.linalg.solve(system, features.t() @ labels)
                torch.testing.assert_close(model.analytic_adaptor.weight.t(), expected, atol=2e-5, rtol=2e-4)
                self.assertEqual(local.tolist(), [0] * 5)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                model.text_features = F.normalize(torch.randn(3, 8), dim=-1)
                torch.testing.assert_close(model.zero_shot(x),
                                           100.0 * model.encode_images(x) @ model.text_features.t())
                model.train()
                self.assertFalse(model.clip_model.training)
                cfg.trained_class_num = 3
                cfg.current_class_num = 4
                with self.assertRaisesRegex(ValueError, 'training samples'):
                    model.analytic_adaption(1, cfg, [])

    def test_reference_fusion_seen_and_unseen(self):
        # Include negative analytic scores: unseen padding must not be confused
        # with selecting a class, and unselected rows must stay exactly CLIP.
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(fusion_weight=0.8)
        logits = torch.tensor([[5., 1., 0.], [0., 1., 5.]])
        trainer.model = unittest.mock.Mock()
        trainer.model.zero_shot.return_value = logits
        trainer.model.return_value = torch.tensor([[-0.2, 0.4]])
        trainer._learned_class_indices = lambda: [0, 1]
        x = torch.zeros(2, 3, 8, 8)
        result = trainer._predict(x)
        expected = logits.softmax(-1)
        expected[0] = 0.2 * expected[0] + 0.8 * torch.tensor([-0.2, 0.4, 0.])
        torch.testing.assert_close(result, expected)
        self.assertEqual(trainer.model.call_args.args[0].shape[0], 1)
        trainer.model.zero_shot.return_value = logits[1:]
        trainer.model.reset_mock()
        torch.testing.assert_close(trainer._predict(x[1:]), logits[1:].softmax(-1))
        trainer.model.assert_not_called()
        # A standalone task may occupy non-prefix global class columns.
        trainer._learned_class_indices = lambda: [1, 2]
        trainer.model.zero_shot.return_value = logits
        trainer.model.return_value = torch.tensor([[0.3, -0.1], [0.1, 0.6]])
        expected = 0.2 * logits.softmax(-1) + 0.8 * torch.tensor([[0., 0.3, -0.1], [0., 0.1, 0.6]])
        torch.testing.assert_close(trainer._predict(x, use_selector=False), expected)

    def test_trainer_original_loaders_resume_and_config(self):
        base = tiny_clip()
        registry = {'first': dataset_type(['cat', 'dog']), 'second': dataset_type(['bird'])}
        with tempfile.TemporaryDirectory() as directory, \
             patch('scenario_datasets.build_functions.dataset_list', registry), \
             patch('trainer.load_clip_to_cpu', side_effect=lambda *args: copy.deepcopy(base)), \
             contextlib.redirect_stdout(io.StringIO()):
            cfg = _C.clone()
            cfg.merge_from_file('configs/model/clip_vit_b16.yaml')
            # Old shell-script epoch arguments remain accepted but are unused.
            cfg.merge_from_list(['num_epochs', '40', 'hidden_dim', '12'])
            cfg.dataset_sequence = ['first', 'second']
            cfg.dataset = 'first'
            cfg.resolution = 8
            cfg.num_workers = 0
            cfg.batch_size = 2
            cfg.output_dir = directory
            cfg.model_dir = directory
            cfg.continue_train_first = True
            first = Trainer(cfg)
            self.assertEqual(first.indices, [0, 2])
            first.train()
            images, _ = next(iter(first.TAIL_test_loader))
            expected = first.model(images)
            cfg.dataset = 'second'
            cfg.continue_train_first = False
            second = Trainer(cfg)
            torch.testing.assert_close(second.model(images), expected)
            torch.testing.assert_close(second.model.R, first.model.R)
            second.train()
            self.assertEqual(second.seen_tasks, ['first', 'second'])
            self.assertEqual(second.model.analytic_adaptor.out_features, 3)
            self.assertEqual(second._learned_class_indices(), [0, 1, 2])
            # Restoration is independent of the random expansion reinitialization.
            restored = Trainer.__new__(Trainer)
            restored.cfg, restored.device = cfg, torch.device('cpu')
            restored.indices, restored.merged_classnames = second.indices, second.merged_classnames
            restored.model = RAILCLIP(cfg, copy.deepcopy(base))
            restored.load_model4CL(directory)
            torch.testing.assert_close(restored.model(images), second.model(images))
            cfg.regularization = 2.0
            with self.assertRaisesRegex(ValueError, 'regularization'):
                restored.load_model4CL(directory)
            cfg.regularization = 0.1
            # A completed checkpoint cannot silently train the last task twice.
            with self.assertRaisesRegex(ValueError, 'Continual tasks'):
                Trainer(cfg)

    def test_zero_shot_original_entry(self):
        registry = {'first': dataset_type(['cat', 'dog'])}
        with patch('scenario_datasets.build_functions.dataset_list', registry), \
             patch('trainer.load_clip_to_cpu', return_value=tiny_clip()), \
             contextlib.redirect_stdout(io.StringIO()):
            cfg = _C.clone()
            cfg.merge_from_file('configs/model/zs_clip_vit_b16.yaml')
            cfg.dataset_sequence = ['first']
            cfg.resolution = 8
            cfg.num_workers = 0
            trainer = Trainer(cfg)
            trainer.test_zero_shot()
            self.assertEqual(trainer.evaluator._total, 4)


if __name__ == '__main__':
    unittest.main()
