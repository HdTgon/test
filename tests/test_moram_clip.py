"""CPU tests with random tiny CLIP; no weights or dataset downloads."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load_file('lada_clip_model', ROOT / 'clip/model.py')
moram = load_file('moram_clip', ROOT / 'models/moram_clip.py')

class CLIPCompatibility(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.base = base.CLIP(16, 8, 1, 64, 4, 5, 32, 64, 1, 1).float().eval()
        self.model = moram.moram_CLIPModel(self.base).eval()
        self.image = torch.randn(2, 3, 8, 8)
        self.text = torch.tensor([[1, 2, 31, 0, 0], [1, 3, 4, 31, 0]])

    def test_pretrained_paths_and_causal_mask_are_preserved(self):
        image, _ = self.model.encode_image(self.image)
        text, _ = self.model.encode_text(self.text)
        torch.testing.assert_close(image, self.base.encode_image(self.image))
        torch.testing.assert_close(text, self.base.encode_text(self.text))

    def test_optional_modalities_and_feature_contract(self):
        for image, text in ((self.image, None), (None, self.text), (self.image, self.text)):
            out = self.model(image, text)
            self.assertEqual(len(out), 3)
            if image is not None:
                self.assertEqual(out[0].shape, (2, 16))
            if text is not None:
                self.assertIsInstance(out[1], torch.Tensor)
                torch.testing.assert_close(out[1].norm(dim=-1), torch.ones(2))
        self.model.output_dict = True
        self.assertEqual(self.model(self.image, self.text)['text_features'].shape, (2, 16))

    def test_backward(self):
        image, _ = self.model.encode_image(self.image)
        text, _ = self.model.encode_text(self.text)
        (image.square().mean() + text.square().mean()).backward()
        self.assertIsNotNone(self.model.visual.conv1.weight.grad)
        self.assertIsNotNone(self.model.token_embedding.weight.grad)

if __name__ == '__main__':
    unittest.main()
