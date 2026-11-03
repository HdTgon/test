import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np

from lada_clip import clip
from lada_clip.model import CLIP
from models.moram import get_moram, moram_CLIPModel, apply_lora

# Independent reference from original LADA at test root.
LADA_ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('reference_model', LADA_ROOT / 'clip/model.py')
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)

class LADAParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.original = CLIP(16, 8, 1, 64, 4, 77, 49408, 64, 1, 1).float().eval()
        self.tmp = tempfile.TemporaryDirectory()
        self.checkpoint = str(Path(self.tmp.name) / 'tiny.pt')
        torch.save(self.original.state_dict(), self.checkpoint)
        # Same build/precision path as trainer.load_clip_to_cpu.
        self.reference = reference_module.build_model(copy.deepcopy(self.original.state_dict())).float().eval()
        self.cfg = dict(backbone_type='ViT-B-16', pretrained_weight='openai',
                        checkpoint_path=self.checkpoint, zero_shot=True, device='cpu', prec='fp32',
                        target_encoder='all', target_modules=['q_proj','k_proj','v_proj','out_proj','ffn_in','ffn_out'],
                        rank=4, topk=2)
        self.image = torch.randn(3, 3, 8, 8)
        self.text = clip.tokenize(['a photo of a cat.', 'a photo of a dog.', 'a photo of a plane.'])

    def tearDown(self):
        self.tmp.cleanup()

    def assert_features(self, model, wrapped=False):
        image = model.encode_image(self.image)
        text = model.encode_text(self.text)
        if wrapped:
            image, text = image[0], text[0]
        torch.testing.assert_close(image, self.reference.encode_image(self.image), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(text, self.reference.encode_text(self.text), atol=2e-6, rtol=2e-5)
        logits = model.logit_scale.exp() * F.normalize(image,dim=-1) @ F.normalize(text,dim=-1).T
        expected, _ = self.reference(self.image,self.text)
        torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(logits.argmax(-1), expected.argmax(-1))

    def test_zero_shot_loads_exact_same_weights_and_features(self):
        model, train, test, tokenizer = get_moram(self.cfg)
        for key, value in self.reference.state_dict().items():
            torch.testing.assert_close(model.clip.state_dict()[key], value, rtol=0, atol=0)
        self.assert_features(model)
        self.assertIs(train,test)
        self.assertIs(tokenizer,clip.tokenize)
        out = model(self.image,None)
        self.assertEqual(out[0].shape,(3,16))
        self.assertIsNone(out[1])

    def test_real_moram_attention_and_ffn_zero_delta(self):
        self.cfg['zero_shot'] = False
        model, _, _, _ = get_moram(self.cfg)
        self.assert_features(model,wrapped=True)
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all('lora_' in n for n in trainable))
        model.train()
        image, _ = model.encode_image(self.image)
        text, _ = model.encode_text(self.text)
        (image.square().mean()+text.square().mean()).backward()
        grads = [p.grad for n,p in model.named_parameters() if 'lora_B' in n]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum()>0 for g in grads))

    def test_eot_causal_mask_and_eval_repeatability(self):
        model,_,_,_ = get_moram(self.cfg)
        text=self.text.clone()
        eot=text.argmax(-1)
        for row,pos in enumerate(eot):
            text[row,pos+1:]=12
        torch.testing.assert_close(model.encode_text(text),model.encode_text(self.text))
        torch.testing.assert_close(model.encode_image(self.image),model.encode_image(self.image),rtol=0,atol=0)

    def test_tokenizer_and_preprocessing_files_match_lada(self):
        import hashlib
        for name in ('model.py','clip.py','simple_tokenizer.py','bpe_simple_vocab_16e6.txt.gz'):
            self.assertEqual(hashlib.sha256((LADA_ROOT/'clip'/name).read_bytes()).digest(),
                             hashlib.sha256((Path(clip.__file__).parent/name).read_bytes()).digest())
        model,_,transform,_=get_moram(self.cfg)
        image=Image.fromarray(np.zeros((12,16,3),dtype=np.uint8))
        self.assertEqual(tuple(transform(image).shape),(3,8,8))

if __name__=='__main__':
    unittest.main()
