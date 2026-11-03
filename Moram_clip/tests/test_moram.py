import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from Moram_clip import apply_lora, get_moram, increment_task
from Moram_clip.clip.model import CLIP
from Moram_clip.injection import MoRAMLinearTensor, PlainMultiheadAttentionLoRA
from Moram_clip.moram_lora import MoRAM_Linear


class MoRAMCLIPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.base=CLIP(16,8,2,64,4,7,32,64,1,2).float().eval()
        self.image=torch.randn(3,3,8,8)
        self.text=torch.tensor([[1,2,31,0,0,0,0],[1,4,5,31,0,0,0],[1,31,0,0,0,0,0]])
        self.cfg=dict(rank=4,topk=2,temp=0.1,prune_thre=0,target_encoder='all',
                      target_modules=['q_proj','k_proj','v_proj','out_proj','ffn_in','ffn_out'])

    def test_original_lada_files_unchanged(self):
        root=Path(__file__).resolve().parents[2]
        for original in (root/'clip').iterdir():
            if original.is_file():
                self.assertEqual(original.read_bytes(),(root/'Moram_clip/clip'/original.name).read_bytes())

    def test_zero_delta_preserves_features_logits_and_causal_mask(self):
        model=apply_lora(copy.deepcopy(self.base),self.cfg).eval()
        self.assertIs(type(model),CLIP)
        for method,x in [('encode_image',self.image),('encode_text',self.text)]:
            torch.testing.assert_close(getattr(model,method)(x),getattr(self.base,method)(x),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(model(self.image,self.text)[0],self.base(self.image,self.text)[0],atol=2e-5,rtol=2e-5)
        changed=self.text.clone()
        for i,pos in enumerate(changed.argmax(-1)):
            changed[i,pos+1:]=10
        torch.testing.assert_close(model.encode_text(changed),model.encode_text(self.text))

    def test_lora_only_training_changes_output_and_backbone_stays_frozen(self):
        model=apply_lora(copy.deepcopy(self.base),self.cfg).train()
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(n.endswith(('.lora_A','.lora_B')) for n in trainable))
        backbone={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        before=model.encode_image(self.image).detach()
        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.05)
        model(self.image,self.text)[0].square().mean().backward()
        factors=[p for n,p in model.named_parameters() if n.endswith('.lora_B')]
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in factors))
        opt.step()
        self.assertFalse(torch.allclose(before,model.encode_image(self.image)))
        for n,p in model.named_parameters():
            if n in backbone: torch.testing.assert_close(p,backbone[n],rtol=0,atol=0)

    def test_target_selection_and_repeated_injection(self):
        for target in ['vision','text','all']:
            model=apply_lora(copy.deepcopy(self.base),dict(self.cfg,target_encoder=target,target_modules=['ffn_in']))
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            self.assertTrue(names)
            if target=='vision': self.assertTrue(all(n.startswith('visual.') for n in names))
            if target=='text': self.assertTrue(all(not n.startswith('visual.') for n in names))
            with self.assertRaises(ValueError): apply_lora(model,self.cfg)

    def test_continual_tasks_archive_factors_without_cuda(self):
        model=apply_lora(copy.deepcopy(self.base),self.cfg)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,MoRAM_Linear): module.lora_B.normal_(std=0.01)
        increment_task(model)
        for module in model.modules():
            if isinstance(module,MoRAM_Linear):
                self.assertEqual(module.old_rank,4)
                self.assertFalse(module.old_lora_A.requires_grad)
                self.assertTrue(module.lora_A.requires_grad)
                self.assertEqual(module.lora_B.count_nonzero().item(),0)
        self.assertTrue(torch.isfinite(model(self.image,self.text)[0]).all())
        increment_task(model)
        self.assertTrue(all(m.old_rank==8 for m in model.modules() if isinstance(m,MoRAM_Linear)))

    def test_loader_frozen_reference_and_trainable_are_independent(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint=d+'/tiny.pt';torch.save(self.base.state_dict(),checkpoint)
            cfg=dict(self.cfg,checkpoint_path=checkpoint,device='cpu',prec='fp32')
            frozen,*_=get_moram(cfg,zero_shot=True)
            model,*_=get_moram(cfg)
            self.assertIs(type(frozen),CLIP)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
            self.assertTrue(any(p.requires_grad for p in model.parameters()))
            self.assertNotEqual(frozen.visual.conv1.weight.data_ptr(),model.visual.conv1.weight.data_ptr())
            torch.testing.assert_close(frozen(self.image,self.text)[0],model(self.image,self.text)[0],atol=2e-5,rtol=2e-5)

    def test_lora_rank_zero_rank_one_and_dtype(self):
        for rank in [0,1,4]:
            module=MoRAM_Linear(8,6,r=rank,topk=8,dtype=torch.float64)
            x=torch.randn(3,8,dtype=torch.float64)
            y,loss=module(x)
            torch.testing.assert_close(y,torch.nn.functional.linear(x,module.weight,module.bias))
            self.assertEqual(loss,0)
        model=apply_lora(self.base.double(),self.cfg)
        for m in model.modules():
            if isinstance(m,MoRAM_Linear): self.assertEqual(m.weight.dtype,torch.float64)

if __name__=='__main__': unittest.main()
