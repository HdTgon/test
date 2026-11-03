import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from clip.model import CLIP
from trainer import Trainer, build_optimizer, moram_losses
from utils.config import _C
from utils.scheduler import cosine_lr
from scenario_datasets.utils import Datum


class TinyDataset:
    def __init__(self,root,num_shots):
        self.classnames=['class_a','class_b']
        self.train_x=[Datum(impath=Image.fromarray(np.full((10,10,3),30+i*40,dtype=np.uint8)),label=i%2,classname=self.classnames[i%2]) for i in range(4)]
        self.test=self.train_x


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cfg=_C.clone()
        self.cfg.dataset_sequence=['aircraft','caltech101']
        self.cfg.dataset='aircraft'
        self.cfg.device='cpu';self.cfg.resolution=8;self.cfg.num_workers=0
        self.cfg.iterations=2;self.cfg.batch_size=2;self.cfg.print_freq=1
        self.cfg.output_dir=self.tmp.name
        self.cfg.target_modules=['q_proj','v_proj','ffn_in','ffn_out']
        self.cfg.prune_thre=0.0
        self.cfg.checkpoint_path=str(Path(self.tmp.name)/'base.pt')
        torch.manual_seed(13)
        torch.save(CLIP(16,8,1,64,4,77,49408,64,1,1).state_dict(),self.cfg.checkpoint_path)
        self.dataset_patch=patch.dict('scenario_datasets.build_functions.dataset_list',aircraft=TinyDataset,caltech101=TinyDataset)
        self.dataset_patch.start();self.addCleanup(self.dataset_patch.stop)

    def test_two_tasks_save_resume_and_zeroshot(self):
        first=self.cfg.clone();first.dataset='aircraft';first.continue_train=True;first.continue_train_first=True
        trainer=Trainer(first)
        initial={n:p.detach().clone() for n,p in trainer.model.named_parameters() if not p.requires_grad}
        trainer.train()
        self.assertEqual(trainer.completed_tasks,1)
        self.assertIsInstance(trainer.test_w_selector(),dict)
        for n,p in trainer.model.named_parameters():
            if n in initial: torch.testing.assert_close(p,initial[n],rtol=0,atol=0)
        checkpoint=str(Path(self.tmp.name)/'checkpoint.pth.tar')
        checkpoint_state=torch.load(checkpoint,weights_only=True)
        self.assertNotIn('model',checkpoint_state)
        self.assertIn('lora_ranks',checkpoint_state)
        self.assertTrue(all(not k.endswith(('.weight','.bias')) for k in checkpoint_state['state_dict']))
        resume=self.cfg.clone();resume.dataset='caltech101';resume.model_dir=self.tmp.name;resume.continue_train=True
        resumed=Trainer(resume)
        self.assertEqual(resumed.completed_tasks,1)
        self.assertTrue(any('old_lora_A' in n for n,_ in resumed.model.named_parameters()))
        self.assertTrue(all(not p.requires_grad for n,p in resumed.model.named_parameters() if 'old_lora_' in n))
        resumed.train()
        self.assertEqual(resumed.completed_tasks,2)
        self.assertEqual(len(resumed.seen_prompts),4)
        self.assertEqual(resumed.image_prototypes.shape,(4,16))
        self.assertEqual(resumed.zero_shot_text_prototypes.shape,(4,16))
        self.assertEqual(resumed.prototype_counts.tolist(),[2,2,2,2])
        from clip import clip
        from models.models import MoRAM_CLIP
        final=self.cfg.clone()
        restored=Trainer.__new__(Trainer)
        restored.cfg=final;restored.device=torch.device('cpu')
        base,_=clip.load(final.checkpoint_path,device='cpu',jit=False)
        restored.model=MoRAM_CLIP(final,base)
        restored.load_model4CL(self.tmp.name)
        saved={n:p for n,p in resumed.model.state_dict().items() if n not in ('prompts','text_features')}
        for n,p in saved.items(): torch.testing.assert_close(p,restored.model.state_dict()[n],rtol=0,atol=0)
        zero=self.cfg.clone();zero.zero_shot=True
        reference=Trainer(zero)
        self.assertFalse(any(p.requires_grad for p in reference.model.parameters()))
        self.assertIsInstance(reference.test_zero_shot(),dict)

    def test_encoder_modes_update_and_losses_finite(self):
        for target in ['vision','text','all']:
            cfg=self.cfg.clone();cfg.target_encoder=target
            trainer=Trainer(cfg)
            before={n:p.clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
            with patch.object(trainer.optim,'step',wraps=trainer.optim.step) as step:
                trainer.train()
                self.assertEqual(step.call_count,2)
            self.assertTrue(all(torch.isfinite(p).all() for p in trainer.model.parameters()))
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in trainer.model.named_parameters() if n in before))

    def test_zero_shot_prototype_cache_and_centroid(self):
        trainer=Trainer(self.cfg)
        cache_path=Path(self.tmp.name)/'zero_shot_prototypes/aircraft.pth'
        self.assertTrue(cache_path.exists())
        cached=torch.load(cache_path,weights_only=True)
        self.assertEqual(cached['counts'].tolist(),[2,2])
        torch.testing.assert_close(cached['image_prototypes'].norm(dim=-1),torch.ones(2))
        sums=torch.zeros(2,16)
        with torch.no_grad():
            for images,labels in trainer.train_loader4updating:
                features=F.normalize(trainer.clip_model.encode_image(images),dim=-1)
                sums.index_add_(0,labels,features)
        torch.testing.assert_close(cached['image_prototypes'],F.normalize(sums/2,dim=-1))
        with patch.object(trainer.clip_model,'encode_image',side_effect=AssertionError('cache hit must not encode')):
            reused=trainer.load_or_extract_zero_shot_prototypes(self.tmp.name)
            torch.testing.assert_close(reused['image_prototypes'],cached['image_prototypes'],rtol=0,atol=0)
        rng=torch.get_rng_state()
        with patch.object(trainer.clip_model,'encode_image',wraps=trainer.clip_model.encode_image) as encode:
            rebuilt=trainer.load_or_extract_zero_shot_prototypes(self.tmp.name,force=True)
            self.assertGreater(encode.call_count,0)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        torch.testing.assert_close(rebuilt['image_prototypes'],cached['image_prototypes'],rtol=0,atol=0)
        trainer.cfg.num_shots=2
        with patch.object(trainer.clip_model,'encode_image',wraps=trainer.clip_model.encode_image) as encode:
            trainer.load_or_extract_zero_shot_prototypes(self.tmp.name)
            self.assertGreater(encode.call_count,0)

    def test_upstream_optimizer_schedule_and_loss_recipe(self):
        parameter=torch.nn.Parameter(torch.randn(2,3))
        model=torch.nn.Linear(3,2)
        optimizer=build_optimizer(self.cfg,model)
        self.assertEqual(optimizer.defaults['betas'],(0.9,0.9999))
        self.assertEqual(optimizer.defaults['weight_decay'],self.cfg.wd)
        scheduler=cosine_lr(optimizer,0.01,2,6)
        for step in range(6):
            scheduler(step)
            expected=0.01*(step+1)/2 if step<2 else 0.005*(1+np.cos(np.pi*(step-2)/4))
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected)
        image=F.normalize(torch.randn(3,8),dim=-1);text=F.normalize(torch.randn(4,8),dim=-1)
        labels=torch.tensor([2,3,2]);scale=torch.tensor(14.)
        loss,ce_i,ce_t,logits=moram_losses(image,text,scale,labels,2,'all',0.)
        expected_logits=scale*image@text.T;expected_logits[:,:2]=float('-inf')
        target=labels[:,None].eq(torch.arange(4)[None,:]).float().T
        expected=F.cross_entropy(expected_logits,labels)+F.cross_entropy(scale*text@image.T,target)
        torch.testing.assert_close(loss,expected,atol=0,rtol=0)
        self.assertTrue(torch.isfinite(loss))

class MoRAMUtilityTests(unittest.TestCase):
    def test_lora_only_roundtrip_ranks_dtype_and_frozen_history(self):
        from models.models import MoRAM_Linear
        from utils.moram_utils import torch_save_lora, torch_load_lora
        torch.manual_seed(10)
        model=torch.nn.Sequential(MoRAM_Linear(8,6,r=4,topk=2,lora_threshold=0)).double()
        with torch.no_grad(): model[0].lora_B.normal_()
        model[0].increment_task()
        with torch.no_grad(): model[0].lora_B.normal_()
        model[0].lora_scaling=0.37
        model.eval()
        x=torch.randn(3,8,dtype=torch.float64)
        expected=model(x)[0]
        loaded=torch.nn.Sequential(MoRAM_Linear(8,6,r=4,topk=2,lora_threshold=0)).double().eval()
        with torch.no_grad():
            loaded[0].weight.copy_(model[0].weight)
            loaded[0].bias.copy_(model[0].bias)
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'lora.pth')
            torch_save_lora(model,path,MoRAM_Linear)
            state=torch.load(path,weights_only=True)
            self.assertEqual(set(state),{'state_dict','lora_ranks'})
            self.assertTrue(state['state_dict'])
            self.assertFalse(any(k.endswith(('.weight','.bias')) for k in state['state_dict']))
            torch_load_lora(loaded,path,MoRAM_Linear)
            torch.testing.assert_close(loaded(x)[0],expected,rtol=0,atol=0)
            self.assertEqual((loaded[0].rank,loaded[0].old_rank),(4,4))
            self.assertEqual(loaded[0].lora_A.dtype,torch.float64)
            self.assertFalse(loaded[0].old_lora_A.requires_grad)
            self.assertTrue(loaded[0].lora_A.requires_grad)
            self.assertEqual(loaded[0].lora_scaling,0.37)
            del state['state_dict']['0.lora_A']
            torch.save(state,path)
            with self.assertRaises(ValueError):torch_load_lora(loaded,path,MoRAM_Linear)

class LADACompatibilityTests(unittest.TestCase):
    def test_entry_and_experiment_config(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'main.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        source=ast.unparse(main)
        for text in ['cfg.model_dir = cfg.output_dir', 'setup_logger(cfg.output_dir, cfg.dataset)', 'trainer.train()']:
            self.assertIn(text,source)
        cfg=_C.clone();cfg.merge_from_file(str(root/'configs/data/TAIL.yaml'));cfg.merge_from_file(str(root/'configs/model/moram_vit_b16.yaml'))
        self.assertEqual((cfg.rank,cfg.topk,cfg.iterations,cfg.lr,cfg.prune_thre),(16,16,500,5e-4,0.2))
        self.assertEqual(cfg.batch_size,64)
        self.assertEqual(set(cfg.target_modules),{'q_proj','k_proj','v_proj','out_proj','ffn_in','ffn_out'})
        from result_process import extract_task_accuracies
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'log_aircraft.txt';path.write_text('* Task 0 Accuracy: 25.0%\n* Task 1 Accuracy: 50.0%\n')
            self.assertEqual(extract_task_accuracies(path,2),[25.,50.])

if __name__=='__main__' :unittest.main()
