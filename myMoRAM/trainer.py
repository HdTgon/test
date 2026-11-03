"""LADA data workflow with the uploaded MoRAM iteration-based training recipe."""
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from clip import clip
from models.models import ZeroShotCLIP, MoRAM_CLIP, increment_task
from utils.scheduler import cosine_lr


def build_optimizer(cfg, model):
    # Exact upstream MoRAM optimizer, including beta2=0.9999 (not LADA's default).
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.wd, betas=(0.9, 0.9999))


def moram_losses(image_features, embeddings, scale, labels, seen_count, target_encoder, smoothing):
    logits = scale * image_features @ embeddings.t()
    if seen_count:
        logits = logits.clone()
        logits[:, :seen_count] = float('-inf')
    ce_i = F.cross_entropy(logits, labels.long(), label_smoothing=smoothing)
    mask = labels[:, None].eq(torch.arange(embeddings.shape[0], device=labels.device)[None, :]).float().t()
    ce_t = F.cross_entropy(scale * embeddings @ image_features.t(), mask)
    loss = ce_i + ce_t if target_encoder in ('vision', 'all') else ce_i
    return loss, ce_i, ce_t, logits


class Trainer:
    def __init__(self, cfg):
        self.cfg=cfg
        if cfg.device=='auto':
            self.device=torch.device(f'cuda:{cfg.gpu}' if cfg.gpu is not None else 'cuda') if torch.cuda.is_available() else torch.device('cpu')
        else: self.device=torch.device(cfg.device)
        if cfg.prec!='fp32':
            raise ValueError('Use prec=fp32: upstream MoRAM training has no autocast/scaler; mixed precision is not silently enabled')
        if cfg.print_freq<=0:
            raise ValueError('print_freq must be positive')
        if cfg.iterations<=0 or cfg.warmup_length<0 or cfg.warmup_length>cfg.iterations:
            raise ValueError('Require iterations>0 and 0<=warmup_length<=iterations')
        if cfg.ls and len(cfg.dataset_sequence)>1:
            raise ValueError('Upstream masks old logits to -inf: nonzero label smoothing makes later-task loss infinite; use ls=0')
        if not cfg.dataset_sequence or len(set(cfg.dataset_sequence))!=len(cfg.dataset_sequence):
            raise ValueError('dataset_sequence must be nonempty and contain no duplicate tasks')
        backbone=cfg.backbone.removeprefix('CLIP-')
        backbone={'ViT-B-16':'ViT-B/16','ViT-B-32':'ViT-B/32'}.get(backbone,backbone)
        base,_=clip.load(cfg.checkpoint_path or backbone,device='cpu',jit=False)
        base.float()
        self.model=(ZeroShotCLIP(base) if cfg.zero_shot else MoRAM_CLIP(dict(cfg),base)).to(self.device).eval()
        del base
        self.tokenizer=clip.tokenize
        self.seen_prompts=[]
        self.completed_tasks=0
        self.accuracy_table=[]
        self.build_data_loader()
        if cfg.resume:
            if cfg.zero_shot: raise ValueError('resume is for MoRAM training/evaluation, not frozen zero-shot')
            self.load_checkpoint(cfg.resume)

    def build_data_loader(self):
        from scenario_datasets.build_functions import build_TAIL_testloader
        cfg=self.cfg
        mean=(0.48145466,0.4578275,0.40821073);std=(0.26862954,0.26130258,0.27577711)
        self.transform_test=transforms.Compose([
            transforms.Resize(cfg.resolution,interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.resolution),transforms.Lambda(lambda im:im.convert('RGB')),
            transforms.ToTensor(),transforms.Normalize(mean,std)])
        self.test_loader,self.merged_classnames,self.indices=build_TAIL_testloader(
            cfg.root,cfg.dataset_sequence,self.transform_test,cfg.batch_size,cfg.num_workers)

    def current_loader(self, task):
        from scenario_datasets.build_functions import build_cur_task_data_loader
        cfg=self.cfg
        if task in ('mnist','eurosat'):
            transform=self.transform_test
        else:
            transform=transforms.Compose([
                transforms.RandomResizedCrop(cfg.resolution,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),transforms.Lambda(lambda im:im.convert('RGB')),
                transforms.ToTensor(),transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])
        train,_,_,names=build_cur_task_data_loader(cfg.root,task,transform,self.transform_test,cfg.num_shots,cfg.batch_size,cfg.num_workers)
        return train,names

    def prompts(self, names):
        return [self.cfg.template.format(name.replace('_',' ')) for name in names]

    def train_task(self, loader, names, task):
        cfg=self.cfg
        if len(loader)==0: raise ValueError(f'Empty training loader: {task}')
        optimizer=build_optimizer(cfg,self.model)
        scheduler=cosine_lr(optimizer,cfg.lr,cfg.warmup_length,cfg.iterations)
        tokens=self.tokenizer(self.prompts(names)).to(self.device)
        seen_count=len(self.seen_prompts)
        self.model.eval()
        with torch.no_grad():
            seen=F.normalize(self.model.encode_text(self.tokenizer(self.seen_prompts).to(self.device)),dim=-1) if seen_count else None
            current=F.normalize(self.model.encode_text(tokens),dim=-1) if cfg.target_encoder=='vision' else None
        iterator=iter(loader)
        losses=[]
        print(f'Task {task}: iterations={cfg.iterations}, trainable={sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        for iteration in range(cfg.iterations):
            self.model.train()
            scheduler(iteration)
            try: images,labels=next(iterator)
            except StopIteration:
                iterator=iter(loader);images,labels=next(iterator)
            images=images.to(self.device);labels=labels.to(self.device)+seen_count
            if cfg.target_encoder in ('text','all'):
                current=F.normalize(self.model.encode_text(tokens),dim=-1)
            embeddings=torch.cat([seen,current],dim=0) if seen is not None else current
            output=F.normalize(self.model.encode_image(images),dim=-1)
            # Wrapper scale is already exp; do not exponentiate it twice.
            loss,ce_i,ce_t,logits=moram_losses(output,embeddings,self.model.logit_scale,labels,seen_count,cfg.target_encoder,cfg.ls)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss in {task} iteration {iteration}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if iteration%cfg.print_freq==0 or iteration==cfg.iterations-1:
                accuracy=100*logits.argmax(-1).eq(labels).float().mean().item()
                print(f'{task} step {iteration+1}/{cfg.iterations} loss={loss.item():.6f} CE_i={ce_i.item():.6f} CE_t={ce_t.item():.6f} lr={optimizer.param_groups[0]["lr"]:.8f} acc={accuracy:.2f}')
        return losses

    def train(self):
        cfg=self.cfg
        for i,task in enumerate(cfg.dataset_sequence):
            if i<self.completed_tasks: continue
            if cfg.dataset and cfg.dataset!=task:
                raise ValueError('For a later dataset resume the preceding task checkpoint; otherwise leave dataset empty for the full sequence')
            loader,names=self.current_loader(task)
            expected=self.merged_classnames[self.indices[i]:self.indices[i]+len(names)]
            if list(names)!=list(expected): raise ValueError(f'Train/eval class order differs for {task}')
            if i: increment_task(self.model)
            self.train_task(loader,names,task)
            self.seen_prompts.extend(self.prompts(names))
            self.completed_tasks=i+1
            self.accuracy_table.append(self.evaluate())
            os.makedirs(cfg.output_dir,exist_ok=True)
            np.savetxt(os.path.join(cfg.output_dir,'accuracies.csv'),np.asarray(self.accuracy_table),
                       delimiter=',',header=','.join(cfg.dataset_sequence),comments='',fmt='%.6f')
            self.save_checkpoint(os.path.join(cfg.output_dir,f'{task}.pth'))
            if cfg.dataset: break

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        embeddings=F.normalize(self.model.encode_text(self.tokenizer(self.prompts(self.merged_classnames)).to(self.device)),dim=-1)
        correct=[0]*len(self.indices);total=[0]*len(self.indices)
        boundaries=torch.tensor(self.indices,device=self.device)
        for images,labels in self.test_loader:
            labels=labels.to(self.device)
            output=F.normalize(self.model.encode_image(images.to(self.device)),dim=-1)
            pred=(self.model.logit_scale*output@embeddings.t()).argmax(-1)
            task_ids=torch.bucketize(labels,boundaries,right=True)-1
            for i in range(len(self.indices)):
                mask=task_ids==i
                total[i]+=int(mask.sum());correct[i]+=int((pred[mask]==labels[mask]).sum())
        if any(n==0 for n in total): raise ValueError('An evaluation task has no samples')
        accuracies=[100*c/n for c,n in zip(correct,total)]
        print('Task accuracies:',dict(zip(self.cfg.dataset_sequence,accuracies)))
        print(f'Average Accuracy: {np.mean(accuracies):.2f}%')
        return accuracies

    def test_zero_shot(self):
        return self.evaluate()

    def save_checkpoint(self,path):
        os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
        torch.save(dict(model=self.model.state_dict(),completed_tasks=self.completed_tasks,
            seen_prompts=self.seen_prompts,accuracy_table=self.accuracy_table,config=dict(self.cfg),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),path)

    def load_checkpoint(self,path):
        # Only load trusted local checkpoints: metadata contains RNG Python objects.
        state=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('dataset_sequence','backbone','checkpoint_path','rank','topk','temp','prune_thre','target_encoder','target_modules','template'):
            if state['config'][key]!=self.cfg[key]: raise ValueError(f'Checkpoint config mismatch: {key}')
        completed=state['completed_tasks']
        if not 1<=completed<=len(self.cfg.dataset_sequence): raise ValueError('Invalid completed_tasks')
        for _ in range(completed-1): increment_task(self.model)
        self.model.load_state_dict(state['model'],strict=True)
        self.completed_tasks=completed;self.seen_prompts=state['seen_prompts'];self.accuracy_table=state['accuracy_table']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resumed after {completed} tasks from {path}')
