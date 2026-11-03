import os
import time
import datetime
import math
import hashlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from clip import clip
from scenario_datasets.build_functions import build_cur_task_data_loader, build_TAIL_testloader
from models.models import ZeroShotCLIP, MoRAM_CLIP, MoRAM_Linear, increment_task
from utils.meter import AverageMeter
from utils.evaluator import Evaluator
from utils.moram_utils import cosine_lr, torch_save_lora, torch_load_lora

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

def load_clip_to_cpu(backbone_name, prec, checkpoint_path=""):
    backbone_name = backbone_name.removeprefix("CLIP-")
    if checkpoint_path:
        model_path = checkpoint_path
    else:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = clip.build_model(state_dict or model.state_dict())
    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        model.float()
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        if cfg.prec != 'fp32':
            raise ValueError('MoRAM recipe requires prec=fp32')
        if cfg.iterations <= 0 or not 0 <= cfg.warmup_length <= cfg.iterations:
            raise ValueError('Invalid iterations/warmup_length')
        if cfg.print_freq <= 0:
            raise ValueError('print_freq must be positive')
        if cfg.ls and len(cfg.dataset_sequence) > 1:
            raise ValueError('Use ls=0 with old-class -inf masking')
        self.cfg = cfg
        self.seen_prompts = []
        self.completed_tasks = 0
        self.text_prototypes = None
        self.image_prototypes = None
        self.zero_shot_text_prototypes = None
        self.prototype_counts = None
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        print("mean:", mean)
        print("std:", std)

        if cfg.dataset in ["mnist", "eurosat"]:
            self.transform_train = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        self.transform_test = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        build_TAIL_testloader_return = build_TAIL_testloader(root=root, 
                                                             dataset_sequence=cfg.dataset_sequence, 
                                                             transform_test=self.transform_test,
                                                             batch_size=cfg.batch_size,
                                                             num_workers=cfg.num_workers)
        self.TAIL_test_loader, self.merged_classnames, self.indices = build_TAIL_testloader_return
        
        if not cfg.zero_shot:
            build_cur_task_data_loader_return = build_cur_task_data_loader(root=root,
                                                                        dataset_name=cfg.dataset, 
                                                                        transform_train=self.transform_train, 
                                                                        transform_test=self.transform_test, 
                                                                        num_shots=cfg.num_shots, 
                                                                        batch_size=cfg.batch_size,
                                                                        num_workers=cfg.num_workers)
            self.train_loader, self.train_loader4updating, self.test_loader, self.classnames = build_cur_task_data_loader_return

            self.num_classes = len(self.classnames)

    def build_model(self):
        cfg = self.cfg
        print("Building model")
        print(f"Loading CLIP (backbone: {cfg.backbone})")
        self.clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec, cfg.checkpoint_path)
        if cfg.zero_shot:
            self.model = ZeroShotCLIP(self.clip_model).to(self.device)
            self.model.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        else:
            self.model = MoRAM_CLIP(cfg, self.clip_model).to(self.device)
            if cfg.resume:
                self.load_model4CL(os.path.dirname(cfg.resume), load_path=cfg.resume)
            elif cfg.continue_train and not cfg.continue_train_first:
                self.load_model4CL(cfg.model_dir)
            task_index = cfg.dataset_sequence.index(cfg.dataset)
            if task_index != self.completed_tasks:
                raise ValueError('Task/checkpoint order differs; check continue_train_first and model_dir')
            if self.completed_tasks:
                increment_task(self.model)
            self.total_num_classes = self.num_classes + len(self.seen_prompts)
            self.model.init_prompts(self.classnames, ['a photo of a {}.'], clip.tokenize, self.device)
            self.current_zero_shot_prototypes = self.load_or_extract_zero_shot_prototypes(
                cfg.zero_shot_prototype_dir or cfg.output_dir)
            self.build_optimizer()
            self.build_criterion()
            torch.cuda.empty_cache()

    def build_optimizer(self):
        cfg = self.cfg
        print(f"Total params: {sum(p.numel() for p in self.model.parameters())}")
        print(f"Tuned MoRAM params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        self.optim = build_optimizer(cfg, self.model)
        self.sched = cosine_lr(self.optim, cfg.lr, cfg.warmup_length, cfg.iterations)

    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.ls)

    def train(self):
        cfg = self.cfg
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)
        time_start = time.time()
        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise ValueError('Empty training loader')
        num_epochs = math.ceil(cfg.iterations / num_batches)
        seen_count = len(self.seen_prompts)
        self.model.eval()
        with torch.no_grad():
            seen_embeddings = F.normalize(self.model.encode_text(clip.tokenize(self.seen_prompts).to(self.device)), dim=-1) if seen_count else None
            current_embeddings = F.normalize(self.model.encode_text(self.model.prompts), dim=-1) if cfg.target_encoder == 'vision' else None

        for epoch_idx in range(num_epochs):
            self.model.train()
            end = time.time()
            for batch_idx, batch in enumerate(self.train_loader):
                iteration = epoch_idx * num_batches + batch_idx
                if iteration >= cfg.iterations:
                    break
                data_time.update(time.time() - end)
                image = batch[0].to(self.device)
                label = batch[1].to(self.device) + seen_count
                self.sched(iteration)
                if cfg.target_encoder in ('text', 'all'):
                    current_embeddings = F.normalize(self.model.encode_text(self.model.prompts), dim=-1)
                embeddings = torch.cat((seen_embeddings, current_embeddings), dim=0) if seen_count else current_embeddings
                image_features = F.normalize(self.model.encode_image(image), dim=-1)
                loss, ce_i, ce_t, output = moram_losses(image_features, embeddings, self.model.logit_scale,
                                                       label, seen_count, cfg.target_encoder, cfg.ls)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite MoRAM loss')
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)
                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches or iteration + 1 == cfg.iterations:
                    nb_remain = cfg.iterations - iteration - 1
                    eta = str(datetime.timedelta(seconds=int(batch_time.avg * nb_remain)))
                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"iteration [{iteration + 1}/{cfg.iterations}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))
                end = time.time()
        torch.cuda.empty_cache()
        print("Finish training")
        elapsed = str(datetime.timedelta(seconds=round(time.time() - time_start)))
        print(f"Time elapsed: {elapsed}")
        self.model.eval()
        with torch.no_grad():
            features = self.model.encode_text(self.model.prompts)
            self.text_prototypes = torch.cat((self.text_prototypes, features), dim=0) if self.text_prototypes is not None else features
        self.seen_prompts += ['a photo of a {}.'.format(name.replace('_', ' ')) for name in self.classnames]
        for attribute, key in [('image_prototypes', 'image_prototypes'),
                               ('zero_shot_text_prototypes', 'text_prototypes'),
                               ('prototype_counts', 'counts')]:
            current = self.current_zero_shot_prototypes[key].to(self.device)
            previous = getattr(self, attribute)
            setattr(self, attribute, torch.cat((previous, current), dim=0) if previous is not None else current)
        self.completed_tasks += 1
        self.save_model(cfg.output_dir)
        self.test_cur_task()
        if cfg.continue_train:
            self.test_wo_selector()

    @torch.no_grad()
    def test_zero_shot(self):
        self.model.eval()
        self.evaluator.reset(self.indices)

        print("==================================================================================")
        print(f"Evaluate on the test set")
        data_loader = self.TAIL_test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_cur_task(self):
        self.model.eval()
        indices = [0]
        self.evaluator.reset(indices)

        print("==================================================================================")
        print(f"Evaluate on the current task test set")
        data_loader = self.test_loader
        text_features = self.model.encode_text(self.model.prompts)

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image, 
                                text_features=text_features)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_w_selector(self):
        '''
        Use the original CLIP as a selector to determine whether the input image belongs to a seen class or an unseen class;
        For images predicted as seen classes by the selector, use the trained model for prediction;
        For images predicted as unseen classes by the selector, use the original CLIP for prediction.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader

        # Load zero shot CLIP
        clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec, cfg.checkpoint_path)
        frozen_clip = ZeroShotCLIP(clip_model)
        frozen_clip.to(self.device)
        frozen_clip.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        frozen_clip.eval()

        # For seen class names(ID), use the text features from the trained model
        id_text_features = self.text_prototypes
        id_num = id_text_features.shape[0]

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            zs_output = frozen_clip(image)
            zs_pred = zs_output.max(1)[1]
            
            id_indices = (zs_pred < id_num).nonzero(as_tuple=True)[0]
            ood_indices = (zs_pred >= id_num).nonzero(as_tuple=True)[0]

            if id_indices.numel() > 0:
                id_image = image[id_indices]
                id_output = self.model(id_image, 
                                       text_features=id_text_features)
                # Pad zeros at the end of id_output to match zs_output size
                pad_size = zs_output.shape[1] - id_output.shape[1]
                if pad_size > 0:
                    padding = torch.zeros(id_output.shape[0], pad_size).to(id_output.device)
                    id_output_padded = torch.cat([id_output, padding], dim=1)
                else:
                    id_output_padded = id_output
            else:
                id_output_padded = None

            if ood_indices.numel() > 0:
                ood_output = zs_output[ood_indices]
            else:
                ood_output = None

            # Combine id_output_padded and ood_output to form output, ensuring that the order matches the images in the batch
            output = torch.zeros_like(zs_output)
            if id_output_padded is not None:
                output[id_indices] = id_output_padded
            if ood_output is not None:
                output[ood_indices] = ood_output

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_wo_selector(self):
        '''
        Use text prototypes extracted by the trained model as text features for seen classes;
        Use text features extracted by the original CLIP as text features for unseen classes.
        Concatenate these as the complete set of text features, and input them together with LADA features into the trained model for prediction.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader

        # Load zero shot CLIP
        clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec, cfg.checkpoint_path)
        frozen_clip = ZeroShotCLIP(clip_model)
        frozen_clip.to(self.device)
        frozen_clip.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        frozen_clip.eval()

        # For the seen class names(ID), use the text features from the trained model; 
        # for the unseen class names(OOD), use the text features from the original CLIP
        origin_text_features = frozen_clip.text_features
        id_text_features = self.text_prototypes
        id_num = id_text_features.shape[0]
        ood_text_features = origin_text_features[id_num:,:]
        text_features = torch.cat((id_text_features, ood_text_features), dim=0)

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image, 
                                text_features=text_features)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def load_or_extract_zero_shot_prototypes(self, directory, force=False):
        """Cache frozen CLIP class centroids from the deterministic training split.

        One normalized mean image embedding and one frozen text embedding per
        class. These are reference data only; no replay/GMM/LADA loss is added.
        """
        cfg = self.cfg
        digest = hashlib.sha256()
        for name, value in self.clip_model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        samples = hashlib.sha256()
        for item in self.train_loader4updating.dataset.data_source:
            samples.update(str(item.label).encode())
            if isinstance(item.impath, str):
                stat = os.stat(item.impath)
                samples.update(f'{os.path.realpath(item.impath)}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
            else:
                samples.update(item.impath.tobytes())
        metadata = dict(version=1, dataset=cfg.dataset, classnames=list(self.classnames),
                        root=os.path.realpath(cfg.root), num_shots=cfg.num_shots,
                        resolution=cfg.resolution, backbone_sha256=digest.hexdigest(),
                        samples_sha256=samples.hexdigest(), template='a photo of a {}.',
                        normalization='mean_of_unit_image_features_then_unit_normalize')
        cache_dir = os.path.join(directory, 'zero_shot_prototypes')
        cache_path = os.path.join(cache_dir, cfg.dataset + '.pth')
        if os.path.isfile(cache_path) and not force:
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            if cached.get('metadata') == metadata:
                print(f'Loaded zero-shot class prototypes from {cache_path}')
                return cached
            print('Zero-shot prototype metadata changed; recomputing')
        rng_state = torch.get_rng_state()
        reference = self.clip_model.to(self.device).eval()
        try:
            sums = torch.zeros(self.num_classes, reference.visual.output_dim, device=self.device)
            counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
            for image, label in tqdm(self.train_loader4updating, desc='Zero-shot class prototypes', ascii=True):
                label = label.to(self.device)
                features = F.normalize(reference.encode_image(image.to(self.device)).float(), dim=-1)
                sums.index_add_(0, label, features)
                counts.index_add_(0, label, torch.ones_like(label))
            if (counts == 0).any():
                raise ValueError('Cannot extract a prototype for a class with no training samples')
            image_prototypes = F.normalize(sums / counts[:, None], dim=-1)
            if (image_prototypes.norm(dim=-1) == 0).any():
                raise ValueError('A zero-shot class mean is zero')
            prompts = ['a photo of a {}.'.format(name.replace('_', ' ')) for name in self.classnames]
            text_prototypes = F.normalize(reference.encode_text(clip.tokenize(prompts).to(self.device)).float(), dim=-1)
            cached = dict(metadata=metadata, image_prototypes=image_prototypes.cpu(),
                          text_prototypes=text_prototypes.cpu(), counts=counts.cpu())
        finally:
            reference.cpu()
            torch.set_rng_state(rng_state)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(cached, cache_path)
        print(f'Saved zero-shot class prototypes to {cache_path}')
        return cached

    @torch.no_grad()
    def save_model(self, directory):
        checkpoint = {
            'text_prototypes': self.text_prototypes,
            'image_prototypes': self.image_prototypes,
            'zero_shot_text_prototypes': self.zero_shot_text_prototypes,
            'prototype_counts': self.prototype_counts,
            'seen_prompts': self.seen_prompts,
            'completed_tasks': self.completed_tasks,
            'config': dict(self.cfg),
        }
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, 'checkpoint.pth.tar')
        torch_save_lora(self.model, save_path, MoRAM_Linear, extra_state=checkpoint)

    @torch.no_grad()
    def load_model4CL(self, directory, load_path=None):
        load_path = load_path or os.path.join(directory, 'checkpoint.pth.tar')
        print('Loading model weights from {}'.format(load_path))
        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
        for key in ('dataset_sequence', 'backbone', 'checkpoint_path', 'rank', 'topk', 'temp',
                    'prune_thre', 'target_encoder', 'target_modules'):
            if checkpoint['config'][key] != self.cfg[key]:
                raise ValueError(f'Checkpoint config mismatch: {key}')
        completed = checkpoint['completed_tasks']
        if not 1 <= completed <= len(self.cfg.dataset_sequence):
            raise ValueError('Invalid completed_tasks')
        torch_load_lora(self.model, load_path, MoRAM_Linear)
        self.text_prototypes = checkpoint['text_prototypes'].to(self.device)
        for key in ('image_prototypes', 'zero_shot_text_prototypes', 'prototype_counts'):
            value = checkpoint.get(key)
            if value is None:
                raise ValueError('Checkpoint lacks zero-shot prototypes; restart from the first task')
            setattr(self, key, value.to(self.device))
        self.seen_prompts = checkpoint['seen_prompts']
        self.completed_tasks = completed
