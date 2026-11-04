import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from clip import clip
from models.models import RAILCLIP, ZeroShotCLIP
from scenario_datasets.build_functions import build_cur_task_data_loader, build_TAIL_testloader
from utils.evaluator import Evaluator


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.removeprefix("CLIP-")
    model_path = clip._download(clip._MODELS[backbone_name])
    try:
        archive = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = archive.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = clip.build_model(state_dict)
    if prec not in ("fp16", "fp32", "amp"):
        raise ValueError(f"Unsupported CLIP precision: {prec}")
    if prec != "fp16" or not torch.cuda.is_available():
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
            self.device = torch.device(f"cuda:{cfg.gpu}")
        self.cfg = cfg
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
        if not 0 <= cfg.fusion_weight <= 1:
            raise ValueError("fusion_weight must be in [0, 1]")
        print(f"Loading CLIP (backbone: {cfg.backbone})")
        self.clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec).to(self.device)
        if cfg.zero_shot:
            self.model = ZeroShotCLIP(self.clip_model).to(self.device)
        else:
            self.model = RAILCLIP(cfg, self.clip_model).to(self.device)
            self.seen_tasks = []
            self.seen_classnames = []
            if cfg.continue_train and not cfg.continue_train_first:
                self.load_model4CL(cfg.model_dir)
            if cfg.dataset not in cfg.dataset_sequence:
                raise ValueError("dataset must be in dataset_sequence")
            self.previous_num_classes = len(self.seen_classnames)
            if cfg.continue_train:
                expected = list(cfg.dataset_sequence[:len(self.seen_tasks)])
                if (len(self.seen_tasks) >= len(cfg.dataset_sequence)
                        or self.seen_tasks != expected
                        or cfg.dataset != cfg.dataset_sequence[len(self.seen_tasks)]):
                    raise ValueError("Continual tasks must follow dataset_sequence and the checkpoint prefix")
            self.total_num_classes = self.previous_num_classes + self.num_classes
        self.model.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        self.model.eval()
        # Analytic state is updated serially; DataParallel must not replicate it.

    def train(self):
        """One pass per augmentation, without an optimizer or prototype replay."""
        cfg = self.cfg
        if cfg.zero_shot:
            raise RuntimeError("Use test_zero_shot for a zero-shot model")
        if cfg.augmentation_time < 1:
            raise ValueError("augmentation_time must be positive")
        task_cfg = SimpleNamespace(
            current_class_num=self.total_num_classes,
            trained_class_num=self.previous_num_classes,
            regularization=cfg.regularization,
        )

        def training_batches():
            for _ in range(cfg.augmentation_time):
                yield from tqdm(self.train_loader, desc=f"RAIL task {len(self.seen_tasks) + 1}", ascii=True)

        start = time.time()
        self.model.eval()
        self.model.analytic_adaption(len(self.seen_tasks), task_cfg, training_batches())
        self.seen_tasks.append(cfg.dataset)
        self.seen_classnames.extend(self.classnames)
        print(f"Analytic training finished in {time.time() - start:.1f}s; learned {self.total_num_classes} classes")
        self.save_model(cfg.output_dir)
        self.test_cur_task()
        if cfg.continue_train:
            self.test_w_selector()

    @torch.no_grad()
    def _predict(self, images, use_selector=True):
        """Reference RAIL fusion: softmax CLIP + raw analytic scores.

        Only examples whose zero-shot winner is learned are fused. Analytic
        columns are scattered into the original TAIL global class ordering.
        """
        output = self.model.zero_shot(images).softmax(dim=-1)
        class_indices = self._learned_class_indices()
        learned = torch.tensor(class_indices, device=images.device, dtype=torch.long)
        mask = torch.isin(output.argmax(dim=-1), learned) if use_selector else torch.ones(
            images.shape[0], device=images.device, dtype=torch.bool)
        if mask.any():
            analytic = self.model(images[mask])
            padded = torch.zeros_like(output[mask])
            padded[:, learned] = analytic.to(padded.dtype)
            weight = self.cfg.fusion_weight
            output[mask] = (1 - weight) * output[mask] + weight * padded
        return output

    def _learned_class_indices(self):
        indices = []
        for task in self.seen_tasks:
            task_id = list(self.cfg.dataset_sequence).index(task)
            start = self.indices[task_id]
            stop = self.indices[task_id + 1] if task_id + 1 < len(self.indices) else len(self.merged_classnames)
            indices.extend(range(start, stop))
        if len(indices) != self.model.analytic_adaptor.out_features:
            raise ValueError("Analytic classifier does not match the learned task classes")
        return indices

    @torch.no_grad()
    def _evaluate(self, loader, indices, predict, label_offset=0):
        self.model.eval()
        self.evaluator.reset(indices)
        for images, labels in tqdm(loader, ascii=True):
            images = images.to(self.device)
            labels = labels.to(self.device) + label_offset
            self.evaluator.process(predict(images), labels)
        return self.evaluator.evaluate()

    def test_zero_shot(self):
        return self._evaluate(self.TAIL_test_loader, self.indices, self.model)

    def test_cur_task(self):
        # Task-local labels and classifier columns; keep the existing evaluator.
        start = self.previous_num_classes
        return self._evaluate(self.test_loader, [0], lambda images: self.model(images)[:, start:])

    def test_w_selector(self):
        return self._evaluate(self.TAIL_test_loader, self.indices, self._predict)

    def test_wo_selector(self):
        # Explicit ablation: fuse every sample, including unseen-domain images.
        return self._evaluate(self.TAIL_test_loader, self.indices,
                              lambda images: self._predict(images, use_selector=False))

    @torch.no_grad()
    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        checkpoint = {
            "format": "primal_RAIL_v1",
            "backbone": self.cfg.backbone,
            "feature_expansion": self.model.feature_expansion,
            "hidden_dim": self.model.hidden_dim,
            "regularization": self.cfg.regularization,
            "seen_tasks": self.seen_tasks,
            "seen_classnames": self.seen_classnames,
            "expansion_state": self.model.expansion_layer.state_dict(),
            "R": self.model.R,
            "analytic_weight": self.model.analytic_adaptor.weight,
        }
        path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"RAIL checkpoint saved to {path}")

    @torch.no_grad()
    def load_model4CL(self, directory):
        path = os.path.join(directory, "checkpoint.pth.tar")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format") != "primal_RAIL_v1":
            raise ValueError("Expected a RAIL checkpoint; old LADA checkpoints cannot resume analytic training")
        for key, expected in (("backbone", self.cfg.backbone),
                              ("feature_expansion", self.model.feature_expansion),
                              ("hidden_dim", self.model.hidden_dim),
                              ("regularization", self.cfg.regularization)):
            if checkpoint[key] != expected:
                raise ValueError(f"Checkpoint {key} does not match configuration")
        tasks = checkpoint["seen_tasks"]
        if not tasks or tasks != list(self.cfg.dataset_sequence[:len(tasks)]):
            raise ValueError("Checkpoint tasks must be a prefix of dataset_sequence")
        count = self.indices[len(tasks)] if len(tasks) < len(self.indices) else len(self.merged_classnames)
        if checkpoint["seen_classnames"] != self.merged_classnames[:count]:
            raise ValueError("Checkpoint class ordering does not match the dataset sequence")
        weight = checkpoint["analytic_weight"]
        if weight.shape != (count, self.model.hidden_dim) or checkpoint["R"].shape != (self.model.hidden_dim, self.model.hidden_dim):
            raise ValueError("Checkpoint analytic state has incompatible dimensions")
        self.model.expansion_layer.load_state_dict(checkpoint["expansion_state"])
        self.model.analytic_adaptor = nn.Linear(self.model.hidden_dim, count, bias=False,
                                               device=self.device, dtype=weight.dtype)
        self.model.analytic_adaptor.weight.copy_(weight)
        self.model.analytic_adaptor.requires_grad_(False)
        self.model.R = checkpoint["R"]
        self.seen_tasks = list(tasks)
        self.seen_classnames = list(checkpoint["seen_classnames"])
        print(f"Resumed RAIL checkpoint with {count} classes from {path}")
