import os
import time
import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from clip import clip
from models.models import DualRAILCLIP, ZeroShotCLIP
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
            self.device = torch.device("cuda:{}".format(cfg.gpu))

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
        print("Building model")
        print(f"Loading CLIP (backbone: {cfg.backbone})")
        self.clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec).to(self.device)
        if cfg.zero_shot:
            self.model = ZeroShotCLIP(self.clip_model).to(self.device)
        else:
            self.model = DualRAILCLIP(cfg, self.clip_model).to(self.device)
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
            regularization=cfg.dual_regularization,
        )

        def training_batches():
            for _ in range(cfg.augmentation_time):
                yield from tqdm(self.train_loader, desc=f"DualRAIL task {len(self.seen_tasks) + 1}", ascii=True)

        time_start = time.time()
        self.model.eval()
        self.model.analytic_adaption(len(self.seen_tasks), task_cfg, training_batches())
        self.seen_tasks.append(cfg.dataset)
        self.seen_classnames.extend(self.classnames)
        torch.cuda.empty_cache()

        print("Finish training")
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # DualRAIL refits its kernel coefficients during fitting; no prototypes to update.
        # Save model
        self.save_model(cfg.output_dir)
        self.test_cur_task()
        if cfg.continue_train:
            self.test_w_selector()

    def _learned_class_indices(self):
        indices = []
        for task in self.seen_tasks:
            task_id = list(self.cfg.dataset_sequence).index(task)
            start = self.indices[task_id]
            stop = self.indices[task_id + 1] if task_id + 1 < len(self.indices) else len(self.merged_classnames)
            indices.extend(range(start, stop))
        if len(indices) != self.model.num_classes:
            raise ValueError("Analytic classifier does not match the learned task classes")
        return indices

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

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            # The loader uses task-local labels. Select this task's RAIL columns.
            output = self.model(image)[:, self.previous_num_classes:]

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_w_selector(self):
        '''
        Use frozen CLIP to select images from learned classes, then fuse their
        CLIP probabilities and analytic scores as in dual RAIL. Images from
        unseen classes retain their zero-shot probabilities.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader
        # RAIL already owns the frozen CLIP and all zero-shot text features.
        learned_classes = torch.tensor(self._learned_class_indices(), device=self.device)

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            zs_output = self.model.zero_shot(image).softmax(dim=-1)
            zs_pred = zs_output.max(1)[1]
            id_indices = torch.isin(zs_pred, learned_classes).nonzero(as_tuple=True)[0]

            output = zs_output.clone()
            if id_indices.numel() > 0:
                id_image = image[id_indices]
                id_output = self.model(id_image)
                # Map analytic columns into the existing TAIL class ordering.
                id_output_padded = torch.zeros_like(zs_output[id_indices])
                id_output_padded[:, learned_classes] = id_output.to(zs_output.dtype)
                output[id_indices] = ((1 - cfg.fusion_weight) * zs_output[id_indices]
                                      + cfg.fusion_weight * id_output_padded)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_wo_selector(self):
        '''
        RAIL ablation: fuse analytic scores for all images without selecting
        learned-domain images. Retain the original standalone evaluation loop.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader
        learned_classes = torch.tensor(self._learned_class_indices(), device=self.device)

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            zs_output = self.model.zero_shot(image).softmax(dim=-1)
            analytic_output = self.model(image)
            analytic_output_padded = torch.zeros_like(zs_output)
            analytic_output_padded[:, learned_classes] = analytic_output.to(zs_output.dtype)
            output = ((1 - cfg.fusion_weight) * zs_output
                      + cfg.fusion_weight * analytic_output_padded)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        checkpoint = {
            "format": "dual_RAIL_v1",
            "backbone": self.cfg.backbone,
            "gamma": self.model.gamma,
            "dual_regularization": self.model.regularization,
            "seen_tasks": self.seen_tasks,
            "seen_classnames": self.seen_classnames,
            "feature_memory": self.model.feature_memory,
            "target_memory": self.model.target_memory,
            "alpha": self.model.alpha,
        }
        path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"DualRAIL checkpoint saved to {path}")

    @torch.no_grad()
    def load_model4CL(self, directory):
        path = os.path.join(directory, "checkpoint.pth.tar")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format") != "dual_RAIL_v1":
            raise ValueError("Expected a DualRAIL checkpoint; primal RAIL checkpoints cannot resume dual training")
        for key, expected in (("backbone", self.cfg.backbone),
                              ("gamma", self.model.gamma),
                              ("dual_regularization", self.model.regularization)):
            if checkpoint[key] != expected:
                raise ValueError(f"Checkpoint {key} does not match configuration")
        tasks = checkpoint["seen_tasks"]
        if not tasks or tasks != list(self.cfg.dataset_sequence[:len(tasks)]):
            raise ValueError("Checkpoint tasks must be a prefix of dataset_sequence")
        count = self.indices[len(tasks)] if len(tasks) < len(self.indices) else len(self.merged_classnames)
        if checkpoint["seen_classnames"] != self.merged_classnames[:count]:
            raise ValueError("Checkpoint class ordering does not match the dataset sequence")
        features = checkpoint["feature_memory"]
        targets = checkpoint["target_memory"]
        alpha = checkpoint["alpha"]
        if (features.ndim != 2 or features.shape[1] != self.model.feature_dim
                or targets.shape != (features.shape[0],)
                or alpha.shape != (features.shape[0], count)
                or targets.dtype != torch.long or targets.numel() == 0
                or targets.min() < 0 or targets.max() >= count):
            raise ValueError("Checkpoint kernel state has incompatible dimensions or labels")
        self.model.feature_memory = features
        self.model.target_memory = targets
        self.model.alpha = alpha
        self.seen_tasks = list(tasks)
        self.seen_classnames = list(checkpoint["seen_classnames"])
        print(f"Resumed DualRAIL checkpoint with {count} classes from {path}")
