# myMoRAM

以 LADA 项目为基础，只替换模型及 MoRAM 必需的训练逻辑。

- `main.py` 与 LADA 原版一致。
- `scenario_datasets/` 保留 LADA 的数据流程。
- `models/` 仅 `models.py`，包含 ZeroShotCLIP、MoRAM_CLIP 和 routed LoRA。
- `clip/` 基于 LADA，FFN 前向增加 MoRAM tuple 解包。
- `trainer.py` 保留 LADA 的方法结构、数据加载、日志及测试流程；模型、优化器、调度器、损失改用 MoRAM。
- 不提取、不缓存、不保存任何视觉/文本类别原型，没有 DPT、GMM 或 LADA 分支。

## 参数

`configs/model/moram_vit_b16.yaml` 对应用户实际实验脚本：

```yaml
backbone: "CLIP-ViT-B/16"
prec: "fp32"
iterations: 500
lr: 0.0005
wd: 0.0
ls: 0.0
warmup_length: 0
target_encoder: "all"
target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "ffn_in", "ffn_out"]
rank: 16
topk: 16
temp: 0.01
prune_thre: 0.2
```

batch_size=64；数据、seed、gpu、num_workers 等沿用 LADA 配置。qkvoinout 已展开为上述六种目标。

## 运行

安装配套 torch/torchvision 和 requirements.txt，按 LADA 说明准备 X-TAIL 数据，将 configs/data 中的 root 设置为数据目录。

```bash
cd myMoRAM
pip install -r requirements.txt
bash scripts/run_TAIL_16shot.sh
```

和 LADA 一样，每次 main 只运行一个 dataset：

```bash
python main.py -d TAIL -m moram_vit_b16 \
  dataset aircraft continue_train_first True output_dir TAIL_16shot
python main.py -d TAIL -m moram_vit_b16 \
  dataset caltech101 output_dir TAIL_16shot
python result_process.py -d TAIL --output_dir TAIL_16shot
```

root 和本地 backbone 可用 `root /path/to/data checkpoint_path /path/to/ViT-B-16.pt` 覆盖。提供 order2、fullshot 和 zero-shot 脚本；冻结 zero-shot 使用 `python main.py -d TAIL -m zs_clip_vit_b16`。

## 训练与评估

MoRAM 使用固定迭代、AdamW（betas=(0.9,0.9999)）、原版 warmup/cosine、旧类别 logits 屏蔽及 image/text 双向 CE。vision/all 使用两项之和，text-only 使用 image CE。fp32 无 autocast；多任务使用 ls=0 避免与 -inf 屏蔽冲突。

历史类别与 prompts 从 dataset_sequence 和 LADA 类别排列推导，不依赖 checkpoint 元数据。历史文本只在训练当前任务时临时编码缓存；评估按需用当前训练模型编码已见类别，用冻结 CLIP 编码未见类别，不存储文本原型。logit_scale 只取一次指数。

AverageMeter、ETA、当前任务/zero-shot/带和不带 selector 的评估与 LADA 方法结构保持一致。Evaluator 仅额外保护无预测任务的 precision 零分母。

## 只保存 MoRAM

`utils/moram_utils.py` 来自上游 MoRAM XTAIL/utils.py；三个学习率函数与原版一致。Trainer 直接调用其中的调度和保存/加载函数，scheduler.py 仅作兼容 re-export。

```python
@torch.no_grad()
def save_model(self, directory):
    save_path = os.path.join(directory, 'checkpoint.pth.tar')
    torch_save_lora(self.model, save_path, MoRAM_Linear)

@torch.no_grad()
def load_model4CL(self, directory, load_path=None):
    load_path = load_path or os.path.join(directory, 'checkpoint.pth.tar')
    if not os.path.exists(load_path):
        raise FileNotFoundError(...)
    torch_load_lora(self.model, load_path, MoRAM_Linear)
```

Checkpoint **只有** `state_dict` 与 `lora_ranks`：不保存整个 CLIP、原型、配置、任务进度或 prompts。LoRA ranks 中包含当前/历史 rank、init_rank、lora_scaling；张量包含当前/历史 A/B 与 gate 等 MoRAM 状态。

恢复先读取相同的原始 CLIP 权重，再加载 MoRAM。保持同样任务顺序、backbone、目标层和超参；rank 检查只能校验历史数量，不能辨认相同长度但不同顺序的任务。后续任务通过 model_dir/checkpoint.pth.tar 加载，之后递增一次并重建 optimizer。加载保持设备/dtype，历史参数冻结。只支持任务边界衔接，不支持迭代中途恢复。

## 验证

```bash
cd myMoRAM
PYTHONDONTWRITEBYTECODE=1 /workspace/moram-venv/bin/python -m unittest discover -s tests -v
```

5 项 CPU 测试通过：三种编码器训练更新、两任务 LoRA-only 衔接及各评估路径、optimizer/schedule/loss 数值、非零历史/当前 LoRA 保存恢复与 rank/dtype/冻结、实际实验配置和 LADA 入口兼容。Checkpoint 测试验证其只有两个 MoRAM 字段。

测试用随机小型 CLIP、内存 PIL 图像和真实 LADA 数据加载流程。正式权重、真实 X-TAIL、GPU 和最终准确率尚未验证。
