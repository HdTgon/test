# myMoRAM：保持 LADA 项目流程，只替换模型与训练配方

`main.py` 与 MaolinLuo/LADA 原文件逐字一致。沿用 LADA 的 YACS 配置、随机种子设置、日志文件命名、model_dir、zero-shot 分支和单任务 Trainer 调用。

- `scenario_datasets/`：LADA 原始数据流程。
- `models/`：仅 `models.py`，为已完成的 ZeroShotCLIP / MoRAM_CLIP 与自定义 LoRA。
- `clip/`：LADA CLIP，仅 FFN 前向增加 MoRAM tuple 解包。
- `trainer.py`：接入模型，并使用用户上传的 MoRAM 训练逻辑；不使用 LADA 的 Text_Tuner/DPT/LADA 分类分支、epoch/OneCycleLR。
- `result_process.py`：LADA 原始结果汇总；评估输出兼容其日志格式。

## 按你的实际实验脚本设置参数

`configs/model/moram_vit_b16.yaml` 默认对应你提供的 shell：

| 参数 | 默认值 |
|---|---|
| backbone / prec | CLIP-ViT-B/16 / fp32 |
| rank / topk | 16 / 16 |
| target_encoder | all |
| target_modules | q_proj, k_proj, v_proj, out_proj, ffn_in, ffn_out（qkvoinout） |
| lr / iterations | 0.0005 / 500 |
| temp / prune_thre | 0.01 / 0.2 |
| wd / ls / warmup_length | 0 / 0 / 0 |
| batch_size | 64（LADA 与原 MoRAM 默认一致） |

`method=moram` 是固定模型选择；`RAIL` 在你给的脚本中仅为实验目录名，不额外引入 RAIL 算法。数据、seed、gpu、num_workers 等保留 LADA 对应设置（seed=0，gpu=0，num_workers=8）。

## 与 LADA 相同的运行方式

先安装配套 torch/torchvision 与 requirements.txt，按上游 LADA 说明准备 X-TAIL 数据，将 `configs/data/TAIL.yaml` 和 `TAIL_order2.yaml` 的 root 改成你的数据根目录。

```bash
cd myMoRAM
pip install -r requirements.txt
bash scripts/run_TAIL_16shot.sh
```

和 LADA 一样每条 main 命令只训练一个 dataset，脚本依次调用十次：

```bash
# 第一任务：从 OpenAI backbone 开始，不读旧的持续学习 checkpoint。
python main.py -d TAIL -m moram_vit_b16 \
  num_shots 16 dataset aircraft continue_train_first True output_dir TAIL_16shot

# 第二任务：配置中 continue_train=True，自动读 output/TAIL_16shot/checkpoint.pth.tar。
python main.py -d TAIL -m moram_vit_b16 \
  num_shots 16 dataset caltech101 output_dir TAIL_16shot

# 十个任务结束后使用原 LADA 汇总方式。
python result_process.py -d TAIL --output_dir TAIL_16shot
```

root 可像 LADA 一样用 YACS overrides 覆盖：`root /absolute/path/to/data`。同一份本地 OpenAI 权重可通过 `checkpoint_path /absolute/path/to/ViT-B-16.pt` 指定，后续任务应保持相同配置。

其他脚本：`run_TAIL_16shot_order2.sh`、`run_TAIL_fullshot.sh`、`run_TAIL_fullshot_order2.sh`；仅替换模型名称并移除不再用于 MoRAM 的 num_epochs 参数，其余任务顺序与 LADA 一致。

```bash
bash scripts/run_TAIL_zeroshot.sh
# 等价于 python main.py -d TAIL -m zs_clip_vit_b16
```

## 训练保留 MoRAM

- 每任务 500 个 optimizer steps，数据耗尽后重建 iterator。
- AdamW：lr=5e-4、wd=0、betas=(0.9,0.9999)。
- 原 MoRAM cosine_lr，每步更新，支持 warmup_length。
- 任务切换时归档旧 LoRA、新建当前 A/B、冻结历史因子，再重建 optimizer。
- 历史文本特征每任务缓存；vision-only 缓存当前文本；text/all 每步重新编码当前文本。
- 标签加历史类别数，旧类别 image logits 设为 -inf。
- image CE 与上游相同的 text multi-hot 软目标 CE；vision/all 使用两项之和，text-only 使用 image CE。
- logit_scale 仅取一次指数；不引入 LADA 的训练损失、原型采样和调度器。

prec=fp32 与原 MoRAM 无 autocast 的训练保持一致；本实现拒绝其他训练精度。默认单进程选择一个 device，没有额外加入多 GPU 调度。多任务时要求 ls=0，避免 label smoothing 与旧类 -inf 屏蔽组合产生无限损失。

## Checkpoint 与评估

与 LADA 相同：通过 `model_dir/checkpoint.pth.tar` 读取前任务结果；main 未指定 model_dir 时使用当前 output_dir。模型与历史 LoRA、已完成任务、历史 prompts、配置一同保存。任务顺序和 LoRA 配置需一致；检查任务索引，防止误跳过历史训练。

checkpoint 仅支持任务边界恢复。可选 resume 字段用于显式指定 checkpoint 文件，但标准脚本不需要它。只加载可信本地 checkpoint。

评估严格保留 LADA 的 test_cur_task / test_zero_shot / test_wo_selector / test_w_selector 结构。不带 selector 时，已见类别使用每任务结束缓存的训练模型文本特征，未见类别使用冻结 CLIP 文本特征（与 LADA 相同），不再将全部类别用当前模型重新编码。输出 `* Task i Accuracy: ...%`，由 LADA result_process.py 解析。非当前任务内的局部候选类别评估。

## 验证

```bash
cd myMoRAM
PYTHONDONTWRITEBYTECODE=1 /workspace/moram-venv/bin/python -m unittest discover -s tests -v
```

4 项 CPU 集成测试通过：实际参数配置与 LADA 入口约定、三种编码器模式的真实更新、两任务独立启动/自动 checkpoint 衔接/再次恢复/zero-shot，以及 MoRAM optimizer、cosine schedule 和上传损失公式；同时执行当前任务、zero-shot、带/不带 selector 评估。测试使用小型随机 CLIP、内存 PIL 图片及真实 LADA DataLoader/增强流程。

未验证正式权重、真实 X-TAIL、GPU 或论文精度；不能据此判断实际 zero-shot gap 已消除。

来源：LADA `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`；MoRAM `aaa51d8ec67171e1bc5e1a1168351588561256c4`，以及用户上传训练逻辑。


## trainer.py 与 LADA 原版的必要差异

Trainer 的方法集合与 LADA 原版完全相同，没有另建 current_loader/train_task/evaluate 等训练控制接口。

- `__init__`：保持设备、数据/模型构建、Evaluator 初始化顺序；补 MoRAM 历史状态与配方检查。
- `build_data_loader`、`test_zero_shot`：与 LADA 原版方法完全一致。
- `load_clip_to_cpu`：仍按 LADA 构建；增加本地 checkpoint 路径并修复普通 state_dict 上错误调用 eval 的问题。
- `build_model`：PeftModelFromCLIP 替换成 MoRAM_CLIP；历史 A/B 恢复/递增替代 DPT/LADA 构建。
- `build_optimizer` / `build_criterion` / `train`：只改 MoRAM 必需的参数组、AdamW、cosine、固定迭代数和双向损失；保留 LADA 的 epoch/batch 骨架、AverageMeter、ETA、训练后保存及评估顺序。epoch 数根据 iteration cap 推导，不使用 cfg.num_epochs 改写 MoRAM 的总步数。
- 其余三个测试方法：保留 LADA 流程，只移除 text_tuner、lada_features、classifier 等已不存在的参数，文本原型改为 Trainer 缓存。
- `save_model` / `load_model4CL`：同名、同 checkpoint 路径，内容替换为 MoRAM 模型/历史因子与文本缓存。不兼容上一版开发中的 checkpoint 格式，需重新开始实验。

重新使用原 LADA `utils/evaluator.py`；只给 task selection precision 的零分母增加保护，未改变有预测样本时的计算。checkpoint 保持任务边界恢复，不额外承诺恢复随机状态或迭代中间恢复。

## Zero-shot 类别原型缓存

`Trainer.load_or_extract_zero_shot_prototypes(directory, force=False)` 在构建训练模型时自动调用。它使用**未注入 LoRA 的原始 CLIP**，从 `train_loader4updating` 提取特征（训练 split、确定性 test transform，不读取测试集图像作为训练原型）。

默认缓存文件：`output/<run>/zero_shot_prototypes/<dataset>.pth`。可设置 `zero_shot_prototype_dir /path/to/cache` 来跨实验共享，实际文件位于该目录的 `zero_shot_prototypes/` 下。

每类图像先做单位归一化，再按类别求均值，最后再归一化，得到 `[num_classes, embedding_dim]` 的视觉原型。缓存内容为：

- `image_prototypes`：上述 frozen zero-shot 视觉原型。
- `text_prototypes`：冻结 CLIP 使用统一类别模板的单位文本原型。
- `counts`：每类样本数量。
- `metadata`：类别排列、样本来源、权重摘要、分辨率等，用于检查缓存是否匹配。

每次先读缓存，匹配则不再跑视觉编码；不匹配时重新提取。需要手动重提取可调用：

```python
trainer.load_or_extract_zero_shot_prototypes(cfg.output_dir, force=True)
```

提取会恢复 CPU 随机状态，避免“命中缓存/重新提取”改变后续训练的 shuffle 随机序列。视觉原型来自 few-shot 设置选中的训练样本，不是 GMM 中心；不生成无意义的 covariance、lada_features 或 classifier。

`save_model/load_model4CL` 同步保存/恢复累计的 `image_prototypes`、`zero_shot_text_prototypes`、`prototype_counts`，顺序与已完成任务及其 classnames 一致。checkpoint 的 `text_prototypes` 仍为 LADA 评估流程使用的**训练后文本缓存**，与冻结参考文本原型分开。

这些 zero-shot 原型仅作为对照保存，不参与 MoRAM replay 或损失；模型 checkpoint 仍需保存实际 LoRA 和历史因子，保证持续学习。没有这些原型字段的旧开发 checkpoint 需要从第一任务重新开始。

现有 5 项 CPU 测试通过，新增验证缓存命中不做图像编码、手工均值结果一致、强制重算、元数据变化失效、随机状态恢复，以及两任务原型累计与 checkpoint 加载。
