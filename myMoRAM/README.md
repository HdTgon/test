# myMoRAM：LADA 数据与项目结构 + MoRAM 模型和训练

## 组成

- LADA：`main.py` 的配置/日志方式、`scenario_datasets/` 的数据划分、few-shot 采样和全序列评估标签偏移、图像增强、CLIP 权重与 tokenizer。
- `models/` **只有 `models.py`**：上一轮完成的 ZeroShotCLIP、MoRAM_CLIP、自定义 routed LoRA、attention/FFN 注入、任务递增。
- `clip/model.py` 只增加 FFN tuple 解包；无 LoRA 时与原版输出一致。
- `trainer.py`：按照用户上传的 MoRAM 训练实现接入上述模型，没有 LADA 的 Text_Tuner、DPT、原型采样或 LADA 分类分支。

来源：LADA `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`；MoRAM `aaa51d8ec67171e1bc5e1a1168351588561256c4`；同时参考本次上传的训练代码。

## 安装与运行

使用配套版本的 torch/torchvision（原 MoRAM 为 2.7.1 / 0.22.1；按机器 CUDA 配置安装）。`requirements.txt` 提供其余依赖及包名，不固定硬件 wheel。

```bash
cd myMoRAM
pip install -r requirements.txt
python main.py -d TAIL -m moram_vit_b16 \
  root /absolute/path/to/X-TAIL output_dir moram_16shot
```

数据准备、目录和 LADA 完全一致，见上游 LADA README 的 X-TAIL 数据说明。默认从 aircraft 到 sun397 依次训练，**一次启动完成全序列**，不再用 LADA 的每任务 epoch 脚本。

使用同一份本地 OpenAI 权重：

```bash
python main.py -d TAIL -m moram_vit_b16 \
  root /path/to/data checkpoint_path /path/to/ViT-B-16.pt \
  output_dir experiment1
```

纯冻结 zero-shot：

```bash
python main.py -d TAIL -m moram_vit_b16 \
  root /path/to/data zero_shot True output_dir baseline
```

第二种顺序使用 `-d TAIL_order2`；full-shot 追加 `num_shots -1`。

常用覆盖项：`target_encoder vision`、`target_modules "['q_proj','v_proj','ffn_in','ffn_out']"`、`iterations 1000`、`rank 4`、`topk 4`、`lr 0.001`、`device cuda:0`。CPU 调试使用 `device cpu num_workers 0`。

## 训练逻辑保持 MoRAM

- 每任务固定 `iterations`；数据耗尽后重新迭代，不使用 LADA epochs。
- AdamW：`lr=cfg.lr`、`weight_decay=cfg.wd`、`betas=(0.9,0.9999)`。
- 原版 `cosine_lr`：每步更新，支持线性 warmup。
- 第一任务直接训练；后续任务归档旧 A/B、新建零增量因子、冻结历史因子，并在此后重建优化器。
- 历史文本特征每任务开始缓存一次；vision-only 的当前文本也缓存；text/all 每步重新编码当前文本，保留梯度。
- 训练标签加历史类别数；旧类别的 image-to-text logits 屏蔽为 `-inf`。
- image CE：可配置 label smoothing；text CE 使用上游相同的类别-样本 multi-hot 软目标（**没有额外按类别频数归一化**）。
- vision/all 使用 `CE_i + CE_t`；text-only 使用 `CE_i`。
- wrapper.logit_scale 已是 exp 倍率，训练器不会再次 exp。

默认 `prec=fp32`，与上传的 MoRAM 无 autocast/scaler 训练一致；本版明确拒绝其他精度，不暗中改用 LADA AMP。单进程使用选定 device；未加入上游无效的 `.module.encode_*` 式 DataParallel 包装，不声称多 GPU 支持。

非零 label smoothing 与后续任务 -inf 屏蔽不兼容；多任务要求 `ls=0`（上游默认即为 0），否则明确报错。

## 评估与输出

沿用 LADA 的全序列类别排列、全序列测试集、统一模板 `a photo of a {}.` 和类别名下划线替换。每个任务后重新编码全部类别，打印每任务准确率及平均准确率。候选类别始终是完整序列，所以它与仅在当前任务内评估的准确率不可直接比较。

- `output/<run>/log_sequence.txt`：LADA Logger 记录控制台。
- `accuracies.csv`：每行对应已完成的一个训练任务，每列对应测试任务。
- `<dataset>.pth`：模型及已完成任务、历史 prompts、配置、随机状态的任务边界 checkpoint。

日志使用 LADA 控制台/文件方式，没有强制 TensorBoard 依赖。

## 恢复

```bash
python main.py -d TAIL -m moram_vit_b16 \
  root /path/to/data checkpoint_path /path/to/ViT-B-16.pt \
  resume output/experiment1/aircraft.pth output_dir experiment1
```

恢复后自动重建历史 LoRA 因子，再严格加载权重，从下一任务继续；不是从头重训。支持**任务边界恢复**，不支持任务中间迭代恢复。checkpoint 含 Python 随机状态，仅加载你信任的本地文件。

`dataset aircraft` 可仅跑第一任务。运行后续单任务需指定前一个任务的 resume，否则报错，防止跳过历史任务。架构、任务顺序、模板和 LoRA 配置必须与 checkpoint 一致。可以在已完成的 checkpoint 上继续 evaluate；换机器/库版本不保证逐位可复现。

## 验证

```bash
cd myMoRAM
PYTHONDONTWRITEBYTECODE=1 /workspace/moram-venv/bin/python -m unittest discover -s tests -v
```

当前 CPU 集成测试使用随机小型 CLIP 和内存 PIL 图像，通过**真实 LADA DatasetWrapper、DataLoader、增强、全任务标签偏移**运行；不下载权重或真实数据。3 项集成测试覆盖：

1. vision/text/all 训练与实际参数更新。
2. 两任务训练、任务递增、每任务评估、保存、首次恢复继续第二任务、再次恢复检查所有权重，以及冻结 zero-shot。
3. MoRAM optimizer betas、warmup/cosine 数值、上传代码的损失公式一致性。

正式 OpenAI 权重、真实 X-TAIL、GPU 和论文指标尚未运行，因此不能据此判断实际 zero-shot gap 或训练精度已经消除。
