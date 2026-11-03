# LADA 结构的 ZeroShotCLIP 与 MoRAM_CLIP

`models.py` 是唯一的模型/LoRA 实现文件，已合并原 `moram_lora.py` 与 `injection.py`，旧文件已删除。

文件按 LADA `models/models.py` 的思路组织：

- `ZeroShotCLIP(clip_model: CLIP)`：冻结参考。保留类别名下划线替换、逐模板归一化、模板均值再归一化及图文相似度分类流程。文本编码直接调用原始 LADA CLIP，不依赖 Peft_Text / Text_Tuner。
- `MoRAM_CLIP(cfg, clip_model: CLIP)`：替代 PeftModelFromCLIP；只注入 MoRAM LoRA，没有 DPT、LADA 分类分支或 Text_Tuner。
- 同一文件包含 `MoRAM_Linear`、`PlainMultiheadAttentionLoRA`、`apply_lora(model, cfg)`、冻结参数和任务递增逻辑。没有重复定义 CLIP、ViT 或 Transformer。
- 两个封装都复制传入 backbone，不改动调用方的 clip_model。`image_encoder` 是内部原始 visual；`clip_model` 是内部 backbone，包含全部文本编码参数。
- `clip/` 使用 LADA 实现，仅在 ResidualAttentionBlock.forward 中逐层执行 FFN，解包 MoRAM_Linear 的 `(features, loss)`。未注入时计算结果与原版逐元素一致。

## 已有 clip_model

```python
from Moram_clip.models import ZeroShotCLIP, MoRAM_CLIP

cfg = {
    'target_encoder': 'all',  # vision / text / all
    'target_modules': ['q_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'rank': 4,
    'topk': 2,
    'temp': 0.01,
    'prune_thre': 1.0,
}
reference = ZeroShotCLIP(clip_model).to(device).eval()
model = MoRAM_CLIP(cfg, clip_model).to(device)

reference.init_text_features(classnames, ['a photo of a {}.'], tokenize, device)
reference_logits = reference(images)

model.init_prompts(classnames, ['a photo of a {}.'], tokenize, device)
model.train()
logits = model(images)  # [batch, classes]，每次重新计算文本，允许文本 LoRA 梯度
# 也可显式输入 token IDs，或外部预先计算的文本特征：
logits = model(images, text=tokens)
logits = model(images, text_features=external_text_features)
```

`encode_image(images)` / `encode_text(tokens)` 仍返回未归一化特征张量，不是 `(features, loss)`。

**封装的 forward 按 LADA 分类接口返回单个 logits 张量，不再返回原始 CLIP 的双向 logits tuple。** `logit_scale` 属性返回已经 exp 的倍率，与 LADA ZeroShotCLIP 一致，不要再调用 `.exp()`。需要原始参数时读取 `model.clip_model.logit_scale`。

`ZeroShotCLIP` 的所有参数冻结；`MoRAM_CLIP` 仅当前 LoRA A/B 可训练。`init_text_features` 用于固定文本分类器，不跟踪梯度；训练文本 LoRA 时请使用 `init_prompts` 或显式 tokens，避免使用缓存文本特征。

## 配套加载入口

```python
from Moram_clip import get_moram

cfg.update(backbone_type='ViT-B/16', pretrained_weight='openai', prec='fp32', device='cuda')
# cfg['checkpoint_path'] = '/path/to/ViT-B-16.pt'
reference, train_transform, test_transform, tokenize = get_moram(cfg, zero_shot=True)
model, _, _, _ = get_moram(cfg)
```

loader 分别返回 ZeroShotCLIP / MoRAM_CLIP，默认 eval；训练前调用 model.train()。默认不从 cfg['zero_shot'] 推断模式，显式参数控制。`prec=amp` 加载 fp32 参数，autocast 由调用者控制。两个 transform 都是 CLIP 确定性预处理。

`apply_lora(model, cfg)` 保留低层原地注入接口，其第一个参数应是**原始 CLIP backbone**（例如 wrapper.clip_model）；一般直接用 MoRAM_CLIP 构造器即可。

## 持续学习

```python
from Moram_clip import increment_task
increment_task(model)
# 新 Parameter 已替换，必须重建 optimizer。
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
```

历史因子冻结。联合新旧 A 的路由可能在新任务初始化时改变，即使新 B 是零，也不保证切换任务前后输出相同。

## 来源与兼容处理

LADA `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`：原始 clip/ 与 ZeroShotCLIP 分类流程。
MoRAM `aaa51d8ec67171e1bc5e1a1168351588561256c4`：路由 LoRA 和 Q/K/V/out、FFN 注入。

仅修复设备/精度、attention 返回布局、rank=0/top-k 边界、CPU 任务递增；FFN 直接注入原始 MoRAM_Linear，在残差块中解包其返回值，不再使用额外的线性层子类。当前上游正则 loss 恒为 0，在这里丢弃。原始路由算法保留。attention 只覆盖 LADA 三维自注意力，不是通用 MultiheadAttention 替代品。

## 验证与限制

```bash
cd /workspace/test
/workspace/moram-venv/bin/python -m unittest discover -s Moram_clip/tests -v
```

9 项 CPU 小型随机模型测试通过，包括普通 FFN 与原版逐元素一致、直接注入 MoRAM_Linear 及 tuple 接口、初始 LoRA 输出一致、梯度/更新隔离、目标选择、任务递增、加载与封装、多模板分类及文本 LoRA 训练。正式预训练权重、GPU、数据集准确率未验证。旧版本 checkpoint 的包装参数路径改变，不能假定整模型 checkpoint 直接兼容。

依赖：torch、torchvision、numpy、Pillow、ftfy、regex、tqdm。无需外部 loralib 或 open_clip。
