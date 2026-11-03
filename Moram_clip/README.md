# LADA CLIP + MoRAM 自定义 LoRA

CLIP 完整使用 MaolinLuo/LADA 的 `clip/`（提交 `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`），**原始 model.py 未修改**。

只从 Artificer-AI-Lab/MoRAM 提取以下部分（提交 `aaa51d8ec67171e1bc5e1a1168351588561256c4`）：

- `XTAIL/loralib/moram.py` → `moram_lora.py`：MoRAM_Linear 的 A/B 参数、归一化投影、top-k softmax 路由、eval 阈值、历史任务因子。
- `XTAIL/models/moram.py` → `injection.py`：只提取 PlainMultiheadAttentionLoRA，以及按 q/k/v/out、FFN 目标注入的逻辑。

不引入 MoRAM 的 CLIPModel、VisionTransformer、Transformer、ResidualAttentionBlock 封装，也不引入 DeFT tuner 或训练器。

## 直接使用

在 `test` 根目录运行，或将整个 `Moram_clip` 文件夹复制到你的项目：

```python
from Moram_clip import get_moram

cfg = {
    'backbone_type': 'ViT-B/16',  # 兼容 ViT-B-16
    'pretrained_weight': 'openai',
    # 'checkpoint_path': '/absolute/path/ViT-B-16.pt',
    'prec': 'fp32',
    'device': 'cuda',             # 也支持 cpu
    'target_encoder': 'all',      # vision / text / all
    'target_modules': ['q_proj', 'v_proj', 'ffn_in', 'ffn_out'],
    'rank': 4,
    'topk': 2,
    'temp': 0.01,
    'prune_thre': 1.0,
}

frozen_clip, _, test_transform, tokenizer = get_moram(cfg, zero_shot=True)
moram_clip, train_transform, _, _ = get_moram(cfg)
moram_clip.train()  # loader 默认 eval；训练前显式切换

# images: [N,3,H,W]，tokens: tokenizer(prompts).to(cfg['device'])
image_features = moram_clip.encode_image(images)  # Tensor，未归一化
text_features = moram_clip.encode_text(tokens)   # Tensor，未归一化
logits_per_image, logits_per_text = moram_clip(images, tokens)
```

**接口直接保留 LADA 语义**，不返回 `(features, loss)`，不使用旧 MoRAM 的 feature-forward 三元组。若你的 main 沿用这些旧接口，需要相应改成上面的调用；本次未修改 main。

- `zero_shot=True`：完全原始的冻结 CLIP，不注入 MoRAM。
- 默认：载入同一 backbone 后注入，仅当前 `lora_A/lora_B` 可训练，其余参数冻结。两次 get_moram 创建独立模型。
- 支持 q_proj / k_proj / v_proj / out_proj / ffn_in / ffn_out，默认 q_proj + v_proj；不支持的目标直接报错。
- 使用包内 `.clip` 相对导入，不依赖根目录旧 clip 或 moram.py。
- `prec=amp` 加载 fp32 参数，不自动开启 autocast。训练和评估时由调用者控制。
- 两个 transform 都是原始 CLIP 确定性预处理；如需训练数据增强，由数据流水线另行提供。

## 已有 backbone 上注入

```python
from Moram_clip import apply_lora
from Moram_clip.clip import clip

backbone, preprocess = clip.load('/path/to/ViT-B-16.pt', device='cpu', jit=False)
model = apply_lora(backbone, cfg).to(cfg['device'])
```

这里原地修改传入模型。如果要另保留冻结参考，请先复制或分别加载；重复注入会报错。

## 持续学习

```python
from Moram_clip import increment_task
increment_task(moram_clip)
# 必须重建 optimizer：当前 A/B 是新创建的 Parameter。
optimizer = torch.optim.AdamW([p for p in moram_clip.parameters() if p.requires_grad], lr=1e-3)
```

历史 A/B 自动冻结；路由在新旧因子上联合进行，与上游逻辑相同。新任务即使 B 初始化为零，加入新 A 后也可能改变联合路由权重，不能假设切换任务前后输出相等。

## 最小兼容改动

- FFN 使用 MoRAMLinearTensor，取原 MoRAM `(output, 0)` 中的张量，让 LADA 原始 Sequential 无需改动。上游当前有效实现的正则 loss 始终是 0。
- attention 仍使用上游分离 Q/K/V + SDPA 的实现，恢复原 bias、设备、精度和文本因果 mask。禁止未实现的 padding mask / attention weights 请求。
- 修复 attention 的 batch_first 返回维度与 train 返回值。
- increment_task 去掉硬编码 CUDA；rank=0 可直接退化到普通线性层；top-k 不超过可用 rank。常规 rank/topk 配置的路由计算不变。

## 验证

依赖：torch、torchvision、numpy、Pillow、ftfy、regex、tqdm。没有 open_clip 或外部 loralib 依赖。

```bash
cd /workspace/test
/workspace/moram-venv/bin/python -m unittest discover -s Moram_clip/tests -v
```

当前实例 7 项 CPU 小型随机模型测试通过：原始 LADA 文件未改变、零增量图像/文本/logits 一致、文本因果 mask、只训练 LoRA、更新确实改变输出且 backbone 不变、目标选择、冻结与训练模型独立、任务递增及 rank/精度兼容。

还独立对照了上游 MoRAM_Linear：非零 A/B，在 train/eval、有/无历史因子四种组合下输出逐元素相同。

尚未验证正式预训练权重、实际数据集准确率和 GPU。该适配面向 LADA ViT 的三维自注意力，不是通用 MultiheadAttention 的完整替代。已训练旧 MoRAM 整模型 checkpoint 的参数路径与本包不同，不能直接假定互通；持续学习快照的自动恢复不在本轮实现范围。
