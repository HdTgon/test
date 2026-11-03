# 基础 CLIP（DeFT，无 tuner）

来源：HotanLee/DeFT，提交 `254490a2b566e3a4fbcce937d116cf04558dabf2`。

- `CLIP.py`：保留 DeFT `CLIP_ViT` 的手写 attention、FFN、CLS pooling；外层补回图像 projection 和标准 CLIP 相似度计算。
- `text_encoder.py`：保留单路文本 Transformer、EOT pooling 和 text projection；直接接收 token IDs。
- `clip/`：DeFT 自带的底层模型定义、checkpoint 构建、BPE tokenizer 和标准图像预处理，原样复制。
- 未包含 `load_clip.py`。`peft_modelus.py` 的全部类均用于调优，移除后不再需要该文件。
- 无 ViT_Tuner、PromptLearner、VPT、Adapter、LoRA、SSF 或额外分类头。仅支持 DeFT 该视觉流程对应的 ViT backbone。

依赖：PyTorch、torchvision、numpy、Pillow、ftfy、regex、tqdm。

从仓库根目录使用：

```python
import torch
from CLIP import CLIP
from CLIP.clip import clip

# 可替换为本地 OpenAI 格式权重路径。使用本目录 DeFT 的底层 loader，
# 不使用 DeFT 的 load_clip.py（其包含 tuner、冻结与优化器逻辑）。
backbone, preprocess = clip.load('ViT-B/16', device='cpu', jit=False)
model = CLIP(backbone).float().to('cuda').eval()
tokens = clip.tokenize(['a photo of a cat.', 'a photo of a dog.']).to('cuda')
# image 是 preprocess(PIL_image).unsqueeze(0).to('cuda')
with torch.no_grad():
    image_features = model.encode_image(image)  # Tensor，未归一化
    text_features = model.encode_text(tokens)   # Tensor，未归一化
    logits_per_image, logits_per_text = model(image, tokens)
```

CPU 推理把上述 `'cuda'` 换成 `'cpu'`。默认不主动冻结参数，`eval()` 只切换运行模式；需要冻结时使用 `model.requires_grad_(False)`。

本轮仅增加此目录，尚未将根目录旧 `moram.py` 切换到本实现。

验证：4 项 CPU 小型随机 checkpoint 检查通过，覆盖与 DeFT 原版 `tuner=None` 图像路径及单 prompt 文本路径逐元素一致、标准 CLIP logits、反向传播、冻结、精度切换及本地权重加载/预处理。未下载正式预训练权重，未验证数据集准确率或 GPU。
