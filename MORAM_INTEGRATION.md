# LADA CLIP 与 MoRAM 融合

## 源码来源

本目录导入 MaolinLuo/LADA，提交 `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`，保留上游 LICENSE。未复制上游 `.git`，test 的 origin 不变。

## 构建调用链

- `trainer.py:load_clip_to_cpu`：下载 OpenAI 权重，调用 `clip/model.py:build_model`，根据配置调整精度。
- `clip/model.py:build_model`：从 checkpoint 张量尺寸推断 ViT/ResNet 与文本 Transformer 结构，实例化 CLIP，转换权重精度并加载参数。
- `models/models.py:PeftModelFromCLIP`：组合图像编码器、文本调优器、DPT 原型以及 LADA 分类分支。这部分是上层学习算法，不是 CLIP backbone 构建本身。
- 新增 `models/moram_clip.py:get_moram`：使用同一套本地 `clip.load` / `build_model`，随后套用上传的 MoRAM 封装并注入 MoRAM 层。原 trainer 仍使用上游入口，没有自动切换算法。

## 使用

先安装 README 的运行依赖。导入 `models.moram_clip` 还会执行上游 `models/__init__.py`，因此需要完整依赖。

```python
from models.moram_clip import get_moram
import your_moram_layers as lora  # 必须替换为你实际的 MoRAM_Linear 模块

cfg = {
    'backbone_type': 'ViT-B/16',
    'pretrained_weight': 'openai',
    'zero_shot': False,
    'target_encoder': 'all',
    'target_modules': ['q_proj', 'v_proj', 'ffn_in'],
    'rank': 8,
    'device': 'cuda',  # 可用 cpu 做小规模检查
}
model, train_transform, test_transform, tokenizer = get_moram(cfg, lora_module=lora)
image_features, image_regularization = model.encode_image(images)
text_features, text_regularization = model.encode_text(tokenizer(texts).to(cfg['device']))
```

两个 transform 均为 LADA 的确定性 CLIP 预处理，并非 OpenCLIP 的随机训练增强。`checkpoint_path` 可指定本地 OpenAI 格式 checkpoint；不支持直接使用其他 OpenCLIP 权重标签。

修复了上传封装中 OpenAI CLIP 缺失属性、文本 forward 返回 tuple、单模态 forward 解包、patchnorm 属性拼写、attention 替换设备/精度及 train 返回值等兼容问题。保留 encode_image / encode_text 的 `(features, loss)` 协议，CLIP forward 返回特征张量。

## 验证与限制

```sh
/workspace/moram-venv/bin/python -m unittest discover -s tests -v
```

测试用随机小型 CLIP，不下载预训练权重，检查原始编码数值一致、文本因果 mask、单/双模态以及反向传播。

附件没有自定义 `MoRAM_Linear`，所以尚不能验证实际 MoRAM 路由、正则化与训练。需要提供该模块才能完成这部分。上传的 attention 实现主要用于 CLIP 的三维自注意力，不是通用 MultiheadAttention 替代品（例如 key_padding_mask、need_weights 尚未实现）。未验证数据集训练、GPU 和预训练权重下载。


后续已导入完整 MoRAM 并完成独立模型对齐，见 `MoRAM/CLIP_ALIGNMENT.md`。请使用 `MoRAM/XTAIL/models/moram.py`；本文件所述的旧融合入口保留作先前版本。
