# MoRAM 模型对齐 LADA

## 文件位置

- 原 LADA：`test/` 根目录。
- 导入的 MoRAM：`test/MoRAM/`，来自 Artificer-AI-Lab/MoRAM 的提交 `aaa51d8ec67171e1bc5e1a1168351588561256c4`，保留 LICENSE。
- 本次修复的 X-TAIL 模型：`MoRAM/XTAIL/models/moram.py`。
- 原样复制的 LADA CLIP：`MoRAM/XTAIL/lada_clip/`，来源 LADA `c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`。
- `TRACE/` 只是上游原始代码，本次未改其模型。

## 接入你已经修改的 main

独立目录中的 main 是 MoRAM 上游版本，**不含你本地已经完成的数据集修改**。把本目录的 `models/moram.py`、`lada_clip/` 放到你修改过的 MoRAM XTAIL 项目对应位置即可；`loralib/` 已随完整 MoRAM 导入，保持上游实现。保留你自己的 main 和 scenario_datasets。loader 名称及四项返回值不变，不需要改 factory。

新入口直接使用本地 loralib，不再需要上一轮 `test/models/moram_clip.py` 的外部模块注入。以本目录 `XTAIL/models/moram.py` 为准。

## 已统一的模型路径

1. 移除模型构建中的 OpenCLIP factory，使用 LADA 原版 OpenAI CLIP `load` / `build_model`，包括 QuickGELU、LayerNorm、EOT pooling、因果 attention mask、图像 CLS pooling。
2. `ViT-B-16` 映射为 `ViT-B/16`。`pretrained_weight` 必须是 `openai`；其他权重标签报错，避免无声切换权重。官方模型下载继续检查原始 SHA256。
3. 增加 `checkpoint_path`，可让两个项目读取**同一个实际 checkpoint 文件**。本地文件的 SHA256 可由验证脚本打印，便于记录实验。
4. 默认 `prec=fp32`；`prec=amp` 同样加载 fp32 参数，符合 LADA trainer 的加载行为。AMP 运算上下文仍由调用者控制。`prec=fp16` 使用原 CLIP convert_weights。两边必须用相同精度和 autocast 设置才能比较。
5. zero-shot 不注入 LoRA，用轻量适配器保持 MoRAM 原有 feature forward API；编码实际执行 LADA 原始模型。非 zero-shot 注入上游真实 MoRAM_Linear，保留路由算法和训练函数。
6. 修复原封装的单模态解包、text_features tuple、OpenAI CLIP 属性兼容、替换 attention 的设备/精度、batch_first 返回布局等问题。LoRA attention 仍为分离 Q/K/V 的 SDPA 实现，浮点数最后几位在其他设备上可能存在误差。

返回的 train/test transform 都使用 LADA 的标准确定性 CLIP 预处理。你自己的 main 如果已接入 LADA 数据增强，应继续使用你已统一的数据流水线。

## 验证

当前实例独立 CPU 验证环境：`/workspace/moram-venv`。Python 3.12、torch 2.15.0+cpu、torchvision 0.29.2+cpu；未在上游 README 固定版本或 GPU 上验证。

```bash
cd /workspace/test/MoRAM/XTAIL
/workspace/moram-venv/bin/python -m unittest discover -s tests -v
```

4 项通过：同 checkpoint 权重精确相等、图像/文本/logits 数值对比、真实 MoRAM attention + FFN 初始零增量以及 LoRA 反向传播、EOT/因果 mask、tokenizer/预处理源码一致性。使用随机小型 CLIP checkpoint，不是正式预训练模型。

另已实际运行校验脚本：zero-shot 和初始 MoRAM 相对参考模型的 image/text/logits 最大绝对误差均为 **0.0**（CPU、小型 checkpoint）。这不代表实际数据集准确率已验证。

使用你实验的真实权重再检查：

```bash
cd /workspace/test/MoRAM/XTAIL
python verify_clip_parity.py --checkpoint /absolute/path/ViT-B-16.pt --device cuda
```

脚本从同一文件分别加载参考 backbone、zero-shot 和初始 MoRAM，输出文件 SHA256 和特征/logits 最大误差；失败会返回非零状态。验证的是初始 LoRA，不会加载已训练 adapter，也不需要数据集。`prec` 在脚本中固定 fp32，确保比较条件一致。

## 剩余 gap 的定位边界

原项目 main 使用数据集各自模板；LADA zero-shot 使用统一 `a photo of a {}.`，并把类别名下划线替换为空格。即使模型完全一致，prompt、类别排列、候选类别集合（跨任务或单任务）、归一化、图像像素、eval/autocast 条件不同仍会影响准确率。此次未修改你的数据或评估逻辑，也未断言这些是你剩余 gap 的原因。

正式 OpenAI 权重、X-TAIL 数据集准确率、GPU/混合精度及持续学习完整训练未运行；你的修改版 main 尚未提供，所以不能直接复现其 gap。旧 OpenCLIP 格式的整模型 checkpoint 不保证可直接载入；本次针对共同 OpenAI backbone 与初始 LoRA 对齐。
