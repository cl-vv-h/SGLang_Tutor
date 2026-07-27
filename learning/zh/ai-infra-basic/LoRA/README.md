# LoRA 系列教学入口

这个目录用小型 Transformer 和简化 Linear 模块解释参数高效微调。阅读目标是理解 adapter 如何插入模型、哪些参数参与训练、推理时如何合并，以及 QLoRA/DoRA/AdaLoRA 分别解决什么问题。

## 文件说明

| 文件 | 主题 | 核心问题 |
|---|---|---|
| [lora_tutorial.py](./lora_tutorial.py) | LoRA 基础实现 | 低秩矩阵 A/B 如何注入 Linear，如何 freeze base model，如何 merge/unmerge |
| [qlora_tutorial.py](./qlora_tutorial.py) | QLoRA 教学实现 | 4-bit base weight 与 LoRA 分支如何一起训练，量化 buffer 如何组织 |
| [dora_tutorial.py](./dora_tutorial.py) | DoRA 教学实现 | 权重方向和 magnitude 拆分后，adapter 学的是什么 |
| [adalora_tutorial.py](./adalora_tutorial.py) | AdaLoRA 教学实现 | 如何根据重要性动态分配 rank，把有限参数预算用在更重要的模块上 |

## 推荐阅读顺序

1. 先读 `lora_tutorial.py`，抓住 `LoRALinear`、参数冻结、模块替换、训练和 merge 的主流程。
2. 再读 `qlora_tutorial.py`，理解 base 权重量化后为什么仍然可以只训练 adapter。
3. 然后读 `dora_tutorial.py`，比较 DoRA 和 LoRA 在权重分解目标上的差异。
4. 最后读 `adalora_tutorial.py`，理解 rank 不一定要固定，adapter 参数预算可以动态分配。

## 和 SGLang 的连接点

- LoRA serving 不只关心训练，还关心线上 adapter 的注册、加载、卸载和混批约束。
- 多 LoRA batching 需要在同一个 batch 中为不同请求选择不同 adapter。
- 推理 kernel 需要把 base output 和 LoRA delta 高效组合，避免 adapter 数量增加后吞吐快速下降。
- QLoRA 与量化相关，适合理解低显存微调和量化权重服务之间的边界。

## 运行示例

```bash
python learning/ai-infra-basic/LoRA/lora_tutorial.py
python learning/ai-infra-basic/LoRA/qlora_tutorial.py
python learning/ai-infra-basic/LoRA/dora_tutorial.py
python learning/ai-infra-basic/LoRA/adalora_tutorial.py
```
