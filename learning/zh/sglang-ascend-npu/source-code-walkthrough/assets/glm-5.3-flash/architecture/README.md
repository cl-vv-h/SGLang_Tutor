# GLM-5.3-Flash 架构图资源

**简体中文** | [English](../../../../../../en/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/README.md)

这些图配套[端到端教程第2.3节](../../../examples/03-glm-5.3-flash-end-to-end.md#23-全模型数据流)，由 [Archify](https://github.com/tt-a1i/archify) 生成。每张图同时保留可编辑 JSON、自包含交互 HTML 和正文直接显示的 PNG 与矢量 SVG。HTML 需下载后用浏览器打开；GitHub 的代码页面不会执行其中的交互。SVG 来自 Archify 的原生整图导出，不包含当前节点聚焦等临时界面状态。

| 顺序 | 静态 PNG | 矢量 SVG | 交互图 | 源码 |
|---|---|---|---|---|
| 1 | [完整模型与45层排列](01-model-overview.png) | [SVG](01-model-overview.svg) | [HTML](01-model-overview.html) | [JSON](01-model-overview.architecture.json) |
| 2 | [一个 Decoder 层的完整结构](02-decoder-layer.png) | [SVG](02-decoder-layer.svg) | [HTML](02-decoder-layer.html) | [JSON](02-decoder-layer.architecture.json) |
| 3 | [KDA 层内部：投影、卷积、门与状态](03-kda-internals.png) | [SVG](03-kda-internals.svg) | [HTML](03-kda-internals.html) | [JSON](03-kda-internals.architecture.json) |
| 4 | [DSA 层内部：KPool 检索与 NoPE MLA](04-dsa-mla-internals.png) | [SVG](04-dsa-mla-internals.svg) | [HTML](04-dsa-mla-internals.html) | [JSON](04-dsa-mla-internals.architecture.json) |
| 5 | [层内 FFN：dense 与 MoE 两种选择](05-ffn-moe-internals.png) | [SVG](05-ffn-moe-internals.svg) | [HTML](05-ffn-moe-internals.html) | [JSON](05-ffn-moe-internals.architecture.json) |
| 6 | [视觉编码器与 Vision block 内部结构](06-vision-internals.png) | [SVG](06-vision-internals.svg) | [HTML](06-vision-internals.html) | [JSON](06-vision-internals.architecture.json) |

## 固定基线与阅读约定

- SGLang：`92580581da6fc8542a997a00c1a8b71b03497680`；节点携带该 revision 下的源码路径和行号。
- 模型配置：`zai-org/GLM-5.3-Flash`，revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`。
- Archify：3.2.0，skill 仓库 revision `d6059331abbddb49bca380ff46dca8ce33158b04`。
- 图中宽度、head 数与 state 表示完整逻辑模型，不是正文 TP=8 的单 rank shape。箭头表示数据或状态依赖，不等同于 kernel launch 或多流调度。
- 只有45个 target Decoder 层。MTP 是独立可选分支；Vision block 的普通残差也不要当成四路 mHC。

## 怎样维护

修改对应的 `.architecture.json` 后，用固定版本 Archify 和相同 revision 的干净 SGLang checkout 重新生成；不要直接修改生成的 HTML。以下以中文版总图为例，在教程仓库根目录执行，把两个 `/path/to/` 路径替换成实际路径：

```bash
node /path/to/archify/archify/bin/archify.mjs finalize architecture \
  learning/zh/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/01-model-overview.architecture.json \
  learning/zh/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/01-model-overview.html \
  --repo-root /path/to/sglang-at-pinned-revision --quality showcase --json
```

生成通过后，在 HTML 的“导出 → SVG · 自动”和“导出 → PNG”保存完整静态图，再替换同名 SVG 与 PNG。同步修改英文版，检查图中数据来源、可选分支、数值、连线与中英文拓扑一致。`manifest.json` 记录发布资源的 SHA-256 和工具检查结果；资源变化后应重新生成记录，旧记录不能作为新文件的验证证据。工具自动检查不等于真实模型在 GPU/NPU 上执行或性能验证。

HTML 内嵌的 Archify viewer 与字体使用声明保存在 [licenses/NOTICE](./licenses/NOTICE) 和相邻许可证文件中；图中模型内容由本教程撰写。
