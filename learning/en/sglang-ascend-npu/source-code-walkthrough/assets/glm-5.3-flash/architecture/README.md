# GLM-5.3-Flash Architecture Assets

**English** | [简体中文](../../../../../../zh/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/README.md)

These diagrams accompany [section 2.3 of the end-to-end walkthrough](../../../examples/03-glm-5.3-flash-end-to-end.md#23-whole-model-dataflow), generated with [Archify](https://github.com/tt-a1i/archify). Each retains editable JSON, self-contained interactive HTML, and a directly embedded PNG plus vector SVG. Download HTML and open it in a browser; GitHub's code page does not execute its interactions. SVGs use Archify's native full-diagram export without temporary viewer state such as node focus.

| Order | Static PNG | Vector SVG | Interactive | Source |
|---|---|---|---|---|
| 1 | [Complete model and the 45-layer order](01-model-overview.png) | [SVG](01-model-overview.svg) | [HTML](01-model-overview.html) | [JSON](01-model-overview.architecture.json) |
| 2 | [Complete structure of one Decoder layer](02-decoder-layer.png) | [SVG](02-decoder-layer.svg) | [HTML](02-decoder-layer.html) | [JSON](02-decoder-layer.architecture.json) |
| 3 | [Inside KDA: projections, convolution, gates, and state](03-kda-internals.png) | [SVG](03-kda-internals.svg) | [HTML](03-kda-internals.html) | [JSON](03-kda-internals.architecture.json) |
| 4 | [Inside DSA: KPool retrieval and NoPE MLA](04-dsa-mla-internals.png) | [SVG](04-dsa-mla-internals.svg) | [HTML](04-dsa-mla-internals.html) | [JSON](04-dsa-mla-internals.architecture.json) |
| 5 | [Inside the FFN: dense and MoE choices](05-ffn-moe-internals.png) | [SVG](05-ffn-moe-internals.svg) | [HTML](05-ffn-moe-internals.html) | [JSON](05-ffn-moe-internals.architecture.json) |
| 6 | [Vision encoder and Vision block internals](06-vision-internals.png) | [SVG](06-vision-internals.svg) | [HTML](06-vision-internals.html) | [JSON](06-vision-internals.architecture.json) |

## Pinned Baselines and Reading Conventions

- SGLang: `92580581da6fc8542a997a00c1a8b71b03497680`; nodes cite source paths and line ranges at that revision.
- Model configuration: `zai-org/GLM-5.3-Flash`, revision `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.
- Archify: 3.2.0, skill repository revision `d6059331abbddb49bca380ff46dca8ce33158b04`.
- Widths, head counts, and states denote the complete logical model, not the chapter's TP=8 per-rank shapes. Arrows show data or state dependencies, not kernel launches or stream scheduling.
- There are 45 target Decoder layers. MTP is a separate optional branch; ordinary Vision block residuals are not four-stream mHC.

## Maintenance

Edit the matching `.architecture.json`, then regenerate with the pinned Archify version and a clean SGLang checkout at the same revision. Do not edit generated HTML directly. From the tutorial repository root, replace the two `/path/to/` placeholders with actual paths:

```bash
node /path/to/archify/archify/bin/archify.mjs finalize architecture \
  learning/en/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/01-model-overview.architecture.json \
  learning/en/sglang-ascend-npu/source-code-walkthrough/assets/glm-5.3-flash/architecture/01-model-overview.html \
  --repo-root /path/to/sglang-at-pinned-revision --quality showcase --json
```

After successful generation, use “Export → SVG · Auto” and “Export → PNG” in the HTML to save full static diagrams and replace the matching SVG and PNG. Update Chinese too; check source evidence, optional branches, values, edges, and bilingual topology. `manifest.json` records published resource SHA-256 hashes and tool checks; regenerate it after changes. An old receipt is not evidence for new bytes. Automated artifact checks do not validate full-model execution or performance on GPU/NPU.

Notices for the embedded Archify viewer and fonts are retained in [licenses/NOTICE](./licenses/NOTICE) and adjacent license files. The model content is authored by this tutorial.
