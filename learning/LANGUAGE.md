# Language / 语言

This repository supports both **Chinese (中文)** and **English** for all learning materials.

## Switching Languages

Each directory contains parallel versions of every file:

- `filename.md` — Chinese original (中文原版)
- `filename_EN.md` — English translation (英文翻译)
- `filename_EN.py` — English version of annotated Python code (英文注释版 Python 代码)

Look for the language switcher at the top of each page:

> **中文** | [English](./README_EN.md)

## Translation Status

| Directory | Status | Notes |
|---|---|---|
| `learning/` | ✅ Complete | Main hub + LANGUAGE.md |
| `ai-infra-basic/` | ✅ Complete | All .md content + all sub-READMEs. 13 .py demo files available (code comments in Chinese) |
| `scheduler-architecture/` | ✅ Complete | README + all 4 content files |
| `tp-worker-model-runner/` | ✅ Complete | README + all 3 content files |
| `sglang-source-reading/` | ✅ Complete | README + all 13 content files |
| `sglang-ascend-npu/` | ✅ Complete | README + all 26 content files (00-15 + source-code-walkthrough subfiles) |
| `ascend-kernel-infra/` | ✅ Complete | README + all 26 content files (01-02, ROADMAP, foundations 01-03, ascend-c 01-04, triton-ascend 01-05, sgl-kernel-npu 01-08, torch_npu 01, reference 3 files) |

**Summary**: 🎉 **All 94 .md content files are now fully bilingual!** Every file has both Chinese (original) and English versions with language switchers at the top.

## Contributing Translations

To add an English translation for a file:

1. Create `filename_EN.md` alongside the original `filename.md`
2. Add the language switcher at the very top:
   ```
   [中文](./filename.md) | [English](./filename_EN.md)
   ```
3. Add the corresponding language switcher to the Chinese original:
   ```
   **中文** | [English](./filename_EN.md)
   ```
4. Translate the content, preserving all Mermaid diagrams, code blocks, and technical terms

## Naming Convention

- Markdown files: `original_name_EN.md`
- Python files with Chinese comments: `original_name_EN.py`
- SVG/asset files: shared (no duplication needed)
