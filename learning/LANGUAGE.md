# Language / 语言

This repository supports both **Chinese (中文)** and **English** for all learning materials.

## Switching Languages

All content is organized by language directory:

```
learning/
├── README.md          ← Language selection page
├── LANGUAGE.md         ← This guide
├── zh/                 ← Chinese (中文原版)
│   ├── README.md
│   ├── ai-infra-basic/
│   ├── scheduler-architecture/
│   └── ...
└── en/                 ← English (英文翻译)
    ├── README.md
    ├── ai-infra-basic/
    ├── scheduler-architecture/
    └── ...
```

To switch languages, navigate between `zh/` and `en/` at any level — the directory structure is identical in both. The root `README.md` serves as a language selection landing page.

## Translation Status

| Directory | Status |
|---|---|
| `zh/` (Chinese) | ✅ Complete — all original content |
| `en/` (English) | ✅ Complete — 106 .md files translated |
| `en/ai-infra-basic/` | ✅ All .md content. Python demos remain Chinese-only |
| `en/scheduler-architecture/` | ✅ All content |
| `en/tp-worker-model-runner/` | ✅ All content |
| `en/sglang-source-reading/` | ✅ All 13 files |
| `en/sglang-ascend-npu/` | ✅ All 26 files |
| `en/ascend-kernel-infra/` | ✅ All 26 files |

## Contributing Translations

English translations live in `en/` with identical file names and directory structure as Chinese originals in `zh/`:

1. Copy the file from `zh/path/to/file.md` to `en/path/to/file.md`
2. Translate the content, preserving Mermaid diagrams, code blocks, and technical terms
3. No language switcher needed — files are language-isolated by directory

## Naming Convention

- Directory structure: identical between `zh/` and `en/`
- File names: identical between `zh/` and `en/` (no `_EN` suffix)
- SVG/asset files: duplicated in both directories for self-contained language packages
