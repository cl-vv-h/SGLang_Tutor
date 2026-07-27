# Language and Translation Convention / 语言与翻译规范

This repository supports English and Simplified Chinese learning materials.

本仓库同时维护英文和简体中文教学材料。

## Directory Convention

```text
learning/
|-- README.md
|-- LANGUAGE.md
|-- en/
|   |-- README.md
|   `-- <topic>/<same-relative-path>
`-- zh/
    |-- README.md
    `-- <topic>/<same-relative-path>
```

- `learning/en/` contains English teaching prose.
- `learning/zh/` contains Simplified Chinese teaching prose.
- Corresponding files use identical relative paths and file names.
- Language suffixes such as `_EN.md` or `_ZH.md` are not used inside the language directories.
- Assets are kept within each language tree so links remain self-contained.

## README Convention

- The repository root uses `README.md` for English and `README.zh-CN.md` for Simplified Chinese.
- Each language-specific README begins with a link to its counterpart.
- `learning/README.md` is a compact bilingual language selector.
- `README_GITHUB.md` contains absolute links for external publishing and does not duplicate the full project introduction.

## Updating Content

When adding or moving a teaching file:

1. Create or move the file at the same relative path under both `learning/en/` and `learning/zh/`.
2. Update the nearest README index in both languages.
3. Preserve code blocks, Mermaid diagrams, filenames, and technical identifiers during translation.
4. Verify all relative Markdown links after the change.
5. Keep topic order and table structure aligned where practical.

Python demos and source snapshots use the same filenames in both trees. Language-specific prose should be translated, while code identifiers remain unchanged. Files explicitly named `*-annotated-cn.py` are Chinese-annotated reference snapshots retained for structural and link compatibility.
