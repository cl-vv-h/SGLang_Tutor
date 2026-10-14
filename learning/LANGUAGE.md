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

## GitHub Math / GitHub 公式规范

Use fenced `math` blocks for display equations and GitHub's protected inline syntax for inline equations. This keeps Markdown from modifying TeX escapes, underscores, and comparison symbols. Use upright function names with `\mathrm{...}`; GitHub's current renderer rejects `\operatorname{...}`. Inside formulas, prefer `\lt` and `\gt` to literal angle brackets.

块公式统一使用 `math` 代码围栏，行内公式使用 GitHub 的反引号保护语法。这样可以避免 Markdown 改写 LaTeX 转义、下划线和比较符号。函数名使用 `\mathrm{...}`，当前 GitHub 渲染器会拒绝 `\operatorname{...}`；公式里的比较符号使用 `\lt`、`\gt`。

````markdown
```math
s_d = \sum_{j=0}^{d-1} n_j
y = \mathrm{RMSNorm}(x)
```

Inline / 行内：$`n_d`$，$`T \gt 1`$，$`\mathcal{V}_r(i)`$。
````

Run the repository checker before committing teaching changes. It scans tracked Markdown files, skips ordinary code fences and inline code, and reports noncanonical delimiters, known incompatible macros, and unprotected comparisons. The formatter option applies mechanical fixes. Real GitHub browser rendering should also be checked when new macros or environments are introduced.

提交教程前运行仓库检查工具。它扫描已跟踪的 Markdown 文件，跳过普通代码块和行内代码，检查公式分隔符、已知不兼容宏及未保护的比较符号；格式化选项可以批量修复。新增宏或公式环境时，还应在 GitHub 浏览器页面验证实际渲染。

```bash
python3 learning/tools/check_github_math.py
python3 learning/tools/check_github_math.py --fix
```

See [GitHub's mathematical-expression documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) for supported Markdown delimiters. Standard LaTeX validity alone does not guarantee that GitHub's restricted renderer accepts a macro.

分隔符语法可参考 [GitHub 公式文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。标准 LaTeX 能编译，并不保证 GitHub 的受限渲染器接受对应宏。
