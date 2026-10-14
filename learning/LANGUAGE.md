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

Mathematical expressions in prose, tables, and derivations must be typeset as math, not hidden in inline code or `text` fences. This includes complexity estimates, memory/cost calculations, inequalities, and tensor shapes. Preserve executable code, source excerpts, configuration examples, logs, and flow diagrams as code/diagrams. A slash used to list alternatives (such as Q/K) is not division. A trailing shape annotation is not a tensor index. Distinguish matrix multiplication, elementwise multiplication, and outer products explicitly when rewriting a derivation.

正文、表格和推导中的数学表达式必须使用正式公式，不再用行内代码或 `text` 围栏代替；复杂度、显存/成本估算、不等式和张量形状也包括在内。可执行代码、源码摘录、配置示例、日志和流程图保留原格式。列举两个对象时的斜杠（例如 Q/K）不是除法；运算末尾的 shape 注释不是张量下标。改写推导时，应明确区分矩阵乘法、逐元素乘法与外积。

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

For legacy ASCII equations, `formalize_math.py` proposes conversions using a restricted grammar. Its report identifies ambiguous blocks for manual classification. Review changes before using `--fix`; it is a migration aid, not a proof of mathematical correctness. In particular, numbered algorithms, ASCII matrix layouts, broadcasting, and overloaded symbols need human review. Run its regression tests and the convention checker after editing.

对于历史遗留的 ASCII 表达式，`formalize_math.py` 可用受限语法提出转换建议，并在报告中列出需要人工分类的混合块。使用 `--fix` 前必须复核建议；它是迁移辅助工具，不是数学正确性证明。编号算法、ASCII 矩阵排版、广播和多义符号尤其需要人工判断。修改后运行回归测试及公式格式检查。

```bash
python3 learning/tools/formalize_math.py --report /tmp/math-review.json
python3 -m unittest discover -s learning/tools -p 'test_formalize_math.py'
python3 learning/tools/check_github_math.py
```

See [GitHub's mathematical-expression documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) for supported Markdown delimiters. Standard LaTeX validity alone does not guarantee that GitHub's restricted renderer accepts a macro.

分隔符语法可参考 [GitHub 公式文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。标准 LaTeX 能编译，并不保证 GitHub 的受限渲染器接受对应宏。
