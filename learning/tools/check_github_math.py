"""Audit and normalize GitHub math without touching ordinary code fences.

Run from the repository root:
    python3 learning/tools/check_github_math.py
    python3 learning/tools/check_github_math.py --fix
    python3 learning/tools/check_github_math.py --json /tmp/github-math.json

This checks Markdown conventions and known GitHub renderer incompatibilities.
It does not replace checking actual browser rendering after publication.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$")
MATH_LANGUAGES = {"math", "latex", "tex"}
BLOCK_END = {"$$": "$$", r"\[": r"\]"}
PROTECTED_INLINE = re.compile(r"\$`([^`\n]+)`\$")
PAREN_INLINE = re.compile(r"\\\((.*?)\\\)")
DOLLAR_INLINE = re.compile(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$")
SAME_LINE_BLOCK = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]")
UNPROTECTED_TEX = re.compile(r"\\(?:frac|sum|mathcal|mathrm|operatorname)\b")


@dataclass(frozen=True)
class Formula:
    start: int
    end: int
    line: int
    kind: str
    body: str

    @property
    def display(self):
        return self.kind not in {"inline", "protected", "paren"}


def scan(text):
    """Extract formulas with offsets; skip code and nested Markdown examples."""
    lines = text.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    formulas, issues = [], []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\r\n")
        fence = FENCE.match(line)
        if fence:
            marker, info = fence.groups()
            language = info.strip().split()[0] if info.strip() else ""
            end = i + 1
            while end < len(lines):
                closer = FENCE.match(lines[end].rstrip("\r\n"))
                if (
                    closer
                    and closer[1][0] == marker[0]
                    and len(closer[1]) >= len(marker)
                    and not closer[2].strip()
                ):
                    break
                end += 1
            if language in MATH_LANGUAGES:
                if end == len(lines):
                    issues.append((i + 1, "unclosed math fence"))
                else:
                    formulas.append(
                        Formula(
                            offsets[i], offsets[end] + len(lines[end].rstrip("\r\n")),
                            i + 1, language, "".join(lines[i + 1 : end]).rstrip("\r\n"),
                        )
                    )
            i = min(end + 1, len(lines))
            continue
        stripped = line.strip()
        if stripped in BLOCK_END:
            end = i + 1
            while end < len(lines) and lines[end].strip() != BLOCK_END[stripped]:
                end += 1
            if end == len(lines):
                issues.append((i + 1, "unclosed display formula"))
            else:
                formulas.append(
                    Formula(
                        offsets[i], offsets[end] + len(lines[end].rstrip("\r\n")),
                        i + 1, "display", "".join(lines[i + 1 : end]).rstrip("\r\n"),
                    )
                )
            i = min(end + 1, len(lines))
            continue
        # Scan prose left to right so dollar signs in inline code stay untouched.
        position = 0
        while position < len(line):
            match = SAME_LINE_BLOCK.match(line, position)
            kind = "display" if match else None
            body = next((g for g in match.groups() if g is not None), "") if match else ""
            if not match:
                for pattern, candidate_kind in (
                    (PROTECTED_INLINE, "protected"),
                    (PAREN_INLINE, "paren"),
                    (DOLLAR_INLINE, "inline"),
                ):
                    match = pattern.match(line, position)
                    if match:
                        kind, body = candidate_kind, match[1]
                        break
            if match:
                formulas.append(
                    Formula(offsets[i] + position, offsets[i] + match.end(), i + 1, kind, body)
                )
                position = match.end()
            elif line[position] == "`":
                run = re.match(r"`+", line[position:])[0]
                end = line.find(run, position + len(run))
                position = len(line) if end < 0 else end + len(run)
            else:
                raw_tex = UNPROTECTED_TEX.match(line, position)
                if raw_tex:
                    issues.append((i + 1, "LaTeX command outside a math delimiter"))
                position += 1
        i += 1
    return formulas, issues


def normalize_body(body):
    # GitHub currently rejects operatorname; upright function names use mathrm.
    body = re.sub(r"\\operatorname\*?\s*\{", r"\\mathrm{", body)
    # Literal angle brackets can become HTML entities before MathJax receives them.
    body = body.replace("<", r"\lt ").replace(">", r"\gt ")
    return body


def replacement(formula):
    body = normalize_body(formula.body)
    if formula.display:
        return "```math\n" + body + "\n```"
    return "$`" + body + "`$"


def normalize(text, formulas):
    for formula in reversed(formulas):
        text = text[: formula.start] + replacement(formula) + text[formula.end :]
    return text


def convention_issues(formulas):
    issues = []
    for formula in formulas:
        if formula.kind not in {"math", "protected"}:
            issues.append((formula.line, "use a math fence or protected inline math"))
        if formula.body != normalize_body(formula.body):
            issues.append((formula.line, "unsupported macro or unprotected angle bracket"))
    return issues


def markdown_paths():
    result = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z", "*.md", "*.mdx", "*.markdown"]
    )
    return [ROOT / name.decode() for name in result.split(b"\0") if name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="normalize existing formulas")
    parser.add_argument("--json", type=Path, help="export the formula inventory and issues")
    args = parser.parse_args()
    paths = markdown_paths()
    report = {"scanned_files": len(paths), "changed_files": [], "files": []}
    for path in paths:
        original = path.read_text(encoding="utf-8")
        formulas, parse_issues = scan(original)
        if args.fix and not parse_issues:
            updated = normalize(original, formulas)
            if updated != original:
                # A formatter-style mechanical rewrite; no code fences are changed.
                path.write_text(updated, encoding="utf-8")
                report["changed_files"].append(path.relative_to(ROOT).as_posix())
                formulas, parse_issues = scan(updated)
        issues = parse_issues + convention_issues(formulas)
        if formulas or issues:
            report["files"].append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "formulas": [
                        {"line": f.line, "display": f.display, "body": f.body}
                        for f in formulas
                    ],
                    "issues": [{"line": line, "message": msg} for line, msg in issues],
                }
            )
    report["formula_count"] = sum(len(f["formulas"]) for f in report["files"])
    report["issue_count"] = sum(len(f["issues"]) for f in report["files"])
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Scanned {report['scanned_files']} Markdown files; "
        f"{report['formula_count']} formulas in {len(report['files'])} files; "
        f"{len(report['changed_files'])} files normalized; {report['issue_count']} issues."
    )
    for f in report["files"]:
        for issue in f["issues"]:
            print(f"{f['path']}:{issue['line']}: {issue['message']}")
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
