"""Conservatively migrate ASCII tutorial equations to GitHub math.

This is a formatter, not a symbolic algebra engine. It only translates a
restricted expression grammar, leaves real code/Mermaid fences untouched, and
reports ambiguous text for human review. Run without --fix to inspect proposals.
Identifiers retain their original names so formulas map back to code. Products
retain an explicit multiplication symbol; elementwise/outer products require
author review because their meaning cannot be inferred from Python syntax.
"""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GREEK = dict(alpha=r"\alpha", beta=r"\beta", gamma=r"\gamma",
             delta=r"\delta", eps=r"\epsilon", epsilon=r"\epsilon",
             theta=r"\theta", sigma=r"\sigma", tau=r"\tau")
FUNCTIONS = {"exp", "log", "sqrt", "softmax", "sigmoid", "softplus", "silu",
             "SiLU", "RMSNorm", "LayerNorm", "ReLU", "relu", "tanh", "abs",
             "min", "max", "sum", "mean", "floor", "ceil", "round", "clip",
             "concat", "flatten", "reshape", "TopK", "topk", "diag", "norm",
             "RoPE", "GELU", "gelu", "sin", "cos", "dot", "outer", "f",
             "clamp", "Attention", "MLA", "FFN", "MoE", "DenseSwiGLU", "SparseMoE",
             "RMSNormGated", "recurrent_update", "zeros", "GDNStep", "O"}


def name(value):
    if re.search(r"[^\x00-\x7f]", value):
        return r"\text{" + value.replace("_", r"\_") + "}"
    if value in GREEK:
        return GREEK[value]
    # Only split a single unambiguous underscore: multiword code identifiers
    # remain upright with escaped underscores, rather than inventing indices.
    if value.count("_") == 1:
        base, sub = value.split("_")
        if len(base) == 1 or base in GREEK or base in {"W", "KV", "Q", "K", "V", "D", "H", "N", "S", "L"}:
            return name(base) + "_{" + name(sub) + "}"
    if len(value) == 1:
        return value
    return r"\mathrm{" + value.replace("_", r"\_") + "}"


def emit(node):
    if isinstance(node, ast.Name):
        if node.id in {"True", "False", "None"}:
            raise ValueError("configuration or code, not algebra")
        return name(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return str(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        return r"\left[" + ",".join(emit(x) for x in node.elts) + r"\right]"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        body = emit(node.operand)
        if isinstance(node.operand, ast.BinOp):
            body = r"\left(" + body + r"\right)"
        return ("-" if isinstance(node.op, ast.USub) else "+") + body
    if isinstance(node, ast.Attribute) and node.attr == "T":
        return "{" + emit(node.value) + r"}^{\mathsf T}"
    if isinstance(node, ast.Subscript):
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        def index(x):
            if isinstance(x, ast.Slice) and x.lower is None and x.upper is None and x.step is None:
                return ":"
            return emit(x)
        return "{" + emit(node.value) + "}_{" + ",".join(index(x) for x in indices) + "}"
    if isinstance(node, ast.BinOp):
        a, b = emit(node.left), emit(node.right)
        if isinstance(node.op, ast.Div):
            return r"\frac{" + a + "}{" + b + "}"
        if isinstance(node.op, ast.FloorDiv):
            return r"\left\lfloor\frac{" + a + "}{" + b + r"}\right\rfloor"
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Name) and node.right.id == "T":
                return "{" + a + r"}^{\mathsf T}"
            return "{" + a + "}^{" + b + "}"
        if isinstance(node.op, ast.Mod):
            return a + r"\bmod " + b
        if isinstance(node.op, (ast.Mult, ast.MatMult)):
            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, (ast.Add, ast.Sub)):
                a = r"\left(" + a + r"\right)"
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
                b = r"\left(" + b + r"\right)"
            elementwise = any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in {"SiLU", "silu", "sigmoid", "gelu", "GELU"} for x in (node.left, node.right))
            elementwise = elementwise or (isinstance(node.left, ast.Name) and node.left.id == "weight" and isinstance(node.right, ast.Name) and node.right.id == "o")
            operator = r"\odot " if elementwise else r"\cdot "
            return a + (operator if isinstance(node.op, ast.Mult) else r"\," ) + b
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if isinstance(node.op, ast.Sub) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
                b = r"\left(" + b + r"\right)"
            return a + ("+" if isinstance(node.op, ast.Add) else "-") + b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = node.func.id
        if func not in FUNCTIONS and not re.fullmatch(r"[A-Za-z](?:_\w+|\d+)?", func):
            raise ValueError("unknown call: " + func)
        args = [emit(x) for x in node.args]
        if func == "sqrt" and len(args) == 1:
            return r"\sqrt{" + args[0] + "}"
        if func == "outer" and len(args) == 2:
            return args[0] + r"\,{" + args[1] + r"}^{\mathsf T}"
        if func == "dot" and len(args) == 2:
            return r"\left\langle " + args[0] + "," + args[1] + r"\right\rangle"
        return name(func) + r"\left(" + ",".join(args) + r"\right)"
    if isinstance(node, ast.Compare):
        result = emit(node.left)
        operators = {ast.Lt: r"\lt", ast.Gt: r"\gt", ast.LtE: r"\le", ast.GtE: r"\ge", ast.Eq: "=", ast.NotEq: r"\ne"}
        for op, right in zip(node.ops, node.comparators):
            if type(op) not in operators:
                raise ValueError("non-numeric comparison")
            result += " " + operators[type(op)] + " " + emit(right)
        return result
    raise ValueError("unsupported expression")


def expression(source):
    source = source.strip().replace("×", "*").replace("·", "*")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", source):
        return source.replace(",", r"\,")
    source = source.replace("^", "**").replace("≤", "<=").replace("≥", ">=")
    for ascii_name, greek in (("gamma", "γ"), ("beta", "β"), ("alpha", "α"), ("tau", "τ")):
        source = source.replace(greek, ascii_name)
    source = re.sub(r"\b([A-Za-z])_\(([^()]+)\)", r"\1[\2]", source)
    source = re.sub(r"\b([A-Za-z])_\{([^{}]+)\}", r"\1[\2]", source)
    source = re.sub(r"\b([A-Za-z]\w*)\s+dot\s+([A-Za-z]\w*)\b", r"dot(\1,\2)", source)
    # Implicit multiplication in the standard state-space notation.
    source = re.sub(r"\b([ABCD](?:_t)?)\s+([hx]\[|[hx]_t\b)", r"\1 @ \2", source)
    source = re.sub(r"\b([ABCD])\s+(state\[|x_t\b)", r"\1 @ \2", source)
    return emit(ast.parse(source, mode="eval").body)


def literal_algebra(source, labels=True):
    """Typeset mathematical pseudonotation without evaluating/rearranging it.

    Used for sums, conditional probabilities, units and implicit products. The
    order/grouping of the authored expression stays intact. Source statements,
    prose definitions, flow arrows and incomplete expressions are excluded.
    """
    if re.search(r"->|<-|=>|::|\.\.\.,|\b(?:self|torch|tl|dist|config|req|batch|metadata)\.|\b(?:None|True|False|return|assert|for|if|else|let)\b", source):
        raise ValueError("source or flow")
    source = source.strip()
    if re.search(r"(?:<<|>>)\s*\d|\bpacked\b|\b0x[0-9A-Fa-f]+\b", source):
        raise ValueError("bitwise source expression")
    if re.match(r"(?:dtype|device|backend|attention_backend|model_path|path)\s*=", source):
        raise ValueError("configuration rather than mathematics")
    if source.endswith((":", "=", "+", ",")):
        raise ValueError("incomplete")
    if not re.search(r"=|≈|∝|\b(?:sum|outer|dot)\b|\bO\(|\bin\s+R", source):
        raise ValueError("not mathematical notation")
    # A definition such as `Nq = number of query heads` is explanatory prose.
    rhs = " ".join(re.split(r"=|≈|∝", source)[1:])
    if not re.search(r"[\d+*/^()\[\]×]|\bsum_|\bdot\b|\bouter\b|\bmod\b", rhs) and not re.search(r"\bsum_|Σ", source) and not re.search(r"\b[a-z]_\w+\s+W_", rhs):
        raise ValueError("prose definition")
    comment = ""
    if "#" in source:
        source, comment = source.split("#", 1)
    # Never merge commas here: [448,512] is a shape, not the integer 448512.
    source = source.replace("Σ_", "sum_")
    label = None
    if labels:
        prefix = re.match(r"^([^=：:]+)(?=[=：:])", source)
        if prefix and re.search(r"\S\s+\S", prefix[1].strip()) and not re.search(r"[+*@^()\[\]{}<>!-]|\b(?:dot|outer|sum_)\b", prefix[1]):
            label = prefix[1].strip()
            source = "MathLabel" + source[prefix.end():]
    pieces, pos, stack = [], 0, []
    token = re.compile(r"\s+|[A-Za-z][A-Za-z0-9_]*|[\u4e00-\u9fff]+|\d+(?:\.\d+)?|\.\.\.|<=|>=|!=|<<|>>|==|//|\|\||.")
    matches = list(token.finditer(source))
    j = 0
    while j < len(matches):
        value = matches[j][0]; j += 1
        if value.isspace():
            pieces.append(" "); continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
            if value in {"sum", "Σ"} or value.startswith("sum_"):
                index = value[4:] if value.startswith("sum_") else ""
                if value == "sum_" and j < len(matches) and matches[j][0] == "(":
                    pieces.append(r"\sum_{"); stack.append(("(", "}")); j += 1
                elif index:
                    if index == "over_tokens":
                        index = "token"
                    pieces.append(r"\sum_{" + name(index) + "}")
                else:
                    pieces.append(r"\sum")
            elif value in {"outer", "dot"}:
                pieces.append(r"\otimes" if value == "outer" else r"\cdot")
            elif value == "in":
                pieces.append(r"\in")
            elif value == "mod":
                pieces.append(r"\bmod")
            elif value.endswith("_") and j < len(matches) and matches[j][0] in {"(", "{"}:
                pieces.append(name(value[:-1]) + "_{"); stack.append((matches[j][0], "}")); j += 1
            else:
                pieces.append(name(value))
        elif re.fullmatch(r"[\u4e00-\u9fff]+", value):
            pieces.append(r"\text{" + value + "}")
        elif value in {"(", "[", "{"}:
            pieces.append({"(": r"\left(", "[": r"\left[", "{": "{"}[value]); stack.append((value, {"(": r"\right)", "[": r"\right]", "{": "}"}[value]))
        elif value in {")", "]", "}"}:
            if not stack or stack[-1][0] != {")": "(", "]": "[", "}": "{"}[value]:
                raise ValueError("unbalanced expression")
            pieces.append(stack.pop()[1])
        elif value == "^":
            if j == len(matches):
                raise ValueError("missing exponent")
            next_value = matches[j][0]
            if next_value in {"(", "{"}:
                pieces.append("^{"); stack.append((next_value, "}")); j += 1
            else:
                if next_value == "-" and j + 1 < len(matches) and matches[j + 1][0].isdigit():
                    next_value += matches[j + 1][0]; j += 1
                if "_" in next_value:
                    exponent, subscript = next_value.split("_", 1)
                    pieces.append("^{" + name(exponent) + "}")
                    if subscript:
                        pieces.append("_{" + name(subscript) + "}"); j += 1
                    elif j + 1 < len(matches) and matches[j + 1][0] == "{":
                        pieces.append("_{"); stack.append(("{", "}")); j += 2
                    else:
                        raise ValueError("incomplete exponent/subscript")
                else:
                    pieces.append("^{" + (r"\mathsf T" if next_value == "T" else name(next_value) if next_value.isalpha() else next_value) + "}"); j += 1
        elif value == "||":
            pieces.append(r"\Vert")
        elif value in {"*", "@", "≤", "≥", "<=", ">=", "<", ">", "<<", ">>", "≈", "∝", "∈", "Σ", "|", "~", "×", "!="}:
            pieces.append({"*": r"\cdot ", "@": r"\,", "≤": r"\le ", "≥": r"\ge ", "<=": r"\le ", ">=": r"\ge ", "<": r"\lt ", ">": r"\gt ", "<<": r"\ll ", ">>": r"\gg ", "≈": r"\approx ", "∝": r"\propto ", "∈": r"\in ", "Σ": r"\sum ", "|": r"\mid ", "~": r"\sim ", "×": r"\times ", "!=": r"\ne "}[value])
        elif value in {"+", "-", "=", ",", ".", ":", "/", "'"} or re.fullmatch(r"\d+(?:\.\d+)?", value):
            pieces.append(value)
        else:
            raise ValueError("unrecognized mathematical token: " + value)
    if stack:
        raise ValueError("unclosed expression")
    if comment:
        pieces.append(r"\qquad\text{" + comment.strip().replace("_", r"\_") + "}")
    result = "".join(pieces).strip()
    if label:
        result = result.replace(r"\mathrm{MathLabel}", r"\text{" + label.replace("_", r"\_") + "}")
    return result


def equation(source):
    shape_row = re.fullmatch(r"\s*([A-Za-z]\w*(?:_\([^()]+\))?)\s*:\s*(\[[\w, +*/-]+\])\s*", source)
    if shape_row:
        return r"\mathrm{shape}\left(" + expression(shape_row[1]) + r"\right)=" + expression(shape_row[2])
    if re.fullmatch(r"\s*\[[\w, +*/-]+\]\s*", source):
        return expression(source)
    # `B=1, L=8` means two assignments, not B=[1,L]=8. Split only
    # top-level commas, never commas inside a vector or function call.
    depth, start, chunks = 0, 0, []
    for index, character in enumerate(source):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            chunks.append(source[start:index]); start = index + 1
    chunks.append(source[start:])
    if len(chunks) > 1 and all("=" in chunk for chunk in chunks):
        return r",\qquad ".join(equation(chunk) for chunk in chunks)
    # A shape annotation is not a tensor subscript. Keep it separate from the
    # operation rather than letting Python parse `W [B,S,D]` as indexing.
    shape = re.search(r"\s{2,}(\[[^\]]+\])\s*$", source)
    suffix = ""
    if shape:
        suffix = r"\qquad\mathrm{shape}:" + expression(shape[1])
        source = source[:shape.start()]
    if "#" in source:
        source, comment = source.split("#", 1)
        suffix += r"\qquad\text{" + comment.strip().replace("_", r"\_") + "}"
    source = source.strip()
    source = re.sub(r"\s+outer\s+([A-Za-z]\w*)", r" @ \1.T", source)
    # Slash-separated alternatives in prose are not division. A bare ratio
    # must be authored explicitly or have an arithmetic context.
    # A display equation can have several equality/approximation steps.
    parts = re.split(r"(?<![<>=!])=(?!=)|≈|~(?=\s)", source)
    if len(parts) == 1:
        return expression(source) + suffix
    signs = re.findall(r"(?<![<>=!])=(?!=)|≈|~(?=\s)", source)
    result = expression(parts[0])
    for op, rhs in zip(signs, parts[1:]):
        result += (r"\approx " if op in {"≈", "~"} else "=") + expression(rhs)
    return result + suffix


def looks_math(source):
    # Do not convert configuration values, filesystem names, or API calls.
    if re.search(r"\b(?:self|torch|tl|dist|batch|req|config|metadata)\.|::|--|\b(?:return|assert|for|if|else|let|True|False|None)\b", source):
        return False
    if re.fullmatch(r"[\w]+/[\w]+", source.strip()):
        return False
    if re.fullmatch(r"\s*(?:[A-Za-z]\w*\s*:\s*)?\[[\w, +*/-]+\]\s*", source):
        return True
    return bool(re.search(r"(?<![<>=!])=(?!=)|≈|[<>]=|[<>]|[\s\d][*/@^]|\b(?:sqrt|softmax|exp|sigmoid|O)\(", source))


def math_block(body):
    return "```math\n" + body + "\n```"


def migrate(text):
    lines = text.splitlines(keepends=True)
    output, changes, unresolved = [], 0, []
    i = 0
    while i < len(lines):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)\n?$", lines[i])
        if fence:
            marker, lang = fence.groups()
            end = i + 1
            while end < len(lines) and not re.fullmatch(r" {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", lines[end]):
                end += 1
            if lang.strip() not in {"text", "plaintext", ""} or end == len(lines):
                output.extend(lines[i:end + 1]); i = end + 1; continue
            body = lines[i + 1:end]
            # Preserve mixed source/dataflow blocks. Only standalone algebra
            # blocks are mechanically migrated, then ambiguous blocks reviewed.
            rows = [row.strip() for row in body if row.strip()]
            joined = []
            for row in rows:
                if joined and (row.startswith(("+", "*", "=", "/")) or joined[-1].endswith(("=", "+")) or sum(joined[-1].count(c) for c in "([") > sum(joined[-1].count(c) for c in ")]")):
                    joined[-1] += " " + row
                else:
                    joined.append(row)
            if re.search(r"\b(?:self|torch|tl|dist|batch|req|config|metadata)\.|::|\b(?:return|assert|for|if|else|let)\b|\.\.\.,|^\s*\)\s*$", "\n".join(rows), re.M):
                output.extend(lines[i:end + 1])
                i = end + 1; continue
            segments, pending, converted_count = [], [], 0
            def preserved(rows):
                if all(re.fullmatch(r"[\w\s:：-]+", row) for row in rows):
                    return "\n\n".join(rows)
                return "```text\n" + "\n".join(rows) + "\n```"
            for row in joined:
                try:
                    if not looks_math(row):
                        raise ValueError("not algebra")
                    try:
                        converted = equation(row)
                    except (ValueError, SyntaxError):
                        converted = literal_algebra(row)
                    if pending:
                        segments.append(preserved(pending)); pending = []
                    segments.append(math_block(converted)); converted_count += 1
                except (ValueError, SyntaxError):
                    pending.append(row)
            if pending:
                segments.append(preserved(pending))
            if converted_count:
                output.append("\n\n".join(segments) + "\n"); changes += converted_count
            else:
                output.extend(lines[i:end + 1])
            if any(looks_math(row) for row in pending):
                unresolved.append({"line": i + 1, "body": "\n".join(pending)})
            i = end + 1; continue
        line = lines[i]
        # Never interpret a backtick inside existing protected math as code.
        def inline(match):
            nonlocal changes
            if match[1]:
                return match[0]
            source = match[2]
            integer_range = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", source)
            if integer_range:
                changes += 1
                return "$`\\{" + integer_range[1] + r",\ldots," + integer_range[2] + r"\}`$"
            if not looks_math(source):
                return match[0]
            try:
                try:
                    converted = equation(source)
                except (ValueError, SyntaxError):
                    converted = literal_algebra(source)
            except (ValueError, SyntaxError):
                return match[0]
            changes += 1
            return "$`" + converted + "`$"
        line = re.sub(r"(\$`[^`\n]+`\$)|(?<!`)`([^`\n]+)`(?!`)", inline, line)
        output.append(line); i += 1
    return "".join(output), changes, unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "learning/**/*.md"]).decode().splitlines()
    report = []
    for relative in paths:
        if not relative.startswith(("learning/zh/", "learning/en/")):
            continue
        path = ROOT / relative
        text = path.read_text()
        updated, changes, unresolved = migrate(text)
        if changes or unresolved:
            report.append({"path": relative, "converted": changes, "unresolved": unresolved})
        if args.fix and updated != text:
            path.write_text(updated)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Scanned {len(paths)} files; proposed {sum(r['converted'] for r in report)} conversions in {sum(bool(r['converted']) for r in report)} files.")
    print(f"{sum(len(r['unresolved']) for r in report)} ambiguous code/text blocks require classification, not blind rewriting.")


if __name__ == "__main__":
    main()
