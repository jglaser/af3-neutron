"""Reject prose nobody reads: over-long docstrings, comment walls, AI tells.

Design rationale belongs in commit messages and PR descriptions, where it is
dated and reviewable. Docstrings say what a caller must know to call the thing.

Run ``python tools/lint_docs.py --report`` to rank offenders instead of failing.
Configure under ``[tool.lint_docs]`` in pyproject.toml. Suppress one site with a
trailing ``# lint-docs: allow`` on the ``def``/``class`` line.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

TELLS_FILE = Path(__file__).with_name("ai_tells.txt")


def load_tells(path: Path = TELLS_FILE) -> re.Pattern[str]:
    """Compile the blocklist. Kept out of this file so it cannot flag itself."""
    patterns = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return re.compile("|".join(rf"\b{p}" for p in patterns), re.IGNORECASE)


AI_TELL_RE = load_tells()

# Headers that invite an essay. A docstring section named "Notes" or "Why" is a
# commit message that lost its way.
ESSAY_HEADER_RE = re.compile(
    r"^\s*(notes?|discussion|rationale|motivation|background|caveats?|why\b[^\n]*|"
    r"implementation\s+notes?|measured\b[^\n]*|scaling\b[^\n]*|design\b[^\n]*)\s*$",
    re.IGNORECASE,
)
UNDERLINE_RE = re.compile(r"^\s*[-=~^]{3,}\s*$")

# Single-asterisk emphasis: an AI fingerprint per the detector's lookbehind rule.
EMPHASIS_RE = re.compile(r"(?<!\*)\*([^*\n]{1,80}?)\*(?!\*)")

# Zero-width characters and Cyrillic homoglyphs -- the detector's
# normalization-flag signal, and a reliable sign of pasted generated text.
# Written as escapes so this file does not trip its own check.
INVISIBLE_RE = re.compile("[\u200b-\u200d\ufeff\u2060]")
HOMOGLYPH_RE = re.compile("[\u0400-\u04ff]")

ALLOW_PRAGMA = "lint-docs: allow"


@dataclass
class Config:
    """Thresholds; every field is overridable from ``[tool.lint_docs]``."""

    max_docstring_lines: int = 12
    max_module_docstring_lines: int = 20
    max_comment_block: int = 8
    max_emphasis: int = 2
    max_prose_ratio: float = 0.30
    exclude: list[str] = field(default_factory=lambda: ["build", ".venv", "__pycache__"])
    ignore: list[str] = field(default_factory=list)
    # A test docstring naming the invariant under test is the test's spec, so the
    # whole-file ratio is meaningless there. The per-docstring budget still applies.
    ratio_exclude: list[str] = field(default_factory=lambda: ["tests/*", "*/tests/*"])


@dataclass
class Finding:
    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def load_config(root: Path) -> Config:
    """Read ``[tool.lint_docs]`` from pyproject.toml, if present."""
    cfg = Config()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return cfg
    table = tomllib.loads(pyproject.read_text()).get("tool", {}).get("lint_docs", {})
    for key, value in table.items():
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown [tool.lint_docs] key: {key}")
        setattr(cfg, key, value)
    return cfg


def _nonblank(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _allowed(lines: list[str], lineno: int) -> bool:
    """True if the definition at ``lineno`` carries the suppression pragma."""
    return ALLOW_PRAGMA in lines[lineno - 1] if 0 < lineno <= len(lines) else False


def _essay_headers(doc: str) -> list[str]:
    """Section headers whose body is prose rather than an interface contract."""
    found = []
    doc_lines = doc.splitlines()
    for i, line in enumerate(doc_lines):
        if not ESSAY_HEADER_RE.match(line):
            continue
        underlined = i + 1 < len(doc_lines) and UNDERLINE_RE.match(doc_lines[i + 1])
        if underlined or line.rstrip().endswith(":"):
            found.append(line.strip())
    return found


def check_docstrings(path: Path, source: str, cfg: Config) -> list[Finding]:
    """Per-docstring budget, essay headers, AI tells and fingerprints."""
    out: list[Finding] = []
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(str(path), exc.lineno or 1, "DOC001", f"syntax error: {exc.msg}")]

    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, kinds):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        is_module = isinstance(node, ast.Module)
        lineno = node.body[0].value.lineno
        name = "module docstring" if is_module else f"{getattr(node, 'name', '?')}()"
        if _allowed(lines, getattr(node, "lineno", 1)):
            continue

        budget = cfg.max_module_docstring_lines if is_module else cfg.max_docstring_lines
        length = _nonblank(doc)
        if length > budget:
            out.append(
                Finding(
                    str(path),
                    lineno,
                    "DOC101",
                    f"{name} docstring is {length} lines (max {budget}); "
                    f"move rationale to the commit message",
                )
            )
        for header in _essay_headers(doc):
            out.append(
                Finding(
                    str(path),
                    lineno,
                    "DOC102",
                    f"{name} docstring has essay section {header!r}; "
                    f"docstrings state the contract, not the reasoning",
                )
            )
        emphasis = len(EMPHASIS_RE.findall(doc))
        if emphasis > cfg.max_emphasis:
            out.append(
                Finding(
                    str(path),
                    lineno,
                    "DOC103",
                    f"{name} docstring has {emphasis} *emphasis* spans "
                    f"(max {cfg.max_emphasis})",
                )
            )
    return out


def check_text(path: Path, source: str, cfg: Config) -> list[Finding]:
    """AI tells and invisible characters, anywhere in the file."""
    out: list[Finding] = []
    for i, line in enumerate(source.splitlines(), start=1):
        if ALLOW_PRAGMA in line:
            continue
        for match in AI_TELL_RE.finditer(line):
            out.append(
                Finding(str(path), i, "DOC201", f"AI tell: {match.group(0).strip()!r}")
            )
        if INVISIBLE_RE.search(line):
            out.append(Finding(str(path), i, "DOC202", "zero-width character"))
        if HOMOGLYPH_RE.search(line):
            out.append(Finding(str(path), i, "DOC203", "Cyrillic homoglyph"))
    return out


def check_comments(path: Path, source: str, cfg: Config) -> list[Finding]:
    """Flag runs of consecutive comment lines that should be a named function."""
    out: list[Finding] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return out

    comment_lines = sorted(
        tok.start[0]
        for tok in tokens
        if tok.type == tokenize.COMMENT and ALLOW_PRAGMA not in tok.line
    )
    run_start = None
    previous = None
    for lineno in comment_lines + [None]:
        if previous is not None and lineno == previous + 1:
            previous = lineno
            continue
        if run_start is not None:
            run = previous - run_start + 1
            if run > cfg.max_comment_block:
                out.append(
                    Finding(
                        str(path),
                        run_start,
                        "DOC301",
                        f"{run} consecutive comment lines (max {cfg.max_comment_block}); "
                        f"extract a named function instead",
                    )
                )
        run_start = previous = lineno
    return out


def prose_stats(source: str) -> tuple[int, int]:
    """Return ``(prose_lines, code_lines)`` for a module."""
    lines = source.splitlines()
    prose = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0, len(lines)

    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, kinds) and ast.get_docstring(node, clean=False):
            const = node.body[0].value
            prose.update(range(const.lineno, (const.end_lineno or const.lineno) + 1))
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                prose.add(tok.start[0])
    except (tokenize.TokenError, IndentationError):
        pass

    prose_lines = sum(1 for n in prose if 0 < n <= len(lines) and lines[n - 1].strip())
    code_lines = sum(1 for i, line in enumerate(lines, 1) if line.strip() and i not in prose)
    return prose_lines, code_lines


def check_ratio(path: Path, source: str, cfg: Config) -> list[Finding]:
    """Catch death by a thousand cuts: whole-file prose share."""
    posix = path.as_posix()
    if any(fnmatch(posix, pattern) for pattern in cfg.ratio_exclude):
        return []
    prose, code = prose_stats(source)
    if code < 50 or not prose:
        return []
    ratio = prose / code
    if ratio <= cfg.max_prose_ratio:
        return []
    return [
        Finding(
            str(path),
            1,
            "DOC401",
            f"{prose} prose lines to {code} code lines = {ratio:.0%} "
            f"(max {cfg.max_prose_ratio:.0%})",
        )
    ]


def iter_files(targets: list[str], cfg: Config) -> list[Path]:
    """Expand targets to Python files, honouring ``exclude``."""
    found: list[Path] = []
    for target in targets:
        path = Path(target)
        candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        found += [
            p
            for p in candidates
            if p.suffix == ".py" and not any(part in cfg.exclude for part in p.parts)
        ]
    return found


def run(targets: list[str], cfg: Config) -> list[Finding]:
    """Apply every enabled check to every target file."""
    findings: list[Finding] = []
    for path in iter_files(targets, cfg):
        source = path.read_text(encoding="utf-8")
        for check in (check_docstrings, check_text, check_comments, check_ratio):
            findings += check(path, source, cfg)
    return [f for f in findings if f.code not in cfg.ignore]


def report(findings: list[Finding], files: list[Path]) -> None:
    """Rank files by offence count instead of failing, for triage."""
    by_file: dict[str, list[Finding]] = {}
    for finding in findings:
        by_file.setdefault(finding.path, []).append(finding)
    print(f"{'file':46} {'total':>5}  by code")
    for path, group in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        codes: dict[str, int] = {}
        for finding in group:
            codes[finding.code] = codes.get(finding.code, 0) + 1
        detail = " ".join(f"{c}:{n}" for c, n in sorted(codes.items()))
        print(f"{path:46} {len(group):5}  {detail}")
    print(f"\n{len(findings)} findings across {len(by_file)}/{len(files)} files")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", default=["."])
    parser.add_argument("--report", action="store_true", help="rank offenders, exit 0")
    parser.add_argument("--root", default=".", help="where to find pyproject.toml")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.root))
    targets = args.targets or ["."]
    findings = run(targets, cfg)

    if args.report:
        report(findings, iter_files(targets, cfg))
        return 0
    for finding in sorted(findings, key=lambda f: (f.path, f.line)):
        print(finding)
    if findings:
        print(f"\n{len(findings)} findings. Prose is not free; cut it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
