"""Tests for the docstring-budget gate in tools/lint_docs.py."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import lint_docs  # noqa: E402


@pytest.fixture
def cfg():
    return lint_docs.Config()


def codes(source, cfg, name="m.py"):
    """Run every check over a source string and return the codes raised."""
    path = Path(name)
    found = []
    for check in (
        lint_docs.check_docstrings,
        lint_docs.check_text,
        lint_docs.check_comments,
        lint_docs.check_ratio,
    ):
        found += check(path, textwrap.dedent(source), cfg)
    return [f.code for f in found]


def test_short_docstring_passes(cfg):
    assert codes('def f():\n    """Do the thing."""\n', cfg) == []


def test_long_docstring_flagged(cfg):
    body = "\n".join(f"    line {i}" for i in range(cfg.max_docstring_lines + 5))
    assert "DOC101" in codes(f'def f():\n    """Summary.\n\n{body}\n    """\n', cfg)


def test_docstring_at_budget_passes(cfg):
    """The budget is inclusive: exactly max_docstring_lines must not fire."""
    body = "\n".join(f"    line {i}" for i in range(cfg.max_docstring_lines - 1))
    assert "DOC101" not in codes(f'def f():\n    """Summary.\n{body}\n    """\n', cfg)


def test_blank_lines_do_not_count_against_budget(cfg):
    padded = "\n".join(["    text", ""] * (cfg.max_docstring_lines - 1))
    assert "DOC101" not in codes(f'def f():\n    """S.\n{padded}\n    """\n', cfg)


def test_module_docstring_gets_larger_budget(cfg):
    n = cfg.max_docstring_lines + 3
    assert n <= cfg.max_module_docstring_lines
    body = "\n".join(f"line {i}" for i in range(n - 1))
    assert "DOC101" not in codes(f'"""Summary.\n{body}\n"""\n', cfg)


def test_essay_header_flagged(cfg):
    src = '''
    def f():
        """Summary.

        Notes
        -----
        Reasons go in the commit message.
        """
    '''
    assert "DOC102" in codes(src, cfg)


def test_numpydoc_parameters_section_is_not_an_essay(cfg):
    """Parameters/Returns are interface contract and must never be flagged."""
    src = '''
    def f(a):
        """Summary.

        Parameters
        ----------
        a : int
            A number.

        Returns
        -------
        int
        """
    '''
    assert "DOC102" not in codes(src, cfg)


def test_essay_header_needs_underline_or_colon(cfg):
    """A prose line merely starting with a header word is not a section."""
    assert "DOC102" not in codes('def f():\n    """Design of the widget is fixed."""\n', cfg)


def test_ai_tell_flagged(cfg):
    assert "DOC201" in codes('def f():\n    """Leverage the cache."""\n', cfg)  # lint-docs: allow


def test_ai_tell_in_comment_flagged(cfg):
    assert "DOC201" in codes("x = 1  # we delve into the cache here\n", cfg)  # lint-docs: allow


def test_ai_tell_is_word_bounded(cfg):
    """'realm' must not fire inside 'realmente'; substrings are not tells."""  # lint-docs: allow
    assert "DOC201" not in codes("realmente = 1\n", cfg)


def test_domain_prose_survives(cfg):
    """Crystallographic wording must not trip the blocklist."""
    src = 'def f():\n    """Scale F_calc against F_obs in the P 3_2 2 1 setting."""\n'
    assert codes(src, cfg) == []


def test_excess_emphasis_flagged(cfg):
    stars = " ".join(f"*w{i}*" for i in range(cfg.max_emphasis + 2))
    assert "DOC103" in codes(f'def f():\n    """Summary {stars}."""\n', cfg)


def test_bold_and_bullets_are_not_emphasis(cfg):
    """Double-asterisk bold and leading-bullet lists must not count."""
    src = '''
    def f():
        """Summary.

        * one item
        * two item
        **bold** is fine.
        """
    '''
    assert "DOC103" not in codes(src, cfg)


def test_comment_wall_flagged(cfg):
    wall = "\n".join(f"# line {i}" for i in range(cfg.max_comment_block + 2))
    assert "DOC301" in codes(f"{wall}\nx = 1\n", cfg)


def test_separated_comment_blocks_are_not_a_wall(cfg):
    """Two short blocks split by code must not merge into one run."""
    half = "\n".join(f"# line {i}" for i in range(cfg.max_comment_block - 1))
    assert "DOC301" not in codes(f"{half}\nx = 1\n{half}\ny = 2\n", cfg)


def test_prose_ratio_flagged(cfg):
    code_lines = "\n".join(f"x{i} = {i}" for i in range(60))
    prose = "\n".join(f"# padding {i}" for i in range(40))
    # interleave so no single run trips DOC301
    body = "\n".join(f"# padding {i}\nx{i} = {i}" for i in range(60))
    assert "DOC401" in codes(body, cfg, name="src/m.py")
    assert code_lines and prose


def test_prose_ratio_skipped_for_tests(cfg):
    body = "\n".join(f"# padding {i}\nx{i} = {i}" for i in range(60))
    assert "DOC401" not in codes(body, cfg, name="tests/test_m.py")


def test_small_files_exempt_from_ratio(cfg):
    """Under 50 code lines the ratio is noise, so it must not fire."""
    body = "\n".join(f"# padding {i}\nx{i} = {i}" for i in range(10))
    assert "DOC401" not in codes(body, cfg, name="src/m.py")


def test_allow_pragma_suppresses_length(cfg):
    body = "\n".join(f"    line {i}" for i in range(cfg.max_docstring_lines + 5))
    src = f'def f():  # {lint_docs.ALLOW_PRAGMA}\n    """S.\n\n{body}\n    """\n'
    assert "DOC101" not in codes(src, cfg)


def test_allow_pragma_suppresses_tell_on_that_line(cfg):
    assert "DOC201" not in codes(f"x = 1  # leverage  # {lint_docs.ALLOW_PRAGMA}\n", cfg)  # lint-docs: allow


def test_invisible_and_homoglyph(cfg):
    assert "DOC202" in codes("x = 1  # a​b\n", cfg)  # lint-docs: allow
    assert "DOC203" in codes("x = 1  # Сyrillic\n", cfg)  # lint-docs: allow


def test_greek_is_allowed(cfg):
    """Cell angles are written with Greek letters; they are not homoglyphs."""
    assert codes("x = 1  # α β γ angles\n", cfg) == []


def test_syntax_error_reported_not_raised(cfg):
    assert "DOC001" in codes("def (:\n", cfg)


def test_unknown_config_key_rejected(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.lint_docs]\nnope = 1\n")
    with pytest.raises(SystemExit):
        lint_docs.load_config(tmp_path)


def test_config_is_read_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.lint_docs]\nmax_docstring_lines = 3\n")
    assert lint_docs.load_config(tmp_path).max_docstring_lines == 3


def test_repo_config_loads():
    """The committed pyproject.toml must parse under the real loader."""
    root = Path(__file__).resolve().parents[1]
    assert lint_docs.load_config(root).max_docstring_lines > 0


def test_linter_is_clean_on_itself():
    """tools/ must satisfy the gate it defines."""
    root = Path(__file__).resolve().parents[1]
    assert lint_docs.run([str(root / "tools")], lint_docs.load_config(root)) == []


def test_tells_file_compiles():
    assert lint_docs.load_tells().search("we leverage this")  # lint-docs: allow
