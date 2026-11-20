"""Safe, read-only source extraction for the code walkthrough.

Only .py files under the repository root are readable. Symbols are located with ast, so "Func" finds a
top-level function or class and "Class.method" finds a method (nested names work the same way).
"""
import ast
import re
from pathlib import PurePath, PureWindowsPath

from checkpoints import ROOT

MAX_LINES = 80


class CodeViewError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def resolve_source(rel):
    """Map a repo-relative path to an existing .py file under ROOT, or raise CodeViewError."""
    if not rel or '\x00' in rel:
        raise CodeViewError(400, 'path is required')
    windows = PureWindowsPath(rel)
    if windows.drive or windows.root or rel.startswith(('/', '\\')) or PurePath(rel).is_absolute():
        raise CodeViewError(400, 'absolute paths are not allowed')
    if any(part == '..' for part in re.split(r'[\\/]', rel)):
        raise CodeViewError(400, 'parent-directory segments are not allowed')
    if not rel.endswith('.py'):
        raise CodeViewError(400, 'only .py files can be read')
    candidate = (ROOT / rel).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        raise CodeViewError(400, 'path resolves outside the repository')
    if not candidate.is_file():
        raise CodeViewError(404, f'no such file: {rel}')
    return candidate


def _read(rel):
    path = resolve_source(rel)
    text = path.read_text(encoding='utf-8', errors='replace')
    return text, text.splitlines()


def _locate(tree, symbol):
    """Walk 'A.B.C' through nested class/function scopes; return the defining ast node or None."""
    scope, node = tree.body, None
    parts = symbol.split('.')
    for depth, name in enumerate(parts):
        match = next((child for child in scope
                      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and child.name == name), None)
        if match is None:
            return None
        node = match
        if depth < len(parts) - 1:
            if not isinstance(match, ast.ClassDef):
                return None
            scope = match.body
    return node


def symbol_source(rel, symbol, context=0):
    text, lines = _read(rel)
    if not symbol or any(not part.isidentifier() for part in symbol.split('.')):
        raise CodeViewError(400, 'symbol must be Name or Class.method')
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        raise CodeViewError(422, f'cannot parse {rel}: {exc.msg}')
    node = _locate(tree, symbol)
    if node is None:
        raise CodeViewError(404, f'symbol {symbol} not found in {rel}')
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    start = max(1, first - context)
    end = min(len(lines), node.end_lineno + context)
    return dict(path=rel, symbol=symbol, start_line=start, end_line=end,
                source='\n'.join(lines[start - 1:end]))


def line_source(rel, start, end):
    _, lines = _read(rel)
    if start < 1 or end < start:
        raise CodeViewError(400, 'need 1 <= start <= end')
    if end - start + 1 > MAX_LINES:
        raise CodeViewError(400, f'at most {MAX_LINES} lines per request')
    if start > len(lines):
        raise CodeViewError(400, f'start is past the end of the file ({len(lines)} lines)')
    end = min(end, len(lines))
    return dict(path=rel, start_line=start, end_line=end, source='\n'.join(lines[start - 1:end]))
