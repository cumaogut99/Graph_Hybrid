"""
Move methods of a class verbatim into mixin modules.

usage: split_class.py plan.json

plan = {
  "source": "src/data/signal_processor.py",
  "class": "SignalProcessor",
  "package": "src/data/signal_processing",      # where mixin modules go
  "package_doc": "...",                          # __init__ docstring if created
  "mixins": [
     {"module": "statistics", "class": "StatisticsMixin",
      "doc": "Statistics of signals and time ranges.",
      "methods": ["calculate_statistics", ...]},
  ]
}
"""
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys

plan = json.load(open(sys.argv[1]))
src_path = plan['source']
src = open(src_path).read()
lines = src.split('\n')
tree = ast.parse(src)

module_name = src_path[:-3].replace('/', '.')
orig_package = module_name.rsplit('.', 1)[0] if '.' in module_name else ''
pkg_dir = plan['package']
pkg_name = pkg_dir.replace('/', '.')

cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == plan['class'])


def _indent(line):
    return len(line) - len(line.lstrip())


def block_range(node):
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    def_indent = _indent(lines[start - 1])
    # comment lines directly above the method, at its indentation, go with it
    while (start - 2 >= 0 and lines[start - 2].strip().startswith('#')
           and _indent(lines[start - 2]) == def_indent):
        start -= 1
    # comments after the last statement, indented as the body, stay with it
    end = node.end_lineno
    probe = end
    while probe < len(lines):
        line = lines[probe]
        if line.strip() == '':
            probe += 1
            continue
        if line.strip().startswith('#') and _indent(line) > def_indent:
            probe += 1
            end = probe
            continue
        break
    return start, end  # 1-based inclusive


by_name = {}
for node in cls.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        by_name.setdefault(node.name, []).append(node)

assigned = {}
for mixin in plan['mixins']:
    for name in mixin['methods']:
        if name not in by_name:
            sys.exit(f'{name} not found in {plan["class"]}')
        if name in assigned:
            sys.exit(f'{name} listed twice')
        assigned[name] = mixin['module']


def absolutize(text):
    def repl(m):
        dots, rest = m.group(2), m.group(3)
        target = importlib.util.resolve_name(dots + rest, orig_package)
        return f'{m.group(1)}from {target} import'
    return re.sub(r'^(\s*)from (\.+)([\w.]*) import', repl, text, flags=re.M)


def seg(node):
    return '\n'.join(lines[node.lineno - 1:node.end_lineno])


# top-level import statements of the original module (copied to each mixin)
header_parts = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        header_parts.append(node)
    elif isinstance(node, ast.Try) and all(isinstance(b, (ast.Import, ast.ImportFrom)) for b in node.body):
        header_parts.append(node)
    elif (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
          and node.test.id == 'TYPE_CHECKING'):
        header_parts.append(node)

remove = []  # (start, end) 1-based inclusive
os.makedirs(pkg_dir, exist_ok=True)
init_path = os.path.join(pkg_dir, '__init__.py')
if not os.path.exists(init_path):
    open(init_path, 'w').write(f'"""{plan["package_doc"]}"""\n')

for mixin in plan['mixins']:
    names = set(mixin['methods'])
    nodes = [n for n in cls.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    blocks = []
    for n in nodes:
        s, e = block_range(n)
        remove.append((s, e))
        blocks.append('\n'.join(lines[s - 1:e]))
    body = absolutize('\n\n'.join(blocks))
    header = absolutize('\n'.join(seg(n) for n in header_parts))
    uses_logger = re.search(r'\blogger\b', body) is not None
    text = (f'"""\n{mixin["doc"]}\n\nPart of {plan["class"]}; moved from {src_path} without changes.\n"""\n\n'
            f'{header}\n\n'
            + ('logger = logging.getLogger(__name__)\n\n' if uses_logger else '')
            + f'\nclass {mixin["class"]}:\n    """{mixin["doc"]}"""\n\n{body}\n')
    out_path = os.path.join(pkg_dir, mixin['module'] + '.py')
    open(out_path, 'w').write(text)

    # drop imports pyflakes reports unused
    for _ in range(3):
        res = subprocess.run([sys.executable, '-m', 'pyflakes', out_path], capture_output=True, text=True)
        reported = re.findall(r"'([^']+)' imported but unused", res.stdout)
        if not reported:
            break
        unused_from, unused_plain = set(), set()
        for r in reported:
            if ' as ' in r:
                unused_from.add(r.split(' as ')[1]); unused_plain.add(r.split(' as ')[1])
            else:
                unused_from.add(r.rsplit('.', 1)[-1]); unused_plain.add(r)
        t = open(out_path).read()
        tl = t.split('\n')
        mt = ast.parse(t)
        edits = []

        def is_unused(node, a):
            if isinstance(node, ast.ImportFrom):
                return (a.asname or a.name) in unused_from
            return (a.asname or a.name) in unused_plain

        for node in mt.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                keep = [a for a in node.names if not is_unused(node, a)]
                if len(keep) != len(node.names):
                    node.names = keep
                    edits.append((node.lineno, node.end_lineno, ast.unparse(node) if keep else None))
            elif isinstance(node, ast.Try):
                if all(is_unused(b, a) for b in node.body for a in b.names):
                    edits.append((node.lineno, node.end_lineno, None))
        for s, e, new in sorted(edits, reverse=True):
            tl[s - 1:e] = [new] if new else []
        open(out_path, 'w').write('\n'.join(tl))
    t = open(out_path).read()
    if 'logging.' not in t:
        t = re.sub(r'^import logging\n', '', t, flags=re.M)
    t = re.sub(r'\n{4,}', '\n\n\n', t)
    open(out_path, 'w').write(t)

# remove blocks from the original, add the mixins to the class bases
for s, e in sorted(remove, reverse=True):
    del lines[s - 1:e]
    # drop one of the blank lines left around the hole
    if s - 1 < len(lines) and s - 2 >= 0 and lines[s - 2].strip() == '' and lines[s - 1].strip() == '':
        del lines[s - 1]

text = '\n'.join(lines)
# collapse blank-line runs inside the class
text = re.sub(r'\n(?:[ \t]*\n){2,}(?=[ \t]+\S)', '\n\n', text)
mixin_names = [m['class'] for m in plan['mixins']]
header_re = re.compile(rf'^class {plan["class"]}\(', re.M)
assert header_re.search(text)
text = header_re.sub(f'class {plan["class"]}({", ".join(mixin_names)}, ', text, count=1)
imports = '\n'.join(f'from {pkg_name}.{m["module"]} import {m["class"]}' for m in plan['mixins'])
# insert after the last top-level import before the class
t2 = ast.parse(text)
last = max(n.end_lineno for n in t2.body
           if isinstance(n, (ast.Import, ast.ImportFrom, ast.Try, ast.If))
           and n.lineno < next(c.lineno for c in t2.body if isinstance(c, ast.ClassDef) and c.name == plan['class']))
tl = text.split('\n')
tl[last:last] = [imports]
open(src_path, 'w').write('\n'.join(tl))

for p in [src_path] + [os.path.join(pkg_dir, m['module'] + '.py') for m in plan['mixins']]:
    res = subprocess.run([sys.executable, '-m', 'pyflakes', p], capture_output=True, text=True)
    bad = [l for l in res.stdout.splitlines() if 'undefined name' in l or 'redefinition' in l]
    print(p, len(open(p).read().split('\n')), 'lines', *bad, sep='\n  ' if bad else ' ')
