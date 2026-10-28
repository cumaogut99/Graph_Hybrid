"""prune_new_unused.py before.py after.py : drop top-level imports unused in after.py but used in before.py"""
import ast, re, subprocess, sys
def unused(p):
    out = subprocess.run([sys.executable, '-m', 'pyflakes', p], capture_output=True, text=True).stdout
    return set(re.findall(r"'([^']+)' imported but unused", out))
new = unused(sys.argv[2]) - unused(sys.argv[1])
bound = {r.split(' as ')[1] if ' as ' in r else r.rsplit('.', 1)[-1] for r in new}
path = sys.argv[2]
t = open(path).read(); tl = t.split('\n'); edits = []
for node in ast.parse(t).body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        keep = [a for a in node.names if (a.asname or a.name.split('.')[-1]) not in bound]
        if len(keep) != len(node.names):
            node.names = keep
            edits.append((node.lineno, node.end_lineno, ast.unparse(node) if keep else None))
for s, e, n in sorted(edits, reverse=True):
    tl[s - 1:e] = [n] if n else []
open(path, 'w').write('\n'.join(tl))
print('removed', sorted(new))
