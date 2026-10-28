"""
remove_methods.py file.py Class name [name ...]

Delete methods of a class, with the comment lines directly above them at
the method's indentation and the trailing comments indented as its body
(the same block split_class.py moves).
"""
import ast
import re
import sys

path, cls_name, names = sys.argv[1], sys.argv[2], set(sys.argv[3:])
lines = open(path).read().split('\n')
tree = ast.parse('\n'.join(lines))
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name)


def indent(line):
    return len(line) - len(line.lstrip())


blocks = []
found = set()
for node in cls.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
        found.add(node.name)
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        def_indent = indent(lines[start - 1])
        while (start - 2 >= 0 and lines[start - 2].strip().startswith('#')
               and indent(lines[start - 2]) == def_indent):
            start -= 1
        end = probe = node.end_lineno
        while probe < len(lines):
            line = lines[probe]
            if line.strip() == '':
                probe += 1
            elif line.strip().startswith('#') and indent(line) > def_indent:
                probe += 1
                end = probe
            else:
                break
        blocks.append((start, end))
missing = names - found
if missing:
    sys.exit(f'not found: {sorted(missing)}')
for start, end in sorted(blocks, reverse=True):
    del lines[start - 1:end]
    if start - 2 >= 0 and start - 1 < len(lines) and lines[start - 2].strip() == '' and lines[start - 1].strip() == '':
        del lines[start - 1]
text = re.sub(r'\n(?:[ \t]*\n){2,}(?=[ \t]+\S)', '\n\n', '\n'.join(lines))
open(path, 'w').write(text)
print(path, 'removed', len(blocks), 'methods')
