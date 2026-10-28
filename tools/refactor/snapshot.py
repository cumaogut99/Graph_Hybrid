"""Dump the syntax tree of every attribute of the given classes, as attribute
lookup resolves it (mixins included).
usage: snapshot.py out.json module:Class [module:Class ...]"""
import sys, os, json, inspect, importlib, textwrap, ast
def norm(src):
    return ast.dump(ast.parse(textwrap.dedent(src)))
sys.path.insert(0, os.getcwd())
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
out = {}
for spec in sys.argv[2:]:
    mod, cls_name = spec.split(':')
    cls = getattr(importlib.import_module(mod), cls_name)
    entry = {}
    for klass in reversed(cls.__mro__):
        if not klass.__module__.startswith('src.'):
            continue
        for name, raw in vars(klass).items():
            entry[name] = raw  # later (more derived) wins, like attribute lookup
    res = {}
    for name, raw in entry.items():
        kind = type(raw).__name__
        fn = raw
        if isinstance(raw, (staticmethod, classmethod)):
            fn = raw.__func__
        if isinstance(raw, property):
            parts = []
            for f in (raw.fget, raw.fset, raw.fdel):
                parts.append(norm(inspect.getsource(f)) if f else None)
            res[name] = [kind, parts]
        elif inspect.isfunction(fn):
            res[name] = [kind, norm(inspect.getsource(fn))]
        else:
            res[name] = [kind, repr(raw) if not hasattr(raw, 'signal') else 'signal']
    out[spec.split(':')[1]] = res
json.dump(out, open(sys.argv[1], 'w'), indent=0, sort_keys=True)
print('classes', {k: len(v) for k, v in out.items()})
