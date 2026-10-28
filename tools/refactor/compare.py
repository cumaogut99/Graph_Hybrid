"""compare.py before.json after.json Class [Class ...]: compare two snapshot.py dumps."""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ok = True
IGNORE = {'__dict__', '__weakref__', '__firstlineno__', '__static_attributes__'}
for cls in sys.argv[3:]:
    x = {k: v for k, v in a[cls].items() if k not in IGNORE}; y = {k: v for k, v in b[cls].items() if k not in IGNORE}
    if set(x) != set(y):
        ok = False; print(cls, 'missing', set(x) - set(y), 'extra', set(y) - set(x))
    diff = [k for k in x if k in y and x[k] != y[k]]
    if diff:
        ok = False; print(cls, 'changed', diff)
    print(cls, len(x), 'attributes', 'IDENTICAL' if not diff and set(x) == set(y) else 'DIFFERENT')
sys.exit(0 if ok else 1)
