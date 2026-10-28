"""comments.py 'old files' -- 'new files': compare multisets of comment lines (stripped)."""
import sys, io, tokenize, collections
args = sys.argv[1:]; i = args.index('--')
def com(paths):
    c = collections.Counter()
    for p in paths:
        for tok in tokenize.generate_tokens(io.StringIO(open(p).read()).readline):
            if tok.type == tokenize.COMMENT:
                c[tok.string.strip()] += 1
    return c
a, b = com(args[:i]), com(args[i+1:])
print('lost', dict(a - b)); print('added', dict(b - a))
