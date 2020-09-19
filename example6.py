#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l import SmallestMaximizerCriterion
from g4l.estimators.ctm import CTM
from g4l.estimators.prune import Prune
#from g4l.estimators.ctm_scanner import CTMScanner
import g4l.tree.generation as gen
import g4l.tree as tree
from g4l.data import Sample
from g4l.util.compression import Compressor
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)

# Create a sample object instance
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
X2 = Sample('examples/example1/publico.txt', [0, 1, 2, 3, 4])
initial_tree = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
initial_tree2 = tree.ContextTree(X2, max_depth=4, tree_initialization_method=gen.incremental)
#tree_a = CTM(initial_tree).execute(1.536489)
#tree_a = CTM(initial_tree).execute(None)


# Tree A:   c = 1.536489   (11 contexts)
# tree_a = "000 1 10 100 2 20 200 3 30 300 4"

import pandas as pd
def do(tt, x, m=10):
  pr = Prune(tt)
  pr.execute(max_trees=m)
  arr = []
  for t in reversed(pr.context_trees):
    compr, q = Compressor(t).compress(x)
    lps = round(t.tree().likelihood.sum(), 2)
    l = len(t.tree())
    a = len(compr)
    qq = len(''.join(q))
    arr.append([l, lps, a, qq])
    #print(l, lps, a, qq)
    #import code; code.interact(local=dict(globals(), **locals()))
  d = pd.DataFrame(arr, columns=['ctx', 'lps', 'compr', 'q'])
  d['ratio'] = d.compr / len(x.data)
  return d

print(do(initial_tree, X, 4))
print(do(initial_tree, X2, 4))
print(do(initial_tree2, X, 4))
print(do(initial_tree2, X2, 4))

from g4l.evaluation import bootstrap as bs
b = bs.Bootstrap(X, '4').resample(4, None)
data = [x for x in b.generate()][0][1]
s = Sample(None, [0, 1, 2, 3, 4], data=data)

print(do(initial_tree, s, 4))
print(do(initial_tree2, s, 4))

from g4l.data import Sample
import random
str_var = list(s.data)
random.shuffle(str_var)
ss = Sample(None, [0, 1, 2, 3, 4], data=''.join(str_var))
print(do(initial_tree, ss, 4))
import code; code.interact(local=dict(globals(), **locals()))
rnd_tree = tree.ContextTree(ss, max_depth=4, tree_initialization_method=gen.incremental)
print(do(rnd_tree, X, 2))
#import code; code.interact(local=dict(globals(), **locals()))
