#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example3.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l import SmallestMaximizerCriterion
from g4l.estimators import CTM
from g4l.estimators import Prune
from g4l.estimators import SMC
import g4l.tree.generation as gen
import g4l.tree as tree
from g4l.data import Sample
import pandas as pd
import time
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


start = time.time()
t_orig = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
end = time.time()
print("Tree creation (original init)", end - start)

start = time.time()
t_incr = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
end = time.time()
print("Tree creation (incremental init)", end - start)
print("-------------")


print('Starting estimation by pruning')
start = time.time()
prune = Prune(t_incr)
prune.execute()
end = time.time()
print("Original init -> CTM")
print("Elapsed (s):", end - start)
print("Generated trees:")
df = pd.DataFrame([], columns=['log-likelihood', 'num_contexts'])
for t in prune.context_trees:
  df.loc[len(df)] = [t.log_likelihood(), t.num_contexts()]
print(df)

# print("Log-Likelihood:", ctm.log_likelihood())

## start = time.time()
## #ctm = CTM(t2).execute(c)
## end = time.time()
## print("Incremental init -> CTM")
## print("Elapsed (s):", end - start)
## print("Nodes:", ctm.to_str())
## print("Log-Likelihood:", ctm.log_likelihood())
## print("-------------")


#ctm = CTM2(t).execute(0.5)

import code; code.interact(local=dict(globals(), **locals()))



#t_orig = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
#t_incr = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
#CTM(t_incr).execute(0.08).to_str()
#CTM2(t_incr).execute(0.8).to_str()
