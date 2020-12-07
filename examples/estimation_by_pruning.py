#!/usr/bin/env python
'''
Estimating context trees using the Pruning method

Usage: python3 ./estimation_by_pruning.py
'''

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from g4l.estimators import CTM
from g4l.estimators import Prune
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
        #logging.FileHandler("./examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)

# Create a sample object instance
X = Sample('./examples/example1/folha.txt', [0, 1, 2, 3, 4])

tree_creation_start = time.time()
t_incr = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
tree_creation_end = time.time()


print('Starting estimation by pruning')
start = time.time()
prune = Prune(t_incr)
prune.execute()
end = time.time()
print('*' * 20)
print("Tree creation (incremental)", round(tree_creation_end - tree_creation_start, 2), 'seconds')
print("Estimation phase:", round(end - start, 2), 'seconds)')
print("Trees constructed:", prune.trees_constructed)
print("Generated trees:")
df = pd.DataFrame([], columns=['log-likelihood', 'num_contexts'])
for t in prune.context_trees:
  df.loc[len(df)] = [t.log_likelihood(), t.num_contexts()]
print(df)
