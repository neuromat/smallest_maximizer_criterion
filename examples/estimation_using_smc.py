#!/usr/bin/env python
'''
Estimating context trees using the Pruning method

Usage: python3 ./estimation_by_pruning.py
'''

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from g4l.estimators import CTM
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
        #logging.FileHandler("./examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)

# Create a sample object instance
X = Sample('./examples/example1/folha.txt', [0, 1, 2, 3, 4])

tree_creation_start = time.time()
t = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
tree_creation_end = time.time()


print('Starting estimation using SMC')
start = time.time()
smc = SMC(t)
smc.execute(penalty_interval=(0.1, 400), epsilon=0.01)
end = time.time()
print('*' * 20)
print("Tree creation (incremental)", round(tree_creation_end - tree_creation_start, 2), 'seconds')
print("Estimation phase:", round(end - start, 2), 'seconds)')
print("Trees constructed:", smc.trees_constructed)
print("Generated trees:")
df = pd.DataFrame([], columns=['log-likelihood', 'num_contexts'])
for t in smc.context_trees:
  df.loc[len(df)] = [t.log_likelihood(), t.num_contexts()]
print(df)
