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
from g4l.estimators.ctm_scanner import CTMScanner
import g4l.tree.generation as gen
import g4l.tree as tree
from g4l.data import Sample
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

import time

start = time.time()
#t = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
end = time.time()
print("Tree creation (original init)", end - start)

start = time.time()
t = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
end = time.time()
print("Tree creation (incremental init)", end - start)
print("-------------")


c = None


start = time.time()
ctm = Prune(t).execute()
end = time.time()
print("Original init -> CTM")
print("Elapsed (s):", end - start)
print("Nodes:", ctm.to_str())
print("Log-Likelihood:", ctm.log_likelihood())
print("-------------")

start = time.time()
#ctm = CTM(t2).execute(c)
end = time.time()
print("Incremental init -> CTM")
print("Elapsed (s):", end - start)
print("Nodes:", ctm.to_str())
print("Log-Likelihood:", ctm.log_likelihood())
print("-------------")


#ctm = CTM2(t).execute(0.5)

#import code; code.interact(local=dict(globals(), **locals()))



#t_orig = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
#t_incr = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
#CTM(t_incr).execute(0.08).to_str()
#CTM2(t_incr).execute(0.8).to_str()
