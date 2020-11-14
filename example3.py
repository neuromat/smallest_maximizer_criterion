#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example3.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l.estimators import SMC
from g4l.estimators import BIC
from g4l.data import Sample
from g4l.data import persistence
import math
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

PATH = '/home/arthur/Documents/Neuromat/projects/SMC/smallest_maximizer_criterion/examples/example2/samples'

A = ['0', '1']


def fetch_samples(model_name, sample_size, max_samples=math.inf):
    i = -1
    key = '%s_%s' % (model_name, sample_size)
    filename = '%s/%s.mat' % (PATH, key)
    for s in persistence.iterate_from_mat(filename, key, A):
        if i > max_samples:
            break
        i += 1
        yield i, s


def sort_trees(context_trees):
    return sorted(context_trees, key=lambda x: -x.num_contexts())

#args = ('model1', '5000', 1)
#for sample_idx, sample in fetch_samples(*args):
#    max_depth = 6
#    c = 0.0536
#    cc = 0.0587
#    bic = BIC(c, max_depth).fit(sample).context_tree
#    bic2 = BIC(cc, max_depth).fit(sample).context_tree
#    print(bic.to_str())
#    print(bic2.to_str())
#
#
#    smc = SMC(max_depth, penalty_interval=(0, 500), epsilon=0.00001)
#    champion_trees = smc.fit(sample).context_trees
#    import code; code.interact(local=dict(globals(), **locals()))
#
#    [print(x.to_str()) for x in champion_trees]
#

data = open('fixtures/sample20000.txt').read()
sample = Sample(None, [0, 1], data=data, separator=None)
max_depth = 6
c = 0.0536
cc = 0.0587

#c = 0
bic = BIC(c, max_depth).fit(sample).context_tree



smc = SMC(max_depth, penalty_interval=(0, 100), epsilon=0.00001)
champion_trees = smc.fit(sample).context_trees

import code; code.interact(local=dict(globals(), **locals()))
#
#
#pp = Prune().fit(sample).context_trees
#BIC(c, max_depth).fit(sample).context_tree

#t_orig = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.original)
#t_incr = tree.ContextTree(X, max_depth=4, tree_initialization_method=gen.incremental)
#CTM(t_incr).execute(0.08).to_str()
#CTM2(t_incr).execute(0.8).to_str()
