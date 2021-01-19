# import pandas as pd
# import numpy as np
# import itertools
# import math
# import sys
# sys.path.insert(0, '../..')
# import matplotlib.pyplot as plt
import pytest
from g4l.data import Sample
#from g4l.tree import ContextTree
#import g4l.tree.generation
#import g4l.tree.integrity
#import g4l.estimators.ctm as ctm
#import g4l.estimators.prune as prune

# max_depth = 4
# X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
#cache_dir = '../example1/cache'


# def test_suffix_property():
#   data_len = X.len()
#   initial_tree = ContextTree(X, max_depth=max_depth, tree_initialization_method=g4l.tree.generation.original)
#   p = prune.Prune(initial_tree)
#   p.execute()
#   for t in p.context_trees:
#     assert(g4l.tree.integrity.satisfies_suffix_property(t)==True)
#
#
# def test_completeness():
#   initial_tree = ContextTree(X, max_depth=max_depth, tree_initialization_method=g4l.tree.generation.incremental)
#   p = prune.Prune(initial_tree)
#   p.execute()
#   for t in p.context_trees:
#     assert(g4l.tree.integrity.satisfies_completeness(t)==True)
#

