# import pandas as pd
# import numpy as np
# import itertools
# import math
# import sys
# sys.path.insert(0, '../..')
# import matplotlib.pyplot as plt
from g4l.data import Sample
from g4l.tree import ContextTree
import g4l.tree.generation
import g4l.tree.integrity
import g4l.estimators.ctm as ctm

max_depth = 4
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
#cache_dir = '../example1/cache'

def test_properness():

  initial_tree = ContextTree(X, max_depth=max_depth, tree_initialization_method=g4l.tree.generation.incremental_strategy)
  t = ctm.CTM(initial_tree).execute(1.536489)

  g4l.tree.integrity.check_properness(t)
  assert capital_case('semaphore') == 'Semaphore'
