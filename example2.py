#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
import itertools
import glob
from scipy import stats
import sys
sys.path.insert(0,'../..')
import g4l.display
from g4l.estimators import SMC
from g4l.models import ContextTree
from g4l.estimators.prune import Prune
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap import Bootstrap
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

cache_dir = 'examples/example1/cache'
max_depth = 4
num_resamples = 200
X_bp = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
X_ep = Sample('examples/example1/publico.txt', [0, 1, 2, 3, 4])


# Instantiates SMC object
smc = SMC(max_depth, penalty_interval=(0, 400), epsilon=0.01, cache_dir=cache_dir)

# Estimates champion trees given a sample
smc.fit(X_bp)


# Select resampling generation strategy
resampling_factory = BlockResampling(X_bp, renewal_point='4')


# Estimates optimal tree using bootstrap procedure
datalen = len(X_bp.data)
bootstrap = Bootstrap(resampling_factory,
                    '%s/resamples/%s' % (cache_dir, 'bp'),
                    num_resamples,
                    resample_sizes=(datalen  * 0.3, datalen * 0.9),
                    alpha=0.01)
opt_idx = bootstrap.find_optimal_tree(smc.context_trees)
print("Optimal Context tree:", smc.context_trees[opt_idx])
