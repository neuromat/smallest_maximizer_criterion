#!/usr/bin/env python
'''
Linguistic case study - experimental method

Usage: ./example2.py
'''

from g4l.estimators.bic import BIC
from g4l.estimators.prune import Prune
from g4l.data import Sample
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap import Bootstrap
import numpy as np

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)


def run_method(X, instance_name='bp'):
    L_path = "%s/L_%s.npy" % (resamples_folder, instance_name)
    n_sizes = (int(X.len() * 0.3), int(X.len() * 0.9)) # 29337, 88011
    method = Prune(max_depth)
    method.fit(X)
    champion_trees = method.context_trees

    bootstrap = Bootstrap(champion_trees, resamples_file, n_sizes)
    # try loading from cache
    try:
        L = np.load(L_path)
    except:
        resample_fctry = BlockResampling(X_bp, resamples_file,
                                         n_sizes,
                                         RENEWAL_POINT)
        resample_fctry.generate(NUM_RESAMPLES, num_cores=NUM_CORES)

        L = bootstrap.calculate_likelihoods(resamples_folder, num_cores=NUM_CORES)
        np.save(L_path, L)

    diffs = bootstrap.calculate_diffs(L)
    opt_idx = bootstrap.find_optimal_tree(diffs, alpha=0.01)
    return champion_trees, opt_idx





# Create a sample object instance
cache_folder = "examples/linguistic_case_study/cache/prune"
samples_folder = "examples/linguistic_case_study"
max_depth = 4
NUM_RESAMPLES = 200
NUM_CORES = 6
RENEWAL_POINT = '4'

X_bp = Sample('%s/folha.txt' % samples_folder, [0, 1, 2, 3, 4])
X_ep = Sample('%s/publico.txt' % samples_folder, [0, 1, 2, 3, 4])
resamples_folder = '%s/resamples' % cache_folder
resamples_file = "%s/resamples.txt" % resamples_folder


champion_trees_bp, opt_idx_bp = run_method(X_bp, instance_name='bp')
champion_trees_ep, opt_idx_ep = run_method(X_ep, instance_name='ep')

import code; code.interact(local=dict(globals(), **locals()))
