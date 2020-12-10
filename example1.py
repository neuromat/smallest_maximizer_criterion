#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

from g4l.estimators.bic import BIC
from g4l.estimators.smc import SMC
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


def run_smc(X, instance_name='bp'):
    L_path = "%s/L_%s.npy" % (resamples_folder, instance_name)
    n_sizes = (int(len(X.data) * 0.3), int(len(X.data) * 0.9)) # 29337, 88011
    smc = SMC(max_depth,
              penalty_interval=(0.1, 400),
              epsilon=0.01,
              cache_dir=cache_folder)
    smc.fit(X)
    champion_trees = smc.context_trees

    # try loading from cache
    try:
        L = np.load(L_path)
    except:
        resample_fctry = BlockResampling(X_bp, resamples_file,
                                         n_sizes,
                                         RENEWAL_POINT)
        resample_fctry.generate(NUM_RESAMPLES, num_cores=NUM_CORES)
        bootstrap = Bootstrap(champion_trees, resamples_file, n_sizes)
        L = bootstrap.calculate_likelihoods(resamples_folder, num_cores=NUM_CORES)
        np.save(L_path, L)

    diffs = bootstrap.calculate_diffs(L)
    opt_idx = bootstrap.find_optimal_tree(diffs, alpha=0.01)
    return champion_trees, opt_idx

# Create a sample object instance
cache_folder = "examples/linguistic_case_study/cache/smc"
samples_folder = "examples/linguistic_case_study"
max_depth = 4
NUM_RESAMPLES = 200
NUM_CORES = 6
RENEWAL_POINT = '4'

X_bp = Sample('%s/folha.txt' % samples_folder, [0, 1, 2, 3, 4])
X_ep = Sample('%s/publico.txt' % samples_folder, [0, 1, 2, 3, 4])
resamples_folder = '%s/resamples' % cache_folder
resamples_file = "%s/resamples.txt" % resamples_folder



bic_tree = BIC(164.648626714437, max_depth).fit(X_bp).context_tree
bic_tree.to_str()
import code; code.interact(local=dict(globals(), **locals()))
# bic_tree = BIC(164.648626714437, max_depth).fit(X_bp).context_tree

# run_smc(X_bp, instance_name='bp')


import code; code.interact(local=dict(globals(), **locals()))
