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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)


def run_smc(X, instance_name='bp'):
    L_path = "%s/L_%s.npy" % (resamples_folder, instance_name)
    n_sizes = (int(X.len() * 0.3), int(X.len() * 0.9)) # 29337, 88011
    smc = SMC(max_depth,
              penalty_interval=(0.1, 800),
              epsilon=0.01,
              cache_dir=cache_folder)
    smc.fit(X)
    champion_trees = smc.context_trees

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
    return champion_trees, opt_idx, smc





# Create a sample object instance
cache_folder = "examples/linguistic_case_study/cache/smc5"
samples_folder = "examples/linguistic_case_study"
max_depth = 4
NUM_RESAMPLES = 200
NUM_CORES = 6
RENEWAL_POINT = '4'

X_bp = Sample('%s/folha.txt' % samples_folder, [0, 1, 2, 3, 4])
X_ep = Sample('%s/publico.txt' % samples_folder, [0, 1, 2, 3, 4])
resamples_folder = '%s/resamples' % cache_folder
resamples_file = "%s/resamples.txt" % resamples_folder


#bic_tree = BIC(0.336811872024198, max_depth).fit(X_ep, df_method='perl').context_tree
#print(bic_tree.to_str(reverse=True))
#import code; code.interact(local=dict(globals(), **locals()))
# bic_tree = BIC(164.648626714437, max_depth).fit(X_bp).context_tree

champion_trees_bp, opt_idx_bp, smc_bp = run_smc(X_bp, instance_name='bp')
champion_trees_ep, opt_idx_ep, smc_ep = run_smc(X_ep, instance_name='ep')

print("Selected tree for BP: ", champion_trees_bp[opt_idx_bp].to_str())
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_bp)]

print("Selected tree for EP: ", champion_trees_ep[opt_idx_ep].to_str())
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_ep)]



import code; code.interact(local=dict(globals(), **locals()))



