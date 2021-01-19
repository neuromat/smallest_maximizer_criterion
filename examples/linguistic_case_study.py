#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from g4l.estimators.bic import BIC
from g4l.estimators.smc import SMC
from g4l.estimators.prune import Prune
from g4l.data import Sample
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap import Bootstrap
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

cache_folder = "linguistic_case_study/cache/smc_perl_g4l_fix"
resamples_folder = '%s/resamples' % cache_folder
samples_folder = "linguistic_case_study"
max_depth = 4
num_resamples = 200
num_cores = 6
penalty_interval = (0.1, 400)
epsilon = 0.01
renewal_point = '4'


def run_smc(X, instance_name='bp'):
    L_path = "%s/L_%s.npy" % (resamples_folder, instance_name)
    resamples_file = "%s/resamples.%s.txt" % (resamples_folder, instance_name)
    data_len = len(X.data)
    n_sizes = (int(data_len * 0.3), int(data_len * 0.9))

    # Execute SMC to estimate champion trees
    smc = SMC(max_depth,
              penalty_interval=penalty_interval,
              epsilon=epsilon, scan_offset=0, df_method='g4l',
              cache_dir=cache_folder, perl_compatible=True)
    smc.fit(X)
    champion_trees = smc.context_trees

    # Use bootstrap
    bootstrap = Bootstrap(champion_trees, resamples_file, n_sizes)
    try:
        # Use precomputed likelihoods when available
        L = np.load(L_path)
    except:
        # Generate samples using block resampling strategy
        resample_fctry = BlockResampling(X, resamples_file,
                                         n_sizes,
                                         renewal_point)
        resample_fctry.generate(num_resamples, num_cores=num_cores)

        # Calculate tree likelihoods for all resamples
        L = bootstrap.calculate_likelihoods(resamples_folder, num_cores=num_cores)
        # Save to cache
        np.save(L_path, L)
    # Select optimal tree among the champion trees using t-test
    opt_idx = bootstrap.find_optimal_tree(L, alpha=0.01)
    return champion_trees, opt_idx, smc



# Create sample objects
X_bp = Sample('%s/folha.txt.bkp' % samples_folder, [0, 1, 2, 3, 4], subsamples_separator='>')
X_ep = Sample('%s/publico.txt.bkp' % samples_folder, [0, 1, 2, 3, 4], subsamples_separator='>')


# Execute the method above for each sample (EP and BP)
champion_trees_ep, opt_idx_ep, smc_ep = run_smc(X_ep, instance_name='ep')
#import code; code.interact(local=dict(globals(), **locals()))
champion_trees_bp, opt_idx_bp, smc_bp = run_smc(X_bp, instance_name='bp')

print("--------------------------")
print("Selected tree for BP: ", champion_trees_bp[opt_idx_bp].to_str())
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_bp)]

print("--------------------------")
print("Selected tree for EP: ", champion_trees_ep[opt_idx_ep].to_str())
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_ep)]

