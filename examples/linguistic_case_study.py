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
import linguistic_case_study as lng

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

cache_folder = "linguistic_case_study/cache/smc_perl_compat"

samples_folder = "linguistic_case_study"
max_depth = 4
num_resamples = 200
num_cores = 6
penalty_interval = (0.1, 400)
epsilon = 0.01
renewal_point = '4'

#(X, sample_file, cache_folder, instance_name='bp', num_cores=1):

# Create sample objects
X_bp = Sample('%s/folha.txt.bkp' % samples_folder, [0, 1, 2, 3, 4], subsamples_separator='>')
X_ep = Sample('%s/publico.txt.bkp' % samples_folder, [0, 1, 2, 3, 4], subsamples_separator='>')


# Execute the method above for each sample (EP and BP)
champion_trees_ep, opt_idx_ep, smc_ep = lng.run_smc(X_ep, cache_folder, instance_name='ep')
#import code; code.interact(local=dict(globals(), **locals()))
champion_trees_bp, opt_idx_bp, smc_bp = lng.run_smc(X_bp, cache_folder, instance_name='bp')

print("--------------------------")
print("Selected tree for BP: ", champion_trees_bp[opt_idx_bp].to_str(reverse=True))
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_bp)]

print("--------------------------")
print("Selected tree for EP: ", champion_trees_ep[opt_idx_ep].to_str(reverse=True))
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_ep)]

#import code; code.interact(local=dict(globals(), **locals()))

#champion_trees_bp, opt_idx_bp, smc_bp = lng.run_smc(X_bp, cache_folder, instance_name='bp', num_cores=4)
#champion_trees_ep, opt_idx_ep, smc_ep = lng.run_smc(X_ep, cache_folder, instance_name='ep', num_cores=4)
