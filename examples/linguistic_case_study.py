#!/usr/bin/env python
'''
Linguistic case study

Usage: python ./linguistic_case_study.py
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

cache_folder = "linguistic_case_study/cache/smc_ml_compat_VER"

samples_folder = "linguistic_case_study"
max_depth = 4
num_resamples = 200
num_cores = 6
penalty_interval = (0.1, 400)
epsilon = 0.01
renewal_point = '4'
perl_compatible = False

#(X, sample_file, cache_folder, instance_name='bp', num_cores=1):

# Create sample objects
bp_cache = os.path.join(cache_folder, 'bp')
ep_cache = os.path.join(cache_folder, 'ep')

X_ep = Sample('%s/publico.txt.bkp' % samples_folder,
              [0, 1, 2, 3, 4],
              max_depth,
              perl_compatible=perl_compatible,
              subsamples_separator='>',
              cache_file=os.path.join(ep_cache, 'sample'))


X_bp = Sample('%s/folha.txt.bkp' % samples_folder,
              [0, 1, 2, 3, 4],
              max_depth,
              subsamples_separator='>',
              perl_compatible=perl_compatible,
              cache_file=os.path.join(bp_cache, 'sample'))

#c = 0.2715885530863805
#bic = BIC(c, 4, df_method='perl', scan_offset=0, keep_data=True, perl_compatible=True)
#bic.fit(X_ep)
#t = bic.context_tree
#import code; code.interact(local=dict(globals(), **locals()))

# Execute the method above for each sample (EP and BP)
champion_trees_ep, opt_idx_ep, smc_ep = lng.run_smc(X_ep, ep_cache, instance_name='ep', perl_compatible=perl_compatible, num_cores=num_cores)
champion_trees_bp, opt_idx_bp, smc_bp = lng.run_smc(X_bp, bp_cache, instance_name='bp', perl_compatible=perl_compatible, num_cores=num_cores)




print("--------------------------")
print("Selected tree for BP: ", champion_trees_bp[opt_idx_bp].to_str(reverse=True))
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_bp)]

print("--------------------------")
print("Selected tree for EP: ", champion_trees_ep[opt_idx_ep].to_str(reverse=True))
[print(x.num_contexts(), '\t', x.to_str(reverse=True)) for x in reversed(champion_trees_ep)]

#import code; code.interact(local=dict(globals(), **locals()))

#champion_trees_bp, opt_idx_bp, smc_bp = lng.run_smc(X_bp, cache_folder, instance_name='bp', num_cores=4)
#champion_trees_ep, opt_idx_ep, smc_ep = lng.run_smc(X_ep, cache_folder, instance_name='ep', num_cores=4)
