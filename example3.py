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



bic_tree = BIC(154, max_depth).fit(X_bp).context_tree

#print(bic_tree.to_str())
import pandas as pd
for i, row in pd.read_csv('tests/perl_bic_results_folha.csv').iterrows():
    c = row[0]
    print(i, 'c = ', c)
    bic_tree = BIC(c, max_depth).fit(X_bp).context_tree
    if not (bic_tree.to_str() == row[1].strip()):
        print('diff ', c)
        print(bic_tree.to_str())
        print(row[1].strip())
        #import code; code.interact(local=dict(globals(), **locals()))



# bic_tree = BIC(164.648626714437, max_depth).fit(X_bp).context_tree

#champion_trees_bp, opt_idx_ep = run_smc(X_bp, instance_name='bp')
#champion_trees_ep, opt_idx_ep = run_smc(X_ep, instance_name='ep')

#import code; code.interact(local=dict(globals(), **locals()))
