#!/usr/bin/env python
'''
Linguistic case study

Usage: python ./linguistic_case_study.py
'''
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from g4l.sample import Sample
import pandas as pd
from g4l.bootstrap import Bootstrap
from g4l.smc_bic import SMC
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

cache_folder = "tmp/linguistic_case_study/smc2"
samples_folder = "linguistic_case_study"
max_depth = 4
num_resamples = 200
num_cores = 4
penalty_interval = (0.1, 400)
epsilon = 0.01
alpha = 0.01
renewal_point = '4'
perl_compatible = True
n_sizes = (0.3, 0.9)


def create_sample(file, cache_fld, instance_name):
    return Sample('%s/publico.txt.bkp' % samples_folder,
                  [0, 1, 2, 3, 4],
                  max_depth,
                  perl_compatible=perl_compatible,
                  subsamples_separator='>',
                  cache_file=os.path.join(cache_fld, instance_name, 'sample'))


def run_smc(source_file, cache_fld, instance_name):
    X = create_sample(source_file, cache_fld)
    bootstrap = Bootstrap()
    smc = SMC(bootstrap,
              penalty_interval=penalty_interval,
              n_sizes=n_sizes,
              alpha=alpha,
              epsilon=epsilon,
              cache_fld=cache_fld,
              perl_compatible=perl_compatible,
              num_cores=num_cores)
    smc.fit(X)
    print("--------------------------")
    optimal_tree = smc.optimal_tree
    champion_trees = smc.context_trees
    print("Selected tree (%s): %s" % (instance_name, optimal_tree.to_str(reverse=True)) )
    print("--------------------------")
    print("Champion trees:")

    df_trees = pd.DataFrame(columns=['c', 'num_contexts', 'tree'])
    for i, t in enumerate(champion_trees):
        c = smc.thresholds[i]
        df_trees.loc[len(df_trees)] = [c, t.num_contexts(), t.to_str(reverse=True)]

    print(df_trees.sort_values('num_contexts', ascending=True))
#import code; code.interact(local=dict(globals(), **locals()))
