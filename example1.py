#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l import SmallestMaximizerCriterion
from g4l.estimators.ctm_scanner import CTMScanner
from g4l.evaluation.bootstrap import Bootstrap
from g4l.evaluation.t_test import TTest
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

# Create a sample object instance
X = Sample('examples/example1/folha_small.txt', [0, 1, 2, 3, 4])

# Define the champion trees strategy to be used
#ctm_scan = CTMScanner(penalty_interval=(0.1, 400), epsilon=0.01)
ctm_scan = CTMScanner(penalty_interval=(0.1, 400), epsilon=0.01)


# Instantiates SMC by passing the strategies that will
# be used to generate the candidate trees
smc = SmallestMaximizerCriterion(ctm_scan, max_depth=3, read_cache_dir='examples/example1/cache', write_cache_dir='examples/example1/cache')


# Define the champion trees strategy to be used
num_resamples = 50
bootstrap = Bootstrap(X, partition_string='4')
small_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.3)
large_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.9)
t_test = TTest(small_resamples, large_resamples, alpha=0.01)

# Run estimator
smc.fit(X, t_test, processors=3)

# Returns the best tree
logging.info('best tree: %s' % smc.best_tree().to_str())

# Evaluates a new sample with the model with previously fitted params
#smc.score(Xb)


# optional one-liner call
#smc.fit(X, max_depth=4).score(Xb)
#     import code; code.interact(local=dict(globals(), **locals()))
