#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''


# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l.estimators.ctm2 import CTM2
from g4l.estimators.ctm import CTM
from g4l.estimators.smc import SMC
from g4l.estimators.prune import Prune
from g4l.models import ContextTree
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
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
#r = CTM2(0.01, 4).fit(X).context_tree.to_str()
r = CTM2(0.01, 4).fit(X).context_tree.to_str()
import code; code.interact(local=dict(globals(), **locals()))

smc = SMC(4, penalty_interval=(0, 800), epsilon=0.00001)
smc.fit(X)
[print(x.to_str()) for x in smc.context_trees]



# Define the champion trees strategy to be used
#ctm_scan = CTMScanner(penalty_interval=(0.1, 400), epsilon=0.01)

# Define the champion trees strategy to be used
#num_resamples = 200
#bootstrap = Bootstrap(X, partition_string='4')
#small_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.3)
#large_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.9)
#t_test = TTest(small_resamples, large_resamples, alpha=0.01)

# Run estimator
#smc.fit(X, t_test, processors=3)

# Returns the best tree
#logging.info('best tree: %s' % smc.best_tree().to_str())

# Evaluates a new sample with the model with previously fitted params
#smc.score(Xb)


# optional one-liner call
#smc.fit(X, max_depth=4).score(Xb)
#     import code; code.interact(local=dict(globals(), **locals()))
