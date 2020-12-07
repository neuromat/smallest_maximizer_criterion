#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html
import sys
sys.path.insert(0, '../..')
from g4l.estimators import BIC
from g4l.estimators.prune import Prune
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap import Bootstrap
from g4l.estimators import SMC
from g4l.estimators import Prune
from g4l.models import integrity
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
from examples.example2 import models
model1 = models.get_model('model1')
logging.info("Generating sample")
dt = model1.generate_sample(20000, [0, 1])
logging.debug("Generating sample - done")
X = Sample(None, [0, 1], data=dt)
num_resamples = 10
cache_dir = 'examples/example1/cache'
#tt = BIC(0, 6).fit(X).context_tree
#print(tt.to_str())


#import code; code.interact(local=dict(globals(), **locals()))
# logging.info("Estimating trees using SMC")
# smc = SMC(6, penalty_interval=(0, 400), epsilon=0.00001, cache_dir=None)
# smc.fit(X)
# [print(x.to_str()) for x in smc.context_trees]


logging.info("Estimating trees using Pruning strategy")
prune = Prune(6)
prune.fit(X)
print("Estimated trees:")
[print(x.log_likelihood(), '\t', x.to_str()) for x in prune.context_trees]

datalen = len(X.data)
resampling_factory = BlockResampling(X, renewal_point='1')
bootstrap = Bootstrap(resampling_factory,
                      '%s/resamples/%s' % (cache_dir, 'bp'),
                      num_resamples,
                      resample_sizes=(datalen * 0.3, datalen * 0.9),
                      num_cores=6,
                      alpha=0.01)

opt_idx = bootstrap.find_optimal_tree(prune.context_trees)
print('Selected tree:', opt_idx)
print(prune.context_trees[opt_idx].to_str())


# Tree A:   c = 1.536489   (11 contexts)
# tree_a = "000 1 10 100 2 20 200 3 30 300 4"

#pruned_trees = Prune(initial_tree).execute()

import code; code.interact(local=dict(globals(), **locals()))
