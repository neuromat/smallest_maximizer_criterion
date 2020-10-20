#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l.estimators import BIC
#from g4l.estimators import BIC2
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
dt = model1.generate_sample(10000, [0, 1])
logging.debug("Generating sample - done")
X = Sample(None, [0, 1], data=dt)


tt = BIC(0, 6).fit(X).context_tree
print(tt.to_str())

#import code; code.interact(local=dict(globals(), **locals()))
integrity.check_admissibility(tt, X)
##for c in [0, 0.5, 200.0]:
  #print(c, CTM(c, 6).fit(X).context_tree.to_str())

#import code; code.interact(local=dict(globals(), **locals()))
logging.info("Estimating trees using SMC")
smc = SMC(6, penalty_interval=(0, 400), epsilon=0.00001, cache_dir=None)
smc.fit(X)
[print(x.to_str()) for x in smc.context_trees]


logging.info("Estimating trees using Pruning strategy")
prune = Prune(6)
prune.fit(X)

[print(x.to_str()) for x in prune.context_trees]

# Tree A:   c = 1.536489   (11 contexts)
# tree_a = "000 1 10 100 2 20 200 3 30 300 4"

#pruned_trees = Prune(initial_tree).execute()

import code; code.interact(local=dict(globals(), **locals()))
