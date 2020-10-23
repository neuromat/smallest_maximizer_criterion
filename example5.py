#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l.estimators.bic import BIC
from g4l.estimators.prune import Prune
#from g4l.estimators.ctm_scanner import CTMScanner
#import g4l.tree.generation as gen
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

filename = "/home/arthur/Documents/Neuromat/projects/SMC/arquivo/data/model1_5000.csv"

ff = [x.replace(',', '') for x in open(filename).read().split('\n')]
samp = ff[-2]
X = Sample(None, [0, 1], data=ff[-2])
print(BIC(0, 6).fit(X).context_tree.to_str())

#initial_tree = ContextTree.init_from_sample(X, max_depth=6)
#p = Prune(6).fit(X)

import code; code.interact(local=dict(globals(), **locals()))
