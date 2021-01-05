#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html


#from g4l.estimators.ctm_scanner import CTMScanner
#import g4l.tree.generation as gen
from g4l.estimators.smc import SMC
from g4l.estimators.bic import BIC
from g4l.data import Sample
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)


# Create a sample object instance

filename = "/home/arthur/Documents/Neuromat/projects/SMC/arquivo/data/model1_5000.csv"
max_depth = 6
ff = [x.replace(',', '') for x in open(filename).read().split('\n')]
samp = ff[0]
X = Sample(None, [0, 1], data=samp)

c = 0.16698
print(BIC(c, 6, df_method='csizar_and_talata').fit(X).context_tree.to_str())
b = BIC(c, 6).fit(X).context_tree
df = b.df

smc = SMC(max_depth,
          penalty_interval=(0.1, 400),
          epsilon=0.01,
          cache_dir=None,
          callback_fn=None,
          df_method='csizar_and_talata')
smc.fit(X)
for tree in smc.context_trees:
    print(tree.to_str())

import code; code.interact(local=dict(globals(), **locals()))
