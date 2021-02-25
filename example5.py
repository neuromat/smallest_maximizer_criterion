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
max_depth = 6
num_cores = 6
sample_idx = 2
from g4l.util.mat import MatSamples
fld = 'examples/simulation_study/samples'
X = MatSamples(fld, 'model1',
               5000, [0, 1],
               max_depth,
               scan_offset=6).sample_by_idx(1)
#
c = 0
#b = BIC(c, 6, scan_offset=0, df_method='perl', perl_compatible=True).fit(X).context_tree
#b = BIC(c, 6, scan_offset=6, df_method='perl', keep_data=True, perl_compatible=False).fit(X).context_tree
#print(b.to_str())
#df = b.df
#import code; code.interact(local=dict(globals(), **locals()))


smc = SMC(max_depth,
          penalty_interval=(0, 100),
          epsilon=0.01,
          cache_dir='/home/arthur/tmp/smc/001',
          callback_fn=None,
          df_method='csizar_and_talata',
          perl_compatible=False)
smc.fit(X)
n_sizes = (X.len() * 0.3, X.len() * 0.9)
RENEWAL_POINT = 1
tree_found, _ = smc.optimal_tree(200, n_sizes, 0.01, RENEWAL_POINT, num_cores=num_cores)

import code; code.interact(local=dict(globals(), **locals()))

for i, tree in enumerate(smc.context_trees):
    print(smc.thresholds[i], '\t', tree.to_str(), tree.log_likelihood())

print("Found: ", tree_found)
smc.clean()
import code; code.interact(local=dict(globals(), **locals()))
