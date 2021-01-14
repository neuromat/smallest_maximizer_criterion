#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html

from g4l.estimators.smc import SMC
from g4l.estimators.bic import BIC
from g4l.data import Sample
import logging

# Create a sample object instance
X = Sample('examples/linguistic_case_study/publico.txt', [0, 1, 2, 3, 4])


c = 0
b = BIC(c, 4, scan_offset=0, df_method='csizar_and_talata', perl_compatible=False).fit(X).context_tree
df = b.df
df = df.drop('comp_aux', axis=1)
print(df[df.depth <= 1] )
print("BIC tree (c=%s):" % c, b.to_str())
import code; code.interact(local=dict(globals(), **locals()))

max_depth = 4
smc = SMC(max_depth,
          penalty_interval=(0, 400),
          epsilon=0.01,
          cache_dir=None,
          callback_fn=None,
          scan_offset=0,
          df_method='csizar_and_talata',
          perl_compatible=False)
smc.fit(X)
for tree in smc.context_trees:
    print(tree.to_str(), tree.log_likelihood())

#
