#!/usr/bin/env python
'''
This is an example of how to estimate context trees using the BIC method

Usage: python ./estimation_using_bic.py
'''

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from g4l.estimators import BIC
from g4l.data import Sample


# Create a sample object instance
X = Sample('./linguistic_case_study/folha.txt', [0, 1, 2, 3, 4])
max_depth = 4
c = 0.34
# Instantiate BIC object and use the 'fit' method
bic = BIC(c, max_depth, df_method='g4l', scan_offset=6).fit(X)

print("\nNodes: \n", bic.context_tree.tree()[['node', 'freq', 'likelihood']])
print("\n\nTransition probs: \n", bic.context_tree.transition_probs.head(10))
print("\n\nResulting tree: ", bic.context_tree.to_str())
print("Number of contexts: ", bic.context_tree.num_contexts())
