#import code; code.interact(local=dict(globals(), **locals()))

#!/usr/bin/env python
'''
Linguistic case study

Usage: ./example1.py
'''

# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html


from g4l.models import ContextTree
from collections import defaultdict
import numpy as np
import logging

t = ContextTree.load_from_file('fixtures/2.tree')
t.to_str(reverse=False)




from ete3 import TreeNode



def connect_node(node, dic, df, parent_idx):
    if parent_idx < 0:
        return
    parent_node = df.loc[parent_idx]
    try:
        parent = dic[parent_node.node]
    except KeyError:
        parent = TreeNode(name=parent_node.node)
        parent.add_feature('freq', parent_node.freq)
        parent.add_feature('context', parent_node.node)
        parent.add_feature('idx', parent_node.node)
        dic[parent_node.node] = parent
    if len(parent.search_nodes(name=node.name))==0:
        parent.add_child(node)
    connect_node(parent, dic, df, parent_node.parent_idx)


dic = dict()
root_node = t.df.set_index('node').loc['']
root = TreeNode(name='')
root.add_feature('freq', root_node.freq)
root.add_feature('context', root_node.index)
root.add_feature('idx', root_node.index)
dic[''] = root

df = t.df.set_index('node_idx')
for i, x in t.tree().iterrows():
    l = TreeNode(name=x.node)
    l.add_feature('freq', x.freq)
    l.add_feature('idx', x.node)
    l.add_feature('context', x.node)
    dic[x.node] = l
    connect_node(l, dic, df, x.parent_idx)



import code; code.interact(local=dict(globals(), **locals()))


#d = defaultdict(lambda: [])
# Create a sample object instance

import code; code.interact(local=dict(globals(), **locals()))
#
