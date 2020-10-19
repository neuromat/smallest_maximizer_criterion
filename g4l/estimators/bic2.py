import numpy as np
from g4l.models import ContextTree
from .base import Base

class BIC2(Base):
    context_tree = None

    def __init__(self, c, max_depth):
        self.c = c
        self.max_depth = max_depth
        assert(max_depth > 0)

    def fit(self, X):
        """ Estimates Context Tree model using BIC """
        full_tree = ContextTree.init_from_sample(X, self.max_depth)
        S_nodes = full_tree.get_nodes(max_depth=self.max_depth)
        nodes = self.assign_values(S_nodes[S_nodes.freq >= 1])

    def assign_values(self, df):
        df['val'] = df.likelihood
        df['indicator'] = 0
        for d in reversed(range(1, self.max_depth)):
            parents = df[df.depth == d]
            ch = df[df.parent_idx.isin(parents.node_idx)]
            ch_vals = ch.groupby([df.parent_idx]).apply(lambda x: x.val.sum())
            ch_vals.name = 'val2'
            ch_vals = ch_vals.to_frame().reset_index()
            ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
            ch_vals = ch_vals.set_index('node_idx')
            df = df.set_index('node_idx').combine_first(ch_vals)
            df.reset_index(inplace=True)
            parents = df[df.depth == d]
            df.loc[df.depth == d, 'val'] = parents[['val', 'val2']].max(axis=1)
            df.loc[df.depth == d, 'indicator'] = (parents.val2 > parents.val).astype(int)
            #df.drop('val2', axis='columns', inplace=True)
        return df
