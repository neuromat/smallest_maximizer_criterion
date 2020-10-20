import pandas as pd
import re
import math
import numpy as np
from g4l.models import ContextTree
from .base import Base

class BIC(Base):
    context_tree = None

    def __init__(self, c, max_depth):
        self.c = c
        self.max_depth = max_depth
        assert(max_depth > 0)

    def fit(self, X):
        """ Estimates Context Tree model using BIC """
        t = ContextTree.init_from_sample(X, self.max_depth)
        t.df.loc[t.df.active==1, 'active'] = 0
        self._bic(t, X)
        self.context_tree = t
        return self

    def fit_tree(self, tree):
        """ Estimates Context Tree model using BIC """
        t = tree.copy()
        self._bic(t, t.sample)
        self.context_tree = t
        return self

    def _bic(self, t, X):
        #import code; code.interact(local=dict(globals(), **locals()))
        self._apply_penalization(self.c, t.df, X)
        self._remove_non_contributive_nodes(t)
        self._select_active_contexts(t.df)
        self._cleanup(t.df)

    def _apply_penalization(self, c, data_frame, sample):
        data_frame['lps'] = data_frame.likelihood
        if c is not None:
            df = len(sample.A)-1
            n = len(sample.data)
            data_frame.lps -= np.log(n) * (df * c)

    def _cleanup(self, df):
        df.drop('remove_node', axis='columns', inplace=True)
        df.drop('lps', axis='columns', inplace=True)

    def _remove_non_contributive_nodes(self, t):
        df = t.df
        df['remove_node'] = 0
        df['lps2'] = None
        df['lps2'] = df['lps2'].astype(np.float64)
        for l in list(reversed(range(1, self.max_depth))):
            #import code; code.interact(local=dict(globals(), **locals()))
            parents = df[(df.depth==l) & (df.freq > 1)]
            child_nodes = df[(df.freq > 1) & (df.parent_idx.isin(parents.node_idx))]

            # >>>>>>> likelihood -> lps
            sum_lps_res = child_nodes.groupby([df.parent_idx]).apply(lambda x: x.lps.sum())
            sum_lps_res.name = 'lps2'
            sum_lps_res = sum_lps_res.to_frame().reset_index()
            sum_lps_res.rename(columns={'parent_idx':'node_idx'}, inplace=True)
            df = df.set_index('node_idx').combine_first(sum_lps_res.set_index('node_idx'))
            df.reset_index(inplace=True)
            parents = df[(df.depth==l) & (df.freq > 1)]

            # remove todo nó cuja soma da verossimilhança dos filhos é maior do que a própria verossimilhança
            nodes_to_remove = parents[(parents.num_child_nodes > 1) & (parents.lps2 > parents.lps)].node_idx
            df.loc[df.node_idx.isin(nodes_to_remove), 'remove_node'] = 1
        df.drop('lps2', axis='columns', inplace=True)
        t.df = df

    def _select_active_contexts(self, df):
        # TMP
        for i in range(5):
            df.loc[df.node_idx.isin(df[df.remove_node==1].parent_idx), 'remove_node'] = 1

        df.loc[(df.depth==1) & (df.remove_node==0), 'active'] = 1
        for depth in range(2, self.max_depth + 1):
            for idx, row in df[(df.depth==depth) & (df.remove_node==0)].iterrows():
                subnodes_str = [row.node[-(depth - m):] for m in range(1, depth)]
                subnodes = df[df['node'].isin(subnodes_str)]
                if subnodes['remove_node'].product() == 1 and row.freq > 0:
                    df.loc[(df.node==row.node), 'active'] = 1
