import numpy as np
from g4l.models import ContextTree
from .base import Base


class BIC(Base):

    def __init__(self, c, max_depth):
        self.c = c
        self.max_depth = max_depth
        assert(max_depth > 0)

    def fit(self, X):
        """ Estimates Context Tree model using BIC """
        full_tree = ContextTree.init_from_sample(X, self.max_depth,
                                                 force_admissible=False)

        full_tree.df['likelihood_pen'] = full_tree.df.likelihood
        penalization_term = np.log(len(X.data)) * ((len(X.A)-1) * self.c)
        full_tree.df.likelihood_pen -= penalization_term

        df = self.assign_values(full_tree.df[full_tree.df.freq >= 1])
        full_tree.df = df
        full_tree.prune_unique_context_paths()
        self.context_tree = full_tree
        return self

    def assign_values(self, df):
        df['val'] = df.likelihood_pen
        df['active'] = 0
        df['indicator'] = 0

        for d in reversed(range(1, self.max_depth)):
            parents = df[(df.depth == d) & (df.freq >= 1)]
            ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
            ch_vals = ch.groupby([df.parent_idx]).apply(lambda x: x.val.sum())
            ch_vals.name = 'val2'
            ch_vals = ch_vals.to_frame().reset_index()
            ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
            ch_vals = ch_vals.set_index('node_idx')
            df = df.set_index('node_idx').combine_first(ch_vals)
            df.reset_index(inplace=True)
        max_val = df.loc[df.depth <= self.max_depth-1][['val', 'val2']]
        max_val = max_val.max(axis=1)
        df.loc[df.depth <= self.max_depth-1, 'val'] = max_val
        df.drop('val2', axis='columns', inplace=True)
        df.loc[(df.depth <= self.max_depth - 1) & (df.val > df.likelihood_pen), 'indicator'] = 1

        df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
        for d in range(2, self.max_depth + 1):
            candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
            for idx, row in candidate_nodes.iterrows():
                #if row.node == '1010':
                #    import code; code.interact(local=dict(globals(), **locals()))
                node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
                suffixes = df[df['node'].isin(node_suffixes)]
                if suffixes['indicator'].product() == 1 and row.freq > 0:
                    df.loc[(df.node == row.node), 'active'] = 1
        df.drop('val', axis='columns', inplace=True)
        return df
