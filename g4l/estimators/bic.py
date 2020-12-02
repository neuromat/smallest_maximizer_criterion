import numpy as np
from g4l.models import ContextTree
from .base import Base


class BIC(Base):
    """
    Implements the tree estimator described by Csiszár and Talata (2006)
    in the paper Context tree estimation for not necessarily finite memory
    processes, Via BIC and MDL (see section IV)
    """

    def __init__(self, c, max_depth):
        self.c = c
        self.max_depth = max_depth
        assert(max_depth > 0)

    def fit(self, X):
        """ Estimates Context Tree model using BIC """
        full_tree = ContextTree.init_from_sample(X, self.max_depth,
                                                 force_admissible=False)

        full_tree.df['likelihood_pen'] = full_tree.df.likelihood
        penalization_term = np.log(len(X.data)) * (((1-len(X.A))/2) * self.c)
        #penalization_term = np.log(len(X.data)) * ((len(X.A)-1) * self.c)
        full_tree.df.likelihood_pen += penalization_term


        df = self.assign_values(full_tree.df[full_tree.df.freq >= 1])
        full_tree.df = df
        full_tree.prune_unique_context_paths()
        self.context_tree = full_tree
        return self

    def assign_values(self, df):
        df['val'] = df.likelihood_pen
        df['active'] = 0
        df['indicator'] = 0
        df.loc[df.depth == self.max_depth, 'val2'] = df[df.depth == self.max_depth].val
        df.loc[df.depth == self.max_depth, 'chosen'] = df[df.depth == self.max_depth].val

        for d in reversed(range(0, self.max_depth)):
            parents = df[(df.depth == d) & (df.freq >= 1)]
            ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
            ch_vals = ch.groupby([df.parent_idx]).apply(lambda x: x.chosen.sum())
            ch_vals.name = 'val2'
            ch_vals = ch_vals.to_frame().reset_index()
            ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
            ch_vals = ch_vals.set_index('node_idx')
            df = df.set_index('node_idx').combine_first(ch_vals)
            df.reset_index(inplace=True)
            depth_df = df.loc[df.depth == d]
            df.loc[df.depth == d, 'chosen'] = depth_df[['val', 'val2']].max(axis=1)
            #df[['node', 'likelihood_pen', 'p_chapeu', 'produtoria_filhos', 'chosen', 'indicator', 'active']]

        #import code; code.interact(local=dict(globals(), **locals()))
        #max_val = df.loc[df.depth <= self.max_depth-1][['val', 'val2']]
        #max_val = max_val.max(axis=1)
        #df.loc[df.depth <= self.max_depth-1, 'val'] = max_val
        df['p_chapeu'] = df.val
        df['produtoria_filhos'] = df.val2

        df.loc[(df.depth <= self.max_depth - 1) & (df.val2 > df.likelihood_pen), 'indicator'] = 1
        df.drop('val2', axis='columns', inplace=True)

        #df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
        for d in range(self.max_depth + 1):
            candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
            for idx, row in candidate_nodes.iterrows():
#                if row.node=='1000':
#                    import code; code.interact(local=dict(globals(), **locals()))
                node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
                if row.depth==1:
                    node_suffixes += ['']
                suffixes = df[df['node'].isin(node_suffixes)]
                #if suffixes['indicator'].product() == 1 and row.freq > 0:
                if suffixes['indicator'].product() == 1 and row.indicator == 0:
                    df.loc[(df.node == row.node), 'active'] = 1

        #import code; code.interact(local=dict(globals(), **locals()))
        #df[['node', 'likelihood_pen', 'p_chapeu', 'produtoria_filhos', 'chosen', 'indicator', 'active']]
        #df.to_csv('/home/arthur/tmp/99.csv', index=False)
        df.drop('val', axis='columns', inplace=True)
        return df
