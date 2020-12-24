import numpy as np
from g4l.models import ContextTree
from .base import Base
from g4l.models.builders import incremental


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

    def fit(self, X, df_method='perl'):
        """ Estimates Context Tree model using BIC """

        full_tree = ContextTree.init_from_sample(X, self.max_depth,
                                                 force_admissible=False,
                                                 initialization_method=incremental)
        full_tree.df['likelihood_pen'] = full_tree.df.likelihood

        # ---- remove it -------------
        #import code; code.interact(local=dict(globals(), **locals()))
        d = full_tree.transition_probs
        num_nodes = d[d.freq > 0].groupby(['idx']).count().next_symbol
        full_tree.df.num_child_nodes = num_nodes

        if df_method == 'g4l':
            degr_freedom = full_tree.df.num_child_nodes
        elif df_method == 'perl':
            # The perl version of this algorithm uses df = |A|-1
            # perl: $LPS = $LPMLS - log($LD) * ( ( $LA - 1 ) * $Pena );
            degr_freedom = len(X.A)-1
        elif df_method == 'csizar_and_talata':
            #degr_freedom = (1 - len(X.A)) / 2
            degr_freedom = (1 - len(X.A)) / 2
        else:
            degr_freedom = full_tree.df.num_child_nodes

        penalization_term = -(np.log(len(X.data)) * self.c * degr_freedom)

        if df_method == 'g4l':
            degr_freedom = full_tree.df.num_child_nodes
            penalization_term = -(np.log(len(X.data)) * self.c * degr_freedom)
        elif df_method == 'perl':
            # The perl version of this algorithm uses df = |A|-1
            # perl: $LPS = $LPMLS - log($LD) * ( ( $LA - 1 ) * $Pena );
            degr_freedom = len(X.A)-1
            penalization_term = -(np.log(len(X.data)) * self.c * degr_freedom)
        if df_method == 'csizar_and_talata':  # tirar, ajustar para o método acima
            penalization_term = np.log(len(X.data)) * (((1-len(X.A))/2) * self.c)  # TODO: tirar

        # /---- remove it -------------
        #penalization_term = np.log(len(X.data)) * (((1-len(X.A))/2) * self.c)



        # import code; code.interact(local=dict(globals(), **locals()))
        # likelihood_pen => n^{-c \dot df(w)}L_{w}(X^{n}_{1})
        full_tree.df.likelihood_pen += penalization_term


        df = self.assign_values(full_tree.df[full_tree.df.freq >= 1], full_tree.transition_probs)
        full_tree.df = df
        full_tree.prune_unique_context_paths()
        #full_tree.df = full_tree.df[['node', 'node_idx', 'parent_idx', 'freq', 'likelihood', 'depth', 'active', 'likelihood_pen']]
        self.context_tree = full_tree
        return self

    def calc_sum_v_children(self, df, level):
        parents = df[(df.depth == level) & (df.freq >= 1)]
        ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
        ch_vals = ch.groupby([df.parent_idx]).apply(lambda x: x['v_node'].sum())
        ch_vals.name = 'sum_v_children'
        ch_vals = ch_vals.to_frame().reset_index()
        ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
        ch_vals = ch_vals.set_index('node_idx')
        df = df.set_index('node_idx').combine_first(ch_vals)
        return df.reset_index(drop=False)


    #import code; code.interact(local=dict(globals(), **locals()))
    def assign_values(self, df, tt):

        # v_node = V^{c}_{w}(X^{n}_{1})
        df.loc[df.depth == self.max_depth, 'v_node'] = df.likelihood_pen
        df['active'] = 0
        df['indicator'] = 0
        #df.loc[df.depth == self.max_depth, 'val2'] = df[df.depth == self.max_depth].val
        #df.loc[df.depth == self.max_depth, 'chosen'] = df[df.depth == self.max_depth].val

        for d in reversed(range(0, self.max_depth)):
            df = self.calc_sum_v_children(df, d)
            depth_df = df.loc[df.depth == d]
            df.loc[df.depth == d, 'v_node'] = depth_df[['likelihood_pen', 'sum_v_children']].max(axis=1)

        cond = (df.depth < self.max_depth) & (df.sum_v_children > df.likelihood_pen)

        # indicator => \delta_{w}^{c}(X^{n}_{1})
        df.loc[cond, 'indicator'] = 1

        #import code; code.interact(local=dict(globals(), **locals()))
        df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
        for d in range(self.max_depth + 1):
            candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
            for idx, row in candidate_nodes.iterrows():
                node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
                if row.depth == 1:
                    node_suffixes += ['']
                suffixes = df[df['node'].isin(node_suffixes)]
                if suffixes['indicator'].product() == 1 and row.indicator == 0:
                    df.loc[(df.node == row.node), 'active'] = 1

        #import code; code.interact(local=dict(globals(), **locals()))
        return df
