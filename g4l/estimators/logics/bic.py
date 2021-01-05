import numpy as np
from g4l.models import ContextTree
from g4l.models.builders import incremental


def fit(X, c, max_depth, df_method):
    """ Estimates Context Tree model using BIC """
    full_tree = ContextTree.init_from_sample(X, max_depth,
                                             force_admissible=False,
                                             initialization_method=incremental)
    df, transition_probs = full_tree.df, full_tree.transition_probs
    # likelihood_pen => n^{-c \dot df(w)}L_{w}(X^{n}_{1})
    deg_f = degrees_of_freedom(df_method, full_tree)
    #df['likelihood_pen'] = df.likelihood
    penalty = penalty_term(len(X.data), c, deg_f)
    df['likelihood_pen'] = penalty + df.likelihood
    full_tree.df = assign_values(max_depth, df[df.freq >= 1])
    full_tree.prune_unique_context_paths()
    return clean_columns(full_tree)


def calc_sum_v_children(df, level):
    parents = df[(df.depth == level) & (df.freq >= 1)]
    ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
    ch_vals = ch.groupby([df.parent_idx]).apply(lambda x: x['v_node'].sum())
    ch_vals.name = 'sum_v_children'
    ch_vals = ch_vals.to_frame().reset_index()
    ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
    ch_vals = ch_vals.set_index('node_idx')
    df = df.set_index('node_idx').combine_first(ch_vals)
    return df.reset_index(drop=False)


def assign_values(max_depth, df):

    # v_node = V^{c}_{w}(X^{n}_{1})
    df.loc[df.depth == max_depth, 'v_node'] = df.likelihood_pen
    df['active'] = 0
    df['indicator'] = 0

    for d in reversed(range(0, max_depth)):
        df = calc_sum_v_children(df, d)
        depth_df = df.loc[df.depth == d]
        df.loc[df.depth == d, 'v_node'] = depth_df[['likelihood_pen', 'sum_v_children']].max(axis=1)

    cond = (df.depth < max_depth) & (df.sum_v_children > df.likelihood_pen)

    # indicator => \delta_{w}^{c}(X^{n}_{1})
    df.loc[cond, 'indicator'] = 1
    df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1

    for d in range(max_depth + 1):
        candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
        for idx, row in candidate_nodes.iterrows():
            node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
            if row.depth == 1:
                node_suffixes += ['']
            suffixes = df[df['node'].isin(node_suffixes)]
            if suffixes['indicator'].product() == 1 and row.indicator == 0:
                df.loc[(df.node == row.node), 'active'] = 1
    return df


def degrees_of_freedom(meth, t):
    return {'g4l': g4l,
            'perl': original_perl,
            'csizar_and_talata': csizar_and_talata
            }[meth](t)


def g4l(t):
    """
    The paper considers the degrees of freedom as
    the number of possible transitions from a node to other symbols
    """
    d = t.transition_probs
    num_nodes = d[d.freq > 0].groupby(['idx']).count().next_symbol
    #t.df.num_child_nodes = num_nodes
    return -num_nodes


def csizar_and_talata(t):
    return ((1 - len(t.sample.A)) / 2)


def original_perl(t):
    """
    The perl version of this algorithm uses df = |A|-1
    perl: $LPS = $LPMLS - log($LD) * ( ( $LA - 1 ) * $Pena );
    """
    return -(len(t.sample.A)-1)


def penalty_term(sample_len, c, degr_freedom):
    return np.log(sample_len) * c * degr_freedom


def clean_columns(t):
    """
    Removes unused columns
    """
    # fs = ['node', 'node_idx', 'parent_idx', 'freq', 'likelihood', 'depth', 'active', 'likelihood_pen']
    # t.df = t.df[fs]
    return t
