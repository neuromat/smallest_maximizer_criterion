import numpy as np
from g4l.models import ContextTree
from g4l.models.builders import incremental
import logging
from tqdm import tqdm


def fit(X, c, max_depth, df_method, scan_offset, comp, keep_data=False, clean=True):
    """ Estimates Context Tree model using BIC """
    full_tree = ContextTree.init_from_sample(X, max_depth,
                                             force_admissible=False,
                                             initialization_method=incremental,
                                             scan_offset=scan_offset)
    df, transition_probs = full_tree.df, full_tree.transition_probs
    # likelihood_pen => n^{-c \dot df(w)}L_{w}(X^{n}_{1})
    deg_f = degrees_of_freedom(df_method, full_tree)
    #df['likelihood_pen'] = df.likelihood
    penalty = penalty_term(X.len(), c, deg_f)
    df['likelihood_pen'] = penalty + df.likelihood
    full_tree.df = assign_values(max_depth, df[df.freq >= 1], comp)
    full_tree.prune_unique_context_paths()
    tt = clean_columns(full_tree, keep_data)
    return tt


def calc_sum_v_children(df, level, max_depth):
    parents = df[(df.depth == level) & (df.freq >= 1)]
    ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
    ch_vals = ch.groupby([df.parent_idx]).v_sum.sum()
    ch_vals.name = 'v_sum'
    ch_vals = ch_vals.to_frame().reset_index()
    ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
    ch_vals = ch_vals.set_index('node_idx')
    df = df.set_index('node_idx').combine_first(ch_vals)
    return df.reset_index(drop=False)


def assign_values(max_depth, df, comp=False):
    # v_node = V^{c}_{w}(X^{n}_{1})
    #import code; code.interact(local=dict(globals(), **locals()))
    df = df.copy()
    df.loc[df.depth == max_depth, 'v_node'] = df.likelihood_pen
    df.loc[df.depth == max_depth, 'v_node_sum'] = df.likelihood_pen
    df['active'] = 0
    df['indicator'] = 0
    df.set_index('node_idx', inplace=True)

    for d in reversed(range(0, max_depth)):
        parents = df[(df.depth == d) & (df.freq >= 1)]
        ch = df[(df.parent_idx.isin(parents.index)) & (df.freq >= 1)]
        ch_vals = ch.groupby([df.parent_idx]).v_node.sum()
        df.loc[ch_vals.index, 'v_node_sum'] = ch_vals
        df.loc[ch_vals.index, 'v_node'] = df.loc[ch_vals.index][['likelihood_pen', 'v_node_sum']].max(axis=1)
        df['comp_aux'] = df.v_node_sum > df.v_node

        #ch2 = df[(df.parent_idx.isin(parents.index)) & (df.freq > 1)]

        #df = calc_sum_v_children(df, d, max_depth)
        #depth_df = df.loc[df.depth == d]
        #df.loc[df.depth == d, 'v_node'] = depth_df[['likelihood_pen', 'v_node']].max(axis=1)
    cond = (df.depth < max_depth) & (df.v_node > df.likelihood_pen)
    # indicator => \delta_{w}^{c}(X^{n}_{1})
    df.loc[cond, 'indicator'] = 1


    if comp==True:
        # Make the code compatible with the perl version
        # this one fits condition in line 581
        # ( $TREE[ $l + 1 ][ ( $m + $j ) ][2] > 1 )
        for i, r in df.iterrows():
            ch = df[df.parent_idx==i]
            df.loc[df.index==i, 'num_child_nodes'] = len(ch[ch.freq > 1])
        df.loc[(cond & (df.num_child_nodes <= 1)), 'indicator'] = 0
        df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
    else:
        if df[df.node==''].indicator.values[0] == 0:
            #import code; code.interact(local=dict(globals(), **locals()))
            df.loc[df.node=='', 'active'] = 1
            return df

    nd = df.set_index('node')
    for d in range(max_depth + 1):
        candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
        itr = candidate_nodes.iterrows()
        if logging.getLogger().level == logging.DEBUG:
            itr = tqdm(itr, total=len(candidate_nodes))
        for idx, row in itr:
            node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
            if comp==False:
                node_suffixes += ['']
            #if row.depth == 1:
               #node_suffixes += ['']
            #import code; code.interact(local=dict(globals(), **locals()))
            suffixes = nd.loc[node_suffixes]
            if suffixes['indicator'].product() == 1 and row.indicator == 0:
                df.loc[idx, 'active'] = 1
    #import code; code.interact(local=dict(globals(), **locals()))
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


def clean_columns(t, keep_data):
    logging.debug('Cleaning...')
    """
    Removes non-relevant info
    """
    if not keep_data:
        t.df = t.tree()
    node_idxs = t.df.node_idx.unique()
    tr = t.transition_probs.set_index('idx').loc[node_idxs]
    tr = tr[tr.prob > 0]
    t.transition_probs = tr

    # fs = ['node', 'node_idx', 'parent_idx', 'freq', 'likelihood', 'depth', 'active', 'likelihood_pen']
    # t.df = t.df[fs]
    return t
