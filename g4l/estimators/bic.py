import logging
import numpy as np
from tqdm import tqdm

"""BIC estimator logics

This file contains the main methods for BIC estimator
"""


def assign_values(max_depth, df, comp=False):
    # v_node = V^{c}_{w}(X^{n}_{1})
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
    cond = (df.depth < max_depth) & (df.v_node > df.likelihood_pen)
    # indicator => \delta_{w}^{c}(X^{n}_{1})
    df.loc[cond, 'indicator'] = 1

    if comp == True:
        # Make the code compatible with the perl version
        # this one fits condition in line 581
        # ( $TREE[ $l + 1 ][ ( $m + $j ) ][2] > 1 )
        for i, r in df.iterrows():
            ch = df[df.parent_idx == i]
            df.loc[df.index == i, 'num_child_nodes'] = len(ch[ch.freq > 1])
        df.loc[(cond & (df.num_child_nodes <= 1)), 'indicator'] = 0
        df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
    else:
        if df[df.node == ''].indicator.values[0] == 0:
            df.loc[df.node == '', 'active'] = 1
            return df

    nd = df.set_index('node')
    for d in range(max_depth + 1):
        candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
        itr = candidate_nodes.iterrows()
        if logging.getLogger().level == logging.DEBUG:
            itr = tqdm(itr, total=len(candidate_nodes))
        for idx, row in itr:
            node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
            if comp == False:
                node_suffixes += ['']
            suffixes = nd.loc[node_suffixes]
            if suffixes['indicator'].product() == 1 and row.indicator == 0:
                df.loc[idx, 'active'] = 1
    return df


def penalty_term(sample_len, c, degr_freedom):
    """Returns the penalty value to be applied to node likelihoods

    Arguments:
        sample_len {int} -- The sample length
        c {float} -- The penalization constant
        degr_freedom {float} -- Degrees of freedom

    Returns:
        float -- Penalization value
    """
    return np.log(sample_len) * c * degr_freedom


def clean_columns(tree, keep_data):
    logging.debug('Cleaning...')
    """
    Removes non-relevant info
    """

    # removes non-context nodes unless keep_data is True
    if not keep_data:
        tree.df = tree.tree()

    # removes rows from transitions table where
    # transition probability is 0
    node_idxs = tree.df.node_idx.unique()
    tr = tree.transition_probs.set_index('idx').loc[node_idxs]
    tr = tr[tr.prob > 0]
    tree.transition_probs = tr
    return tree
