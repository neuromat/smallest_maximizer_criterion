import numpy as np
import pandas as pd
from operator import add


def nodes_and_transitions(sample):
    """Creates tables with contexts and transition probabilites

    For a given sample, this method returns two Pandas dataframes.
    The first one contains the node counts and likelihoods,
    while the second one contains the transition probabilities.

    === Example of node's initial DataFrame.
    node_idx    node   freq  active  depth  parent_idx   likelihood
           8    1010   5900       0      4           9 -3596.440982
           9     010   8405       0      3          10 -5121.113204
          10      10   8405       0      2          11 -5121.113204
          11       0  11589       0      1          52 -6814.372550


    === Example of transition probabilities DataFrame ===

    idx next_symbol  freq      prob   likelihood
      8           0  1761  0.298475 -2129.173188
      8           1  4139  0.701525 -1467.267794
      9           0  2506  0.298156 -3032.608109
      9           1  5899  0.701844 -2088.505095
     10           0  2506  0.298156 -3032.608109


    Arguments:
        sample {Sample} -- Sample object

    Returns:
        tuple(DataFrame, DataFrame) -- Counts and transitions tables
    """

    df = pd.DataFrame()
    # count frequencies of each unique subsequence of size 1..max_depth
    df, transition_probs = count_subsequence_frequencies(df, sample)
    # create depth-related info columns
    df = remove_last_level(df, sample.max_depth)
    # create parent relationship between nodes
    df = bind_parent_nodes(df)
    # calculate nodes likelihoods
    df = calc_node_likelihood(df, transition_probs)
    # remove unuseful data
    df = cleanup(df, sample.max_depth)
    return df, transition_probs


def remove_last_level(df, max_depth):
    """ Removes max_depth + 1 nodes

    Node frequencies are initially calculated for nodes
    with length max_depth + 1 in order to efficiently extract
    the transition. Since transitions were already calculated
    we can drop the nodes with length == (max_depth + 1)
    """
    return df[df.depth <= max_depth]


def count_subsequence_frequencies(df, sample):
    cols = {'index': 'node', 'N': 'freq'}
    df = sample.F['N'].to_frame().reset_index().rename(columns=cols).copy()
    df['active'] = 0
    df = create_indexes(df)
    df2 = sample.F[[i for i in range(len(sample.A))]].astype(int)
    df2['idx'] = df.reset_index().set_index('node')['node_idx']
    df2 = df2.melt(id_vars=['idx'], var_name='next_symbol', value_name='freq')
    df2 = df2.sort_values(['idx', 'next_symbol']).reset_index(drop=True).set_index('idx')
    df2.freq = df2.freq.astype(int)
    df2['prob'] = df2.freq / df.freq
    transition_probs = df2.reset_index()
    return df, transition_probs


def create_indexes(df):
    df['depth'] = df.node.str.len()
    df.reset_index(inplace=True, drop=True)
    df.index.name = 'node_idx'
    return df


def calc_node_likelihood(df, transition_probs):
    probs = transition_probs
    L = probs.freq[probs.freq > 0] * np.log(probs.prob[probs.freq > 0])
    transition_probs['likelihood'] = L

    fn = lambda s: s.likelihood.sum()
    df['likelihood'] = transition_probs.groupby(['idx']).apply(fn)
    return df


def cleanup(df, max_depth):
    df.reset_index(inplace=True)
    df = df[df.depth <= max_depth]
    return df


def calculate_num_child_nodes(df):
    """
    Given a context tree dataframe, this method creates a column with
    the number of (immediate) children
    """

    df['active_children'] = 0
    num_child_nodes = (df[df.depth >= 1]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .count()).node_idx

    active_children = (df[(df.depth >= 1) & (df.active == 1)]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .sum()).active
    if df.index.name != 'node_idx':
        df.set_index('node_idx', inplace=True)
    df['num_child_nodes'] = num_child_nodes
    df['active_children'] = active_children
    df.reset_index(inplace=True, drop=False)
    return df


def bind_parent_nodes(df):
    """
    Creates a parent_idx column with references to the parent node index
    """

    pa = df.node.str.slice(start=1)
    parent_idxs = df.reset_index().set_index('node', drop=False).loc[pa.values].node_idx
    df['parent_idx'] = parent_idxs.values
    df.loc[df.node == '', 'parent_idx'] = -1
    return df
