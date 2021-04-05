""" Methods to build the main tables that represent the tree.
    - Nodes DataFrame
    - Transitions DataFrame


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

"""

import numpy as np
import pandas as pd
#from operator import add


def nodes_and_transitions(sample):
    """Creates tables with contexts and transition probabilites

    For a given sample, this method returns two Pandas dataframes.
    The first one contains the node counts and likelihoods,
    while the second one contains the transition probabilities.

    Arguments:
        sample {Sample} -- Sample object

    Returns:
        tuple(DataFrame, DataFrame) -- Counts and transitions tables
    """

    # creates the main DataFrames (nodes and transitions)
    df = pd.DataFrame()
    df = create_nodes_table(sample)
    transition_probs = create_transitions_table(df, sample)
    df = calc_node_likelihoods(df, transition_probs)

    # remove unuseful data
    #df.reset_index(inplace=True)
    return df, transition_probs


def remove_last_level(df, max_depth):
    """ Removes max_depth + 1 nodes

    Node frequencies are initially calculated for nodes
    with length max_depth + 1 in order to efficiently extract
    the transition. Since transitions were already calculated
    we can drop the nodes with length == (max_depth + 1)
    """

    df = df[df.depth <= max_depth]


def create_nodes_table(sample):
    """Creates the nodes table given a sample

    Builds the node table. The node counts were already precomputed
    in the sample object (see sample.py#_calculate_substr_frequencies)

    Arguments:
        sample {Sample} -- A Sample object
    """
    cols = {'index': 'node', 'N': 'freq'}
    df = sample.F['N'].to_frame().reset_index().rename(columns=cols).copy()
    df['active'] = 0
    create_indexes(df)
    remove_last_level(df, sample.max_depth)
    bind_parent_nodes(df)

    return df


def create_transitions_table(df, sample):
    """Creates the transitions table

    This method creates the following transition table:

    idx next_symbol  freq      prob
      8           0  1761  0.298475
      8           1  4139  0.701525

    - where idx is the node index (in the nodes table)
    - next symbol is a possible transition from this node
    - freq is the number of occurrences of this node in the sample
    - prob is the probability of transition from this node to next_symbol

    Arguments:
        df {[type]} -- [description]
        sample {[type]} -- [description]

    Returns:
        DataFrame -- the transitions table
    """

    tbl = sample.F[[i for i in range(len(sample.A))]].astype(int)

    tbl['idx'] = df.reset_index().set_index('node')['node_idx']
    tbl = tbl.melt(id_vars=['idx'], var_name='next_symbol', value_name='freq')
    tbl = tbl.sort_values(['idx', 'next_symbol']).reset_index(drop=True).set_index('idx')
    tbl.freq = tbl.freq.astype(int)
    tbl['prob'] = tbl.freq / df.freq
    transition_probs = tbl.reset_index()
    return transition_probs


def create_indexes(df):
    # creates a column with the node length
    df['depth'] = df.node.str.len()
    df.reset_index(inplace=True, drop=True)
    # sets node_idx column as the index field
    df.index.name = 'node_idx'


def calc_node_likelihoods(df, transition_probs):
    """Calculates node log-likelihoods

    Given the node and transition tables, calculates the
    log-likelihood of each node.

    Arguments:
        df {DataFrame} -- node table
        transition_probs {DataFrame} -- transitions table

    Returns:
        DataFrame -- returns the (modified) node table
    """

    # For each row with freq > 0, compute N * log(p)
    # in a column named likelihood
    probs = transition_probs
    L = probs.freq[probs.freq > 0] * np.log(probs.prob[probs.freq > 0])
    transition_probs['likelihood'] = L

    # then we sum all the log-likelihoods grouped by each node.
    # the results are stored in the node table, associated to
    # their respective nodes.
    fn = lambda s: s.likelihood.sum()
    df['likelihood'] = transition_probs.groupby(['idx']).apply(fn)
    return df


def calculate_num_child_nodes(df):
    """ Calculates and stores the number of child nodes of all nodes

    Given a context tree dataframe, this method creates a column in
    the node table counting the number of child nodes

    Arguments:
        df {DataFrame} -- the node table

    Returns:
        DataFrame -- the (modified) node table
    """

    num_child_nodes = (df[df.depth >= 1]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .count()).node_idx

    if df.index.name != 'node_idx':
        df.set_index('node_idx', inplace=True)
    df['num_child_nodes'] = num_child_nodes
    df.reset_index(inplace=True, drop=False)
    return df


def bind_parent_nodes(df):
    """
#    Adds a column with reference to the parent node index for each node
    """

    # gets the parent node string
    pa = df.node.str.slice(start=1)

    # set the table index to the node column
    node_df = df.reset_index().set_index('node', drop=False)

#    # adds a column with the parent_idxs related to their child nodes
    parent_idxs = node_df.loc[pa.values].node_idx
    df['parent_idx'] = parent_idxs.values
    df.loc[df.node == '', 'parent_idx'] = -1
