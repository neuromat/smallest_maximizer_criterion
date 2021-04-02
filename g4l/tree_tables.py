import numpy as np
import pandas as pd
from operator import add


def nodes_and_transitions(sample):
    """
    Creates DataFrame object for contexts and transition probabilites
    given the sample:
    """
    df = pd.DataFrame()
    # count frequencies of each unique subsequence of size 1..max_depth
    df, transition_probs = count_subsequence_frequencies(df, sample)
    # create depth-related info columns
    df = remove_last_level(df, sample.max_depth)
    # create parent relationship between nodes
    df = bind_parent_nodes(df)
    # calculate nodes likelihoods
    df = calculate_likelihood(df, transition_probs)
    # remove unuseful data
    df = cleanup(df, sample.max_depth)
    return df, transition_probs


def remove_last_level(df, max_depth):
    return df[df.depth <= max_depth]


def sum_log_likelihoods(df_children):
    return (df_children.freq * np.log(df_children.node_prob)).sum()


def transition_sum_log_probs(df_children):
    return np.sum(np.log(df_children[df_children.node_prob > 0].node_prob))


def merge_freqs(dicts):
    d = dicts[0]
    for d2 in dicts[1:]:
        for k in d2.keys():
            d[k] += d2[k]
    return d


def merge_trans(dicts):
    d = dicts[0]
    for d2 in dicts[1:]:
        for k in d2.keys():
            list(map(add, d[k], d2[k]))
            d[k] += d2[k]
    return d


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
    df.index.name = 'depth_idx'
    df.reset_index(inplace=True)
    df.index.name = 'node_idx'
    return df


def calculate_transition_probs(df, dct_transition, dct_node_freq, A):
    node_idxs = (df[['node']].reset_index()
                             .set_index('node', drop=True)
                             .to_dict()['node_idx'])
    transition_columns = ['idx', 'next_symbol', 'freq', 'prob']
    probs = pd.DataFrame(columns=transition_columns)
    # TODO: make it more efficient

    for node in dct_transition.keys():
        for a in A:
            node_idx = node_idxs[node]
            fr = dct_transition[node][A.index(a)]
            prob = fr/dct_node_freq[node]
            probs.loc[len(probs)] = [node_idx, a, fr, prob]
    probs.freq = probs.freq.astype(int)
    return probs


def calculate_likelihood(df, transition_probs):
    x = transition_probs
    transition_probs['likelihood'] = x.freq[x.freq > 0] * np.log(x.prob[x.freq > 0])
    df['likelihood'] = transition_probs.groupby(['idx']).apply(lambda s: s.likelihood.sum())
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
    Connects nodes to their parents through the 'parent_idx' column
    """
    pa = df.node.str.slice(start=1)
    parent_idxs = df.reset_index().set_index('node', drop=False).loc[pa.values].node_idx
    df['parent_idx'] = parent_idxs.values
    df.loc[df.node == '', 'parent_idx'] = -1
    return df
