import numpy as np
import pandas as pd
from collections import Counter
from . import resources as rsc


def run(sample, max_depth):
    """
    Creates contexts and transition probabilites given the
    sample and a maximum depth value:
    """

    df = pd.DataFrame()
    # count frequencies of each unique subsequence of size 1..max_depth
    df = count_subsequence_frequencies(df, sample, max_depth)
    # create depth-related info columns
    df = create_indexes(df)
    # calculate transition probabilities
    transition_probs = calculate_transition_probs(df)
    df = remove_last_level(df, max_depth)
    # create parent relationship between nodes
    df = rsc.bind_parent_nodes(df)
    # remove invalid nodes
    #prune_unique_context_paths(df) (moved to context_tree)
    # calculate nodes likelihoods
    df = calculate_likelihood(df, transition_probs)
    # remove unuseful data
    df = cleanup(df, max_depth)
    return df, transition_probs


def remove_last_level(df, max_depth):
    return df[df.depth <= max_depth]


def sum_log_likelihoods(df_children):
    return (df_children.freq * np.log(df_children.node_prob)).sum()


def transition_sum_log_probs(df_children):
    return np.sum(np.log(df_children[df_children.node_prob > 0].node_prob))


def count_subsequence_frequencies(df, sample, max_depth):
    sample_data = sample.data
    # for each position in a sliding window of size max_depth over sample_data,
    #for d in range(1, context_tree.max_depth + 1):
    for d in range(1, max_depth + 2):
        # create a dataframe with all subsequences and their frequencies
        substr_freqs = Counter([sample_data[i:i + d] for i in range(len(sample_data)-d+1)])
        df_tmp = pd.DataFrame.from_dict(substr_freqs, orient='index').reset_index()
        df_tmp = df_tmp.rename(columns={'index':'node', 0:'freq'})
        df_tmp.node = substr_freqs.keys()
        df = df.append(df_tmp)
    df['active'] = 0
    return df


def create_indexes(df):
    df['depth'] = df.node.str.len()
    # depth_idx is an index for all nodes with same depth
    df.index.name = 'depth_idx'
    # create a unique index per node
    df.reset_index(inplace=True)
    df.index.name = 'node_idx'
    return df


def calculate_transition_probs(df):
    nodes = df[['node', 'freq', 'depth']].copy()
    nodes['prev'] = nodes.node.str.slice(stop=-1)
    nodes['next_symbol'] = nodes.node.str.slice(start=-1)
    nodes.reset_index(drop=False, inplace=True)
    nodes.set_index(['prev'], inplace=True)
    nodes['idx'] = nodes.set_index(['node']).node_idx.astype(int)
    nodes = nodes[nodes.depth > 1]
    nodes['idx'] = nodes['idx'].astype(int)
    transition_probs = nodes[['idx', 'next_symbol', 'freq']].reset_index(drop=True)
    all_transition_freqs = transition_probs.groupby(['idx']).apply(lambda  x: x.freq.sum())
    transition_probs.set_index(['idx'], inplace=True)
    transition_probs['prob'] = transition_probs.freq / all_transition_freqs.loc[transition_probs.index]
    transition_probs = transition_probs[transition_probs.freq > 0]
    return transition_probs





def calculate_likelihood(df, transition_probs):
    x = transition_probs
    transition_probs['likelihood'] = x.freq[x.freq > 0] * np.log(x.prob[x.freq > 0])
    df['likelihood'] = transition_probs.groupby(['idx']).apply(lambda s: s.likelihood.sum())
    return df


def cleanup(df, max_depth):
    df.reset_index(inplace=True)
    df = df[df.depth <= max_depth]
    return df
