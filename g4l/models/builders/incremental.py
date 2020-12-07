import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from . import resources as rsc


def run(sample, max_depth):
    """
    Creates contexts and transition probabilites given the
    sample and a maximum depth value:
    """

    df = pd.DataFrame()
    # count frequencies of each unique subsequence of size 1..max_depth
    df, transition_probs = count_subsequence_frequencies(df, sample, max_depth)
    # create depth-related info columns
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
    dct_transition = defaultdict(lambda: np.zeros(len(sample.A)))
    dct_node_freq = defaultdict(lambda: 0)
    for d in range(max_depth + 1):
        # create a dataframe with all subsequences and their frequencies
        # aqui
        for i in range(max_depth, len(sample_data)):
            node = sample_data[i-d:i]
            #if d==1:
                #import code; code.interact(local=dict(globals(), **locals()))
            a = sample_data[i]
            dct_node_freq[node] += 1
            dct_transition[node][sample.A.index(a)] += 1
    df = pd.DataFrame.from_dict(dct_node_freq, orient='index').reset_index()
    df = df.rename(columns={'index':'node', 0:'freq'})
    df['active'] = 0
    df = create_indexes(df)
    transition_probs = calculate_transition_probs(df, dct_transition, dct_node_freq, sample.A)
    return df, transition_probs


def create_indexes(df):
    df['depth'] = df.node.str.len()
    # depth_idx is an index for all nodes with same depth
    df.index.name = 'depth_idx'
    # create a unique index per node
    df.reset_index(inplace=True)
    df.index.name = 'node_idx'
    return df


def calculate_transition_probs(df, dct_transition, dct_node_freq, A):
    node_idxs = (df[['node']].reset_index()
                             .set_index('node', drop=True)
                             .to_dict()['node_idx'])
    transition_columns = ['idx', 'next_symbol', 'freq', 'prob']
    probs = pd.DataFrame(columns=transition_columns)
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
