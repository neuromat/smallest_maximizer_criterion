import numpy as np
import pandas as pd
from collections import Counter
from operator import add
from collections import defaultdict
from . import resources as rsc
import logging

def run(sample, max_depth, scan_offset):
    """
    Creates contexts and transition probabilites given the
    sample and a maximum depth value:
    """
    # scan_offset = max_depth  # <== SeqROCTM uses it
    df = pd.DataFrame()
    # count frequencies of each unique subsequence of size 1..max_depth
    df, transition_probs = count_subsequence_frequencies(df, sample, max_depth, scan_offset)
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


def count_subsequence_frequencies(df, sample, max_depth, scan_offset):
    #    freqs = []
    #    trns = []
    #    for smpl in sample.subsamples():
    #        sample_data = smpl.data
    #        A = sample.A
    #        # for each position in a sliding window of size max_depth over sample_data,
    #        #for d in range(1, context_tree.max_depth + 1):
    #        dct_transition = defaultdict(lambda: np.zeros(len(A)))
    #        dct_node_freq = defaultdict(lambda: 0)
    #        dct_node_freq[''] = 0
    #        for d in range(max_depth + 1):
    #            # create a dataframe with all subsequences and their frequencies
    #            # aqui
    #            for i in range(scan_offset, len(sample_data)):
    #                node = sample_data[i-d:i]
    #                a = sample_data[i]
    #                if node != '':
    #                    dct_node_freq[node] += 1
    #                    dct_transition[node][A.index(a)] += 1
    #        #dct_node_freq[sample_data[-1]] += 1  # pra compatibilizar com perl
    #        # TODO: melhorar estratégia de contagem do nó vazio
    #        dct_node_freq[''] = sum([dct_node_freq[a] for a in A])
    #        for ii, a in enumerate(A):
    #            dct_transition[''][ii] = dct_node_freq[a]
    #        freqs.append(dct_node_freq)
    #        trns.append(dct_transition)
    #
    #    dct_node_freq = merge_freqs(freqs)
    #    dct_transition = merge_trans(trns)

    cols = {'index': 'node', 'N': 'freq'}
    #df = sample.F.T.sum().to_frame().reset_index().rename(columns=cols)
    df = sample.F['N'].to_frame().reset_index().rename(columns=cols).copy()
    #df = pd.DataFrame.from_dict(dct_node_freq, orient='index').reset_index()
    #df = df.rename(columns={'index':'node', 0:'freq'})
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
