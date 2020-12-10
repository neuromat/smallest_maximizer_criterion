import numpy as np
import pandas as pd
import re
from collections import defaultdict


def run(sample, max_depth):
    df = empty_frame()
    A = sample.A
    sample_data = sample.data
    node_idx = 0
    parent_node_indexes = defaultdict(lambda: None)

    for tree_length in range(1, max_depth + 1):

        # gets all leaves for trees of size 1 to max_depth
        for depth_idx, node in enumerate(all_leaves(tree_length, A)):
            parent_node_indexes[node] = node_idx

            # retrieve parent node idx from dictionary
            parent_idx = parent_node_indexes[node[1:]]

            # check how many occurrences of node exist in data
            freq = calc_node_frequency(node, sample_data)
            # calculate child nodes' probs

            child_freqs, transition_probs = calc_transition_probs(node, freq, A, sample_data)

            log_max_likelihood = calc_lpmls(child_freqs, transition_probs)
            # Creates a new row for tree table
            # - last 2 fields will be further populated
            # - lpmls is the initial value, will be updated later
            transition_sum_log_probs = np.log(transition_probs[transition_probs > 0]).sum()

            row = [tree_length, node_idx, depth_idx, parent_idx, node, freq, log_max_likelihood, transition_probs, transition_sum_log_probs, 0, 0]
            node_idx += 1
            df.loc[len(df)] = row

    grp = df.groupby([df.parent_idx])
    df['num_child_nodes'] = grp.apply(lambda x: len(x[x.freq > 0]))
    return df, transition_probs

def empty_frame():
    return pd.DataFrame(
        columns=['depth', 'node_idx', 'depth_idx', 'parent_idx',
                         'node', 'freq',
                         'likelihood',
                         'transition_probs', 'transition_sum_log_probs', 'remove_node', 'active'])

# Scans sample data returns the number of occurrences of a given node
def calc_node_frequency(node, sample_data):
    pos = [m.start() for m in re.finditer('(?=%s)' % node, sample_data)]
    return len(pos)

def children_freq_prob(node, freq, A, sample_data):
    # returns freqs and probs as a pair of arrays; ex: ([10, 30, 60], [0.1, 0.3, 0.6])
    freqs = np.array([calc_node_frequency(node + str(sym), sample_data) for sym in A])
    probs = freqs/freq
    return (freqs, probs)

def calc_transition_probs(node, freq, A, sample_data):
    if freq > 0:
        child_freqs, transition_probs = children_freq_prob(node, freq, A, sample_data) # ([10, 30, 60], [0.1, 0.3, 0.6])
    else:
        child_freqs, transition_probs = (np.zeros(len(A)), np.zeros(len(A)))
    return child_freqs, transition_probs

def all_leaves(depth, A):
    return np.array([''.join(x)[::-1] for x in all_leaves_by_len(depth, A)])

def all_leaves_by_len(depth, A):
        """Find the list of all strings of 'A' of depth 'depth'"""
        c = [[]]
        def to_str(arr):
                return [str(el) for el in arr]

        for i in range(1, depth+1):
                c = [to_str([x]+y) for x in A for y in c]
        return np.array(c).astype(str)

# Logaritmo da máxima verossimilhança
def calc_lpmls(child_freqs, transition_probs):
    return np.sum(child_freqs[child_freqs > 0] * np.log(transition_probs[child_freqs > 0]))


