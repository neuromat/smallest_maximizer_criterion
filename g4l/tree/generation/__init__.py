import numpy as np
import pandas as pd
import re
from collections import defaultdict
from collections import Counter
import h5py
import tables

def incremental_strategy(context_tree):
  #df = empty_frame()
  df = pd.DataFrame()
  A = context_tree.sample.A
  sample_data = context_tree.sample.data

  for d in range(1, context_tree.max_depth + 2):
    # for each position in a sliding window of size max_depth over sample_data,
    # count frequencies of each unique subsequence
    substr_freqs = Counter([sample_data[i:i + d] for i in range(len(sample_data)-d+1)])
    df_tmp = pd.DataFrame.from_dict(substr_freqs, orient='index').reset_index()
    df_tmp = df_tmp.rename(columns={'index':'node', 0:'node_freq'})
    df_tmp.node = substr_freqs.keys()
    df = df.append(df_tmp)

  df['l'] = df.node.str.len()
  #df.drop('len_idx', axis='columns', inplace=True)
  df.index.name = 'len_idx'
  #df.drop('node_idx', axis='columns', inplace=True)
  df.reset_index(inplace=True)
  df.index.name = 'node_idx'
  df['parent_node'] = df.node.str.slice(stop=-1)
  df.reset_index(inplace=True)
  df.set_index('node', inplace=True)

  #df.reset_index().set_index('node').loc[df.node]

  parent_nodes = df[df.l > 1].parent_node
  parent_nodes_idx = df.loc[parent_nodes].node_idx
  df.reset_index(inplace=True)
  #df.drop('parent_idx', axis='columns', inplace=True)
  parent_nodes_idx = parent_nodes_idx.reset_index().node_idx.values
  l1_values = np.repeat(None, len(df.loc[df.l == 1]))


  df['parent_idx'] = np.concatenate((l1_values, parent_nodes_idx))
  df.drop('parent_node', axis='columns', inplace=True)

  #df.drop('lps', axis='columns', inplace=True)
  parent_node_freqs = df.set_index('node_idx').iloc[df[df.l > 1].parent_idx].node_freq.values
  l1_freqs_sum = np.repeat(df[df.l == 1].node_freq.values.sum(), len(df.loc[df.l == 1]))
  df['node_prob'] = (df.node_freq / np.concatenate((l1_freqs_sum, parent_node_freqs))).astype(float)

  lps_values = df.groupby([df.parent_idx]).apply(sum_log_likelihoods)
  sum_transition_log_probs = df.groupby([df.parent_idx]).apply(transition_sum_log_probs)
  df.set_index('node_idx', inplace=True)
  #df.groupby([df.parent_idx]).apply(sum_log_likelihoods)
  df['lps'] = lps_values
  df['transition_sum_log_probs'] = sum_transition_log_probs
  df.reset_index(inplace=True)
  df = df[df.l <= context_tree.max_depth]

  #import code; code.interact(local=dict(globals(), **locals()))

  df['final'] = 0
  df['flag'] = 0

  return df

def sum_log_likelihoods(df_children):
  return (df_children.node_freq * np.log(df_children.node_prob)).sum()

def transition_sum_log_probs(df_children):
  return np.sum(np.log(df_children[df_children.node_prob > 0].node_prob))


def original_strategy(context_tree):
  df = empty_frame()
  A = context_tree.sample.A
  sample_data = context_tree.sample.data
  node_idx = 0
  parent_node_indexes = defaultdict(lambda: None)
  for tree_length in range(1, context_tree.max_depth + 1):

    # gets all leaves for trees of size 1 to max_depth
    for len_idx, node in enumerate(all_leaves(tree_length, A)):
      parent_node_indexes[node] = node_idx

      # retrieve parent node idx from dictionary
      parent_idx = parent_node_indexes[node[1:]]

      # check how many occurrences of node exist in data
      node_freq = calc_node_frequency(node, sample_data)
      # calculate child nodes' probs
      # TODO: remover abaixo
      #ps = node_freq / (len(context_tree.sample.data) - tree_length + 1)

      # frequencia da passagem pai -> filho
      child_freqs, transition_probs = calc_transition_probs(node, node_freq, A, sample_data)

      log_max_likelihood = calc_lpmls(child_freqs, transition_probs)
      # Creates a new row for tree table
      # - last 2 fields will be further populated
      # - lpmls is the initial value, will be updated later
      transition_sum_log_probs = np.log(transition_probs[transition_probs > 0]).sum()

      row = [tree_length, node_idx, len_idx, parent_idx, node, node_freq, log_max_likelihood, transition_probs, transition_sum_log_probs, 0, 0]
      node_idx += 1
      df.loc[len(df)] = row
  return df

def empty_frame():
  return pd.DataFrame(
    columns=['l', 'node_idx', 'len_idx', 'parent_idx',
             'node', 'node_freq',
             'lps',
             'transition_probs', 'transition_sum_log_probs', 'flag', 'final'])

# Scans sample data returns the number of occurrences of a given node
def calc_node_frequency(node, sample_data):
  pos = [m.start() for m in re.finditer('(?=%s)' % node, sample_data)]
  return len(pos)

def children_freq_prob(node, node_freq, A, sample_data):
  # returns freqs and probs as a pair of arrays; ex: ([10, 30, 60], [0.1, 0.3, 0.6])
  freqs = np.array([calc_node_frequency(node + str(sym), sample_data) for sym in A])
  probs = freqs/node_freq
  return (freqs, probs)

def calc_transition_probs(node, node_freq, A, sample_data):
  if node_freq > 0:
    child_freqs, transition_probs = children_freq_prob(node, node_freq, A, sample_data) # ([10, 30, 60], [0.1, 0.3, 0.6])
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


