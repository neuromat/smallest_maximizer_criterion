import numpy as np
import pandas as pd
import re
from collections import defaultdict
from collections import Counter
import h5py
import tables


def run(context_tree):
  #df = empty_frame()
  df = pd.DataFrame()

  # count frequencies of each unique subsequence
  df = count_subsequence_frequencies(df, context_tree)

  # create depth-related info columns
  df = create_indexes(df)

  calculate_transition_probs(df, context_tree)

  df = remove_last_level(df, context_tree)

  # create parent relationship between nodes
  df = bind_parent_nodes(df)

  # calculate nodes likelihoods
  df = calculate_likelihood(df, context_tree)

  df = tag_nodes(df)

  df = cleanup(df, context_tree)

  return df



def remove_last_level(df, context_tree):
  return df[df.depth <= context_tree.max_depth]

def sum_log_likelihoods(df_children):
  return (df_children.freq * np.log(df_children.node_prob)).sum()

def transition_sum_log_probs(df_children):
  return np.sum(np.log(df_children[df_children.node_prob > 0].node_prob))

def count_subsequence_frequencies(df, context_tree):
  sample_data = context_tree.sample.data
  # for each position in a sliding window of size max_depth over sample_data,
  #for d in range(1, context_tree.max_depth + 1):
  for d in range(1, context_tree.max_depth + 2):

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

def calculate_transition_probs(df, context_tree):
  nodes = df[['node', 'freq', 'depth']]
  nodes['prev'] = nodes.node.str.slice(stop=-1)
  nodes['next_symbol'] = nodes.node.str.slice(start=-1)
  nodes.reset_index(drop=False, inplace=True)
  nodes.set_index(['prev'], inplace=True)
  nodes['idx'] = nodes.set_index(['node']).node_idx.astype(int)
  nodes = nodes[nodes.depth > 1]
  nodes['idx'] = nodes['idx'].astype(int)
  freqs = nodes[['idx', 'next_symbol', 'freq']].reset_index(drop=True)
  all_transition_freqs = freqs.groupby(['idx']).apply(lambda  x: x.freq.sum())
  freqs.set_index(['idx'], inplace=True)
  freqs['prob'] = freqs.freq / all_transition_freqs.iloc[freqs.index]
  context_tree.transitions_df = freqs

def bind_parent_nodes(df):
  df['parent_node'] = df.node.str.slice(start=1)
  df.reset_index(inplace=True)
  df.set_index('node', inplace=True)
  parent_nodes = df[df.depth > 1].parent_node
  parent_nodes_idx = df.loc[parent_nodes].node_idx
  df.reset_index(inplace=True)
  parent_nodes_idx = parent_nodes_idx.reset_index().node_idx.values
  l1_values = np.repeat(None, len(df.loc[df.depth == 1]))
  # this concatenation creates a full column with parent_idxs; None values are
  # set for nodes with depth == 1
  df['parent_idx'] = np.concatenate((l1_values, parent_nodes_idx))
  num_child_nodes = df[df.depth>1].groupby(['parent_idx']).apply(lambda x: x.count().node_idx)
  df.set_index(['node_idx'], inplace=True)
  df['num_child_nodes'] = num_child_nodes
  df.drop('parent_node', axis='columns', inplace=True)
  df.reset_index(inplace=True)
  #import code; code.interact(local=dict(globals(), **locals()))
  return df

def calculate_likelihood(df, context_tree):
  df.set_index(['node_idx'], inplace=True)
  x = context_tree.transitions_df
  context_tree.transitions_df['lps'] = x.freq[x.freq > 0] * np.log(x.prob[x.freq > 0])
  df['lps'] = context_tree.transitions_df.groupby(['idx']).apply(lambda s: s.lps.sum())

  #df['transition_sum_log_likelihoods'] = sum_transition_log_probs
  return df

def tag_nodes(df):
  return df

def cleanup(df, context_tree):
  df.reset_index(inplace=True)
  df = df[df.depth <= context_tree.max_depth]
  #df['final'] = 0
  #df['flag'] = 0
  return df
