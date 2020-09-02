import numpy as np
import math
import pandas as pd
from g4l.estimators.base import Base
from g4l.tree import ContextTree
from g4l.estimators.ctm import CTM
from datetime import datetime
import logging

class Prune(Base):

  def execute(self):
    self.results = pd.DataFrame(columns=['iter_num', 'num_nodes', 'log_likelihood_sum'])
    t = self.apply_ctm()
    self.context_trees = []
    df = self.initialize_pruning(t)
    self.perform_pruning(df)
    self.context_trees = list(reversed(self.context_trees))


  def apply_ctm(self):
    return CTM(self.context_tree).execute(None)

  def initialize_pruning(self, t):
    df = t.df.copy()
    #df['children_contrib'] = df.transition_sum_log_probs
    df.loc[df.lps==0, 'children_contrib'] = -math.inf
    df.loc[df.active==0, 'num_total_leaves'] = 0
    df.loc[df.active==0, 'num_direct_leaves'] = 0
    return df

  def update_parent_counts(self, df, nodes_idx):
    updated_nodes = []
    nodes_to_update = df.loc[nodes_idx].sort_values(['depth'], ascending=False)
    if len(nodes_to_update)==0:
      return
    for depth in reversed(range(1, nodes_to_update.depth.max()+1)):
      num_total_leaves = df.loc[(df.node_idx.isin(nodes_idx)) & (df.depth==depth)].groupby(['parent_idx']).sum().num_total_leaves
      idxs = list(num_total_leaves.index.values)
      current_leaves_count = df.loc[idxs].num_total_leaves.values
      df.loc[idxs, 'num_total_leaves'] = current_leaves_count + num_total_leaves.values
      updated_nodes += idxs
    self.update_parent_counts(df, updated_nodes)

  def update_counts(self, df):
    leaves = df[df.active==1]
    leaf_counts = leaves.groupby(['parent_idx']).count().node_idx
    contrib = leaves.groupby(['parent_idx']).sum().lps
    df.loc[leaf_counts.index, 'num_total_leaves'] = leaf_counts
    df.loc[leaf_counts.index, 'num_direct_leaves'] = leaf_counts
    df.loc[leaf_counts.index, 'children_contrib'] = contrib
    self.update_parent_counts(df, leaf_counts.index.values)

  def perform_pruning(self, df):
    iteration_num = 0

    #trs = self.context_tree.transitions_df
    #children_contrib = trs.groupby(['idx']).apply(lambda x: np.log(x[x.prob > 0].prob).sum())
    #df.set_index(['node_idx'], inplace=True)
    #df['children_contrib'] = children_contrib
    #df.reset_index(drop=False, inplace=True)
    self.add_tree(df)
    while True:
      self.update_counts(df)
      candidate_nodes = df[(df.num_total_leaves==df.num_direct_leaves) & (df.num_total_leaves > 0) & (df.depth > 0)]

      if len(candidate_nodes)==0:
        break
      candidate_children = df[(df.parent_idx.isin(candidate_nodes.node_idx)) & (df.active==1)]
      lps2 = candidate_children.groupby(['parent_idx']).lps.sum()
      diff = (lps2 - candidate_nodes.lps)
      less_contributive_node_idx = diff.sort_values().index[0]
      #if less_contributive_node_idx==5:
      #  import code; code.interact(local=dict(globals(), **locals()))
      self.remove_leaves(df, less_contributive_node_idx)
      self.mark_as_leaf(df, less_contributive_node_idx)
      self.add_tree(df)
      iteration_num += 1
      logdata = (iteration_num, len(df[df.active==1]), less_contributive_node_idx)
      logging.debug("Iteration: %s ; leaves: %s; pruned node_idx: %s" % logdata)
    return self

  def add_tree(self, df):
    new_tree = ContextTree(self.context_tree.sample, max_depth=self.context_tree.max_depth, source_data_frame=df)
    self.context_trees.append(new_tree)

  def remove_leaves(self, df, node_idx):
    df.loc[df.parent_idx==node_idx, 'active'] = 0
    df.loc[df.node_idx==node_idx, 'num_total_leaves'] = 0
    df.loc[df.node_idx==node_idx, 'num_direct_leaves'] = 0

  def mark_as_leaf(self, df, node_idx):
    df.loc[df.node_idx==node_idx, 'active'] = 1

