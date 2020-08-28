import numpy as np
import math
import pandas as pd
from g4l.estimators.base import Base
from g4l.estimators.ctm import CTM
from datetime import datetime
import logging

class Prune(Base):

  def execute(self):
    self.results = pd.DataFrame(columns=['iter_num', 'num_nodes', 'log_likelihood_sum'])
    t = self.apply_ctm()
    df = self.initialize_pruning(t)
    self.tag_nodes(df)
    self.perform_pruning(df)


  def apply_ctm(self):
    return CTM(self.context_tree).execute(None)

  def initialize_pruning(self, t):
    df = t.df.copy()
    df['final'] = 1
    #df['children_contrib'] = df.transition_sum_log_probs
    df.loc[df.lps==0, 'children_contrib'] = -math.inf
    return df

  def tag_nodes(self, df):
    # Tag nodes as N (simple node) as default
    df['type'] = 'N'
    # All nodes without child nodes are tagged as leaves (L)
    df.loc[(~df.node_idx.isin(df.parent_idx.unique())) & (df.active==1), 'type'] = 'L'

    # Tag a node as LP when it's parent of leaf nodes
    parent_node_idx = df.loc[(df.final==1) & (df.type=='L')].parent_idx.unique()
    df.loc[df.node_idx.isin(parent_node_idx), 'type'] = 'LP'
    #lp_nodes = df.loc[(df.type=='LP') & (df.final==1)]

  def perform_pruning(self, df):
    iteration_num = 0

    parent_node_idx = df.loc[(df.active==1) & (df.final==1) & (df.type=='L')].parent_idx.unique()
    df.loc[df.node_idx.isin(parent_node_idx), 'type'] = 'LP' # leaf parent

    trs = self.context_tree.transitions_df
    children_contribs = trs.groupby(['idx']).apply(lambda x: np.log(x[x.prob > 0].prob).sum())
    df.set_index(['node_idx'], inplace=True)
    df['children_contrib'] = children_contribs
    df.reset_index(drop=False, inplace=True)

    while df.final.sum()>0:
      #print('iteration', iteration_num)
      # fetch all nodes connected to leaf nodes and mark them as parent of leaves (LP)

      # for all LP nodes, mark as candidate those ones that connects only with leaf nodes
      lp_nodes = df.loc[(df.type=='LP') & (df.final==1)]
      for idx, lp_node in lp_nodes.iterrows():
        child_nodes = df[df.parent_idx==lp_node.node_idx]
        num_child_nodes = len(child_nodes)
        active_child_nodes = ((child_nodes.active==1) & (child_nodes.final==1)).astype(int).sum()
        leaf_nodes = (child_nodes.type=='L').astype(int).sum()
        # when all child nodes are leaves and active
        if num_child_nodes == active_child_nodes == leaf_nodes:
          df.loc[df.node_idx==lp_node.node_idx, 'type'] = 'LPC' # mark these nodes as candidates (LPC)

      lpc = df[df.type=='LPC']
      if len(df[df.type=='LPC'])==0:
        break


      # teste
      #import code; code.interact(local=dict(globals(), **locals()))

      leaf_parents_lps_sum = df[df.type=='L'].groupby(['parent_idx']).apply(lambda x: x.lps.sum())

      lpc['lps2'] = leaf_parents_lps_sum

      lpcs = lpc[lpc.node_idx.isin(leaf_parents_lps_sum.index)]
      #import code; code.interact(local=dict(globals(), **locals()))
      lpcs = lpcs[lpcs.active==0]
      lpcs['lps_delta'] = lpcs.lps2 - lpcs.lps
      less_contributive_lp_idx = lpcs.sort_values(['lps_delta']).index[0]


      # /teste

      # for all the candidate nodes (LPC), calculate the less contributive
      #less_contributive_lp_idx = np.array(lpc.sort_values(['children_contrib']).node_idx)[0]
      #import code; code.interact(local=dict(globals(), **locals()))

      # Eliminate leaves
      df.loc[df.parent_idx==less_contributive_lp_idx, 'final'] = 0
      df.loc[df.parent_idx==less_contributive_lp_idx, 'active'] = 0

      # The less contributive LPC becomes a leaf
      df.loc[df.node_idx==less_contributive_lp_idx, 'type'] = 'L'
      df.loc[df.node_idx==less_contributive_lp_idx, 'active'] = 1

      grandparent_idx = df.loc[df.node_idx==less_contributive_lp_idx].parent_idx.iloc[0]
      df.loc[df.node_idx==grandparent_idx, 'type'] = 'LP'

      #import code; code.interact(local=dict(globals(), **locals()))
      iteration_num += 1

      active_nodes = df[(df.final==1) & (df.active==1)]

      # ['iter_num', 'num_nodes', 'log_likelihood_sum', 'node_idx']
      #print([iteration_num, len(active_nodes), active_nodes.lps.sum()])
      self.results.loc[len(self.results)] = [iteration_num, len(active_nodes), active_nodes[active_nodes.type=='L'].lps.sum()]
