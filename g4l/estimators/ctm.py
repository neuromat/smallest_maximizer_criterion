import pandas as pd
import re
import math
import numpy as np
import g4l.tree as tr
import g4l.tree.generation as gen
from g4l.estimators.base import Base

class CTM(Base):

  def execute(self, c):
    # TODO: criar a context_tree
    data_frame = self.context_tree.df.copy()
    self.apply_penalization(c, data_frame)
    data_frame = self.remove_non_contributive_nodes(data_frame) # até aqui ok
    self.select_active_contexts(data_frame)
    #champion_tree_df = self.final_tree(data_frame)
    self.cleanup(data_frame)
    new_tree = tr.ContextTree(self.sample,
                          self.context_tree.max_depth,
                          source_data_frame=data_frame,
                          chosen_penalty=c)
    new_tree.tag_nodes()
    return new_tree

  def apply_penalization(self, c, data_frame):
    if c is not None:
      degrees_of_freedom = len(self.A)-1
      n = len(self.data)
      data_frame.lps -= np.log(n) * (degrees_of_freedom * c)

  def cleanup(self, df):
    df.drop('remove_node', axis='columns', inplace=True)

  # find the largest admissible context tree
  def remove_non_contributive_nodes(self, df):
    df['remove_node'] = 0
    df['lps2'] = None
    df['lps2'] = df['lps2'].astype(np.float64)
    for l in list(reversed(range(1, self.context_tree.max_depth))):
      parents = df[(df.depth==l) & (df.freq > 1)]
      child_nodes = df[(df.freq > 1) & (df.parent_idx.isin(parents.node_idx))]
      sum_lps_res = child_nodes.groupby([df.parent_idx]).apply(lambda x: x.lps.sum())
      sum_lps_res.name = 'lps2'
      sum_lps_res = sum_lps_res.to_frame().reset_index()
      sum_lps_res.rename(columns={'parent_idx':'node_idx'}, inplace=True)
      df = df.set_index('node_idx').combine_first(sum_lps_res.set_index('node_idx'))
      df.reset_index(inplace=True)
      parents = df[(df.depth==l) & (df.freq > 1)]
      nodes_to_remove = parents[(parents.num_child_nodes > 1) & (parents.lps2 > parents.lps)].index
      df.loc[nodes_to_remove, 'remove_node'] = 1
      #df.loc[nodes_to_remove, 'lps'] = df.loc[nodes_to_remove].lps2
    #import code; code.interact(local=dict(globals(), **locals()))
    df.drop('lps2', axis='columns', inplace=True)
    return df

  def select_active_contexts(self, df):
    for depth in range(1, self.context_tree.max_depth + 1):
      for idx, row in df[df.depth==depth].iterrows():
        if depth==1:
          if row.remove_node==0:
            df.loc[(df.node==row.node), 'active'] = 1
        else:
          if row.remove_node==0:
            subnodes_str = [row.node[-(depth - m):] for m in range(1, depth)]
            subnodes = df[df['node'].isin(subnodes_str)]
            if subnodes['remove_node'].product() == 1 and row.freq > 0:
              df.loc[(df.node==row.node), 'active'] = 1

  # def final_tree(self, df):
  #   selected_nodes = df.loc[df.active==1].sort_values(
  #     by=['node'],
  #     ascending=(True))
  #   return selected_nodes

  # def __update_summary(self, summary, node, field_name, value):
  #   summary.loc[(summary['node']==node), field_name] = value

