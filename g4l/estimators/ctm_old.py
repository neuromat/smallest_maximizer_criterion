import pandas as pd
import re
import math
import numpy as np
import g4l.tree as tr
import g4l.tree.generation.original as gen
from g4l.estimators.base import Base

class CTM(Base):

  def execute(self, c):
    # TODO: criar a context_tree
    data_frame = self.context_tree.df.copy()
    self.apply_penalization(c, data_frame)
    self.block2(data_frame) # até aqui ok
    self.block3(data_frame)
    champion_tree_df = self.final_tree(data_frame)
    return tr.ContextTree(self.sample,
                          self.context_tree.max_depth,
                          source_data_frame=champion_tree_df,
                          chosen_penalty=c)

  def apply_penalization(self, c, data_frame):
    if c is not None:
      degrees_of_freedom = len(self.A)-1
      n = len(self.data)
      data_frame.lps -= np.log(n) * (degrees_of_freedom * c)

  # TODO: rename this method
  def block2(self, df):
    alphabet_len = len(self.A)
    df['remove_node'] = 0
    # do último ao primeiro nível da árvore:
    for l in list(reversed(range(1, self.context_tree.max_depth))):
      # para cada indice de folha:
      for _node_idx, row in df[df.depth==l].iterrows():
        # para todo nó ocorrente mais de uma vez na amostra:
        row = df[df.node_idx==row.node_idx].iloc[0]
        if row.freq > 1:

          # Locate all child nodes of current node
          child_rows = df[(df.parent_idx==row.node_idx) & (df.freq > 1 )]
          child_nodes_log_likelihood = child_rows.lps.sum()

          # poda
          # if row.node=='3':
          #   import code; code.interact(local=dict(globals(), **locals()))
          if len(child_rows) > 1 and child_nodes_log_likelihood > row.lps:
            self.__update_row(df, l, row.depth_idx, 'lps', child_nodes_log_likelihood)
            self.__update_row(df, l, row.depth_idx, 'remove_node', 1)


  # TODO: rename this method - select_final_nodes
  def block3(self, df):
    data_len = len(self.data)
    for tree_length in range(1, self.context_tree.max_depth + 1):
      # TODO: change enumerate to yield to consume nodes on demand (optimizes memory)

      for node_idx, node in enumerate(gen.all_leaves(tree_length, self.A)):
        try:
          row = df[df.node==node].iloc[0]
        except IndexError:
          continue
        if tree_length==1:
          if row.remove_node==0:
            df.loc[(df.node==node), 'active'] = 1
        else:
          if row.remove_node==0:
            subnodes_str = [node[-(tree_length - m):] for m in range(1, tree_length)]
            subnodes = df[df['node'].isin(subnodes_str)]
            if subnodes['remove_node'].product() == 1 and row.freq > 0:
              df.loc[(df.node==node), 'active'] = 1

  def final_tree(self, df):
    selected_nodes = df.loc[df.active==1].sort_values(
      by=['node'],
      ascending=(True))
    return selected_nodes

  def __locate_row(self, df, l, node_idx):
    # depth_idx is the node index in the list of all nodes with same length
    filtered_row = df[(df['depth']==l) & (df['depth_idx']==node_idx)]
    return filtered_row.iloc[0]

  def __update_row(self, df, l, node_idx, field_name, value):
    df.loc[(df['depth']==l) & (df['depth_idx']==node_idx), field_name] = value

  def __update_summary(self, summary, node, field_name, value):
    summary.loc[(summary['node']==node), field_name] = value

