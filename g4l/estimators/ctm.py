import pandas as pd
import re
import math
import numpy as np
import g4l.tree as tr
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
    degrees_of_freedom = len(self.A)-1
    n = len(self.data)
    data_frame.lps -= np.log(n) * (degrees_of_freedom * c)

  # TODO: rename this method
  def block2(self, df):
    alphabet_len = len(self.A)

    # do último ao primeiro nível da árvore:
    for l in list(reversed(range(1, self.context_tree.max_depth))):
      # para cada indice de folha:
      for node_idx in range(alphabet_len ** l):
        row = self.__locate_row(df, l, node_idx)

        # para todo nó ocorrente mais de uma vez na amostra:
        if row.node_freq > 1:
          m = alphabet_len * node_idx
          # para cada letra do alfabeto:

          # Locate all child nodes of current node
          rows = df[df.l==l+1]
          rows = rows[rows.node_freq > 1]
          rows = rows[rows.len_idx >= m]
          rows = rows[rows.len_idx < m+alphabet_len]
          child_nodes_log_likelihood = rows.lps.sum()

          # poda
          if len(rows) > 1 and child_nodes_log_likelihood > row.lps:
            self.__update_row(df, l, node_idx, 'lps', child_nodes_log_likelihood)
            self.__update_row(df, l, node_idx, 'flag', 1)

  # TODO: rename this method - select_final_nodes
  def block3(self, df):
    data_len = len(self.data)
    for tree_length in range(1, self.context_tree.max_depth + 1):
      # TODO: change enumerate to yield to consume nodes on demand (optimizes memory)
      for node_idx, node in enumerate(self.context_tree.all_leaves(tree_length)):
        row = df[df.node==node].iloc[0]
        if tree_length==1:
          if row.flag==0:
            df.loc[(df.node==node), 'final'] = 1
        else:
          if row.flag==0:
            subnodes_str = [node[-(tree_length - m):] for m in range(1, tree_length)]
            subnodes = df[df['node'].isin(subnodes_str)]
            if subnodes['flag'].product() == 1 and row.node_freq > 0:
              df.loc[(df.node==node), 'final'] = 1

  def final_tree(self, df):
    selected_nodes = df.loc[df.final==1].sort_values(
      by=['node'],
      ascending=(True))
    return selected_nodes

  def __locate_row(self, df, l, node_idx):
    # len_idx is the node index in the list of all nodes with same length
    filtered_row = df[(df['l']==l) & (df['len_idx']==node_idx)]
    return filtered_row.iloc[0]

  def __update_row(self, df, l, node_idx, field_name, value):
    df.loc[(df['l']==l) & (df['len_idx']==node_idx), field_name] = value

  def __update_summary(self, summary, node, field_name, value):
    summary.loc[(summary['node']==node), field_name] = value


