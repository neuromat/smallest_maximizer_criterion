from . import generation
from . import integrity
import numpy as np
import pandas as pd
import g4l
import re
from collections import defaultdict
import h5py

class ContextTree():
  sample = None
  max_depth = None
  df = None
  transitions_df = None

  def __init__(self, sample, max_depth=4, source_data_frame=None, tree_initialization_method=generation.incremental):
    self.sample = sample
    self.max_depth = max_depth
    self.transitions_df = pd.DataFrame(columns=['node_idx', 'symbol', 'freq', 'prob'])
    self.df = None
    if source_data_frame is not None:
      self.df = source_data_frame.copy()
    else:
      self.df = tree_initialization_method.run(self)

  @classmethod
  def load(cls, file_path, file_name):
    metadata_file = '%s/%s.metadata.h5' % (file_path, file_name)
    df_file = '%s/%s.df.pkl' % (file_path, file_name)
    hf = h5py.File(metadata_file, 'r')
    sample = g4l.data.Sample(hf.attrs['sample.filename'], hf.attrs['sample.A'])
    max_depth = hf.attrs['max_depth']
    df = pd.read_pickle(df_file)
    hf.close()
    return ContextTree(sample, max_depth, df)

  def save(self, file_path, file_name):
    metadata_file = '%s/%s.metadata.h5' % (file_path, file_name)
    df_file = '%s/%s.df.pkl' % (file_path, file_name)

    self.df.to_pickle(df_file)
    hf = h5py.File(metadata_file, 'w')
    hf.attrs['sample.filename'] = self.sample.filename
    hf.attrs['sample.A'] = self.sample.A
    hf.attrs['max_depth'] = self.max_depth
    hf.close()

  def evaluate_sample(self, data):
    sample = g4l.data.Sample('', self.sample.A, data=data)
    new_tree = ContextTree(sample, max_depth=self.max_depth, source_data_frame=self.df)
    new_tree.df.node_freq = new_tree.df.likelihood = new_tree.df.ps = 0
    new_tree.df.transition_probs = None
    new_tree.df['transition_probs'] = self.df['transition_probs'].astype('object')
    new_tree.df.reset_index(drop=True, inplace=True)
    new_tree.calculate_node_frequency()
    new_tree.calculate_node_prob()
    return new_tree

  # TODO: add `reverse=False` parameter to display contexts as root->leaf
  def to_str(self):
    return ' '.join(self.leaves())

  def num_contexts(self):
    return len(self.leaves())

  def log_likelihood(self):
    return self.tree().likelihood.sum()

  def tree(self):
    return self.contexts().sort_values(
          by=['node'],
          ascending=(True))

  def contexts(self, active_only=True):
    df = self.df
    r = df[~df.node_idx.isin(df[df.active==1].parent_idx)]
    if active_only==True:
      r = r[r.active==1]
    return r

  def leaves(self):
    return np.sort(list(self.tree()['node']))

  def equals_to(self, context_tree):
    return self.to_str()==context_tree.to_str()

  def calculate_node_frequency(self):
    for i, row in self.df.iterrows():
      self.df.at[i, 'freq'] = gen.calc_node_frequency(row.node, self.sample.data)

  def calculate_node_prob(self):
    sample_len = len(self.sample.data)
    transition_probs_idx = self.df.columns.get_loc('transition_probs')
    # calculates prob of occurrence for each node
    # self.df.ps = self.df.node_freq / (sample_len - self.df.l + 1)
    # calculates child nodes' probs
    for i, row in self.df.iterrows():
      fr, pb = gen.children_freq_prob(row.node, row.node_freq, self.A, self.sample.data)
      self.df.at[i, 'transition_probs'] = list(pb)
      self.df.at[i, 'likelihood'] = gen.calc_lpmls(fr, pb)
