import numpy as np
import pandas as pd
import re
from collections import defaultdict
import g4l
import h5py
import tables

class ContextTree():
  sample = None
  max_depth = None
  df = None
  chosen_penalty = None

  def __init__(self, sample, max_depth=4, source_data_frame=None, chosen_penalty=None):
    self.sample = sample
    self.chosen_penalty = chosen_penalty
    self.max_depth = max_depth
    self.df = None
    if source_data_frame is not None:
      self.df = source_data_frame.copy()
    else:
      self.initial_dataframe()

  @classmethod
  def load(cls, file_path, file_name):
    metadata_file = '%s/%s.metadata.h5' % (file_path, file_name)
    df_file = '%s/%s.df.pkl' % (file_path, file_name)

    hf = h5py.File(metadata_file, 'r')
    sample = g4l.data.Sample(hf.attrs['sample.filename'], hf.attrs['sample.A'])
    max_depth = hf.attrs['max_depth']
    chosen_penalty = hf.attrs['chosen_penalty']
    df = pd.read_pickle(df_file)
    hf.close()
    return ContextTree(sample, max_depth, df, chosen_penalty)

  def save(self, file_path, file_name):
    metadata_file = '%s/%s.metadata.h5' % (file_path, file_name)
    df_file = '%s/%s.df.pkl' % (file_path, file_name)

    self.df.to_pickle(df_file)
    hf = h5py.File(metadata_file, 'w')
    hf.attrs['sample.filename'] = self.sample.filename
    hf.attrs['sample.A'] = self.sample.A
    hf.attrs['max_depth'] = self.max_depth
    hf.attrs['chosen_penalty'] = (self.chosen_penalty or -1)
    hf.close()

  def evaluate_sample(self, data):
    sample = g4l.data.Sample('', self.sample.A, data=data)
    new_tree = ContextTree(sample, max_depth=self.max_depth, source_data_frame=self.df)
    new_tree.df.node_freq = new_tree.df.lps = new_tree.df.ps = 0
    new_tree.df.child_probs = None
    new_tree.df['child_probs'] = self.df['child_probs'].astype('object')
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
    return self.df.lps.sum()

  def leaves(self):
    return np.sort(list(self.df['node']))

  def equals_to(self, context_tree):
    return self.to_str()==context_tree.to_str()

  def initial_dataframe(self):
    df = self.empty_frame()
    node_idx = 0
    parent_node_indexes = defaultdict(lambda: None)
    for tree_length in range(1, self.max_depth + 1):
      # gets all leaves for trees of size 1 to max_depth
      for node in self.all_leaves(tree_length):
        parent_node_indexes[node] = node_idx

        # retrieve parent node idx from dictionary
        parent_idx = parent_node_indexes[node[1:]]

        # check how many occurrences of node exist in data
        node_freq = self.calc_node_frequency(node)
        # calculate child nodes' probs
        # TODO: remover abaixo
        ps = node_freq / (len(self.sample.data) - tree_length + 1)

        # frequencia da passagem pai -> filho
        child_freqs, child_probs = self.calc_transition_probs(node, node_freq)

        log_max_likelihood = self.calc_lpmls(child_freqs, child_probs)
        # Creates a new row for tree table
        # - last 2 fields will be further populated
        # - lpmls is the initial value, will be updated later

        row = [tree_length, node_idx, parent_idx, node, node_freq, log_max_likelihood, ps, child_probs, 0, 0]
        node_idx += 1
        df.loc[len(df)] = row
    #TODO: implement calculate_node_frequency()
    #TODO: implement calculate_node_prob()
    self.df = df
    return self.df

  def empty_frame(self):
    return pd.DataFrame(
      columns=['l', 'node_idx', 'parent_idx',
               'node', 'node_freq',
               'lps', 'ps',
               'child_probs', 'flag', 'final'])

  def calculate_node_frequency(self):
    for i, row in self.df.iterrows():
      self.df.at[i, 'node_freq'] = self.calc_node_frequency(row.node)

  def calculate_node_prob(self):
    sample_len = len(self.sample.data)
    child_probs_idx = self.df.columns.get_loc('child_probs')
    # calculates prob of occurrence for each node
    self.df.ps = self.df.node_freq / (sample_len - self.df.l + 1)
    # calculates child nodes' probs
    for i, row in self.df.iterrows():
      fr, pb = self.children_freq_prob(row.node, row.node_freq)
      self.df.at[i, 'child_probs'] = list(pb)
      self.df.at[i, 'lps'] = self.calc_lpmls(fr, pb)


  def children_freq_prob(self, node, node_freq):
    # returns freqs and probs as a pair of arrays; ex: ([10, 30, 60], [0.1, 0.3, 0.6])
    freqs = np.array([self.calc_node_frequency(node + str(sym)) for sym in self.sample.A])
    probs = freqs/node_freq
    return (freqs, probs)

  def calc_node_frequency(self, node):
    pos = [m.start() for m in re.finditer('(?=%s)' % node, self.sample.data)]
    return len(pos)

  # calculates $LPMLS = $LPMLS + @NSa[$j] * log( @PSa[$j] );

  # Logaritmo da máxima verossimilhança
  def calc_lpmls(self, child_freqs, child_probs):
    return np.sum(child_freqs[child_freqs > 0] * np.log(child_probs[child_freqs > 0]))

  def calc_transition_probs(self, node, node_freq):
    if node_freq > 0:
      child_freqs, child_probs = self.children_freq_prob(node, node_freq) # ([10, 30, 60], [0.1, 0.3, 0.6])
    else:
      child_freqs, child_probs = (np.zeros(len(self.sample.A)), np.zeros(len(self.sample.A)))
    return child_freqs, child_probs

  def all_leaves(self, depth):
    return np.array([''.join(x)[::-1] for x in self._all_leaves_by_len(depth)])

  def _all_leaves_by_len(self, depth):
      """Find the list of all strings of 'A' of depth 'depth'"""
      c = [[]]
      def to_str(arr):
          return [str(el) for el in arr]

      for i in range(1, depth+1):
          c = [to_str([x]+y) for x in self.sample.A for y in c]
      return np.array(c).astype(str)
