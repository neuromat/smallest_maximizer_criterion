from .builders import incremental
from . import persistence
import numpy as np
import pandas as pd

class ContextTree():
  sample = None
  max_depth = None
  df = None
  transition_probs = None

  def __init__(self, max_depth, contexts_dataframe, transition_probs, source_sample=None):
    self.max_depth = max_depth
    self.df = contexts_dataframe
    self.transition_probs = transition_probs
    self.sample = source_sample


  @classmethod
  def init_from_sample(cls, X, max_depth, initialization_method=incremental):
    """ Builds a complete, admissible initial tree from a given sample """

    contexts, transition_probs = initialization_method.run(X, max_depth)
    return ContextTree(max_depth, contexts, transition_probs, X)

  @classmethod
  def load_from_file(cls, file_path):
    """ Loads model data from file """

    X, max_depth, contexts, transition_probs = persistence.load_model(file_path)
    return ContextTree(max_depth, contexts, transition_probs, X)


  def copy(self):
    """ Creates a complete copy of the model """
    return ContextTree(self.max_depth,
                      self.df.copy(),
                      self.transition_probs.copy(),
                      source_sample=self.sample)

  def save(self, file_path):
    """ Saves model in a file """

    persistence.save_model(self, file_path)

  def evaluate_sample(self, new_sample):
    new_tree = self.copy()
    new_tree.df.node_freq = new_tree.df.likelihood = new_tree.df.ps = 0
    new_tree.df.transition_probs = None
    new_tree.df['transition_probs'] = self.df['transition_probs'].astype('object')
    new_tree.df.reset_index(drop=True, inplace=True)
    new_tree.calculate_node_frequency()
    new_tree.calculate_node_prob()
    return new_tree

  def __str__(self):
    return self.to_str()

  def to_str(self):
    """ Represents context tree as a string

    TODO: add `reverse=False` parameter to display contexts as root->leaf
    """

    return ' '.join(self.leaves())

  def num_contexts(self):
    """ Returns the number of contexts """

    return len(self.leaves())

  def log_likelihood(self):
    """ Returns the total log likelihood for all active contexts """

    return self.tree().likelihood.sum()

  def tree(self):
    """ Returns the tree with all active contexts ascending by nodes"""

    return self.contexts().sort_values(
          by=['node'],
          ascending=(True))

  def contexts(self, active_only=True):
    """ Returns the tree with all active contexts"""

    df = self.df
    r = df[~df.node_idx.isin(df[df.active==1].parent_idx)]
    if active_only==True:
      r = r[r.active==1]
    return r

  def leaves(self):
    return np.sort(list(self.tree()['node']))

  def equals_to(self, context_tree):

    """ Matches the current context tree to another one """
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
