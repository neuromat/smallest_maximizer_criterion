import random
import pandas as pd
import numpy as np
from scipy import stats
from g4l.models.builders import incremental

class Bootstrap():
  def __init__(self, original_sample, resample_factory, temp_folder, num_resamples, resample_sizes=(100, 500), alpha=0.01):
    self.original_sample = original_sample
    self.resample_factory = resample_factory
    self.temp_folder = temp_folder
    self.num_resamples = num_resamples
    self.resample_sizes = resample_sizes
    self.alpha = alpha

  def find_optimal_tree(champion_trees):
    diffs = self._initialize_diffs(len(champion_trees))
    for j, sz in enumerate(self.resample_sizes):
      self._generate_resamples(j, sz)
    l_current = np.zeros((self.num_resamples, 2))
    for j in self.resample_sizes:
      for b, resample in self._get_resamples(j):
        tree = champion_trees[0]
        l_current[b, j] = self.__calc_likelihood(tree, resample)
    for t, tree in enumerate(champion_trees):
      l_next = np.zeros((self.num_resamples, 2))
      for j in self.resample_sizes:
        for b, resample in self._get_resamples(j):
          l_next[b, j] = self.__calc_likelihood(tree, resample)
          diffs[j][t, b] = (l_current[b, j] - l_next[b, j])/(self.resample_sizes**0.9)
      l_current = l_next
    pvalue = 1
    t = len(champion_trees)
    while (pvalue > alpha) and (t > 0):
      t-=1
      d1, d2 = diffs
      _, pvalue = stats.ttest_ind(d1[t], d2[t], alpha=self.alpha)
    return champion_trees[t+1]


  def _generate_resamples(j, sz):
    for b in range(self.num_resamples): # TODO: run in parallel
      self.resample_factory.generate(sz, self.resample_file(j))


  def _calc_likelihood(tree, resample_data):
    resample = Sample(None, self.original_sample.A, data=resample_data)
    likelihood = tree.sample_likelihood(resample)

  def _initialize_diffs(num_trees):
    m = np.zeros((num_trees-1, self.num_resamples))
    return (m, m.copy())

  def _resample_file(j):
    return "%s/resamples.n%s.csv" % (self.temp_folder, j)

  def _get_resamples(j):
    pass

  def run_t_test(alpha):
    pass

  def _generate_likelihood_df(self):
    return pd.DataFrame(['tree_idx', 'sample_idx', 'n', 'L'])

  def freqs(self):
    pass
    #df = incremental.count_subsequence_frequencies(pd.DataFrame(), sample, max_depth)
