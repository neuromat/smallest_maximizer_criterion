from abc import ABCMeta, abstractmethod
import logging
import os

from .util import parallel as prl
from .util import persistence as per
from .util.stats import t_test
from numba import jit
import numpy as np


class EstimatorsBase():
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X):
        ''' Estimators must implement the `fit(X)` method '''
        self


class SMCBase(EstimatorsBase):

    def __init__(self, bootstrap_obj, cache_dir, num_cores=None):
        """
        Parameters
        ----------
        cache_dir : str
            When this variable is set with a valid path, the `fit` method will
            work cached; for any set of initial paramaters and sample X, there
            will exist a folder that receives the computed champion trees. In
            any further call using the same arguments, the method will returns
            the cached information.
        df_method : str
            The method used by BIC to calculate degrees_of_freedom. Options:
            - 'perl': uses the same df as the original implementation in perl
            - 'g4l': uses the method as described in the paper (slightly different)
            - 'csizar_and_talata': uses df as described in Csizar and Talata (2006)
        n_sizes : tuple<int, int>
            Bootstrap sample sizes. First element of tuple is the smallest
            size, whereas the second element is the largest one.
        renewal_point : int
            Renewal point used for bootstrap sampling.
        alpha : int
            Stop condition parameter for t-test phase.
        perl_compatible : int
            Makes algorithm compatible with the paper's perl code.

        """

        self.cache_dir = cache_dir or per.tempdir()
        self.num_cores = num_cores
        self.bootstrap = bootstrap_obj
        self.context_trees = []
        self.thresholds = []

    def optimal_tree(self, X, n_sizes, alpha):

        # Instantiates bootstrap
        # generates bootstrap samples
        sizes = (int(X.len() * n_sizes[0]), int(X.len() * n_sizes[1]))
        resamples_file = self.bootstrap.get_resamples(X, max(sizes),
                                                      self.num_cores)
        # calculates likelihoods
        L = self._calculate_likelihoods(resamples_file, sizes,
                                        num_cores=self.num_cores)

        # calculates likelihood deltas
        diffs = self._calculate_diffs(L, sizes)

        # Select optimal tree among the champion trees using t-test
        opt_idx = self._find_change_of_regime(diffs, alpha)

        # returns the selected tree and its index
        self.optimal_tree, self.opt_idx = self.context_trees[opt_idx], opt_idx

    def add_tree(self, new_tree):
        self.context_trees.append(new_tree)

    def _find_change_of_regime(self, diffs, alpha):
        """
        Performs t-test over each pair of deltas
        Returns the index of the champion tree where
        differences on their distributions are significant
        """
        d1, d2 = diffs
        t = d1.shape[0]
        pvalue = 1
        while pvalue > alpha and t > 0:
            t -= 1
            pvalue = t_test(d1[t], d2[t], alternative='greater')
        return t+1

    def _calculate_likelihoods(self, resamples_file, resample_sizes,
                               num_cores=0):
        """
        Calculates likelihoods for all boostrap samples
         for all the champion trees.
        """

        # Get locations for champion trees and likelihoods in the filesystem
        champion_trees_folder = per.save_champion_trees(self.context_trees,
                                                        self.cache_dir)
        temp_folder = os.path.join(self.cache_dir, 'likelihoods')
        filename = os.path.join(temp_folder, 'L.npy')

        try:
            # If likelihoods were already saved (cached), just returns it
            L = np.load(filename)
            return L
        except:
            # Otherwise, perform calculations
            L = [None, None]
            per.create_temp_folder(temp_folder)
            for j, resample_size in enumerate(resample_sizes):
                logging.info("Calculating likelihood j=%s" % (j+1))
                # Calculates likelihoods in parallel
                L[j] = prl.calculate_likelihoods(champion_trees_folder,
                                                 resamples_file,
                                                 int(resample_size),
                                                 num_cores=num_cores)
            np.save(filename, L)
            return np.array(L)

    def _calculate_diffs(self, L, n_sizes):
        num_resample_sizes, num_trees, num_resamples = L.shape
        return jit_calculate_diffs(n_sizes, L,
                                   num_resample_sizes,
                                   num_trees,
                                   num_resamples)


@jit(nopython=True)
def jit_calculate_diffs(resample_sizes, L,
                        num_resample_sizes,
                        num_trees,
                        num_resamples):
    m = np.zeros((num_trees-1, num_resamples))
    diffs = (m, m.copy())
    l_current = np.zeros((num_resamples, 2))
    for j in [0, 1]:
        l_current[:, j] = L[j][0]
    for t in range(num_trees - 1):
        l_next = np.zeros((num_resamples, 2))
        for j in range(num_resample_sizes):
            l_next[:, j] = L[j][t+1]
            for b in range(num_resamples):
                diffs[j][t, b] = (l_current[b, j] - l_next[b, j])
                diffs[j][t, b] /= (resample_sizes[j]**0.9)
        l_current = l_next
    return diffs
