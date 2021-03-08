from abc import ABCMeta, abstractmethod
import os
from g4l.util import parallel as prl
from g4l.util.stats import t_test
import numpy as np
from g4l.bootstrap import Bootstrap
from numba import jit
from pathlib import Path
import shutil


@jit(nopython=True)
def jit_calculate_diffs(resample_sizes, L, num_resample_sizes, num_trees, num_resamples):
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



class Base():
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(X):
        ''' To override '''
        pass


class CollectionBase(Base):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.context_trees = []

    def optimal_tree(self, num_resamples,
                     n_sizes,
                     alpha,
                     renewal_point,
                     num_cores=None):
        # Use bootstrap
        bootstrap = Bootstrap(self.X, self.cache_dir, max(n_sizes),
                              num_resamples, renewal_point,
                              num_cores=num_cores)
        resamples_file = bootstrap.resamples()
        L = self._calculate_likelihoods(resamples_file, n_sizes,
                                        num_cores=num_cores)
        diffs = self._calculate_diffs(L, n_sizes)
        opt_idx = self._find_change_of_regime(diffs, alpha)
        # Select optimal tree among the champion trees using t-test
        ret = self.context_trees[opt_idx], opt_idx
        #try:
        #
        #except:
        #    import code; code.interact(local=dict(globals(), **locals()))
        return ret

    def _find_change_of_regime(self, diffs, alpha):
        d1, d2 = diffs
        t = d1.shape[0]
        pvalue = 1
        #vals = [t_test(d1[ti], d2[ti], alternative='greater') for ti in list(reversed(range(t)))]
        while pvalue > alpha and t > 0:
            t -= 1
            pvalue = t_test(d1[t], d2[t], alternative='greater')
            #if pvalue != pvalue:  # pvalue = 1 if pvalue == nan
            #    pvalue = 1.0
            #print(t, pvalue)
        #import code; code.interact(local=dict(globals(), **locals()))
        return t+1

    def add_tree(self, new_tree):
        self.context_trees.append(new_tree)

    def _calculate_likelihoods(self, resamples_file, resample_sizes,
                               num_cores=0):
        champion_trees_folder = self._save_champion_trees()
        temp_folder = os.path.join(self.cache_dir, 'likelihoods')
        filename = os.path.join(temp_folder, 'L.npy')

        try:
            L = np.load(filename)
            return L
        except:
            L = [None, None]
            for j, resample_size in enumerate(resample_sizes):
                print("Calculating likelihood j=", j+1)
                self.create_temp_folder(temp_folder)
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

    def _save_champion_trees(self):
        trees_folder = os.path.join(self.cache_dir, 'champion_trees')
        try:
            shutil.rmtree(trees_folder)
        except FileNotFoundError:
            pass
        os.makedirs(trees_folder, exist_ok=True)
        for i, tree in enumerate(self.context_trees):
            n = '%06d' % i
            tree.save(os.path.join(trees_folder, '%s.tree' % n))
        return trees_folder

    def create_temp_folder(self, temp_folder):
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
