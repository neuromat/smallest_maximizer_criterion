import numpy as np
from g4l.util import parallel as prl
from g4l.util.stats import t_test
from numba import jit
from pathlib import Path
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


@jit(nopython=True)
def jit_calculate_diffs(resample_sizes, L, num_resample_sizes, num_trees, num_resamples):
    # import code; code.interact(local=dict(globals(), **locals()))
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


class Bootstrap():
    def __init__(self, champion_trees, resample_file, resample_sizes):
        self.champion_trees = champion_trees
        self.resample_file = resample_file
        self.resample_sizes = resample_sizes

    def calculate_likelihoods(self, temp_folder, num_cores=3):
        L = [None, None]
        for j, resample_size in enumerate(self.resample_sizes):
            print("Calculating likelihood j=", j+1)
            self.create_temp_folder(temp_folder)
            L[j] = prl.calculate_likelihoods(temp_folder,
                                             self.champion_trees,
                                             self.resample_file,
                                             int(resample_size),
                                             num_cores=num_cores)
        return np.array(L)

    def create_temp_folder(self, temp_folder):
        Path(temp_folder).mkdir(parents=True, exist_ok=True)

    def calculate_diffs(self, L):
        num_resample_sizes, num_trees, num_resamples = np.array(L).shape
        return jit_calculate_diffs(self.resample_sizes, L, num_resample_sizes, num_trees, num_resamples)

    def find_optimal_tree(self, L, alpha=0.01):
        diffs = self.calculate_diffs(np.array(L))
        #t = len(self.champion_trees)-1
        #import code; code.interact(local=dict(globals(), **locals()))
        d1, d2 = diffs
        t = len(d1)
        pvalue = 1
        while pvalue > alpha and t > 0:
            t -= 1
            pvalue = t_test(d1[t], d2[t], alternative='greater')
        return t+1

    def _initialize_diffs(self, num_trees, num_resamples):
        m = np.zeros((num_trees-1, num_resamples))
        return (m, m.copy())
