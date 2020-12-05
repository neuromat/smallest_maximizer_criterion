import numpy as np
from g4l.util import parallel as prl
from g4l.util.stats import t_test


class Bootstrap():
    def __init__(self, champion_trees, resample_file, resample_sizes):
        self.champion_trees = champion_trees
        self.resample_file = resample_file
        self.resample_sizes = resample_sizes

    def calculate_likelihoods(self, temp_folder, num_cores=3):
        L = [None, None]
        for j, resample_size in enumerate(self.resample_sizes):
            print("Calculating likelihood j=", j+1)
            L[j] = prl.calculate_likelihoods(temp_folder,
                                             self.champion_trees,
                                             self.resample_file,
                                             resample_size,
                                             num_cores=num_cores)
        return np.array(L)

    def calculate_diffs(self, L):
        num_resample_sizes, num_trees, num_resamples = L.shape
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
                    diffs[j][t, b] /= (self.resample_sizes[j]**0.9)
            l_current = l_next
        return diffs

    def find_optimal_tree(self, diffs, alpha=0.01):
        t = len(self.champion_trees)-1
        d1, d2 = diffs
        pvalue = 1
        while pvalue > alpha and t > 0:
            t -= 1
            pvalue = t_test(d1[t], d2[t], alternative='greater')

        #rev_idxs = list(reversed(range(len(self.champion_trees)-1)))
        #res = np.array([t_test(d1[t], d2[t], alternative='greater') for t in rev_idxs])
        #first_occur_idx = (len(res) - np.argmax(res < alpha))+1
        ##first_occur_idx = np.argsort(1 - (res < alpha).astype(int))[0]
        return t+1

    def _initialize_diffs(self, num_trees, num_resamples):
        m = np.zeros((num_trees-1, num_resamples))
        return (m, m.copy())
