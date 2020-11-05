import random
import os
import numpy as np
#from scipy import stats
#from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import ttest_ind
from g4l.util import parallel
import math


class Bootstrap():
    def __init__(self, resample_factory, temp_folder,
                 num_resamples, resample_sizes=(100, 500),
                 alpha=0.01, num_cores=3):
        self.resample_factory = resample_factory
        self.temp_folder = temp_folder
        self.num_resamples = num_resamples
        self.resample_sizes = resample_sizes
        self.num_cores = num_cores
        self.alpha = alpha

    def find_optimal_tree(self, champion_trees):
        assert (champion_trees[0].num_contexts()
                >= champion_trees[-1].num_contexts())

        diffs = self._initialize_diffs(len(champion_trees))
        for j, sz in enumerate(self.resample_sizes):
            self._generate_resamples(j, sz)
        l_current = np.zeros((self.num_resamples, 2))

        for j, sz in enumerate(self.resample_sizes):
            print("Calculating likelihood j=", j+1)
            L = parallel.calculate_likelihoods(self.temp_folder,
                                               champion_trees,
                                               self._resample_file(j),
                                               num_cores=self.num_cores)
            l_current[:, j] = L[0]
        for t, tree in enumerate(champion_trees[1:]):
            l_next = np.zeros((self.num_resamples, 2))
            for j, sz in enumerate(self.resample_sizes):
                l_next[:, j] = L[t+1]
                for b in range(self.num_resamples):
                    diffs[j][t, b] = (l_current[b, j] - l_next[b, j])
                    diffs[j][t, b] /= (self.resample_sizes[j]**0.9)
            l_current = l_next
        pvalue = 1
        t = len(champion_trees)-1

        while (pvalue > self.alpha) and (t > 0):
            t -= 1
            d1, d2 = diffs
            #import code; code.interact(local=dict(globals(), **locals()))
            #_, pvalue, _ = ttest_ind(d1[t], d2[t], alternative='smaller')

            pvalue = self.t_test(d1[t], d2[t], alternative='greater')
            import code; code.interact(local=dict(globals(), **locals()))
        return t+1

    def t_test(self, x, y, alternative='both-sided'):
        _, double_p = ttest_ind(x, y, equal_var=False)
        if alternative == 'both-sided':
            pval = double_p
        elif alternative == 'greater':
            if np.mean(x) > np.mean(y):
                pval = double_p/2.
            else:
                pval = 1.0 - double_p/2.
        elif alternative == 'less':
            if np.mean(x) < np.mean(y):
                pval = double_p/2.
            else:
                pval = 1.0 - double_p/2.
        return pval

    def _generate_resamples(self, j, sz):
        file = self._resample_file(j)
        if os.path.exists(file):
            os.remove(file)
        for b in range(self.num_resamples): # TODO: run in parallel
            self.resample_factory.generate(sz, file)

    def _initialize_diffs(self, num_trees):
        m = np.zeros((num_trees-1, self.num_resamples))
        return (m, m.copy())

    def _resample_file(self, j):
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        return "%s/resamples.n%s.csv" % (self.temp_folder, j)
