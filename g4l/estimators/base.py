from abc import ABCMeta, abstractmethod
import os
import logging
import numpy as np


class Base():
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(X):
        ''' To override '''
        pass


class CollectionBase(Base):
    def __init__(self):
        self.context_trees = []

    def optimal_tree(self, results_folder,  num_resamples,
                     n_sizes,
                     alpha,
                     renewal_point,
                     num_cores=None):
        from g4l.bootstrap.resampling import BlockResampling
        from g4l.bootstrap import Bootstrap
        # Use bootstrap
        champion_trees_folder = self.save_champion_trees(results_folder)
        resamples_file = os.path.join(results_folder, "resamples.txt")
        bootstrap = Bootstrap(champion_trees_folder, resamples_file, n_sizes)
        L_path = "%s/L.npy" % (results_folder)
        if os.path.isfile(L_path):
            # Use precomputed likelihoods when available
            L = np.load(L_path)
        else:
            # Generate samples using block resampling strategy
            resample_fctry = BlockResampling(self.X, resamples_file,
                                             n_sizes,
                                             renewal_point)
            logging.info("Generating bootstrap samples (n: %s)" % num_resamples)
            resample_fctry.generate(num_resamples, num_cores=num_cores)

            # Calculate tree likelihoods for all resamples
            logging.info("Calculating likelihoods")
            L = bootstrap.calculate_likelihoods(results_folder, num_cores=num_cores)
            # Save to cache
            np.save(L_path, L)
        # Select optimal tree among the champion trees using t-test
        opt_idx = bootstrap.find_optimal_tree(L, alpha=alpha)
        return self.context_trees[opt_idx]

    def save_champion_trees(self, results_folder):
        trees_folder = os.path.join(results_folder, 'champion_trees')
        os.makedirs(trees_folder, exist_ok=True)
        for i, tree in enumerate(self.context_trees):
            n = '%06d' % i
            tree.save(os.path.join(trees_folder, '%s.tree' % n))
        return trees_folder

    def add_tree(self, new_tree):
        self.context_trees.append(new_tree)

