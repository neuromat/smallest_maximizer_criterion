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

    def optimal_tree(self, resamples_folder,  num_resamples,
                     n_sizes,
                     alpha,
                     renewal_point,
                     num_cores=None):
        from g4l.bootstrap.resampling import BlockResampling
        from g4l.bootstrap import Bootstrap
        # Use bootstrap
        resamples_file = resamples_folder + "/resamples.txt"
        bootstrap = Bootstrap(self.context_trees, resamples_file, n_sizes)
        L_path = "%s/L.npy" % (resamples_folder)
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
            L = bootstrap.calculate_likelihoods(resamples_folder, num_cores=num_cores)
            # Save to cache
            np.save(L_path, L)
        # Select optimal tree among the champion trees using t-test
        opt_idx = bootstrap.find_optimal_tree(L, alpha=alpha)
        return self.context_trees[opt_idx]

    def add_tree(self, new_tree):
        self.context_trees.append(new_tree)
