""" Implements the Smallest Maximizer Criterion algorithm

Context tree selection and linguistic rhythm retrieval from written texts.
Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2012).
Annals of Applied Statistics, 4(1), 186–209.
https://doi.org/10.1214/11-AOAS511
"""

import logging
from .bic import BIC
from .context_tree import ContextTree
from .smc_base import SMCBase


class SMC(SMCBase):
    def __init__(self, bootstrap_obj, penalty_interval=(0.1, 400),
                 n_sizes=(0.3, 0.9), alpha=0.01,
                 epsilon=0.01, cache_dir=None, df_method='perl',
                 perl_compatible=False, num_cores=None):
        """Initializes the estimator

        Arguments:
            bootstrap_obj {Boostrap} -- A previously configured bootstrap
                                        object

        Keyword Arguments:
            penalty_interval {tuple} -- The interval of BIC constants
                                        to be scanned (default: {(0.1, 400)})
            n_sizes {tuple} -- Bootstrap sample sizes. First element of tuple
                               is the smallest size, whereas the second element
                               is the largest one. (default: {(0.3, 0.9)})
            alpha {number} -- Stop condition parameter for t-test phase.
                              (default: {0.01})
            epsilon {number} -- Stop condition parameter for BIC constant
                                scanning. (default: {0.01})
            cache_dir {str} -- Directory for temporary files. The program
                               creates one by itself if none is set
                               (default: {None})
            df_method {str} -- Degree of freedom strategy (default: {'perl'})
            perl_compatible {bool} -- Makes algorithm compatible with the
                                      original perl implementation.
                                      (default: {False})
            num_cores {int} -- Number of cores for parallel processing
                               (default: {None})
        """

        super().__init__(bootstrap_obj, cache_dir,
                         num_cores=num_cores,
                         n_sizes=n_sizes,
                         alpha=alpha)

        self.penalty_interval = penalty_interval
        self.epsilon = epsilon
        self.df_method = df_method
        self.perl_compatible = perl_compatible
        assert epsilon > 0, 'epsilon must be greater than zero'

    def fit(self, X):
        self.context_trees = []
        self.estimate_trees(X)

        # this methods are defined in the superclass (smc_base)
        self.find_optimal_tree(X)

        return self

    def estimate_trees(self, X):
        """Estimates the set of champion

        This method performs the algorithm describe in the section 4
        of the original paper

        Arguments:
            X {Sample} -- A sample

        Returns:
            SMC -- This object instance itself
        """

        self.thresholds = []
        self.intervals = None
        #if cache.load_cache(estimator, X):
        #    return estimator
        self.initial_tree = ContextTree.init_from_sample(X)
        min_c, max_c = self.penalty_interval
        self.trees_constructed = 0
        tree_a = self.calc_bic(min_c)
        tree_b = tree_f = self.calc_bic(max_c)
        self.add(tree_a, min_c)

        a, b = (min_c, max_c)
        while not tree_a.equals_to(tree_b):
            while abs(b - a) > self.epsilon:
                while not tree_a.equals_to(tree_b) and abs(max_c - min_c) > 10 ** -5:
                    old_b = b
                    old_tree_b = tree_b
                    b = (a + b)/2
                    tree_b = self.strategy_dynamic(b)
                a = b
                b = old_b
                tree_b = old_tree_b
            a = b
            tree_a = tree_b
            self.add(tree_a, b)
            b = max_c
            tree_b = tree_f

    def calc_bic(self, c):
        """
        Estimate a context tree using the BIC estimator

        Parameters
        ----------
        estimator : g4l.data.SMC
            The SMC estimator object
        c : float
            The penalty value ( > 0) used by the bayesian information criteria

        """

        sample = self.initial_tree.sample
        self.trees_constructed += 1

        bic_estimator = BIC(c, df_method=self.df_method,
                            keep_data=True,
                            perl_compatible=self.perl_compatible)
        bic_estimator.fit(sample)
        return bic_estimator.context_tree

    def add(self, t, c):
        """
        Appends a context tree to the list of estimated trees

        Parameters
        ----------
        t : g4l.models.ContextTree
            The resulting context tree
        c : float
            The penalty value used to estimate the tree

        """

        logging.info('%s\t%s' % (c, t.to_str()))
        self.add_tree(t)
        self.thresholds.append(c)

    def strategy_dynamic(self, c):
        """
        This strategy avoids computing trees where the current c is between
        2 already computed values of c that have produced the same tree, for
        speed-up purposes. The original procedure described in the paper is
        found in `strategy_default` method.


        Parameters
        ----------
        c : float
            The penalty value used to estimate the tree
        """

        if self.intervals is None:
            self.intervals = dict()
        t = self._cached_trees(c)
        if not t:
            t = self.calc_bic(c)
            self._add_tree(c, t)
            logging.debug("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        else:
            logging.debug("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        return t

    def strategy_default(self, c):   # pragma: no cover
        """
        Computes the context tree using the BIC estimator for
        a given penalty value `c`


        Parameters
        ----------
        c : float
            The penalty value used to estimate the tree
        """

        bic = self.calc_bic(c)
        logging.debug('c=%s; \t\tt=%s' % (round(c, 4), bic.to_str()))
        return bic

    def _cached_trees(self, k):
        """ Skips tree computation if the tree for the given constant
            is already known (for speed-up purposes)
            if i1 <= k <= i2, where i1, i2 are constant values
            that were already computed and tree_{i1} ==  tree_{i2}
        """

        for a, b in self.intervals.keys():
            if k >= a and b >= k:
                return self.intervals[(a, b)]
        return None

    def _add_tree(self, k, t):
        """ adds the tree to the interval

        [description]

        Arguments:
            k {float} -- BIC constant used to compute the tree
            t {ContextTree} -- a tree
        """

        for i, t2 in enumerate(self.intervals.values()):
            if t.equals_to(t2):
                rng = list(self.intervals)[i]
                rng2 = (min(rng[0], k), max(rng[1], k))
                del(self.intervals[rng])
                self.intervals[rng2] = t
                return
        self.intervals[(k, k)] = t
