from .smc_base import SMCBase
from .context_tree import ContextTree

#from g4l.models import ContextTree
from .bic import BIC
import logging

#from g4l.estimators.base import CollectionBase
#from g4l.estimators.logics import smc as smc
#from tempfile import TemporaryDirectory
#import numpy as np
#import os
#import logging


class SMC(SMCBase):
    """
    Context tree selection using the Smallest Maximizer Criterion
    Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
    Context tree selection and linguistic rhythm retrieval from written texts.
    Annals of Applied Statistics, 4(1), 186–209.
    https://doi.org/10.1214/11-AOAS511

    Download available at https://arxiv.org/abs/0902.3619

    ...

    Attributes
    ----------
    context_trees : list
        All champion trees found by the estimator

    thresholds : list
        The constant values used by BIC estimator to produce the
        context trees in the attribute `context_trees`.


    Methods
    -------
    fit(X)
        Estimates champion trees for the given sample X
    """

    def __init__(self, bootstrap_obj, penalty_interval=(0.1, 400),
                 n_sizes=(0.3, 0.9), alpha=0.01,
                 epsilon=0.01, cache_dir=None, df_method='perl',
                 perl_compatible=False, num_cores=None):
        """
        Parameters
        ----------
        penalty interval : tuple
            minimum and maximum values for the penalization constant used in
            the BIC estimator
        epsilon : float
            This value sets an stop condition when scanning for new trees
            between two penalty intervals in the BIC criteria
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
        perl_compatible : int
            Makes algorithm compatible with the paper's perl code

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
        """
        This method performs the algorithm describe in the section 4
        of the original paper

        Parameters
        ----------
        X : g4l.data.Sample
            A sample object

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
        logging.info('Finished CTM scanning')
        return self

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
        t = self.cached_trees(c)
        if not t:
            t = self.calc_bic(c)
            self._add_tree(c, t)
            logging.debug("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        else:
            logging.debug("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        return t

    def strategy_default(self, c):
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

    def cached_trees(self, k):
        for a, b in self.intervals.keys():
            if k >= a and b >= k:
                return self.intervals[(a, b)]
        return None

    def _add_tree(self, k, t):
        for i, t2 in enumerate(self.intervals.values()):
            if t.equals_to(t2):
                rng = list(self.intervals)[i]
                rng2 = (min(rng[0], k), max(rng[1], k))
                del(self.intervals[rng])
                self.intervals[rng2] = t
                return
        self.intervals[(k, k)] = t
