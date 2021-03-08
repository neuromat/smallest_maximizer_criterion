"""
Implementation of the Smallest Maximizer Criterion as stated in the paper:

Context tree selection using the Smallest Maximizer Criterion
Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
Context tree selection and linguistic rhythm retrieval from written texts.
Annals of Applied Statistics, 4(1), 186–209.
https://doi.org/10.1214/11-AOAS511

"""

from g4l.models import ContextTree
from g4l.estimators import BIC
from g4l.estimators.caching import smc as cache
import logging


def fit(estimator, X):
    """
    This method performs the algorithm describe in the section 4
    of the original paper

    Parameters
    ----------
    X : g4l.data.Sample
        A sample object

    """

    estimator.thresholds = []
    scan_offset = estimator.scan_offset
    perl_compatible = estimator.perl_compatible
    df_method = estimator.df_method
    max_depth = estimator.max_depth
    estimator.intervals = None
    #if cache.load_cache(estimator, X):
    #    return estimator
    estimator.initial_tree = ContextTree.init_from_sample(X, max_depth)
    min_c, max_c = estimator.penalty_interval
    estimator.trees_constructed = 0
    tree_a = calc_bic(estimator, min_c, df_method, scan_offset, perl_compatible)

    tree_b = tree_f = calc_bic(estimator, max_c, df_method, scan_offset, perl_compatible)
    add_tree(estimator, tree_a, min_c)

    a, b = (min_c, max_c)
    while not tree_a.equals_to(tree_b):
        while abs(b - a) > estimator.epsilon:
            while not tree_a.equals_to(tree_b) and abs(max_c - min_c) > 10**-5:
                old_b = b
                old_tree_b = tree_b
                b = (a + b)/2
                tree_b = strategy_dynamic(estimator, b, df_method, scan_offset, perl_compatible)
            a = b
            b = old_b
            tree_b = old_tree_b
        a = b
        tree_a = tree_b
        add_tree(estimator, tree_a, b)
        b = max_c
        tree_b = tree_f
    #if estimator.cache_dir is not None:
    #    cache.save_cache(estimator, X)
    logging.info('Finished CTM scanning')
    return estimator


def calc_bic(estimator, c, df_method, scan_offset, perl_compatible):
    """
    Estimate a context tree using the BIC estimator

    Parameters
    ----------
    estimator : g4l.data.SMC
        The SMC estimator object
    c : float
        The penalty value ( > 0) used by the bayesian information criteria

    """
    sample = estimator.initial_tree.sample
    estimator.trees_constructed += 1

    bic_estimator = BIC(c, estimator.max_depth,
                        df_method=df_method,
                        scan_offset=scan_offset,
                        keep_data=True,
                        perl_compatible=perl_compatible)
    bic_estimator.fit(sample)
    return bic_estimator.context_tree


def add_tree(estimator, t, c):
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
    estimator.add_tree(t)
    estimator.thresholds.append(c)
    try:
        estimator.callback_fn((c, t))
    except TypeError:
        pass


def strategy_dynamic(estimator, c, df_method, scan_offset, perl_compatible):
    """
    This strategy avoids computing trees where the current c is between
    2 already computed values of c that have produced the same tree, for
    speed-up purposes. The original procedure described in the paper is
    found in `strategy_default` method.


    Parameters
    ----------
    estimator : g4l.estimators.SMC
        The resulting context tree
    c : float
        The penalty value used to estimate the tree
    """

    if estimator.intervals is None:
        estimator.intervals = dict()
    t = cached_trees(estimator, c)
    if not t:
        t = calc_bic(estimator, c, df_method, scan_offset, perl_compatible)
        __add_tree(estimator, c, t)
        logging.debug("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
    else:
        logging.debug("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
    return t


def strategy_default(estimator, c, df_method, scan_offset, perl_compatible):
    """
    Computes the context tree using the BIC estimator for
    a given penalty value `c`


    Parameters
    ----------
    estimator : g4l.estimators.SMC
        The resulting context tree
    c : float
        The penalty value used to estimate the tree
    """

    bic = calc_bic(estimator, c, df_method, scan_offset, perl_compatible)
    logging.debug('c=%s; \t\tt=%s' % (round(c, 4), bic.to_str()))
    return bic


def cached_trees(estimator, k):
    for a, b in estimator.intervals.keys():
        if k >= a and b >= k:
            #print("++", k, a, b, ': '[(a, b)].to_str())
            return estimator.intervals[(a, b)]
    return None


def __add_tree(estimator, k, t):
    for i, t2 in enumerate(estimator.intervals.values()):
        if t.equals_to(t2):
            rng = list(estimator.intervals)[i]
            rng2 = (min(rng[0], k), max(rng[1], k))
            del(estimator.intervals[rng])
            estimator.intervals[rng2] = t
            return
    estimator.intervals[(k, k)] = t
