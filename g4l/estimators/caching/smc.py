"""
Caching methods for SMC

"""

import logging
import os
import pickle
from hashlib import md5
from g4l.models import ContextTree
from g4l.util import hashstr

def load_cache(estimator, X):
    """
    Loads previously estimated context trees from file. The result
    folder is unique for each set of SMC parameters (epsilon and
    penalty interval) and sample.

    Parameters
    ----------
    estimator : g4l.estimators.SMC
        The resulting context tree
    X : g4l.data.Sample
        A sample

    """
    cache_folder, cachefile = cache_file(estimator, X)
    try:
        with open(cachefile, 'rb') as f:
            dic = pickle.load(f)
        print("Loaded from cache")
    except FileNotFoundError:
        return False
    estimator.max_depth = dic['max_depth']
    estimator.penalty_interval = dic['penalty_interval']
    estimator.epsilon = dic['epsilon']
    estimator.cache_dir = dic['cache_dir']
    estimator.thresholds = dic['thresholds']
    estimator.context_trees = []
    for i in range(dic['context_trees']):
        t = ContextTree.load_from_file('%s/%s.tree' % (cache_folder, i))
        estimator.context_trees.append(t)
    return True


def cache_file(estimator, X):
    """
    Loads previously estimated context trees from file. The result
    folder is unique for each set of SMC parameters (epsilon and
    penalty interval) and sample.

    Parameters
    ----------
    estimator : g4l.estimators.SMC
        The resulting context tree
    X : g4l.data.Sample
        A sample

    """
    strg = 'SMC%s%s%s' % (X.data,
                          estimator.penalty_interval,
                          estimator.epsilon)
    cachefile = hashstr(strg)
    cachefile = '%s/%s/params.pkl' % (estimator.cache_dir, cachefile)
    folder = os.path.dirname(cachefile)
    return folder, cachefile


def save_cache(estimator, X):
    """
    Persists estimated context trees into a cache folder

    Parameters
    ----------
    estimator : g4l.estimators.SMC
        The resulting context tree
    X : g4l.data.Sample
        A sample

    """
    folder, cachefile = cache_file(estimator, X)
    logging.info("Cached in folder %s" % folder)
    os.makedirs(folder, exist_ok=True)
    dic = {'max_depth': estimator.max_depth,
           'penalty_interval': estimator.penalty_interval,
           'epsilon': estimator.epsilon,
           'cache_dir': estimator.cache_dir,
           'thresholds': estimator.thresholds,
           'context_trees': len(estimator.context_trees)}
    with open(cachefile, 'wb') as f:
        pickle.dump(dic, f)
    for i, t in enumerate(estimator.context_trees):
        t.save('%s/%s.tree' % (folder, i))
