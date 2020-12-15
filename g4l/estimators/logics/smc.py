from g4l.models import ContextTree
from g4l.estimators import BIC
from hashlib import md5
import logging
import os
import pickle


def fit(estimator, X):
    estimator.tresholds = []
    max_depth = estimator.max_depth
    estimator.intervals = None
    if load_cache(estimator, X):
        return estimator
    estimator.initial_tree = ContextTree.init_from_sample(X, max_depth)
    min_c, max_c = estimator.penalty_interval
    estimator.trees_constructed = 0
    tree_a = bic(estimator, min_c)
    tree_b = tree_f = bic(estimator, max_c)
    add_tree(estimator, tree_a, min_c)

    a, b = (min_c, max_c)
    while not tree_a.equals_to(tree_b):
        while abs(b - a) > estimator.epsilon:
            while not tree_a.equals_to(tree_b) and abs(max_c - min_c) > 10**-5:
                old_b = b
                old_tree_b = tree_b
                b = (a + b)/2
                tree_b = strategy_dynamic(estimator, b)
            a = b
            b = old_b
            tree_b = old_tree_b
        a = b
        tree_a = tree_b
        add_tree(estimator, tree_a, b)
        b = max_c
        tree_b = tree_f
    if estimator.cache_dir is not None:
        estimator._save_cache(X)
    logging.info('Finished SMC')
    return estimator


def bic(estimator, c):
    sample = estimator.initial_tree.sample
    estimator.trees_constructed += 1
    bic_estimator = BIC(c, estimator.max_depth).fit(sample)
    return bic_estimator.context_tree


def add_tree(estimator, t, c):
    estimator.add_tree(t)
    estimator.tresholds.append(c)
    try:
        estimator.callback_fn((c, t))
    except TypeError:
        pass


def load_cache(estimator, X):
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
    estimator.tresholds = dic['tresholds']
    estimator.context_trees = []
    for i in range(dic['context_trees']):
        t = ContextTree.load_from_file('%s/%s.tree' % (cache_folder, i))
        estimator.context_trees.append(t)
    return True


def cache_file(estimator, X):
    strg = 'SMC%s%s%s' % (X.data,
                            estimator.penalty_interval,
                            estimator.epsilon)
    cachefile = md5((strg).encode('utf-8')).hexdigest()
    cachefile = '%s/%s/params.pkl' % (estimator.cache_dir, cachefile)
    folder = os.path.dirname(cachefile)
    return folder, cachefile


def _save_cache(estimator, X):
    folder, cachefile = estimator.cache_file(X)
    logging.info("Cached in folder %s" % folder)
    os.makedirs(folder, exist_ok=True)
    dic = {'max_depth': estimator.max_depth,
           'penalty_interval': estimator.penalty_interval,
           'epsilon': estimator.epsilon,
           'cache_dir': estimator.cache_dir,
           'tresholds': estimator.tresholds,
           'context_trees': len(estimator.context_trees)}
    with open(cachefile, 'wb') as f:
        pickle.dump(dic, f)
    for i, t in enumerate(estimator.context_trees):
        t.save('%s/%s.tree' % (folder, i))


def strategy_default(estimator, c):
    bic = bic(estimator, c)
    logging.debug('c=%s; \t\tt=%s' % (round(c, 4), bic.to_str()))
    return bic


def strategy_dynamic(estimator, c):
    # This strategy avoids computing trees where the current c is between
    # 2 already computed values of c that have produced the same tree
    if estimator.intervals is None:
        estimator.intervals = dict()
    t = cached_trees(estimator, c)
    if not t:
        t = bic(estimator, c)
        __add_tree(estimator, c, t)
        print("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
    else:
        print("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
    return t


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
