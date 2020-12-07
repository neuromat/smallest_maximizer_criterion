from .base import CollectionBase
from g4l.models import ContextTree
from . import BIC
import os
from hashlib import md5
import pickle
from collections import Counter
from g4l.models.builders.tree_builder import ContextTreeBuilder
import logging


class SMC(CollectionBase):

    def __init__(self, max_depth, penalty_interval=(0.1, 400),
                 epsilon=0.01, cache_dir=None, callback_fn=None):

        assert max_depth > 0, 'max depth must be greater than zero'
        assert(epsilon > 0), 'epsilon must be greater than zero'
        super().__init__()
        self.max_depth = max_depth
        self.penalty_interval = penalty_interval
        self.epsilon = epsilon
        self.cache_dir = cache_dir
        self.callback_fn = callback_fn
        self.tresholds = []

    def fit(self, X):
        if self._load_cache(X):
            return self
        self.intervals = None
        self.initial_tree = ContextTree.init_from_sample(X, self.max_depth)
        min_c, max_c = self.penalty_interval
        self.trees_constructed = 0
        tree_a = self._bic(min_c)
        tree_b = tree_f = self._bic(max_c)
        self._add_tree(tree_a, min_c)

        a, b = (min_c, max_c)
        while not tree_a.equals_to(tree_b):
            while abs(b - a) > self.epsilon:
                while not tree_a.equals_to(tree_b) and abs(max_c - min_c) > 10**-5:
                    old_b = b
                    old_tree_b = tree_b
                    b = (a + b)/2
                    tree_b = self.strategy_dynamic(b)
                a = b
                b = old_b
                tree_b = old_tree_b
            a = b
            tree_a = tree_b
            self._add_tree(tree_a, b)
            b = max_c
            tree_b = tree_f
        if self.cache_dir is not None:
            self._save_cache(X)
        logging.info('Finished SMC')
        return self

    def _bic(self, c):
        self.trees_constructed += 1
        bic_estimator = BIC(c, self.max_depth).fit(self.initial_tree.sample)
        return bic_estimator.context_tree

    def _add_tree(self, t, c):
        self.add_tree(t)
        self.tresholds.append(c)
        try:
            self.callback_fn((c, t))
        except TypeError:
            pass

    def _load_cache(self, X):
        cache_folder, cachefile = self.cache_file(X)
        try:
            with open(cachefile, 'rb') as f:
                dic = pickle.load(f)
            print("Loaded from cache")
        except FileNotFoundError:
            return False
        self.max_depth = dic['max_depth']
        self.penalty_interval = dic['penalty_interval']
        self.epsilon = dic['epsilon']
        self.cache_dir = dic['cache_dir']
        self.tresholds = dic['tresholds']
        self.context_trees = []
        for i in range(dic['context_trees']):
            t = ContextTree.load_from_file('%s/%s.tree' % (cache_folder, i))
            self.context_trees.append(t)
        return True

    def cache_file(self, X):
        strg = 'SMC' + X.data + str(self.penalty_interval) + str(self.epsilon)
        cachefile = md5((strg).encode('utf-8')).hexdigest()
        cachefile = '%s/%s/params.pkl' % (self.cache_dir, cachefile)
        folder = os.path.dirname(cachefile)
        return folder, cachefile

    def _save_cache(self, X):
        folder, cachefile = self.cache_file(X)
        print("Cached in folder %s" % folder)
        os.makedirs(folder, exist_ok=True)
        dic = {'max_depth': self.max_depth,
               'penalty_interval': self.penalty_interval,
               'epsilon': self.epsilon,
               'cache_dir': self.cache_dir,
               'tresholds': self.tresholds,
               'context_trees': len(self.context_trees)}
        with open(cachefile, 'wb') as f:
            pickle.dump(dic, f)
        for i, t in enumerate(self.context_trees):
            t.save('%s/%s.tree' % (folder, i))

    def strategy_default(self, c):
        bic = self._bic(c)
        logging.debug('c=%s; \t\tt=%s' % (round(c, 4), bic.to_str()))
        return bic

    def strategy_dynamic(self, c):
        # This strategy avoids computing trees where the current c is between
        # 2 already computed values of c that have produced the same tree
        if self.intervals is None:
            self.intervals = dict()
        t = self.__cached_trees(c)
        if not t:
            t = self._bic(c)
            self.__add_tree(c, t)
            print("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        else:
            print("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
        return t


    def __cached_trees(self, k):
        for a, b in self.intervals.keys():
            if k >= a and b >= k:
                #print("++", k, a, b, ': '[(a, b)].to_str())
                return self.intervals[(a, b)]
        return None

    def __add_tree(self, k, t):
        for i, t2 in enumerate(self.intervals.values()):
            if t.equals_to(t2):
                rng = list(self.intervals)[i]
                rng2 = (min(rng[0], k), max(rng[1], k))
                del(self.intervals[rng])
                self.intervals[rng2] = t
                return
        self.intervals[(k, k)] = t
