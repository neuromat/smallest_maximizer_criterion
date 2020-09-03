import numpy as np
import pandas as pd
from . import CollectionBase
from . import CTM
from datetime import datetime
import logging


class SMC(CollectionBase):

  def execute(self, penalty_interval=(0.1, 400), epsilon=0.01):
    self.intervals = None
    min_c, max_c = penalty_interval
    self.trees_constructed = 0

    logging.info('Starting CTM Scanner')
    tree_a = self.__calc_bic(min_c)
    tree_b = tree_f = self.__calc_bic(max_c)
    self.add_tree(tree_a.df)
    a, b = (min_c, max_c)
    while not tree_a.equals_to(tree_b):
      while b - a > epsilon:
        while not tree_a.equals_to(tree_b):
          old_b = b
          old_tree_b = tree_b
          b = (a + b)/2
          tree_b = self.strategy_dynamic(b)
          #tree_b = self.strategy_default(b)
        a = b
        b = old_b
        tree_b = old_tree_b
      a = b
      tree_a = tree_b
      self.add_tree(tree_a.df)
      b = max_c
      tree_b = tree_f
    logging.info('Finished CTM Scanner')

  def __calc_bic(self, c):
    self.trees_constructed += 1
    return CTM(self.context_tree).execute(c)

  def strategy_default(self, c):
    bic =  self.__calc_bic(c)
    logging.debug('c=%s; \t\tt=%s' % (round(c, 4), bic.to_str()))
    return bic

  def strategy_dynamic(self, c):
    # This strategy avoids computing trees where the current c is between
    # 2 already computed values of c that have produced the same tree
    if self.intervals is None:
      self.intervals = dict()
    t = self.__cached_trees(c)
    if not t:
      t = self.__calc_bic(c)
      self.__add_tree(c, t)
      logging.debug("[new] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
    else:
      logging.debug("[skip] c=%s; \t\tt=%s" % (round(c, 4), t.to_str()))
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
