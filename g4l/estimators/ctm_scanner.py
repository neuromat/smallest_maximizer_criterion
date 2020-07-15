import numpy as np
import pandas as pd
from g4l.estimators.base import Base
from g4l.estimators.ctm import CTM
from datetime import datetime

class CTMScanner():
  def __init__(self, penalty_interval=(0.1, 400), epsilon=0.01):
    self.penalty_interval=penalty_interval
    self.epsilon = epsilon

  def execute(self, context_tree):
    estimator = CTMScannerEstimator(context_tree)
    return estimator.execute(self.penalty_interval[0],
                             self.penalty_interval[1],
                             self.epsilon)


class CTMScannerEstimator(Base):

  def execute(self, min_c, max_c, epsilon):
    self.intervals = None
    print("Start:", datetime.now())

    tree_a = self.__calc_bic(min_c)
    tree_b = tree_f = self.__calc_bic(max_c)
    champion_trees = []
    champion_trees.append(tree_a)

    a = min_c
    b = max_c
    #print("c = %s\n" % a, tree_a.to_str())
    #print("c = %s\n" % b, tree_b.to_str())

    while not tree_a.equals_to(tree_b):
      while b - a > epsilon:
        while not tree_a.equals_to(tree_b):
          old_b = b
          old_tree_b = tree_b
          b = (a + b)/2
          tree_b = self.strategy1(b)
          #print("c = %s\n" % b, tree_b.to_str())
        a = b
        b = old_b
        tree_b = old_tree_b
      a = b
      tree_a = tree_b
      champion_trees.append(tree_a)
      #print("-> c = %s\n" % b, tree_a.to_str())

      b = max_c
      tree_b = tree_f
    print("End:", datetime.now())
    return champion_trees

  def __calc_bic(self, c):
    return CTM(self.context_tree).execute(c)



  def strategy1(self, c):
    bic =  self.__calc_bic(c)
    print("** c=", round(c, 4), '\t\t==>', bic.to_str())
    return bic

  #### Verificar se cabe utilizar estratégia com caching

  def strategy2(self, c):
    if self.intervals is None:
      self.intervals = dict()
    t = self.__cached_trees(c)
    if not t:
      t = self.__calc_bic(c)
      self.__add_tree(c, t)
      #print(self.intervals.keys(), c)
      print("** c=", c, '\t\t==>', t.to_str())
    else:
      print("++ c=", c, '\t\t==>', t.to_str())
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
