from abc import ABCMeta, abstractmethod
import g4l.models

class Base():
  __metaclass__ = ABCMeta

  @abstractmethod
  def fit(X):
    ''' To override '''
    pass


class CollectionBase(Base):
  def __init__(self):
    self.context_trees = []

  def add_tree(self, new_tree):
    print("appended", new_tree)
    self.context_trees.append(new_tree)
