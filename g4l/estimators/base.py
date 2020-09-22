from abc import ABCMeta, abstractmethod
import g4l.models

class Base():
  __metaclass__ = ABCMeta

  @abstractmethod
  def fit(X):
    ''' To override '''
    pass


class CollectionBase(Base):
  def __init__(self, selection_criteria=None):
    self.context_trees = []
    self.selection_criteria = selection_criteria

  def optimal_tree(self, args):
    if self.selection_criteria is None:
      return None
    return self.selection_criteria(self.context_trees, *args)

  def add_tree(self, new_tree):
    print("appended", new_tree)
    self.context_trees.append(new_tree)
