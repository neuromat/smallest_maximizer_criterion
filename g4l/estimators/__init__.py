from .ctm import CTM
from .smc import SMC
from .prune import Prune
import g4l.tree


class Base():
  def __init__(self, context_tree):
    self.context_tree = context_tree
    self.sample = self.context_tree.sample
    self.data = self.sample.data
    self.A = self.sample.A


class CollectionBase(Base):
  def __init__(self, context_tree, best_tree_selection_strategy=None):
    super().__init__(context_tree)
    self.context_trees = []
    self.best_tree_selection_strategy = best_tree_selection_strategy

  def best_tree(args):
    if self.best_tree_selection_strategy is None:
      return None
    return self.best_tree_selection_strategy(self.context_trees, *args)

  def add_tree(self, df):
    new_tree = g4l.tree.ContextTree(self.context_tree.sample,
      max_depth=self.context_tree.max_depth,
      source_data_frame=df)
    self.context_trees.append(new_tree)
