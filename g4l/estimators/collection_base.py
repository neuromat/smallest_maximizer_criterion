from g4l.estimators.base import Base
from g4l.tree import ContextTree

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
    new_tree = ContextTree(self.context_tree.sample, max_depth=self.context_tree.max_depth, source_data_frame=df)
    self.context_trees.append(new_tree)
