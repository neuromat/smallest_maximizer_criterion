from . import estimators
from . import data
from . import tree
from . import evaluation
from . import util
from . import display
import hashlib

class SmallestMaximizerCriterion():
  champion_trees = []
  best_tree = None
  initial_tree = None
  X = None
  max_depth = None

  def __init__(self, tree_generation_method, max_depth=4):
    self.max_depth = max_depth
    self.tree_generation_method = tree_generation_method

  def fit(self, X, best_tree_selection_method, dump_folder=None, processors=None, read_cache_dir=None, write_cache_dir=None):
    self.X = X
    self.best_tree_selection_method = best_tree_selection_method
    if read_cache_dir is None:
      self.initial_tree = tree.ContextTree(X, max_depth=self.max_depth)
      self.champion_trees = self.tree_generation_method.execute(self.initial_tree)
    else:
      read_cache()

    self.best_tree = self.best_tree_selection_method.evaluate(self.champion_trees)

    if write_cache_dir is not None:
      self.save(write_cache_dir)
    #bootstrap_dataframe = self.bootstrap(self.sample, partition_string)
    #t_result = self.t_test(bootstrap_dataframe)
    #tree_idx = int(list(t_result[t_result.pvalue < alpha].tree_idx)[0])
    #self.best_tree = champion_trees[tree_idx]
    return self

  def score(self, data):
    pass

  def generate_champion_trees(self, min_c, max_c, epsilon):
    smc = g4l.estimators.SMC(self.initial_tree)
    self.champion_trees = smc.execute(min_c, max_c, epsilon)

  def save(self, dump_folder):
    # TODO: add sample_file to model_folder
    for i, ch in enumerate(self.champion_trees):
      ch.save(dump_folder, 'champion.%s' % i)
    print('saved!')

  def read_cache(self, dump_folder):
    filename = "%s.h5" % hashlib.md5(self.X.filename).digest()
    self.__read_trees_cache()
    best_tree = None
    initial_tree = None
    X = None
    max_depth = None

  def __read_trees_cache():
    self.champion_trees = self.__read_trees_cache()
    self.best_tree

    # TODO: remove sample_file, it should be added when saving
    # TODO: load all configs
    smc = SmallestMaximizerCriterion(self.tree_generation_method, max_depth=self.max_depth)
    smc.champion_trees = []
    amnt_trees = len(glob.glob(model_folder + '/champion*.df.pkl'))
    for i in range(amnt_trees):
        ch = g4l.tree.ContextTree.load(dump_folder, 'champion.%s' % i)
        smc.champion_trees.append(ch)
    return smc
