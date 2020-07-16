from . import estimators
from . import data
from . import tree
from . import evaluation
from . import util
from . import display
import hashlib
import logging
import h5py
import numpy as np
import pandas as pd
import pathlib

class SmallestMaximizerCriterion():
  champion_trees = []
  best_tree_idx = -1
  initial_tree = None
  X = None
  max_depth = None
  evaluation_results = None

  def __init__(self, tree_generation_method, max_depth=4, read_cache_dir=None, write_cache_dir=None):
    logging.debug("Initializing SMC")
    self.max_depth = max_depth
    self.tree_generation_method = tree_generation_method
    self.read_cache_dir = read_cache_dir
    self.write_cache_dir = write_cache_dir

  def fit(self, X, best_tree_selection_method, dump_folder=None, processors=None):
    self.X = X
    self.best_tree_selection_method = best_tree_selection_method
    if self.read_cache_dir is None:
      self.__create_champion_trees(X)
    else:
      try:
        self.read_cache(self.read_cache_dir)
      except OSError:
        logging.warning("Cache file not found - skipping...")
        self.__create_champion_trees(X)

    if self.best_tree_idx < 0:
      self.perform_evaluation()
    return self

  def perform_evaluation(self):
    self.best_tree_idx = -1
    self.evaluation_results = None
    evaluation_obj = self.best_tree_selection_method
    evaluation_obj.evaluate(self.champion_trees)
    self.best_tree_idx = evaluation_obj.best_tree_idx
    self.evaluation_results = evaluation_obj.results
    if self.write_cache_dir is not None:
      self.save(self.write_cache_dir)


  def best_tree(self):
    if self.best_tree_idx < 0:
      return None
    return self.champion_trees[self.best_tree_idx]

  def score(self, data):
    pass

  def generate_champion_trees(self, min_c, max_c, epsilon):
    smc = g4l.estimators.SMC(self.initial_tree)
    self.champion_trees = smc.execute(min_c, max_c, epsilon)

  def save(self, cache_folder_path):
    cache_hash = self.cache_filename()
    cache_folder = "%s/%s" % (cache_folder_path, cache_hash)
    logging.debug("Writing cache to %s" % cache_folder)
    pathlib.Path(cache_folder).mkdir(parents=True, exist_ok=True)
    file = '%s/smc.h5' % cache_folder
    try:
      hf = h5py.File(file, 'w')
    except:
      import code; code.interact(local=dict(globals(), **locals()))
    hf.attrs['champion_trees_len'] = len(self.champion_trees)
    hf.attrs['best_tree_idx'] = self.best_tree_idx
    hf.attrs['max_depth'] = self.max_depth
    if self.best_tree_idx >= 0:
      hf.attrs['evaluation_results_columns'] = ';'.join(self.evaluation_results.columns)
      hf['evaluation_results'] = self.evaluation_results
    hf.close()
    self.initial_tree.save(cache_folder, 'init_tree')
    for i, ch in enumerate(self.champion_trees):
      ch.save(cache_folder, 'champion_tree_%s' % i)

  def read_cache(self, cache_folder_path):
    logging.debug("Reading cache from %s" % cache_folder_path)
    cache_hash = self.cache_filename()
    cache_folder = "%s/%s" % (cache_folder_path, cache_hash)
    file = '%s/smc.h5' % cache_folder
    hf = h5py.File(file, 'r')
    self.best_tree_idx = hf.attrs['best_tree_idx']
    self.max_depth     = hf.attrs['max_depth']
    champion_trees_len = hf.attrs['champion_trees_len']
    if self.best_tree_idx >= 0:
      ev_columns = hf.attrs['evaluation_results_columns'].split(';')
      self.evaluation_results = pd.DataFrame(np.array(hf['evaluation_results']), columns=ev_columns)
    self.initial_tree = tree.ContextTree.load(cache_folder, 'init_tree')
    self.champion_trees = []
    for i in range(champion_trees_len):
      ch = tree.ContextTree.load(cache_folder, 'champion_tree_%s' % i)
      self.champion_trees.append(ch)
    hf.close()

  def cache_filename(self):
    return hashlib.md5(self.X.filename.encode('utf-8')).hexdigest()

  def __create_champion_trees(self, X):
    self.initial_tree = tree.ContextTree(X, max_depth=self.max_depth)
    self.champion_trees = self.tree_generation_method.execute(self.initial_tree)
    if self.write_cache_dir is not None:
      self.save(self.write_cache_dir)
