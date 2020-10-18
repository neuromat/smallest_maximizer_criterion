import sys
import math
import os
import pandas as pd
import numpy as np
from . import models
from g4l.estimators import CTM
from g4l.estimators import SMC
from g4l.estimators import Prune
from g4l.data import persistence
from g4l.data import Sample
from g4l.bootstrap import Bootstrap
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap.resampling import TreeSourceResampling
import logging

#from g4l.data import Sample
#import pandas as pd
#import time


A = ['0', '1']
PATH = os.path.abspath('./examples/example2/samples')
RESAMPLES_FOLDER = os.path.abspath('./examples/example2/tmp/resamples')
RESULTS_FOLDER = os.path.abspath('./examples/example2/results')
SAMPLE_SIZES = [20000, 5000, 10000]
NUM_RESAMPLES = 7
RENEWAL_POINT = 1
N1_FACTOR = 0.3
N2_FACTOR = 0.9
MAX_SAMPLES = math.inf
max_depth = 6

def run_simulation(model_name):
  results_file = "%s/%s.csv" % (RESULTS_FOLDER, model_name)
  if os.path.exists(results_file):
    os.remove(results_file)

  logging.info("Running simulation with %s" % model_name)
  model = models.get_model(model_name)

  for sample_size in SAMPLE_SIZES:
    for sample_idx, sample in fetch_samples(model_name, sample_size, MAX_SAMPLES):
      print('sample:', sample_size, sample_idx)
      resample_factory = BlockResampling(sample, RENEWAL_POINT)
      folder_vars = (RESAMPLES_FOLDER, model_name, sample_size, sample_idx)
      bootstrap = Bootstrap(resample_factory,
                            '%s/%s_%s_%s' % folder_vars,
                            NUM_RESAMPLES,
                            resample_sizes=resample_sizes(sample_size),
                            alpha=0.01)
      print("estimating champion trees")
      champion_trees = prune(sample)

      print("finding optimal trees")
      opt_idx = bootstrap.find_optimal_tree(champion_trees)
      for tree_idx, champion_tree in enumerate(champion_trees):
        opt = int(tree_idx==opt_idx)
        obj = { 'model_name': model_name,
                'sample_size': sample_size,
                'sample_idx': sample_idx,
                'method': 'prune',
                'tree_idx': tree_idx,
                'tree': champion_tree.to_str(),
                'num_contexts': champion_tree.num_contexts(),
                'opt': opt }
        use_header = (not os.path.exists(results_file))
        df = pd.DataFrame.from_dict([obj])
        df.to_csv(results_file, mode='a', index=False, header=use_header)

def resample_sizes(sample_size):
  return tuple(math.floor(f * sample_size) for f in [N1_FACTOR, N2_FACTOR])

def bic(sample):
  pass
  #return sort_trees([CTM(c, max_depth).fit(sample).context_tree])

def smc(sample):
  #import code; code.interact(local=dict(globals(), **locals()))
  return sort_trees(SMC(max_depth, penalty_interval=(0, 500), epsilon=0.00001).fit(sample).context_trees)

def prune(sample):
  return sort_trees(Prune(max_depth).fit(sample).context_trees)

def sort_trees(context_trees):
  return sorted(context_trees, key=lambda x: -x.num_contexts())

def fetch_samples(model_name, sample_size, max_samples=math.inf):
  i = -1
  key = '%s_%s' % (model_name, sample_size)
  filename = '%s/%s.mat' % (PATH, key)
  for s in persistence.iterate_from_mat(filename, key, A):
    if i > max_samples:
      break
    i += 1
    yield i, s
