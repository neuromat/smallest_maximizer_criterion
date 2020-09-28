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
import logging

#from g4l.data import Sample
#import pandas as pd
#import time


A = ['0', '1']
PATH = os.path.abspath('./examples/example2/samples')
RESAMPLES_FOLDER = os.path.abspath('./examples/example2/tmp/resamples')
RESULTS_FOLDER = os.path.abspath('./examples/example2/results')
LIKELIHOODS_FILE = '%s/likelihoods.csv' % RESULTS_FOLDER
#SAMPLE_SIZES = [5000, 10000, 20000]
SAMPLE_SIZES = [5000]
#NUM_RESAMPLES = 100
NUM_RESAMPLES = 3
RENEWAL_POINT = 1
N1_FACTOR = 0.3
N2_FACTOR = 0.9
max_depth = 3
c = 2

def run_simulation(model_name):
  logging.info("Running simulation with %s" % model_name)
  print("aaa")
  model = models.get_model(model_name)
  #if os.path.exists(LIKELIHOODS_FILE):
    #os.remove(LIKELIHOODS_FILE)
    #pass

  for sample_size in SAMPLE_SIZES:
    for sample in get_samples(sample_size):
      resample_factory = BlockResampling(sample, renewal_point=RENEWAL_POINT)
      bootstrap = Bootstrap(sample,
                            resample_factory,
                            RESAMPLES_FOLDER,
                            NUM_RESAMPLES,
                            resample_sizes=(100, 500),
                            alpha=0.01)
      bootstrap.find_optimal_tree(smc(sample))



  import code; code.interact(local=dict(globals(), **locals()))


  df2 = calculate_deltas()
  df_results = None
  fields_to_remove = ['delta_l_norm', 'delta_l', 'resample_size', 'resample_idx']
  print("applying t-test")
  for key, idxs in df2.groupby(['sample_idx', 'tree_idx']).groups.items():
    rows = df2.loc[idxs]
    deltas_n1 = rows[rows.resample_n=='n1'].delta_l_norm
    deltas_n2 = rows[rows.resample_n=='n2'].delta_l_norm
    t, pvalue = stats.ttest_ind(deltas_n1, deltas_n2)
    rs = rows[['resample_n', 'resample_idx']]
    rs = rows.drop(fields_to_remove, axis=1)[:1]
    rs['pvalue'] = pvalue
    if df_results is None:
      df_results = rs
    else:
      df_results = df_results.append(rs)



def calculate_deltas():
  df2 = None
  for resample_size in RESAMPLE_SIZES:
    #estimate_optimal_trees(resample_size, model_name, smc)
    #pass
    # TODO: measure time spent
    # estimate_optimal_trees(resample_size, model_name, bic)
    # prn_trees = estimate_optimal_tree(resample_size, 'model', prune)
    #import code; code.interact(local=dict(globals(), **locals()))

    df = pd.read_csv(LIKELIHOODS_FILE)
    df = df[df.estimation_method=='smc']
    group_fields = ['sample_size', 'model_name', 'estimation_method', 'resample_n', 'sample_idx', 'resample_idx']
    grouped_df = df.groupby(group_fields)

    for keys, idxs in grouped_df.groups.items():
      rows = df.loc[idxs]
      resample_n = rows.resample_n.values[0]
      sample_size = rows.sample_size.max()
      n1 = math.floor(N1_FACTOR * sample_size)
      n2 = math.floor(N2_FACTOR * sample_size)
      resample_size = {'n1': n1, 'n2': n2}[resample_n]
      rows['resample_size'] = resample_size
      rows['tree_idx_2'] = rows.tree_idx.shift(-1)
      rows['tree_2'] = rows.tree.shift(-1)
      rows['delta_l'] = rows.likelihood - rows.likelihood.shift(-1)
      rows['delta_l_norm'] = rows.delta_l / resample_size ** 0.9
      r = rows.drop(['num_contexts', 'likelihood'], axis=1)[~rows.delta_l.isnull()]
      r['tree_idx_2'] = r.tree_idx_2.astype(int)
      if df2 is None:
        df2 = r
      else:
        df2 = df2.append(r)
  return df2
  # delta_means = df2.groupby(['sample_idx', 'resample_n', 'tree_idx']).delta_l_norm.mean()
  # df3 = df2[df2.resample_idx==0].set_index(['sample_idx', 'resample_n', 'tree_idx'])
  # df3['delta_mean'] = delta_means
  # fields_to_remove = ['delta_l_norm', 'delta_l', 'resample_size', 'resample_idx']
  # return df3.reset_index(drop=False).drop(fields_to_remove, axis=1)

def estimate_optimal_trees(sample_size, model_name, estimation_method):
  df = pd.DataFrame(columns=[ 'sample_size', 'model_name',
                              'estimation_method', 'tree_idx',
                              'num_contexts', 'tree',
                              'likelihood', 'resample_sz'])
  resampling_strategy = BlockResampling(renewal_point=RENEWAL_POINT)
  for sample_idx, sample  in fetch_samples(model_name, sample_size):
    resamples_n1, resamples_n2 = generate_resamples(sample,
                                                    resampling_strategy,
                                                    model_name,
                                                    sample_size,
                                                    sample_idx)
    champion_trees = sorted(estimation_method(sample),
                            key=lambda x: x.num_contexts())


    for res_name, resample_set in [('n1', resamples_n1), ('n2', resamples_n2)]:
      kwargs = { 'sample_size': sample_size,
              'model_name': model_name,
              'estimation_method': estimation_method.__name__,
              'resample_n': res_name,
              'sample_idx': sample_idx}
      calculate_sample_likelihoods(champion_trees, resample_set, sample, **kwargs)

def calculate_sample_likelihoods(champion_trees, resamples, sample, **kwargs):
  for tree_idx, tree in enumerate(champion_trees):
    for resample_idx, resample_data in enumerate(resamples):
      kwargs['tree_idx'] = tree_idx
      kwargs['tree'] = tree.to_str()
      kwargs['num_contexts'] = tree.num_contexts()
      kwargs['resample_idx'] = resample_idx
      resample = Sample(None, sample.A, data=resample_data)
      kwargs['likelihood'], _ = tree.sample_likelihood(resample)
      file_exists = os.path.exists(LIKELIHOODS_FILE)
      df = pd.DataFrame([kwargs])
      df.to_csv(LIKELIHOODS_FILE, mode='a',
                                  header=(not file_exists),
                                  index=False)


def generate_resamples(sample, resampling_strategy, model_name, sample_size, resample_idx):
    n1 = math.floor(N1_FACTOR * sample_size)
    n2 = math.floor(N2_FACTOR * sample_size)

    file_str = '%s/%s_%s_%s'  % (RESAMPLES_FOLDER, model_name, sample_size, resample_idx)
    file_n2 = '%s.txt' % (file_str)
    os.remove(file_n2) if os.path.exists(file_n2) else None
    print("Resampling...", model_name, sample_size, resample_idx)
    for i in range(NUM_RESAMPLES):
      resampling_strategy.generate(sample, n2, file_n2)
    print("creating n1")
    resamples_n2 = open(file_n2).read().split('\n')[:-1]
    resamples_n1 = [x[:n1] for x in resamples_n2]
    return resamples_n1, resamples_n2
    #open(file_n2).read

def bic(sample):
  return sort_trees([CTM(c, max_depth).fit(sample).context_tree])

def smc(sample):
  return sort_trees(SMC(max_depth).fit(sample).context_trees)

def prune(sample):
  return sort_trees(Prune(max_depth).fit(sample).context_trees)

def sort_trees(context_trees):
  return sorted(context_trees, key=lambda x: x.num_contexts())

def fetch_samples(model_name, resample_size):
  i = -1
  key = '%s_%s' % (model_name, resample_size)
  filename = '%s/%s.mat' % (PATH, key)
  for s in persistence.iterate_from_mat(filename, key, A):
    i += 1
    yield i, s

