import numpy as np
import pandas as pd
from scipy import stats
import g4l

# workaround to temporarily persist generated trees
def load_trees(dump_fld):
  bic_tree = g4l.tree.ContextTree.load(dump_fld, 'bic_tree')
  champion_trees = []
  for i in range(7):
    ch = g4l.tree.ContextTree.load(dump_fld, 'champion.%s' % i)
    champion_trees.append(ch)
  return (bic_tree, champion_trees)

def add_row(df, resample_idx, tree_idx, t, lml_small, lml):
  # lml = log maximum likelihood
  tree_str = t.to_str()
  df.loc[len(df)] = [resample_idx,
                     tree_idx,
                     t.chosen_penalty,
                     lml_small,
                     len(tree_str.split(' ')),
                     tree_str,
                     lml]

fld = '/home/arthur/Documents/Neuromat/code/SMC/euclid.aoas.1331043393/supplementary/SCRIPTS/python/g4l/fixtures/dump'
bic_tree, champion_trees = load_trees(fld)
print('BIC Tree:', bic_tree.to_str())
print('# SMC Champion Trees:', len(champion_trees))
print('\n\n\n')



# TODO: transform partition_string into a parameter
# - use most_frequent_node when parameter value is None
# ponto de renovação - renewal context
partition_string = '4' # para o exemplo linguistico
#partition_string = get_most_frequent_node(sample)
amnt_new_samples = 10
sample_ratio_1 = 0.1
sample_ratio_2 = 0.3
A = [0, 1, 2, 3, 4]
source_sample = g4l.data.Sample('./g4l/fixtures/folha.txt', A)
len1 = int(len(source_sample.data) * sample_ratio_1)
len2 = int(len(source_sample.data) * sample_ratio_2)

df = pd.DataFrame(columns=['sample_idx', 'tree_idx', 'c', 'sample_size', 'num_contexts', 'tree', 'lml'])
resample_iterator = g4l.evaluation.resample(source_sample, amnt_new_samples, len1, partition_string)
for resample_idx, sample_small in enumerate(resample_iterator):
  for tree_idx, t in enumerate(champion_trees):
    add_row(df, resample_idx, tree_idx, t, 'sm', g4l.evaluation.LML(sample_small, t))

resample_iterator = g4l.evaluation.resample(source_sample, amnt_new_samples, len2, partition_string)
for resample_idx, sample_large in enumerate(resample_iterator):
  for tree_idx, t in enumerate(champion_trees):
    print('---', resample_idx, tree_idx)
    add_row(df, resample_idx, tree_idx, t, 'lg', g4l.evaluation.LML(sample_large, t))

def compute_tree_deltas(df, sample_size, sample_length):
  df = df.copy()
  for resample_idx in range(len(df.sample_idx.unique())):
    condition = (df.sample_size==sample_size) & (df.sample_idx==resample_idx)
    delta_lml = list(df.loc[condition].lml.rolling(window=2).sum()/(sample_length**0.9))
    df.loc[condition, 'delta_lml'] = (delta_lml[1:] + [float('NaN')])
    df.loc[(condition) & (df.delta_lml.notnull())]

  df2 = pd.DataFrame(columns=['sample_size', 'tree1_idx', 'tree2_idx', 'dlml_mean'])
  for tree_idx in df[df.delta_lml.notnull()].tree_idx.unique():
    condition = (df.sample_size==sample_size) & (df.tree_idx == tree_idx)
    dlml_mean = df.loc[condition].delta_lml.mean()
    df2.loc[len(df2)] = [sample_size, tree_idx, tree_idx+1, dlml_mean]
  return df2

def calc_delta(df, sample_size, sample_length):
  df.sort_values(['sample_size', 'sample_idx', 'c'], inplace=True)
  for resample_idx in range(len(df.sample_idx.unique())):
    condition = (df.sample_size==sample_size) & (df.sample_idx==resample_idx)
    df.loc[condition, 'lml_next'] = (list(df.loc[condition].lml)[1:] + [float('NaN')])
    df.loc[condition, 'lml_delta'] = (df.loc[condition].lml - df.loc[condition].lml_next)
    df.loc[condition, 'lml_delta_div_n'] = df.loc[condition].lml_delta / sample_length**0.9

def calc_ttest(df):
  df2 = pd.DataFrame(columns=['tree_idx', 'tvalue','pvalue', 'sd_sm', 'sd_lg'])
  for tree_idx in df[df.lml_delta_div_n.notnull()].tree_idx.unique():
    delta_lml_sm = df.loc[(df.sample_size=='sm') & (df.tree_idx == tree_idx)].lml_delta_div_n
    delta_lml_lg = df.loc[(df.sample_size=='lg') & (df.tree_idx == tree_idx)].lml_delta_div_n
    ttest = stats.ttest_ind(delta_lml_sm, delta_lml_lg)
    df2.loc[len(df2)] = [tree_idx, ttest.statistic, ttest.pvalue, delta_lml_sm.std(), delta_lml_lg.std()]
  return df2

calc_delta(df, 'sm', len1)
calc_delta(df, 'lg', len2)
x = calc_ttest(df)
alpha = 0.01
tree_idx = int(list(x[x.pvalue < alpha].tree_idx)[0])
champion_trees[tree_idx]


#deltas_sm = compute_tree_deltas(df, 'sm', len1)
#deltas_lg = compute_tree_deltas(df, 'lg', len2)
#stats.ttest_ind(deltas_sm.dlml_mean, deltas_lg.dlml_mean)

import code; code.interact(local=dict(globals(), **locals()))

