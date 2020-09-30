import os
from scipy import io
import pandas as pd
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
from examples.example2 import simulation

results_folder = os.path.abspath('./examples/example2/results')

def test_compare_results():
  return
  s = '/home/arthur/Documents/Neuromat/projects/SMC/arquivo/data/redadosparacomparao/champion_M1_5000.mat'
  f = io.loadmat(s)
  df = pd.DataFrame(columns=['sample_idx', 'tree_idx', 'context'])
  for sample_idx, sample in enumerate(f['champion_M1_5000']):
    print('sample')
    for tree_idx, context_tree in enumerate(sample[0][0,:]):
      print('tree:', tree_idx)
      for cell_id, context_cell in enumerate(context_tree):
        print('cell:', cell_id)
        for context_arr_id, context_arr in enumerate(context_cell):
          #print('context_arr_id:', context_arr_id)
          for context_arr_id2, context_arr2 in enumerate(context_arr):
            #print('context_arr_id2:', context_arr_id2)
            ss = ''.join([str(s) for s in list(context_arr2)])
            df.loc[len(df)] = [sample_idx, tree_idx, ss]
  df = df.sort_values(['sample_idx', 'tree_idx', 'context'], ascending=True)
  r = df.groupby(['sample_idx', 'tree_idx']).context.apply(' '.join).reset_index()
  r.to_csv(results_folder + '/mat_compare.csv', index=False)
        #print(''.join(context))

def test_compare_results2():
  import code; code.interact(local=dict(globals(), **locals()))
  df = pd.read_csv(results_folder + '/mat_compare.csv')
  df['num_contexts'] = df.context.apply(lambda x: len(x.split(' ')))
  df = df.sort_values(['sample_idx', 'num_contexts'])
  df = df[['sample_idx', 'num_contexts', 'context']].set_index(['sample_idx', 'num_contexts'])

  df2 = pd.read_csv(results_folder + '/model1.csv')
  df2 = df2[['sample_idx', 'num_contexts', 'tree']].set_index(['sample_idx', 'num_contexts'])



