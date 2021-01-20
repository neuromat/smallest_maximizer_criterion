import os
from scipy import io
import pandas as pd
from collections import Counter
import logging
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
#from examples.example2 import simulation

results_folder = os.path.abspath('./examples/simulation_study/results')
trees_folder = os.path.abspath('./fixtures/champion_trees/2')


def test_compare_trees():
    sample_sizes = [5000]
    #models = ['model1', 'model2']
    models = ['model1']
    for sample_size in sample_sizes:
        for model_name in models:
            print("Executing %s - %s" % (sample_size, model_name))
            result_file = '/SeqROCTM/%s_%s.csv' % (model_name, sample_size)
            #if not os.path.exists(results_folder + result_file):
            #create_champion_trees_file(sample_size, model_name)
    for sample_size in sample_sizes:
        for model_name in models:
            algs = ['smc', 'SeqROCTM']
            f1, f2 = ['%s/%s/%s_%s.csv' % (results_folder, alg, model_name, sample_size) for alg in algs]
            #import code; code.interact(local=dict(globals(), **locals()))
            df1 = pd.read_csv(f1).sort_values(['sample_idx', 'num_contexts'])
            df2 = pd.read_csv(f2)
            df2['num_contexts'] = df2.tree.apply(lambda x: len(x.split()))
            df2 = df2.sort_values(['sample_idx', 'num_contexts'])
            df2['tree'] = df2.tree.apply(lambda x: ' '.join(sorted(x.split())))
            df1['tree'] = df1.tree.apply(lambda x: ' '.join(sorted(x.split())))
            for sample_idx in range(100):
                ok = True
                for tree in df2[df2.sample_idx == sample_idx].tree.values:
                    try:
                        df1[(df1.sample_idx == sample_idx) & (df1.tree == tree)].values[0]
                    except IndexError:
                        ok = False
                        print("===============================")
                        print("sample idx ", sample_idx, "; missing tree: ", tree)
                        print("===============================")
                        print("smc:")
                        print('\n'.join(df1[df1.sample_idx == sample_idx].tree.values))
                        print("seq:")
                        print('\n'.join(df2[df2.sample_idx == sample_idx].tree.values))
                        print("===============================")

                if ok:
                    print('sample ', sample_idx, ' -> OK')



    samples_path = os.path.abspath('./examples/simulation_study/samples')
    mat_file = '%s/model1_5000.mat' % (samples_path)
    f = io.loadmat(mat_file)
    sample = ''.join(f['model1_5000'][98].astype(str))

    import code; code.interact(local=dict(globals(), **locals()))

    print('ok')


def create_champion_trees_file(sample_size, model_name):
    mat_file = '%s/%s_%s.mat' % (trees_folder, model_name, sample_size)
    model_abbrev = {'model1': 'M1', 'model2': 'M2'}
    f = io.loadmat(mat_file)
    df = pd.DataFrame(columns=['sample_idx', 'tree_idx', 'tree'])
    for sample_idx, sample in enumerate(f['champion_%s_%s_bic' % (model_abbrev[model_name], sample_size)]):
        for tree_idx, context_tree in enumerate(sample[0][0, :]):
            for cell_id, context_cell in enumerate(context_tree):
                for context_arr_id, context_arr in enumerate(context_cell):
                    #print('context_arr_id:', context_arr_id)
                    for context_arr_id2, context_arr2 in enumerate(context_arr):
                        #print('context_arr_id2:', context_arr_id2)
                        ss = ''.join([str(s) for s in list(context_arr2)])
                        df.loc[len(df)] = [sample_idx, tree_idx, ss]
    df = df.sort_values(['sample_idx', 'tree_idx', 'tree'], ascending=True)
    rr = df.groupby(['sample_idx', 'tree_idx']).tree.apply(' '.join).reset_index()
    #result_file = '/mat_compare_%s_%s.csv' % (model_name, sample_size)
    result_file = '/SeqROCTM/%s_%s.csv' % (model_name, sample_size)
    rr.to_csv(results_folder + result_file, index=False)


def xtest_compare_results2():
    df = pd.read_csv(results_folder + '/mat_compare.csv')
    df['num_contexts'] = df.context.apply(lambda x: len(x.split(' ')))
    df = df.sort_values(['sample_idx', 'num_contexts'])
    df = df[['sample_idx', 'num_contexts', 'tree']].set_index(['sample_idx', 'num_contexts'])

    df2 = pd.read_csv(results_folder + '/model1.smc.tmp.csv')
    df2 = df2[df2.sample_size==5000]
    df2 = df2[['sample_idx', 'num_contexts', 'tree', 'opt']].set_index(['sample_idx', 'num_contexts'])
    ct = Counter(df2[df2.opt==1].reset_index().num_contexts.values)
    ct[4]/len(df2[df2.opt==1])


    #  import scipy.io as sio
    #  from g4l.models import ContextTree
    #  from g4l.data import Sample
    #  from g4l.estimators import CTM
    #
    #  filename = './examples/example2/samples/model1_5000.mat'
    #  arr = sio.loadmat(filename)['model1_5000']
    #  dt = ''.join([str(x) for x in arr[0]])
    #  sample = Sample(None, [0, 1], data=dt)
    #  ctm = CTM(0, 6)
    #  ctm.fit(sample)
    #  print(ctm.context_tree.to_str())



    df3 = pd.read_csv(results_folder + '/model1.csv')
    df3 = df3[df3.sample_size==5000]
    df3 = df3[['sample_idx', 'num_contexts', 'tree', 'opt']].set_index(['sample_idx', 'num_contexts'])

    import code; code.interact(local=dict(globals(), **locals()))



