import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
#import code; code.interact(local=dict(globals(), **locals()))
from examples.simulation_study import get_model

from g4l.util.mat import MatSamples
from g4l.estimators import SMC
from g4l.bootstrap.bootstrap import Bootstrap

largest_tree = '000000 000010 000100 001000 001010 1 10000 100000 100010 10010 100100 10100 101000 101010'
resamples_folder = os.path.abspath('tests/simulation/fixtures/%s_n%s.txt')
samples_folder = os.path.abspath('./examples/simulation_study/samples')
cache_folder = os.path.abspath('tests/cache')
model_name = 'model1'
model = get_model(model_name)
sample_size = 5000
resample_sizes = (1500, 4500)
sample_idx = 14
max_depth = 6
A = [0, 1]
num_cores = 4


@pytest.fixture(scope='function')
def tree_resources(request):
    sample = load_sample()
    samples_n1, samples_n2 = load_bootstrap_samples()
    cachefld = '%s/smc' % cache_folder
    smc = SMC(max_depth, penalty_interval=(0, 1000),
              epsilon=0.00001,
              df_method='csizar_and_talata',
              cache_dir=cachefld)
    champion_trees = smc.fit(sample).context_trees
    return sample, samples_n1, samples_n2, champion_trees


def test_likelihoods(tree_resources):
    sample, samples_n1, samples_n2, champion_trees = tree_resources
    assert(sample.data.index('0001001010101010100010') == 0)
    assert(champion_trees[0].to_str() == largest_tree)
    assert(len(champion_trees) == 13)
    assert(len(open(samples_n1).read().split('\n')[0]) == resample_sizes[0])
    assert(len(open(samples_n2).read().split('\n')[0]) == resample_sizes[1])
    assert(open(samples_n1).read().split('\n')[0].index('1001000101') == 0)
    assert(open(samples_n2).read().split('\n')[0].index('1001000101') == 0)
    bootstrap = Bootstrap(champion_trees, samples_n2, resample_sizes)
    L = bootstrap.calculate_likelihoods(cache_folder, num_cores=num_cores)
    assert(round(sum(sum(L[0])), 2) == -1398697.55)  # values calculated with SeqROCTM
    assert(round(sum(sum(L[1])), 2) == -4217787.59)
    #optimal_tree_idx = find_optimal_tree(L)
    pass


def test_calculate_diffs(tree_resources):
    sample, samples_n1, samples_n2, champion_trees = tree_resources
    bootstrap = Bootstrap(champion_trees, samples_n2, resample_sizes)
    L = np.load('tests/simulation/fixtures/L.npy')
    diffs = bootstrap.calculate_diffs(L)
    assert(round(sum(diffs[0][0]), 5) == 0.16477)  # values calculated with SeqROCTM
    assert(round(sum(diffs[0][1]), 5) == 0.04217)
    assert(round(sum(diffs[1][0]), 5) == 0.04875)
    assert(round(sum(diffs[1][1]), 5) == 0.02095)
    # tree_folder = 'tests/simulation/fixtures/trees/%s'
    # for t in range(len(champion_trees)):
    #     champion_trees[t].save(tree_folder % t)


def test_find_optimal_tree(tree_resources):
    sample, samples_n1, samples_n2, champion_trees = tree_resources
    bootstrap = Bootstrap(champion_trees, samples_n2, resample_sizes)
    #L = np.load('tests/simulation/fixtures/L.npy')
    L = bootstrap.calculate_likelihoods(cache_folder, num_cores=num_cores)
    diffs = bootstrap.calculate_diffs(L)
    opt_idx = bootstrap.find_optimal_tree(diffs)
    #round_res = [round(x, 4) for x in res]
    #expected = [1.0, 0.2525, 0.202, 0.0057, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0015, 0.0]
    #assert(round_res == expected)
    assert(champion_trees[opt_idx].to_str() == '000 1 10 100')
    #assert(champion_trees[opt_idx+1].to_str() == '000 1 10 100')


def load_sample():
    obj = MatSamples(samples_folder, model_name, sample_size, A)
    return obj.sample_by_idx(sample_idx)


def load_bootstrap_samples():
    #fn = lambda f, s, n: open(f % (s, n)).read().split('\n')[:-1]
    fn = lambda filename, sample_idx, n: filename % (sample_idx, n)
    n1 = fn(resamples_folder, sample_idx, 1)
    n2 = fn(resamples_folder, sample_idx, 2)
    return n1, n2

#open(samples_n1).read().split('\n')[0]
