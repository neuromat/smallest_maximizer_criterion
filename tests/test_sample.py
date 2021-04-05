from tests.fixtures.sample import *


def test_sample_generation(sample, filename, cache_file):
    assert sample.data == open(filename, 'r').read()[:-1]
    assert sample.A == ['0', '1']


def test_cached_sample(cached_sample, filename, cache_file):
    assert cached_sample.cache_file.endswith(cache_file)


def test_counts_consistency(sample):
    first_depth_nodes = sample.F.loc[['0', '1']]
    readed_sample_len = len(sample.data) - sample.max_depth
    assert first_depth_nodes.N.sum() == readed_sample_len
    assert first_depth_nodes[[0, 1]].sum().sum() == first_depth_nodes.N.sum()
    #import code; code.interact(local=dict(globals(), **locals()))


