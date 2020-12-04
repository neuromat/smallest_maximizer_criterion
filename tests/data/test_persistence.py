import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import persistence
from g4l.data import Sample
from g4l.estimators import BIC
from g4l.models import ContextTree
from examples.example2 import models
from g4l.util.mat import MatSamples

A = [0, 1]
samples_folder = os.path.abspath('./examples/example2/samples')
largest_tree = '000000 000010 000100 001000 001010 1 10000 100000 100010 10010 100100 10100 101000 101010'

def xtest_load_samples():
    filename = os.path.abspath('./examples/example2/samples/model1_5000.mat')
    for i in persistence.iterate_from_mat(filename, 'model1_5000', A):
        assert(len(i.data) == 5000)


def test_tree_persistence():
    sample = load_sample(14)
    bic = BIC(0, 6)
    tree = bic.fit(sample).context_tree
    assert(tree.to_str() == largest_tree)
    tree_file = './tests/cache/test_tree.h5'
    tree.save(tree_file)
    tree2 = ContextTree.load_from_file(tree_file)
    assert(tree.to_str() == tree2.to_str())
    assert(tree.df.sum().freq == tree2.df.sum().freq)
    assert(tree.df.sum().active == tree2.df.sum().active)
    assert(tree.df.sum().indicator == tree2.df.sum().indicator)
    assert(tree.df.sum().likelihood == tree2.df.sum().likelihood)
    assert(tree.df.sum().likelihood_pen == tree2.df.sum().likelihood_pen)
    assert(tree.df.sum().chosen == tree2.df.sum().chosen)
    t1_sum = tree.transition_probs.sum()
    t2_sum = tree2.transition_probs.sum()
    assert(t1_sum.idx == t2_sum.idx)
    assert(t1_sum.freq == t2_sum.freq)
    assert(t1_sum.likelihood == t2_sum.likelihood)
    assert(tree.sample.data == tree.sample.data)


def load_sample(idx):
    obj = MatSamples(samples_folder, 'model1', 5000, A)
    return obj.sample_by_idx(idx)
