import g4l.context_tree as ct
from g4l.bic import BIC
from tests.fixtures.sample import *


def test_init_from_file(sample):
    t = ct.ContextTree.init_from_sample(sample)
    assert t.num_contexts() == 21
    #import code; code.interact(local=dict(globals(), **locals()))


def test_sample_generation(sample):
    tree = BIC(0.08).fit(sample).context_tree
    tree.generate_sample(1000, sample.A)
    #import code; code.interact(local=dict(globals(), **locals()))

