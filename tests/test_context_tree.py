import g4l.context_tree as ct
from tests.fixtures.sample import *


def test_init_from_file(sample):
    t = ct.ContextTree.init_from_sample(sample)
    assert t.num_contexts() == 21
    #import code; code.interact(local=dict(globals(), **locals()))

