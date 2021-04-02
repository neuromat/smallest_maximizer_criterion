from g4l.bic import BIC
from tests.fixtures.sample import *
from g4l.util import integrity


def test_tree_estimation(sample):
    tree_0 = '000000 000010 000100 001000 001010 1 10000 100000 100010 10010 100100 10100 101000 101010'
    tree_0_08 = '000 1 10 100'
    tree_2000 = ''

    assert BIC(0.08, keep_data=True).fit(sample).context_tree.to_str() == tree_0_08
    assert BIC(0.0, keep_data=True).fit(sample).context_tree.to_str() == tree_0
    assert BIC(2000, keep_data=True).fit(sample).context_tree.to_str() == tree_2000

    for c in [0.08]:
        tree = BIC(c, keep_data=True).fit(sample).context_tree
        integrity.is_freq_consistent(tree)
        integrity.satisfies_irreductibility(tree)
        integrity.satisfies_completeness(tree, sample)
        integrity.check_admissibility(tree, sample)
        #import code; code.interact(local=dict(globals(), **locals()))

