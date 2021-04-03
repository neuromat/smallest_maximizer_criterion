from g4l.bic import BIC
from tests.fixtures.sample import *
from g4l.util import integrity


def test_tree_estimation(sample):
    t1 = '000000 000010 000100 001000 001010 1 10000 100000 100010 10010 100100 10100 101000 101010'
    t2 = ''
    assert BIC(0.0, keep_data=True).fit(sample).context_tree.to_str() == t1
    assert BIC(2000, keep_data=True).fit(sample).context_tree.to_str() == t2


def test_tree_properties(sample):
    c = 0.08
    tree = BIC(c, keep_data=True).fit(sample).context_tree
    assert tree.to_str() == '000 1 10 100'
    integrity.is_freq_consistent(tree)
    integrity.satisfies_irreductibility(tree)
    integrity.satisfies_completeness(tree, sample)
    integrity.check_admissibility(tree, sample)


def test_perl_compatibility(sample_pl_compat):
    bic = BIC(2000, keep_data=False, perl_compatible=True)
    bic.fit(sample_pl_compat)
    # perl version doesn't have the empty node
    assert bic.context_tree.to_str() == '0 1'


def test_df(sample):
    from g4l.context_tree import ContextTree
    from g4l.util.degrees_of_freedom import degrees_of_freedom as df
    tree = ContextTree.init_from_sample(sample)
    assert len(df('g4l', tree)) == 53  # nodes have different dfs
    assert df('perl', tree) == -1
    assert df('ct06', tree) == -0.5


def test_sample_bp(sample_bp):
    bic = BIC(0.2982733078507897).fit(sample_bp)
    t = '0000 2000 100 200 300 0010 2010 210 20 30 001 201 21 2 3 4'
    assert bic.context_tree.to_str(reverse=True) == t


def test_sample_ep(sample_ep):
    bic = BIC(0.2654890885033804).fit(sample_ep)
    t = '0000 2000 100 200 300 0010 2010 210 20 30 001 201 21 2 3 4'
    assert bic.context_tree.to_str(reverse=True) == t


def test_lipsum(sample_lipsum):
    bic = BIC(0).fit(sample_lipsum)
    s = bic.context_tree.generate_sample(400, sample_lipsum.A)
    print(s)
    #assert bic.context_tree.to_str() == ''


def test_sample_bp_empty_tree(sample_bp):
    bic = BIC(200).fit(sample_bp)
    assert bic.context_tree.to_str() == ''


def test_sample_ep_empty_tree(sample_ep):
    bic = BIC(200).fit(sample_ep)
    assert bic.context_tree.to_str() == ''

