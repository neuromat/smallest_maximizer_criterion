import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import Sample
from g4l.estimators.bic import BIC


def test_sample_likelihood():
    max_depth = 4
    X = Sample('examples/linguistic_case_study/folha.txt', [0, 1, 2, 3, 4])
    tree0 = BIC(0, max_depth).fit(X).context_tree
    L0, buf = tree0.sample_likelihood(X)
    assert(L0 == -76845.38611413536)

    c = 1.536489
    tree = BIC(c, max_depth).fit(X).context_tree
    L1, buf2 = tree.sample_likelihood(X, buf=buf)
    assert(L1 == -77309.69143269697)
    assert(buf == buf2)
