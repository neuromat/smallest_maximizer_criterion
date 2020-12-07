import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import Sample
import g4l.models.integrity as integrity
from g4l.estimators.bic import BIC

max_depth = 4
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
ctm_estimator = BIC(1.536489, max_depth).fit(X)


def test_suffix_property():
    assert(integrity.satisfies_suffix_property(ctm_estimator.context_tree))


def test_completeness():
    assert(integrity.satisfies_completeness(ctm_estimator.context_tree))
