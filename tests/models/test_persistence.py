import os
import sys
import pytest
import tempfile
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import Sample
from g4l.models import ContextTree
from g4l.estimators.ctm import CTM

max_depth = 4
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
c = 1.536489


def test_save_and_load_model():
    t1 = CTM(c, max_depth).fit(X).context_tree

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = tmpdirname + '/saved_model.h5'
        t1.save(filename)
        t2 = ContextTree.load_from_file(filename)
        assert(t1.max_depth == t2.max_depth)
        assert(t1.df.freq.sum() == t2.df.freq.sum())
        assert(t1.df.likelihood.sum() == t2.df.likelihood.sum())
        assert(len(t1.df) == len(t2.df))
        assert(t1.transition_probs.freq.sum() == t2.transition_probs.freq.sum())
        assert(t1.sample.filename == t2.sample.filename)
        assert(t1.sample.separator == t2.sample.separator)
        assert(t1.sample.A == t2.sample.A)
