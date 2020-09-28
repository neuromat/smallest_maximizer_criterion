import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import persistence

def test_load_samples():
  filename = os.path.abspath('./examples/example2/samples/model1_5000.mat')
  for i in persistence.iterate_from_mat(filename, 'model1_5000', [0, 1]):
    assert(len(i.data)==5000)
