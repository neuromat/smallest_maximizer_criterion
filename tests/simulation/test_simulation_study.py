import os
import sys
import pytest
sys.path.insert(0, os.path.abspath('.'))
from examples.example2 import simulation

def test_simulation_study():
  simulation.run_simulation('model1')
  simulation.run_simulation('model2')
