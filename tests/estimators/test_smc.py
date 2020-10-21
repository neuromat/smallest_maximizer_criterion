import os
import sys
import pytest
import logging
sys.path.insert(0, os.path.abspath('.'))
from g4l.data import Sample
import g4l.models.integrity as integrity
from g4l.estimators.smc import SMC

max_depth = 4
X = Sample('examples/example1/folha.txt', [0, 1, 2, 3, 4])
smc_estimator = SMC(max_depth, penalty_interval=(10, 210), epsilon=5).fit(X)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #logging.FileHandler("examples/example1/debug.log"),
        logging.StreamHandler()
    ]
)


def test_suffix_property():
    for t in smc_estimator.context_trees:
      assert(integrity.satisfies_suffix_property(t))


def test_completeness():
    for t in smc_estimator.context_trees:
      assert(integrity.satisfies_completeness(t))
