import os
import random

from g4l.smc_bic import SMC
from tests.fixtures.bootstrap import bootstrap
from tests.fixtures.sample import *


@pytest.mark.parametrize('num_bs_resamples', [100])
def test_smc_bic(bootstrap, sample, tmp_path):
    correct_tree = '000 100 10 1'
    random.seed(12345)
    smc = SMC(bootstrap, penalty_interval=(0.01, 800), cache_dir=str(tmp_path))
    smc.fit(sample)
    assert smc.optimal_tree.to_str(reverse=True) == correct_tree
    out_path = os.path.join(str(tmp_path), 'out')
    smc.save_output(sample, out_path)
    html_report = open(os.path.join(out_path, 'report.html')).read()
    assert html_report.find(correct_tree) >= 0
    #import code; code.interact(local=dict(globals(), **locals()))


@pytest.mark.parametrize('num_bs_resamples', [10])
def test_smc_bic_pl(bootstrap, sample, tmp_path):
    smc = SMC(bootstrap, perl_compatible=True,
              penalty_interval=(0.05, 100),
              cache_dir=str(tmp_path))
    smc.context_trees = []
    smc.estimate_trees(sample)

    #import code; code.interact(local=dict(globals(), **locals()))





