import random

from g4l.smc_lcb import SMC
from tests.fixtures.bootstrap import bootstrap
from tests.fixtures.sample import *


@pytest.mark.parametrize('num_bs_resamples', [50])
def test_smc_lcb(bootstrap, sample, tmp_path):
    correct_tree = '000 100 10 1'
    random.seed(12345)
    smc = SMC(bootstrap, cache_dir=str(tmp_path))
    smc.fit(sample)
    assert smc.optimal_tree.to_str(reverse=True) == correct_tree
    #import code; code.interact(local=dict(globals(), **locals()))



