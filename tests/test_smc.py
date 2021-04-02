from g4l.smc_bic import SMC
from tests.fixtures.sample import *
from tests.fixtures.bootstrap import bootstrap, cache_folder
import random


@pytest.mark.parametrize('num_bs_resamples', [50])
def test_smc_bic(bootstrap, sample):
    random.seed(12345)
    smc = SMC(bootstrap, penalty_interval=(0.01, 800))
    smc.fit(sample)
    assert smc.optimal_tree.to_str() == '000 1 10 100'
    #import code; code.interact(local=dict(globals(), **locals()))



