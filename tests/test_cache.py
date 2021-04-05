import os
import random

from g4l.smc_bic import SMC
from tests.fixtures.bootstrap import bootstrap
from g4l.util.caching import smc as caching
from tests.fixtures.sample import *


@pytest.mark.parametrize('num_bs_resamples', [20])
def test_smc_caching(bootstrap, sample, tmp_path):
    smc = SMC(bootstrap, penalty_interval=(0.01, 800), cache_dir=str(tmp_path))
    smc2 = SMC(bootstrap, penalty_interval=(0.01, 800), cache_dir=str(tmp_path))
    smc.fit(sample)
    caching.save_cache(smc, sample)
    caching.load_cache(smc2, sample)
    t1 = [t.to_str() for t in smc.context_trees]
    t2 = [t.to_str() for t in smc2.context_trees]
    assert [t for t in t1 if t in t2] == t1
    assert [t for t in t2 if t in t1] == t2


