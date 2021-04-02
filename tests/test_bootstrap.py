from tests.fixtures.sample import *
from tests.fixtures.bootstrap import *


def test_resample_generation(bootstrap, sample):
    size = 1000
    num_resamples = 3
    f = bootstrap.get_resamples(sample, size, num_cores=0)
    content = open(f, 'r').read()[:-1]
    assert sum([len(x) for x in content.split('\n')]) == size * num_resamples
    #import code; code.interact(local=dict(globals(), **locals()))

