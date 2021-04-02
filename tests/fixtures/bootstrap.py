import pytest
from g4l.bootstrap import Bootstrap


@pytest.fixture()
def num_bs_resamples():
    yield 3


@pytest.fixture()
def cache_folder():
    yield 'tests/tmp/bootstrap'


@pytest.fixture()
def bootstrap(num_bs_resamples, cache_folder):
    renewal_point = '1'
    yield Bootstrap(cache_folder, num_bs_resamples, renewal_point)
