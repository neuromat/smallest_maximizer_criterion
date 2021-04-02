import pytest
import os
from g4l.bootstrap import Bootstrap


@pytest.fixture()
def num_bs_resamples():
    yield 3


@pytest.fixture()
def bootstrap(num_bs_resamples, tmp_path):
    renewal_point = '1'
    yield Bootstrap(os.path.join(tmp_path, 'bs'),
                    num_bs_resamples,
                    renewal_point)
