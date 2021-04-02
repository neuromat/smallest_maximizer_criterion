import pytest
from g4l.sample import Sample


@pytest.fixture()
def filename():
    yield 'tests/files/sample20000.txt'


@pytest.fixture()
def cache_file():
    yield 'tests/tmp/sample.cache'


@pytest.fixture()
def A():
    yield [0, 1]


@pytest.fixture()
def max_depth():
    yield 6


@pytest.fixture()
def sample(filename, A, max_depth):
    yield Sample(filename, A, max_depth)


@pytest.fixture()
def cached_sample(filename, cache_file, A, max_depth):
    yield Sample(filename, A, max_depth, cache_file=cache_file)

