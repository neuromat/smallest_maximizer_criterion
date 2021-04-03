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
def sample_pl_compat(filename, A, max_depth):
    yield Sample(filename, A, max_depth, perl_compatible=True)


@pytest.fixture()
def sample_bp():
    A = [0, 1, 2, 3, 4]
    max_depth = 4
    filename = 'examples/linguistic_case_study/folha.txt'
    yield Sample(filename, A, max_depth, subsamples_separator='>')


@pytest.fixture()
def sample_ep():
    A = [0, 1, 2, 3, 4]
    max_depth = 4
    filename = 'examples/linguistic_case_study/folha.txt'
    yield Sample(filename, A, max_depth, subsamples_separator='>')


@pytest.fixture()
def cached_sample(filename, cache_file, A, max_depth):
    yield Sample(filename, A, max_depth, cache_file=cache_file)

