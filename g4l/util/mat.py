import os
import math
import scipy.io as sio
from g4l.data import Sample


class MatSamples:
    """
    Loads a set of samples from a Matlab file (.mat)
    Returns: a list of Sample objects
    """

    def __init__(self, folder, model_name,
                 sample_size, A,
                 max_depth):
        self.folder = folder
        self.model_name = model_name
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.A = A
        self.key = '%s_%s' % (model_name, sample_size)
        self.filename = '%s/%s.mat' % (self.folder, self.key)

    def fetch_samples(self, max_samples=math.inf):
        i = -1
        prm = self.filename, self.key, self.A, self.max_depth
        for s in iterate_from_mat(prm):
            if i > max_samples:
                break
            i += 1
            yield i, s

    def sample_by_idx(self, idx):
        samples = [x for x in self.fetch_samples(idx)]
        return samples[idx][1]


def iterate_from_mat(filename, key, A,
                     max_depth, cache_dir=None):

    """ Loads and yields samples from file """

    arr = sio.loadmat(filename)[key]
    for i, s in enumerate(arr):
        dt = ''.join(s.astype(str))
        cache_file = os.path.join(cache_dir or './tmp', 'mat', str(i))
        yield Sample(None, A,
                     max_depth,
                     data=dt,
                     cache_file=cache_file)
