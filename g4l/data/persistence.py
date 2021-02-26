import os
import scipy.io as sio
from g4l.data import Sample


def iterate_from_mat(filename, key, A, max_depth, scan_offset=0,
                     cache_dir=None):
    """ Loads samples from file """
    arr = sio.loadmat(filename)[key]
    for i, s in enumerate(arr):
        dt = ''.join(s.astype(str))
        cache_file = os.path.join(cache_dir, 'mat', str(i))
        yield Sample(None, A,
                     max_depth,
                     data=dt,
                     scan_offset=scan_offset,
                     cache_file=cache_file)
