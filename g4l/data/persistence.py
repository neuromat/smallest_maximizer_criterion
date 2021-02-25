import scipy.io as sio
from g4l.data import Sample


def iterate_from_mat(filename, key, A, max_depth, scan_offset=0):
    """ Loads samples from file """
    arr = sio.loadmat(filename)[key]
    for s in arr:
        dt = ''.join(s.astype(str))
        yield Sample(None, A, max_depth, data=dt, scan_offset=scan_offset)
