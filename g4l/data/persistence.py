import h5py
import pandas as pd
import scipy.io as sio
from g4l.data import Sample

def iterate_from_mat(filename, key, A, max_depth):
  """ Loads samples from file """
  arr = sio.loadmat(filename)[key]
  for s in arr:
    dt = ''.join(s.astype(str))
    yield Sample(None, A, max_depth, data=dt)
