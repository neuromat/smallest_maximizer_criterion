import random
import pandas as pd
import numpy as np
from .base import ResamplingBase


class BlockResampling(ResamplingBase):
  def __init__(self, sample, renewal_point=None):
    self.renewal_point = renewal_point
    self.sample = sample

  def generate(self, resample_size, file):
    data = self.sample.data
    slices = np.array(np.char.split([data], str(self.renewal_point)[0]))[0]
    num_slices = len(slices)
    idxs = np.random.randint(num_slices, size=resample_size)
    resample = ''.join(['%s%s' % (slices[idx], self.renewal_point) for idx in idxs])
    with open(file, 'a') as f:
      f.write(resample[:resample_size] + '\n')


  def __most_frequent_substring(self, source_sample):
    # TODO: implementation
    return '0'
