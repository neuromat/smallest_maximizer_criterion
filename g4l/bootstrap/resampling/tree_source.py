import random
import pandas as pd
import numpy as np
from .base import ResamplingBase

class TreeSourceResampling(ResamplingBase):
  def __init__(self, tree, sample):
    self.tree = tree
    self.sample = sample

  def generate(self, resample_size, file):
    resample = self.tree.generate_sample(resample_size, self.sample.A)
    with open(file, 'a') as f:
      f.write(resample[:resample_size] + '\n')

    #slices = np.array(np.char.split([data], str(self.renewal_point)[0]))[0]
    #num_slices = len(slices)
    #idxs = np.random.randint(num_slices, size=resample_size)
    #resample = ''.join(['%s%s' % (slices[idx], self.renewal_point) for idx in idxs])
    #with open(file, 'a') as f:
    #  f.write(resample[:resample_size] + '\n')

