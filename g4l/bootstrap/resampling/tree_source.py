import random
import pandas as pd
import numpy as np
from .base import ResamplingBase

class TreeSourceResampling(ResamplingBase):
  def __init__(self, tree, A):
    self.tree = tree
    self.A = A

  def __iter(self, num_resamples, size):
    for i in range(num_resamples):
      yield i, self.tree.generate_sample(size, self.A)
